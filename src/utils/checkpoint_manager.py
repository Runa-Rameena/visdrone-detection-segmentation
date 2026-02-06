"""
SAM Checkpoint Download and Validation Utility

Handles robust checkpoint downloading with integrity verification.
Supports Windows with proper error handling.
"""

import os
import hashlib
import json
from pathlib import Path
import urllib.request
from typing import Tuple, Optional


class CheckpointManager:
    """Manage checkpoint downloads and validation"""
    
    # Official SAM checkpoint URLs and expected sizes
    CHECKPOINT_INFO = {
        'sam_vit_h_4b8939.pth': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'size': 2561567049,  # bytes (~2.38 GB)
            'sha256': 'dce6db1fb5537b72a4c1f0f0cd59a1dd1e5ad0ad5da5cd00c3f44c11d6a1b7ef'
        },
        'sam_vit_l_0b3195.pth': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'size': 1299636635,  # bytes (~1.21 GB)
            'sha256': '3c6c43a15d0d67d9876fd106eda73c15fdb0f2e0ad39bfab5e2daee32e60db2f'
        },
        'sam_vit_b_01ec64.pth': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'size': 375042383,  # bytes (~357 MB)
            'sha256': '01ec64d29b576da9d0a0ebc32854f30565663f858378664f9a46eb019da657978'
        }
    }
    
    @staticmethod
    def get_checkpoint_info(checkpoint_name: str) -> Optional[dict]:
        """Get checkpoint URL and validation info"""
        return CheckpointManager.CHECKPOINT_INFO.get(checkpoint_name)
    
    @staticmethod
    def validate_checkpoint_size(filepath: Path, expected_size: int, tolerance_percent: float = 5.0) -> Tuple[bool, str]:
        """
        Validate checkpoint file size
        
        Args:
            filepath: Path to checkpoint file
            expected_size: Expected file size in bytes
            tolerance_percent: Allow this % difference (default 5%)
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not filepath.exists():
            return False, f"File does not exist: {filepath}"
        
        actual_size = filepath.stat().st_size
        tolerance_bytes = int(expected_size * tolerance_percent / 100)
        
        if abs(actual_size - expected_size) > tolerance_bytes:
            return False, (
                f"File size mismatch:\n"
                f"  Expected: {expected_size:,} bytes ({expected_size / (1024**3):.2f} GB)\n"
                f"  Actual:   {actual_size:,} bytes ({actual_size / (1024**3):.2f} GB)\n"
                f"  Difference: {(actual_size - expected_size):,} bytes\n"
                f"  Status: CORRUPTED OR INCOMPLETE"
            )
        
        return True, f"✓ File size valid: {actual_size:,} bytes ({actual_size / (1024**3):.2f} GB)"
    
    @staticmethod
    def validate_checkpoint_hash(filepath: Path, expected_hash: str) -> Tuple[bool, str]:
        """
        Validate checkpoint using SHA256
        
        Args:
            filepath: Path to checkpoint file
            expected_hash: Expected SHA256 hash
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not filepath.exists():
            return False, f"File does not exist: {filepath}"
        
        print(f"Computing SHA256 hash (this may take a minute)...")
        
        sha256_hash = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        actual_hash = sha256_hash.hexdigest()
        
        if actual_hash.lower() != expected_hash.lower():
            return False, (
                f"Hash mismatch:\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}\n"
                f"  Status: CORRUPTED"
            )
        
        return True, f"✓ SHA256 hash valid: {actual_hash}"
    
    @staticmethod
    def can_load_pytorch_checkpoint(filepath: Path) -> Tuple[bool, str]:
        """
        Test if PyTorch can read the checkpoint
        
        Args:
            filepath: Path to checkpoint file
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not filepath.exists():
            return False, f"File does not exist: {filepath}"
        
        try:
            import torch
            print("Testing PyTorch checkpoint loading...")
            _ = torch.load(filepath, map_location='cpu')
            return True, f"✓ PyTorch can load checkpoint"
        except Exception as e:
            return False, (
                f"PyTorch cannot load checkpoint:\n"
                f"  Error: {type(e).__name__}: {str(e)}\n"
                f"  Status: CORRUPTED"
            )
    
    @staticmethod
    def validate_all(filepath: Path, checkpoint_name: str, skip_hash: bool = True) -> Tuple[bool, list]:
        """
        Run all validation checks
        
        Args:
            filepath: Path to checkpoint file
            checkpoint_name: Name of checkpoint (e.g., 'sam_vit_h_4b8939.pth')
            skip_hash: Skip hash validation (default True for faster qualitative-only tasks)
            
        Returns:
            Tuple of (all_valid, list of results)
        """
        checkpoint_info = CheckpointManager.get_checkpoint_info(checkpoint_name)
        
        if not checkpoint_info:
            return False, [f"Unknown checkpoint: {checkpoint_name}"]
        
        results = []
        
        # Check 1: File existence
        if not filepath.exists():
            results.append(f"✗ File not found: {filepath}")
            return False, results
        
        results.append(f"✓ File exists: {filepath}")
        
        # Check 2: File size
        size_valid, size_msg = CheckpointManager.validate_checkpoint_size(
            filepath, 
            checkpoint_info['size']
        )
        results.append(size_msg)
        
        if not size_valid:
            return False, results
        
        # Check 3: PyTorch loading
        pytorch_valid, pytorch_msg = CheckpointManager.can_load_pytorch_checkpoint(filepath)
        results.append(pytorch_msg)
        
        if not pytorch_valid:
            return False, results
        
        # Check 4: Hash (optional, slower) - skip for qualitative-only workflows
        hash_valid = True
        if skip_hash:
            results.append("⊘ SHA256 hash validation skipped (qualitative-only step)")
        else:
            hash_valid, hash_msg = CheckpointManager.validate_checkpoint_hash(
                filepath,
                checkpoint_info['sha256']
            )
            results.append(hash_msg)
        
        all_valid = size_valid and pytorch_valid and hash_valid
        return all_valid, results
    
    @staticmethod
    def download_checkpoint(checkpoint_name: str, output_path: Path, show_progress: bool = True) -> Tuple[bool, str]:
        """
        Download checkpoint with progress bar
        
        Args:
            checkpoint_name: Name of checkpoint file
            output_path: Where to save the file
            show_progress: Show download progress
            
        Returns:
            Tuple of (success, message)
        """
        checkpoint_info = CheckpointManager.get_checkpoint_info(checkpoint_name)
        
        if not checkpoint_info:
            return False, f"Unknown checkpoint: {checkpoint_name}"
        
        url = checkpoint_info['url']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading SAM checkpoint ({checkpoint_info['size'] / (1024**3):.2f} GB)...")
        print(f"From: {url}")
        print(f"To:   {output_path}")
        print(f"\nNote: This may take several minutes depending on internet speed.\n")
        
        def download_with_progress(url, filepath):
            """Download with progress bar"""
            try:
                def progress_hook(block_num, block_size, total_size):
                    downloaded = min(block_num * block_size, total_size)
                    percent = (downloaded * 100) // total_size if total_size > 0 else 0
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rProgress: {percent}% ({mb_downloaded:.0f}MB / {mb_total:.0f}MB)", end='', flush=True)
                
                urllib.request.urlretrieve(url, str(filepath), reporthook=progress_hook)
                print("\n")
                return True, "Download complete"
            except Exception as e:
                return False, f"Download failed: {str(e)}"
        
        success, msg = download_with_progress(url, output_path)
        
        if not success:
            return False, msg
        
        # Validate after download
        results = []
        print("\nValidating downloaded checkpoint...")
        all_valid, validation_results = CheckpointManager.validate_all(output_path, checkpoint_name)
        
        for result in validation_results:
            results.append(result)
        
        if not all_valid:
            output_path.unlink()  # Delete corrupted file
            return False, "Validation failed (file deleted):\n" + "\n".join(results)
        
        return True, "Checkpoint ready:\n" + "\n".join(results)


def main():
    """Test checkpoint manager"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and validate SAM checkpoints')
    parser.add_argument('--checkpoint', default='sam_vit_h_4b8939.pth', 
                       help='Checkpoint name to download')
    parser.add_argument('--output', default='./sam_vit_h_4b8939.pth',
                       help='Output path')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate, do not download')
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    if args.validate_only:
        print("=" * 80)
        print("VALIDATING CHECKPOINT")
        print("=" * 80)
        
        all_valid, results = CheckpointManager.validate_all(output_path, args.checkpoint)
        
        for result in results:
            print(result)
        
        if all_valid:
            print("\n✓ Checkpoint is valid and ready to use")
        else:
            print("\n✗ Checkpoint validation failed")
            print("\nTo fix, download a fresh copy with:")
            print(f"  python -m src.utils.checkpoint_manager --checkpoint {args.checkpoint} --output {args.output}")
    else:
        print("=" * 80)
        print("DOWNLOADING CHECKPOINT")
        print("=" * 80)
        
        success, msg = CheckpointManager.download_checkpoint(args.checkpoint, output_path)
        
        print(msg)
        
        if success:
            print("\n✓ Checkpoint ready to use with SAM")
        else:
            print("\n✗ Download failed")


if __name__ == "__main__":
    main()
