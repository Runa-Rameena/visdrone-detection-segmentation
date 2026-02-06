#!/usr/bin/env python3
"""
MASTER SCRIPT: Run Complete Pipeline
Executes all steps from preprocessing to report generation
"""

import sys
import subprocess
from pathlib import Path

def run_script(script_path: str, description: str, allow_failure: bool = False) -> bool:
    """
    Run a Python script and return success status
    
    Args:
        script_path: Path to script
        description: Description of what script does
        allow_failure: If True, don't stop pipeline on failure
        
    Returns:
        True if successful or if failure is allowed
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"\n✗ {description} failed with error code {e.returncode}"
        
        if allow_failure:
            print(error_msg)
            print(f"⚠️  This is an optional step - continuing pipeline...")
            return True
        else:
            print(error_msg)
            return False
    except Exception as e:
        error_msg = f"\n✗ {description} failed: {str(e)}"
        
        if allow_failure:
            print(error_msg)
            print(f"⚠️  This is an optional step - continuing pipeline...")
            return True
        else:
            print(error_msg)
            return False

def main():
    print("=" * 80)
    print("VISDRONE DETECTION & SEGMENTATION - COMPLETE PIPELINE")
    print("=" * 80)
    print("\nThis script will run the entire pipeline:")
    print("  1. Preprocess dataset")
    print("  2. Run YOLO detection")
    print("  3. Run RT-DETR detection")
    print("  4. Run SAM segmentation")
    print("  5. Evaluate metrics")
    print("  6. Generate final report")
    
    response = input("\nDo you want to continue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Get scripts directory
    scripts_dir = Path(__file__).parent
    
    # Define pipeline steps
    steps = [
        (scripts_dir / "02_preprocess_data.py", "Data Preprocessing", False),
        (scripts_dir / "03_run_yolo.py", "YOLO Inference", False),
        (scripts_dir / "04_run_rfdetr.py", "RT-DETR Inference", False),
        (scripts_dir / "05_run_sam.py", "SAM Segmentation (Optional)", True),  # Allow SAM to fail
        (scripts_dir / "06_evaluate_metrics.py", "Metrics Evaluation", False),
        (scripts_dir / "07_generate_report.py", "Report Generation", False),
    ]
    
    # Track progress
    completed = []
    failed = []
    skipped = []
    
    # Run each step
    for script_path, description, allow_failure in steps:
        if not script_path.exists():
            print(f"\nError: Script not found: {script_path}")
            failed.append(description)
            if not allow_failure:
                break
            continue
        
        success = run_script(str(script_path), description, allow_failure=allow_failure)
        
        if success:
            completed.append(description)
        else:
            failed.append(description)
            
            # Ask if user wants to continue after failure (if not optional)
            if not allow_failure:
                response = input(f"\n{description} failed. Continue with next step? (y/n): ")
                if response.lower() != 'y':
                    break
            else:
                skipped.append(description)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    
    print(f"\nCompleted ({len(completed)}/{len(steps)}):")
    for step in completed:
        print(f"  ✓ {step}")
    
    if skipped:
        print(f"\nSkipped (Optional - {len(skipped)}):")
        for step in skipped:
            print(f"  ⊘ {step}")
            print(f"     Note: To fix, download SAM checkpoint or check repair guide")
    
    if failed:
        print(f"\nFailed ({len(failed)}/{len(steps)}):")
        for step in failed:
            print(f"  ✗ {step}")
    else:
        print(f"\n🎉 All required steps completed successfully!")
        if skipped:
            print(f"\nℹ️  {len(skipped)} optional step(s) skipped (SAM segmentation)")
            print(f"   Metrics evaluation completed without SAM")
        print("\nCheck these outputs:")
        print(f"  • Detections: results/detections/")
        print(f"  • Metrics: results/metrics/")
        print(f"  • Report: results/")
        if not skipped:
            print(f"  • Segmentations: results/segmentations/")
        print("  - RESULTS_REPORT.md (comprehensive report)")
        print("  - results/metrics/ (evaluation metrics)")
        print("  - results/visualizations/ (comparison plots)")
        print("  - results/detections/ (detection images)")
        print("  - results/segmentations/ (segmentation masks)")

if __name__ == "__main__":
    main()