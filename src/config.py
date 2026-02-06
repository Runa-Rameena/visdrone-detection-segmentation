import yaml
import os
from pathlib import Path

class Config:
    def __init__(self, config_path='config.yaml'):
        """Load configuration from YAML file"""
        self.project_root = Path(__file__).parent.parent
        config_file = self.project_root / config_path
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def get_path(self, key):
        """Get absolute path for a configuration key"""
        path = self.get(key)
        if path:
            return self.project_root / path
        return None
    
    @property
    def dataset_raw_path(self):
        return self.get_path('dataset.raw_path')
    
    @property
    def raw_path(self):
        """Alias for dataset_raw_path for evaluation scripts"""
        return self.get_path('dataset.raw_path')
    
    @property
    def dataset_processed_path(self):
        return self.get_path('dataset.processed_path')
    
    @property
    def splits_path(self):
        return self.get_path('dataset.splits_path')
    
    @property
    def results_path(self):
        return self.project_root / 'results'
    
    @property
    def detections_path(self):
        return self.get_path('results.detections_path')
    
    @property
    def segmentations_path(self):
        return self.get_path('results.segmentations_path')
    
    @property
    def metrics_path(self):
        return self.get_path('results.metrics_path')
    
    @property
    def visualizations_path(self):
        return self.get_path('results.visualizations_path')

# Global config instance
config = Config()