import json
import logging
from typing import Dict, Any
import os

class ConfigManager:
    """Enterprise-level configuration manager"""
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file with validation"""
        try:
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['training_data_size', 'model_type', 'random_seed']
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Required configuration field missing: {field}")
            
            return config
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {str(e)}")
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=level, format=format_str)
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific parameters"""
        return self.config.get('model_parameters', {}).get(model_type, {})
 