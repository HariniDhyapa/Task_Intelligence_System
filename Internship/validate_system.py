#!/usr/bin/env python3
"""
Enterprise System Validation Script
Validates all components before deployment
"""

import os
import json
import pickle
import sys
import logging
from typing import Dict, List, Any
import numpy as np
from config_manager import ConfigManager

class SystemValidator:
    """Comprehensive system validation"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.setup_logging()
    
    def setup_logging(self):
        """Setup validation logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SystemValidator')
    
    def validate_configuration(self) -> bool:
        """Validate configuration management"""
        self.logger.info("Validating configuration...")
        
        try:
            # Test ConfigManager initialization
            config_manager = ConfigManager()
            
            # Required configuration fields
            required_fields = [
                'task_limit_per_call', 'model_type', 'random_seed',
                'task_bank_path', 'model_output_path', 'training_dataset_path'
            ]
            
            for field in required_fields:
                if config_manager.get(field) is None:
                    self.errors.append(f"Missing required config field: {field}")
            
            # Validate model parameters
            model_type = config_manager.get('model_type', 'random_forest')
            model_params = config_manager.get_model_params(model_type)
            if not model_params:
                self.errors.append(f"Missing model parameters for type: {model_type}")
            
            self.logger.info("✓ Configuration validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Configuration validation failed: {str(e)}")
            return False
    
    def validate_files(self) -> bool:
        """Validate required files exist"""
        self.logger.info("Validating required files...")
        
        required_files = [
            'config.json',
            'task_bank.json',
            'main.py',
            'task_engine.py','config_manager.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            self.errors.extend([f"Missing required file: {f}" for f in missing_files])
            return False
        
        self.logger.info("✓ All required files present")
        return True
    
    def validate_task_bank(self) -> bool:
        """Validate task bank structure"""
        self.logger.info("Validating task bank...")
        
        try:
            with open('task_bank.json', 'r') as f:
                task_bank = json.load(f)
            
            if not task_bank:
                self.errors.append("Task bank is empty")
                return False
            
            # Validate task structure
            required_task_fields = ['title', 'type', 'karma_reward', 'category']
            for task_id, task in task_bank.items():
                for field in required_task_fields:
                    if field not in task:
                        self.errors.append(f"Task {task_id} missing field: {field}")
                
                # Validate karma reward range
                karma_reward = task.get('karma_reward', 0)
                if not (5 <= karma_reward <= 20):
                    self.warnings.append(f"Task {task_id} karma_reward {karma_reward} outside range [5-20]")
            
            self.logger.info(f"✓ Task bank validated ({len(task_bank)} tasks)")
            return True
            
        except Exception as e:
            self.errors.append(f"Task bank validation failed: {str(e)}")
            return False
    def validate_api_structure(self) -> bool:
        """Validate API code structure"""
        self.logger.info("Validating API structure...")
        
        try:
            # Test imports
            from main import app
            from task_engine import TaskIntelligenceEngine
            from config_manager import ConfigManager
            
            # Check FastAPI app structure
            routes = [route.path for route in app.routes]
            required_routes = ['/generate-tasks', '/health', '/version', '/tasks/available']
            
            for route in required_routes:
                if route not in routes:
                    self.errors.append(f"Missing API route: {route}")
            
            self.logger.info("✓ API structure validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"API structure validation failed: {str(e)}")
            return False
    def run_comprehensive_validation(self) -> bool:
        """Run all validation checks"""
        self.logger.info("=" * 50)
        self.logger.info("STARTING COMPREHENSIVE SYSTEM VALIDATION")
        self.logger.info("=" * 50)
        
        validations = [
            self.validate_files,
            self.validate_configuration,
            self.validate_task_bank,
            self.validate_api_structure
        ]
        
        all_passed = True
        for validation in validations:
            try:
                if not validation():
                    all_passed = False
            except Exception as e:
                self.errors.append(f"Validation error: {str(e)}")
                all_passed = False
        
        # Print results
        self.logger.info("=" * 50)
        self.logger.info("VALIDATION RESULTS")
        self.logger.info("=" * 50)
        
        if self.errors:
            self.logger.error(f"❌ VALIDATION FAILED - {len(self.errors)} errors found:")
            for error in self.errors:
                self.logger.error(f"  • {error}")
        
        if self.warnings:
            self.logger.warning(f"⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                self.logger.warning(f"  • {warning}")
        
        if all_passed and not self.errors:
            self.logger.info("✅ ALL VALIDATIONS PASSED")
            self.logger.info("System is ready for deployment!")
        else:
            self.logger.error("❌ SYSTEM NOT READY")
            self.logger.error("Please fix the errors above before deployment")
        
        return all_passed and not self.errors

def main():
    """Main validation entry point"""
    validator = SystemValidator()
    success = validator.run_comprehensive_validation()
    
    if not success:
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 SYSTEM VALIDATION COMPLETED SUCCESSFULLY!")
    print("Your Task Intelligence Engine is ready for deployment.")
    print("=" * 50)

if __name__ == "__main__":
    main()
