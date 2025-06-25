#!/usr/bin/env python3
"""
Simple Test Runner for Task Intelligence Engine
Runs basic functionality tests without requiring pytest
"""

import sys
import json
import tempfile
import os
from unittest.mock import MagicMock, patch

def test_config_manager():
    """Test ConfigManager functionality"""
    print("Testing ConfigManager...")
    
    # Create test config
    test_config = {
        "task_limit_per_call": 3,
        "model_type": "random_forest",
        "random_seed": 42,
        "training_data_size": 100,
        "logging": {"level": "INFO"}
    }
    
    # Test config loading
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        from config_manager import ConfigManager
        config_manager = ConfigManager(config_path)
        
        # Test basic access
        assert config_manager.get('task_limit_per_call') == 3
        assert config_manager.get('model_type') == 'random_forest'
        
        # Test dot notation
        assert config_manager.get('logging.level') == 'INFO'
        
        # Test default values
        assert config_manager.get('nonexistent', 'default') == 'default'
        
        print("✓ ConfigManager tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ConfigManager test failed: {e}")
        return False
    finally:
        os.unlink(config_path)

def test_api_health():
    """Test API health endpoint"""
    print("Testing API health endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        required_fields = ['status', 'model_loaded', 'config_loaded']
        for field in required_fields:
            assert field in data
        
        print("✓ API health endpoint test passed")
        return True
        
    except Exception as e:
        print(f"❌ API health test failed: {e}")
        return False

def test_task_generation_structure():
    """Test task generation request structure"""
    print("Testing task generation request structure...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test with invalid request
        response = client.post("/generate-tasks", json={})
        assert response.status_code == 422  # Validation error
        
        # Test with valid structure (might fail due to missing model, but structure should be validated)
        test_request = {
            "user_id": "test_user",
            "context": "duel",
            "profile": {
                "branch": "CSE",
                "year": 3,
                "goal_tags": ["SDE"],
                "interests": ["python"]
            },
            "activity": {
                "posts_today": 1,
                "quiz_attempts": 2
            },
            "task_history": []
        }
        
        response = client.post("/generate-tasks", json=test_request)
        # Should be 500 (server error) if model not loaded, not 422 (validation error)
        assert response.status_code in [200, 500]
        
        print("✓ Task generation structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Task generation structure test failed: {e}")
        return False

def test_file_existence():
    """Test that all required files exist"""
    print("Testing file existence...")
    
    required_files = [
        'main.py',
        'config_manager.py',
        'task_engine.py',
        'config.json',
        'task_bank.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✓ All required files exist")
    return True

def run_basic_tests():
    """Run all basic tests"""
    print("🧪 Running Basic Test Suite")
    print("=" * 50)
    
    tests = [
        test_file_existence,
        test_config_manager,
        test_api_health,
        test_task_generation_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL BASIC TESTS PASSED")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)