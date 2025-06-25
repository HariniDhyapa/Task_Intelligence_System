import json
import pickle
import numpy as np
from typing import Dict, List, Any
import logging
from config_manager import ConfigManager

class TaskIntelligenceEngine:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = config_manager.logger
        self.task_bank = self._load_task_bank()
        self.model_data = self._load_model()
        
    def _load_config(self) -> Dict:
        """Load configuration - now handled by ConfigManager"""
        return self.config.config
        
    def _load_task_bank(self) -> Dict:
        """Load task bank from task_bank.json"""
        try:
            task_bank_path = self.config.get('task_bank_path', 'task_bank.json')
            with open(task_bank_path, 'r') as f:
                task_bank = json.load(f)
            self.logger.info(f"Loaded {len(task_bank)} tasks from task bank")
            return task_bank
        except FileNotFoundError:
            self.logger.error("Task bank file not found")
            raise FileNotFoundError("Task bank file not found. Please ensure task_bank.json exists.")
        except Exception as e:
            self.logger.error(f"Error loading task bank: {e}")
            raise
    
    def _load_model(self) -> Dict:
        """Load trained model and preprocessing components"""
        try:
            model_path = self.config.get('model_output_path', 'model.pkl')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.logger.info("Model loaded successfully")
                return model_data
        except FileNotFoundError:
            self.logger.error("Model file not found. Please train the model first using train_model.py")
            raise FileNotFoundError("Model not found. Run 'python train_model.py' first.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def _extract_features(self, profile: Dict, activity: Dict, context: str) -> np.ndarray:
        """Extract features from user profile and activity"""
        try:
            # Validate inputs
            required_profile_fields = ['branch', 'year', 'goal_tags', 'interests']
            for field in required_profile_fields:
                if field not in profile:
                    raise ValueError(f"Missing required profile field: {field}")
            
            # Categorical features
            branch_encoded = self.model_data['label_encoders']['branch'].transform([profile.get('branch', 'CSE')])[0]
            context_encoded = self.model_data['label_encoders']['context'].transform([context])[0]
            
            # TF-IDF features
            goal_text = ' '.join(profile.get('goal_tags', []))
            interest_text = ' '.join(profile.get('interests', []))
            
            goal_features = self.model_data['tfidf_goals'].transform([goal_text]).toarray()[0]
            interest_features = self.model_data['tfidf_interests'].transform([interest_text]).toarray()[0]
            
            # Combine all features
            feature_vector = [
                branch_encoded,
                context_encoded,
                profile.get('year', 1),
                activity.get('posts_today', 0),
                activity.get('quiz_attempts', 0),
                activity.get('karma_spent_today', 0),
                activity.get('karma_earned_today', 0),
                activity.get('buddies_interacted', 0),
                activity.get('login_streak', 0),
                activity.get('karma_earned_today', 0)  # karma_today
            ]
            
            # Add TF-IDF features
            feature_vector.extend(goal_features)
            feature_vector.extend(interest_features)
            
            return np.array(feature_vector).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
    
    def _filter_valid_tasks(self, task_history: List[str], context: str) -> List[str]:
        """Filter tasks based on history and context"""
        valid_tasks = []
        excluded_types = self.config.get('excluded_types', [])
        min_days_between_repeat = self.config.get('min_days_between_repeat', 3)
        
        for task_id, task in self.task_bank.items():
            # Skip if task was recently completed
            if task_id in task_history:
                continue
            
            # Skip excluded types
            if task.get('type') in excluded_types:
                continue
            
            # Context matching (allow some flexibility)
            task_type = task.get('type', 'any')
            if context != 'any' and task_type not in [context, 'any']:
                continue
            
            valid_tasks.append(task_id)
        
        return valid_tasks
    
    def generate_tasks(self, user_id: str, context: str, profile: Dict, 
                      activity: Dict, task_history: List[str]) -> Dict:
        """Generate task recommendations for a user"""
        try:
            # Filter valid tasks
            valid_task_ids = self._filter_valid_tasks(task_history, context)
            
            if not valid_task_ids:
                return {
                    "user_id": user_id,
                    "context": context,
                    "suggested_tasks": [],
                    "status": "no_tasks"
                }
            
            # Extract features and make predictions
            features = self._extract_features(profile, activity, context)
            features_scaled = self.model_data['scaler'].transform(features)
            
            # Get model predictions
            predictions = self.model_data['model'].predict(features_scaled)[0]
            
            # Get recommended task IDs
            recommended_indices = np.where(predictions == 1)[0]
            recommended_task_ids = self.model_data['mlb'].classes_[recommended_indices]
            
            # Filter recommended tasks to only include valid ones
            final_recommended_tasks = [task_id for task_id in recommended_task_ids if task_id in valid_task_ids]
            
            # If no tasks recommended or too few, add some based on simple rules
            if len(final_recommended_tasks) < 2:
                # Add tasks based on simple heuristics
                for task_id in valid_task_ids:
                    if task_id not in final_recommended_tasks:
                        task = self.task_bank[task_id]
                        # Simple matching logic
                        if any(goal in profile.get('goal_tags', []) for goal in task.get('goal_tags', [])):
                            final_recommended_tasks.append(task_id)
                            if len(final_recommended_tasks) >= 3:
                                break
            
            # Limit results based on configuration
            limit = self.config.get('task_limit_per_call', 3)
            final_recommended_tasks = final_recommended_tasks[:limit]
            
            # Create task recommendations
            task_recommendations = []
            reward_range = self.config.get('reward_range', [5, 20])
            
            for task_id in final_recommended_tasks:
                task = self.task_bank[task_id]
                # Ensure karma reward is within configured range
                karma_reward = max(reward_range[0], min(task['karma_reward'], reward_range[1]))
                
                task_recommendations.append({
                    'task_id': task_id,
                    'title': task['title'],
                    'score': 0.85,  # Fixed score since we're using classification
                    'karma_reward': karma_reward,
                    'type': task['type'],
                    'category': task['category'],
                    'goal_tags': task.get('goal_tags', [])
                })
            
            self.logger.info(f"Generated {len(task_recommendations)} tasks for user {user_id}")
            
            return {
                "user_id": user_id,
                "context": context,
                "suggested_tasks": task_recommendations,
                "status": "generated"
            }
            
        except Exception as e:
            self.logger.error(f"Error generating tasks: {e}")
            return {
                "user_id": user_id,
                "context": context,
                "suggested_tasks": [],
                "status": "error"
            }
    
    def get_task_bank(self) -> Dict:
        """Return the complete task bank"""
        return self.task_bank