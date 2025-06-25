import json
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, hamming_loss, jaccard_score
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')
from config_manager import ConfigManager

class TaskRecommendationModelTrainer:
    """Train machine learning model for task recommendations"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = config_manager.logger
        self.load_task_bank()
        self.setup_encoders()
        
    def load_task_bank(self):
        """Load task bank"""
        task_bank_path = self.config.get('task_bank_path', 'task_bank.json')
        with open(task_bank_path, 'r') as f:
            self.task_bank = json.load(f)
        self.task_ids = list(self.task_bank.keys())
        self.logger.info(f"Loaded {len(self.task_ids)} tasks from task bank")
    
    def setup_encoders(self):
        """Setup label encoders and transformers"""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.mlb = MultiLabelBinarizer(classes=self.task_ids)
        
        max_features = self.config.get('tfidf_max_features', 50)
        self.tfidf_goals = TfidfVectorizer(max_features=max_features)
        self.tfidf_interests = TfidfVectorizer(max_features=max_features)
    
    def load_training_data(self, filename: str = None):
        """Load training data from JSON file"""
        if filename is None:
            filename = self.config.get('training_dataset_path', 'training_dataset.json')
        
        self.logger.info(f"Loading training data from {filename}")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        examples = data['examples']
        self.logger.info(f"Loaded {len(examples)} training examples")
        
        return examples
    
    def prepare_features(self, examples):
        """Convert training examples to feature matrix"""
        self.logger.info("Preparing features...")
        
        features = []
        labels = []
        
        # Collect all categorical values for encoding
        branches = [ex['input']['branch'] for ex in examples]
        contexts = [ex['input']['context'] for ex in examples]
        
        # Fit label encoders
        self.label_encoders['branch'] = LabelEncoder()
        self.label_encoders['context'] = LabelEncoder()
        self.label_encoders['branch'].fit(branches)
        self.label_encoders['context'].fit(contexts)
        
        # Prepare text data for TF-IDF
        goal_texts = [' '.join(ex['input']['goal_tags']) for ex in examples]
        interest_texts = [' '.join(ex['input']['interests']) for ex in examples]
        
        # Fit TF-IDF vectorizers
        self.tfidf_goals.fit(goal_texts)
        self.tfidf_interests.fit(interest_texts)
        
        # Create feature vectors
        for ex in examples:
            inp = ex['input']
            
            # Categorical features
            branch_encoded = self.label_encoders['branch'].transform([inp['branch']])[0]
            context_encoded = self.label_encoders['context'].transform([inp['context']])[0]
            
            # TF-IDF features
            goal_features = self.tfidf_goals.transform([' '.join(inp['goal_tags'])]).toarray()[0]
            interest_features = self.tfidf_interests.transform([' '.join(inp['interests'])]).toarray()[0]
            
            # Combine all features
            feature_vector = [
                branch_encoded,
                context_encoded,
                inp['year'],
                inp['posts_today'],
                inp['quiz_attempts'],
                inp['karma_spent_today'],
                inp['karma_earned_today'],
                inp['buddies_interacted'],
                inp['login_streak'],
                inp['karma_today']
            ]
            
            # Add TF-IDF features
            feature_vector.extend(goal_features)
            feature_vector.extend(interest_features)
            
            features.append(feature_vector)
            labels.append(ex['label'])
        
        X = np.array(features)
        
        # Fit and transform labels using MultiLabelBinarizer
        y = self.mlb.fit_transform(labels)
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Label matrix shape: {y.shape}")
        
        return X, y
    
    def train_models(self, X, y):
        """Train different models and select the best one"""
        self.logger.info("Training models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        test_size = self.config.get('test_size', 0.2)
        random_seed = self.config.get('random_seed', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_seed
        )
        
        # Get model parameters from config
        model_type = self.config.get('model_type', 'random_forest')
        model_params = self.config.get_model_params(model_type)
        
        # Define models
        models = {}
        
        if model_type == 'random_forest' or model_type == 'both':
            rf_params = self.config.get_model_params('random_forest')
            models['random_forest'] = MultiOutputClassifier(
                RandomForestClassifier(
                    random_state=random_seed,
                    **rf_params
                )
            )
        
        if model_type == 'gradient_boosting' or model_type == 'both':
            gb_params = self.config.get_model_params('gradient_boosting')
            models['gradient_boosting'] = MultiOutputClassifier(
                GradientBoostingClassifier(
                    random_state=random_seed,
                    **gb_params
                )
            )
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            hamming_loss_score = hamming_loss(y_test, y_pred)
            jaccard_score_avg = jaccard_score(y_test, y_pred, average='macro', zero_division=0)
            
            model_results[name] = {
                'hamming_loss': hamming_loss_score,
                'jaccard_score': jaccard_score_avg,
                'model': model
            }
            
            self.logger.info(f"{name} - Hamming Loss: {hamming_loss_score:.4f}, Jaccard Score: {jaccard_score_avg:.4f}")
            
            # Select best model based on Jaccard score (higher is better)
            if jaccard_score_avg > best_score:
                best_model = model
                best_score = jaccard_score_avg
        
        self.logger.info(f"Best model selected with Jaccard score: {best_score:.4f}")
        
        return best_model, model_results
    
    def save_model(self, model, filename: str = None):
        """Save the trained model and all preprocessing components"""
        if filename is None:
            filename = self.config.get('model_output_path', 'model.pkl')
        
        self.logger.info(f"Saving model to {filename}")
        
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'mlb': self.mlb,
            'tfidf_goals': self.tfidf_goals,
            'tfidf_interests': self.tfidf_interests,
            'task_ids': self.task_ids,
            'feature_names': [
                'branch', 'context', 'year', 'posts_today', 'quiz_attempts',
                'karma_spent_today', 'karma_earned_today', 'buddies_interacted',
                'login_streak', 'karma_today'
            ] + [f'goal_tfidf_{i}' for i in range(self.config.get('tfidf_max_features', 50))] + [f'interest_tfidf_{i}' for i in range(self.config.get('tfidf_max_features', 50))]
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info("Model saved successfully!")
    
    def train_and_save(self):
        """Complete training pipeline"""
        self.logger.info("Starting model training pipeline...")
        
        # Load data
        examples = self.load_training_data()
        
        # Prepare features
        X, y = self.prepare_features(examples)
        
        # Train models
        best_model, results = self.train_models(X, y)
        
        # Save model
        self.save_model(best_model)
        
        self.logger.info("Training completed successfully!")
        return best_model, results

def main():
    """Train the task recommendation model"""
    config_manager = ConfigManager()
    trainer = TaskRecommendationModelTrainer(config_manager)
    model, results = trainer.train_and_save()
    
    print("\nTraining Results Summary:")
    for name, metrics in results.items():
        print(f"{name}: Hamming Loss = {metrics['hamming_loss']:.4f}, Jaccard Score = {metrics['jaccard_score']:.4f}")

if __name__ == "__main__":
    main()
