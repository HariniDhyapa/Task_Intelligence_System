import json
import random
from typing import Dict, List, Any
from datetime import datetime
import numpy as np
from config_manager import ConfigManager

class TrainingDataGenerator:
    """Generate synthetic training data for task recommendation model"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = config_manager.logger
        self.task_bank = self._load_task_bank()
        self.task_ids = list(self.task_bank.keys())
    
    def _load_task_bank(self) -> Dict:
        """Load task bank"""
        task_bank_path = self.config.get('task_bank_path', 'task_bank.json')
        with open(task_bank_path, 'r') as f:
            return json.load(f)
    
    def _generate_user_profile(self) -> Dict[str, Any]:
        """Generate realistic user profile"""
        branches = ['CSE', 'ECE', 'ME', 'CE', 'EE', 'IT', 'BT', 'CH']
        years = [1, 2, 3, 4]
        
        all_goals = ['SDE', 'GRE', 'MBA', 'Research', 'PhD', 'Career', 'Internship']
        all_interests = ['coding', 'python', 'algorithms', 'research', 'academic', 
                        'resume', 'career', 'internships', 'technology', 'AI']
        
        profile = {
            'branch': random.choice(branches),
            'year': random.choice(years),
            'goal_tags': random.sample(all_goals, random.randint(1, 3)),
            'interests': random.sample(all_interests, random.randint(1, 4))
        }
        
        return profile
    
    def _generate_user_activity(self) -> Dict[str, Any]:
        """Generate realistic user activity data"""
        # Create realistic correlations
        base_activity = random.randint(0, 3)  # Base activity level
        
        activity = {
            'posts_today': max(0, base_activity + random.randint(-1, 2)),
            'quiz_attempts': max(0, base_activity + random.randint(-1, 3)),
            'karma_spent_today': random.randint(0, 20),
            'karma_earned_today': random.randint(0, 30),
            'buddies_interacted': max(0, base_activity + random.randint(-1, 3)),
            'login_streak': random.randint(1, 30),
            'karma_today': random.randint(0, 30)
        }
        
        return activity
    
    def _select_relevant_tasks(self, profile: Dict, activity: Dict, context: str) -> List[str]:
        """Select tasks that would be relevant for the given profile and context"""
        relevant_tasks = []
        
        for task_id, task in self.task_bank.items():
            relevance_score = 0
            
            # Context matching
            if task.get('type') == context or context == 'any':
                relevance_score += 3
            
            # Goal alignment
            task_goals = task.get('goal_tags', [])
            user_goals = profile.get('goal_tags', [])
            goal_overlap = len(set(task_goals) & set(user_goals))
            relevance_score += goal_overlap * 2
            
            # Interest alignment
            task_interests = task.get('interests', [])
            user_interests = profile.get('interests', [])
            interest_overlap = len(set(task_interests) & set(user_interests))
            relevance_score += interest_overlap
            
            # Activity-based relevance
            if context == 'remedial' and activity['karma_earned_today'] < 5:
                if 'assessment' in task.get('title', '').lower():
                    relevance_score += 2
            
            if context == 'duel' and activity['buddies_interacted'] > 0:
                if task.get('type') == 'duel':
                    relevance_score += 1
            
            # Add some randomness
            relevance_score += random.uniform(-1, 1)
            
            if relevance_score > 2:  # Threshold for relevance
                relevant_tasks.append((task_id, relevance_score))
        
        # Sort by relevance and return top tasks
        relevant_tasks.sort(key=lambda x: x[1], reverse=True)
        
        # Select 1-4 tasks based on relevance
        num_tasks = min(random.randint(1, 4), len(relevant_tasks))
        selected_tasks = [task[0] for task in relevant_tasks[:num_tasks]]
        
        return selected_tasks
    
    def generate_example(self) -> Dict[str, Any]:
        """Generate a single training example"""
        contexts = ['duel', 'solo', 'remedial']
        context = random.choice(contexts)
        
        profile = self._generate_user_profile()
        activity = self._generate_user_activity()
        relevant_tasks = self._select_relevant_tasks(profile, activity, context)
        
        example = {
            'input': {
                'context': context,
                **profile,
                **activity
            },
            'label': relevant_tasks
        }
        
        return example
    
    def generate_dataset(self, num_examples: int = None) -> Dict[str, Any]:
        """Generate complete training dataset"""
        if num_examples is None:
            num_examples = self.config.get('training_data_size', 500)
        
        self.logger.info(f"Generating {num_examples} training examples...")
        
        examples = []
        for i in range(num_examples):
            if (i + 1) % 100 == 0:
                self.logger.info(f"Generated {i + 1}/{num_examples} examples")
            
            example = self.generate_example()
            examples.append(example)
        
        # Create task mapping for reference
        task_mapping = {task_id: idx for idx, task_id in enumerate(self.task_ids)}
        
        dataset = {
            'examples': examples,
            'task_mapping': task_mapping,
            'metadata': {
                'total_examples': num_examples,
                'num_tasks': len(self.task_ids),
                'task_ids': self.task_ids,
                'generated_at': datetime.now().isoformat(),
                'format': 'input-label pairs for task recommendation'
            }
        }
        
        self.logger.info(f"Generated dataset with {num_examples} examples and {len(self.task_ids)} tasks")
        return dataset
    
    def save_dataset(self, dataset: Dict, filename: str = None) -> None:
        """Save dataset to JSON file"""
        if filename is None:
            filename = self.config.get('training_dataset_path', 'training_dataset.json')
        
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        self.logger.info(f"Dataset saved to {filename}")

def main():
    """Generate training dataset"""
    config_manager = ConfigManager()
    generator = TrainingDataGenerator(config_manager)
    
    # Generate dataset
    dataset = generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset(dataset)
    
    # Print sample
    print("\nSample training example:")
    print(json.dumps(dataset['examples'][0], indent=2))
    
    print(f"\nDataset summary:")
    print(f"- Total examples: {dataset['metadata']['total_examples']}")
    print(f"- Number of tasks: {dataset['metadata']['num_tasks']}")
    print(f"- Generated at: {dataset['metadata']['generated_at']}")

if __name__ == "__main__":
    main()
