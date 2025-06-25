from task_engine import TaskIntelligenceEngine
from config_manager import ConfigManager
import pprint

# Initialize engine
config_manager = ConfigManager()
engine = TaskIntelligenceEngine(config_manager)

# Define test users
test_cases = [
    {
        "user_id": "user_01",
        "context": "solo",
        "profile": {
            "branch": "CSE",
            "year": 2,
            "goal_tags": ["SDE", "Portfolio"],
            "interests": ["coding", "projects"]
        },
        "activity": {
            "posts_today": 0,
            "quiz_attempts": 1,
            "karma_spent_today": 2,
            "karma_earned_today": 15,
            "buddies_interacted": 0,
            "login_streak": 3,
            "karma_today": 10
        },
        "task_history": []
    },
    {
        "user_id": "user_02",
        "context": "duel",
        "profile": {
            "branch": "ECE",
            "year": 1,
            "goal_tags": ["Career", "Networking"],
            "interests": ["mentorship", "internships"]
        },
        "activity": {
            "posts_today": 1,
            "quiz_attempts": 0,
            "karma_spent_today": 0,
            "karma_earned_today": 8,
            "buddies_interacted": 0,
            "login_streak": 1,
            "karma_today": 4
        },
        "task_history": []
    },
    {
        "user_id": "user_03",
        "context": "remedial",
        "profile": {
            "branch": "ME",
            "year": 3,
            "goal_tags": ["GRE", "Research"],
            "interests": ["test_prep", "academic"]
        },
        "activity": {
            "posts_today": 2,
            "quiz_attempts": 3,
            "karma_spent_today": 12,
            "karma_earned_today": 20,
            "buddies_interacted": 1,
            "login_streak": 7,
            "karma_today": 7
        },
        "task_history": []
    },
    {
        "user_id": "user_04",
        "context": "duel",
        "profile": {
            "branch": "CH",
            "year": 4,
            "goal_tags": ["SDE", "Teaching"],
            "interests": ["helping", "tutoring"]
        },
        "activity": {
            "posts_today": 2,
            "quiz_attempts": 2,
            "karma_spent_today": 6,
            "karma_earned_today": 18,
            "buddies_interacted": 2,
            "login_streak": 10,
            "karma_today": 9
        },
        "task_history": ["ch_peer_tutoring_029"]  # Already done
    },
    {
        "user_id": "user_05",
        "context": "solo",
        "profile": {
            "branch": "CE",
            "year": 2,
            "goal_tags": ["Community", "Open Source"],
            "interests": ["open_source", "tech"]
        },
        "activity": {
            "posts_today": 0,
            "quiz_attempts": 0,
            "karma_spent_today": 1,
            "karma_earned_today": 30,
            "buddies_interacted": 0,
            "login_streak": 15,
            "karma_today": 14
        },
        "task_history": []
    }
]

# Run test cases
print("\n==================== TESTING RECOMMENDATIONS ====================\n")
for test in test_cases:
    print(f" Testing for User: {test['user_id']} | Context: {test['context']}")
    output = engine.generate_tasks(
        user_id=test["user_id"],
        context=test["context"],
        profile=test["profile"],
        activity=test["activity"],
        task_history=test["task_history"]
    )
    pprint.pprint(output)
    print("\n" + "-"*70 + "\n")
