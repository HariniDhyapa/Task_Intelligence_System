# Task_Intelligence_System
A pluggable, **offline AI-based microservice** that dynamically recommends relevant tasks based on user profiles, goals, activities, and context (e.g., `solo`, `duel`, `remedial`). Powers features like:
- Smart Challenges
- 1v1 Karma Duels
- Personalized remedial interventions
##  Features
-  Context-aware task recommendation
-  Trained multi-label classification model (RandomForest / Gradient Boosting)
-  Completely offline, works with a saved `.pkl` model
-  FastAPI REST API with Swagger UI
-  Configurable via `config.json`
##  Directory Structure
📦task-intelligence-engine
├── main.py # FastAPI app (API endpoints)
├── task_engine.py # Core task recommendation engine
├── train_model.py # Train & save ML model
├── validate_system.py # System checks before deployment
├── config_manager.py # Loads & validates config
├── config.json # Configuration file
├── task_bank.json # Master list of tasks
├── training_dataset.json # Training examples (user-task mappings)
├── model.pkl # Saved trained model
├── generate_training_data.py # Utility to generate more training data
├── requirements.txt # Python dependencies
├── Dockerfile # Docker setup
├── testing.py # Basic file and setup check
├── testingc.py # Comprehensive user-profile testing
steps:
pip install -r requirements.txt

python train_model.py
  Loads training_dataset.json and task_bank.json
  Trains an ML model
  Saves output to model.pkl
  
python validate_system.py
  Checks for:
  Config integrity
  Required files
  Task bank format
  API route availability
  
Run the FastAPI Server
uvicorn main:app --reload
Localhost
Swagger UI: http://127.0.0.1:8000/docs
Health: http://127.0.0.1:8000/health
Root: http://127.0.0.1:8000/
or
uvicorn main:app --host 0.0.0.0 --port 8000
Docker
Build and run the container:
docker build -t task-intelligence .
docker run -p 8000:8000 task-intelligence


System Validation

Run this to validate configuration, task bank, and API setup:
python validate_system.py


Testing

Basic Test:
python testing.py

Comprehensive Testing:
python testingc.py
