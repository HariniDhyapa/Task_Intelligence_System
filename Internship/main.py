from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import logging
from task_engine import TaskIntelligenceEngine
from config_manager import ConfigManager

# Initialize configuration manager first
try:
    config_manager = ConfigManager()
    logger = config_manager.logger
except Exception as e:
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to initialize ConfigManager: {e}")
    raise

app = FastAPI(
    title="Task Intelligence Engine",
    description="AI-powered task recommendation system for student challenges and duels",
    version="1.0.0"
)

# Pydantic Models
class UserProfile(BaseModel):
    branch: str = Field(..., description="Student's academic branch")
    year: int = Field(..., ge=1, le=4, description="Academic year")
    goal_tags: List[str] = Field(..., description="User goals like GRE, SDE")
    interests: List[str] = Field(..., description="User interests")

class UserActivity(BaseModel):
    posts_today: int = Field(default=0, ge=0)
    quiz_attempts: int = Field(default=0, ge=0)
    karma_spent_today: int = Field(default=0, ge=0)
    karma_earned_today: int = Field(default=0, ge=0)
    buddies_interacted: int = Field(default=0, ge=0)
    login_streak: int = Field(default=0, ge=0)
    last_resume_upload: Optional[str] = Field(default=None)

class TaskGenerationRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    context: str = Field(..., description="Context: duel, solo, remedial")
    profile: UserProfile
    activity: UserActivity
    task_history: List[str] = Field(default=[], description="Previously completed tasks")

class SuggestedTask(BaseModel):
    task_id: str
    title: str
    score: float = Field(..., ge=0, le=1)
    karma_reward: int
    type: str
    category: str
    goal_tags: Optional[List[str]] = None

class TaskGenerationResponse(BaseModel):
    user_id: str
    context: str
    suggested_tasks: List[SuggestedTask]
    status: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    config_loaded: bool

class VersionResponse(BaseModel):
    model_version: str

# Initialize the task engine
try:
    task_engine = TaskIntelligenceEngine(config_manager)
    logger.info("Task Intelligence Engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Task Intelligence Engine: {e}")
    task_engine = None
@app.get("/")
def read_root():
    return {"message": "Welcome to the Task Intelligence Engine API"}
@app.post("/generate-tasks", response_model=TaskGenerationResponse)
async def generate_tasks(request: TaskGenerationRequest):
    """Generate personalized task recommendations based on user profile and context"""
    try:
        if task_engine is None:
            raise HTTPException(status_code=500, detail="Task engine not initialized")
        
        logger.info(f"Generating tasks for user {request.user_id} in context {request.context}")
        
        # Validate context against allowed values
        valid_contexts = ['duel', 'solo', 'remedial', 'streak']
        if request.context not in valid_contexts:
            raise HTTPException(status_code=400, detail=f"Invalid context. Must be one of: {valid_contexts}")
        
        # Generate task recommendations
        result = task_engine.generate_tasks(
            user_id=request.user_id,
            context=request.context,
            profile=request.profile.model_dump(),
            activity=request.activity.model_dump(),
            task_history=request.task_history
        )
        
        return TaskGenerationResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    model_loaded = task_engine is not None
    config_loaded = config_manager is not None
    
    status = "ok" if model_loaded and config_loaded else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        config_loaded=config_loaded
    )

@app.get("/version", response_model=VersionResponse)
async def get_version():
    """Get model version information"""
    return VersionResponse(model_version="1.0.0")

@app.get("/tasks/available")
async def get_available_tasks():
    """Get all available tasks from the task bank"""
    try:
        if task_engine is None:
            raise HTTPException(status_code=500, detail="Task engine not initialized")
        
        return {"tasks": task_engine.get_task_bank()}
    except Exception as e:
        logger.error(f"Error fetching tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))
'''
if __name__ == "__main__":
    host = config_manager.get('server.host', '127.0.0.1')
    port = config_manager.get('server.port', 8000)
    uvicorn.run(app, host=host, port=port)'''