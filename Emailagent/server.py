from fastapi import FastAPI
from models import Action
from env import AdvancedEmailEnv
from tasks import get_task, calculate_grade
from baseline import run_all_baselines

app = FastAPI()

# Global state for the active episode
current_task_emails, current_task_truth = get_task("easy")
global_env = AdvancedEmailEnv(current_task_emails, current_task_truth)

@app.get("/")
def health_check():
    """Automated ping to the Space URL — must return 200"""
    return {"status": "ok", "environment": "AdvancedEmailTriageEnv"}

@app.api_route("/reset", methods=["GET", "POST"])
def reset_env():
    obs = global_env.reset()
    return {"observation": obs.model_dump()}

@app.post("/step")
def step_env(action: Action):
    obs, reward, done, info = global_env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    return global_env.state()

# --- Checklist Required Additional Endpoints ---

@app.api_route("/baseline", methods=["GET", "POST"])
def run_baseline_endpoint():
    """Trigger inference script and return baseline score for all 3 tasks"""
    scores = run_all_baselines()
    return {"baseline_scores": scores}

@app.get("/grader")
def get_grader():
    """Returns grader score after an episode is completed"""
    score = calculate_grade(global_env.state()["processed"], global_env.ground_truth)
    return {"score": score}

@app.get("/tasks")
def get_tasks():
    """Returns list of tasks and the action schema"""
    return {
        "tasks": ["easy", "medium", "hard"],
        "action_schema": Action.model_json_schema()
    }