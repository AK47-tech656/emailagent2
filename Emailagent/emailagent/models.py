from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    email_id: str
    department: str
    priority: str
    chain_of_thought: str

class Observation(BaseModel):
    current_email: Optional[dict] = None
    emails_remaining: int
    feedback: str
    # Fields required for the OpenEnv web server
    reward: float = 0.0
    done: bool = False