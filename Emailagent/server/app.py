import copy
from emailagent.models import Observation, Action
from openenv.core import Environment
from openenv.core.env_server import create_web_interface_app

# ... (Keep your TEST_EMAILS and ANSWER_KEY here) ...

class AdvancedEmailEnv(Environment):
    # ... (Keep your full AdvancedEmailEnv class logic here) ...
    @property
    def state(self) -> dict:
        return {"queue_length": len(self.queue), "done": self.done}
    
    def reset(self) -> Observation:
        self.queue = copy.deepcopy(self.initial_emails)
        self.done = len(self.queue) == 0
        return self._make_obs("Ready.", 0.0)

    def _make_obs(self, feedback: str, reward_val: float) -> Observation:
        current = self.queue[0] if self.queue else None
        return Observation(current_email=current, emails_remaining=len(self.queue), 
                           feedback=feedback, reward=reward_val, done=self.done)

    def step(self, action: Action) -> Observation:
        # ... (Keep your step logic here) ...
        self.queue.pop(0)
        self.done = len(self.queue) == 0
        return self._make_obs("Processed.", 1.0)

# ⭐ THE CRITICAL CHANGE ⭐
def main():
    """This is the entry point function the grader is looking for."""
    return create_web_interface_app(AdvancedEmailEnv, Action, Observation)

# Create the app variable for Uvicorn as well
app = main()
