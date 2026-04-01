import copy
from .models import Observation, Action
from openenv.core import Environment

# --- Dataset ---
TEST_EMAILS = [
    {"id": "003", "subject": "URGENT: Invoice", "body": "Pay in Bitcoin.", "sender": "spam@web.com"},
    {"id": "004", "subject": "Angry", "body": "Refund me now!", "sender": "user@gmail.com"},
    {"id": "005", "subject": "Idea", "body": "Dark mode?", "sender": "dev@tech.com"}
]

ANSWER_KEY = {
    "003": {"department": "spam", "priority": "low"},
    "004": {"department": "billing", "priority": "critical"}, 
    "005": {"department": "support", "priority": "low"}
}

class AdvancedEmailEnv(Environment):
    def __init__(self, emails=TEST_EMAILS, ground_truth=ANSWER_KEY):
        super().__init__() 
        self.initial_emails = copy.deepcopy(emails)
        self.ground_truth = ground_truth
        self.reset()

    def reset(self) -> Observation:
        self.queue = copy.deepcopy(self.initial_emails)
        self.done = len(self.queue) == 0
        return self._make_obs("Ready.", 0.0)

    def _make_obs(self, feedback: str, reward: float) -> Observation:
        current = self.queue[0] if self.queue else None
        return Observation(
            current_email=current,
            emails_remaining=len(self.queue),
            feedback=feedback,
            reward=reward,
            done=self.done
        )

    def step(self, action: Action) -> Observation:
        if self.done: return self._make_obs("Done.", 0.0)
        truth = self.ground_truth[action.email_id]
        reward = 1.0 if action.department == truth["department"] and action.priority == truth["priority"] else 0.0
        self.queue.pop(0)
        self.done = len(self.queue) == 0
        return self._make_obs("Processed.", reward)

# --- Server Binding ---
from openenv.core.env_server import create_web_interface_app
app = create_web_interface_app(AdvancedEmailEnv, Action, Observation)