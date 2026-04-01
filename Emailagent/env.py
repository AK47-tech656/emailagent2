import copy

from models import Observation, Action, Reward

# --- 1. The Chaos Tier Dataset ---
TEST_EMAILS = [
    {
        "id": "email_003",
        "subject": "URGENT: Your Invoice #88492 is Overdue!",
        "body": "Dear customer, your account will be suspended in 24 hours. Click this totally safe link to pay in Bitcoin immediately.",
        "sender": "billing-admin@suspicious-domain.xyz"
    },
    {
        "id": "email_004",
        "subject": "I am so done with your company.",
        "body": "I have been on hold for 3 hours. Your software deleted all my files. Cancel my subscription right now and refund me or I am calling my lawyer.",
        "sender": "angry.user@gmail.com"
    },
    {
        "id": "email_005",
        "subject": "Quick idea",
        "body": "Hey, it would be super cool if your dashboard had a dark mode. Just a thought! Have a good weekend.",
        "sender": "friendly.dev@techcorp.com"
    }
]

# The Answer Key (Ground Truth)
ANSWER_KEY = {
    "email_003": {"department": "spam", "priority": "low"},
    "email_004": {"department": "billing", "priority": "critical"}, 
    "email_005": {"department": "support", "priority": "low"}
}


# --- 2. Your Advanced Environment ---
class AdvancedEmailEnv:
    def __init__(self, emails=TEST_EMAILS, ground_truth=ANSWER_KEY):
        self.initial_emails = copy.deepcopy(emails)
        self.ground_truth = ground_truth
        self.reset()

    def reset(self) -> Observation:
        self.queue = copy.deepcopy(self.initial_emails)
        self.processed = {}
        self.done = len(self.queue) == 0
        return self._make_obs("Environment initialized. Ready for triage.")

    def state(self) -> dict:
        return {
            "queue_length": len(self.queue),
            "processed": self.processed,
            "done": self.done
        }

    def _make_obs(self, feedback: str) -> Observation:
        current = self.queue[0] if self.queue else None
        return Observation(
            current_email=current,
            emails_remaining=len(self.queue),
            feedback=feedback
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        if self.done:
            return self._make_obs("Episode done."), Reward(value=0.0, reason="Done"), True, self.state()

        current_email = self.queue[0]
        if action.email_id != current_email["id"]:
            return self._make_obs("Fatal Error: Processed out of order."), Reward(value=-1.0, reason="ID mismatch"), self.done, self.state()

        truth = self.ground_truth[action.email_id]
        step_reward = 0.0
        feedback_notes = []

        # Dense Reward Shaping
        if action.department == truth["department"]:
            step_reward += 0.5
            feedback_notes.append("Correct department.")
        else:
            step_reward -= 0.3
            feedback_notes.append(f"Wrong department (Expected {truth['department']}).")

        if action.priority == truth["priority"]:
            step_reward += 0.4
            feedback_notes.append("Correct priority.")
        else:
            step_reward -= 0.2
            feedback_notes.append(f"Wrong priority (Expected {truth['priority']}).")

        if len(action.chain_of_thought) > 10:
            step_reward += 0.1
            feedback_notes.append("Good reasoning provided.")
            
        self.processed[action.email_id] = action.model_dump()
        self.queue.pop(0)
        self.done = len(self.queue) == 0

        reward = Reward(value=round(step_reward, 2), reason=" | ".join(feedback_notes))
        return self._make_obs("Email processed."), reward, self.done, self.state()