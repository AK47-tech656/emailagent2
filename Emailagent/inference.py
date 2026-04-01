import os
import json
from openai import OpenAI

from env import AdvancedEmailEnv
from models import Action

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"


SYSTEM_PROMPT = """
You are an autonomous Email Triage Agent. 
Read the provided email observation and route it appropriately.

CRITICAL ROUTING RULES:
- If an email is a phishing scam or spam, department MUST be 'spam' and priority MUST be 'low' (we do not prioritize spam).
- If a customer explicitly demands a refund, cancellation, or threatens legal action over money, department MUST be 'billing' even if they mention technical bugs.
- Feature requests and general ideas should go to 'support' with 'low' priority.
- Server outages or complete data loss for VIPs/CEOs is 'support' with 'critical' priority.

You MUST respond with a valid JSON object containing exactly these 4 keys:
1. "email_id": (string) The ID of the email you are processing.
2. "chain_of_thought": (string) Your step-by-step reasoning for the routing.
3. "department": (string) Must be exactly one of: 'sales', 'support', 'billing', or 'spam'.
4. "priority": (string) Must be exactly one of: 'low', 'normal', 'high', or 'critical'.
"""

def main():
    print("🤖 Booting up Agent...")
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("🏠 Connecting directly to your vanilla local environment...")
    
    # Initialize your environment directly
    env = AdvancedEmailEnv()
    
    # Start the test
    print("✅ Connected!")
    obs = env.reset()
    done = env.done
    
    step = 1
    while not done:
        print(f"\n--- 📧 Step {step}: Processing Email ---")
        print(f"Observation: {obs}")
        
        # Ask the Llama model to make a decision
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current Email Observation: {obs}"}
            ],
            response_format={"type": "json_object"}
        )
        
        response_text = completion.choices[0].message.content
        print(f"🧠 AI Reasoning & Decision:\n{response_text}")
        
        try:
            action_dict = json.loads(response_text)
            action = Action(**action_dict)
        except Exception as e:
            print(f"❌ LLM returned invalid formatting. Error: {e}")
            break

        # Send the action directly to your step function!
        obs, reward, done, state = env.step(action)
        
        # Print your awesome Dense Reward shaping
        print(f"🎯 Reward Score: {reward.value} | Feedback: {reward.reason}")
        step += 1
        
    print("\n✅ Environment complete! No more emails.")

if __name__ == "__main__":
    main()