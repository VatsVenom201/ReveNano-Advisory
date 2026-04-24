import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat():
    print("--- RAIO Text Chat (Type 'exit' to quit) ---")
    
    # Simple agricultural assistant instructions
    instructions = (
    "You are a professional agricultural advisor. Provide clear, actionable, and scientific advice to farmers. "
    "Understand the user's query and respond with practical, field-applicable guidance. "
    "Keep explanations simple but accurate.\n\n"

    "1. UNDERSTANDING THE PROBLEM\n"
    "Explain the user’s issue clearly.\n\n"

    "2. POSSIBLE CAUSES / INSIGHTS\n"
    "List likely scientific or environmental causes.\n\n"

    "3. RECOMMENDED ACTIONS\n"
    "Provide clear step-by-step solutions.\n\n"

    "4. RISKS / PRECAUTIONS\n"
    "Mention risks and what to avoid.\n\n"

    "5. NEXT STEPS / MONITORING\n"
    "Explain how to track progress or improvements.\n\n"

    "6. CONFIDENCE LEVEL\n"
    "State confidence (Low/Medium/High) with a short reason."
    "Do NOT repeat any section. Avoid duplicate explanations. Keep response concise and structured."
    "reply in the language that user used, if in gujarati then reply in gujarati with same meaning. "
)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        if not user_input.strip():
            continue

        try:
            # Use the Responses API with structured input
            response = client.responses.create(
                model="gpt-4.1-mini",
                instructions=instructions,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_input}
                        ]
                    }
                ]
            )
            
            # Extract response clearly
            bot_reply = response.output[0].content[0].text
            direct_reply = response.output
            print(f"\nRAIO: {bot_reply}")
            print("\n********************************\n")
            print(f"\nDirect Reply:\n {direct_reply}\n")
            usage = response.usage
            print(f"Usage for this query: \n{usage}\n")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat()
