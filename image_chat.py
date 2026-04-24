import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_image():
    print("--- RAIO Image Analysis ---")
    
    image_path = input("Enter image file path: ").strip().strip('"').strip("'")

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    question = input("Your question about the image (e.g., 'What disease is this crop showing?'): ").strip()
    if not question:
        print("Error: Question cannot be empty.")
        return

    try:
        # Read image in binary and encode to base64
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        # Agricultural expert instructions
        instructions = (
    "You are a professional agricultural advisor with expertise in crop health analysis. "
    "You will be given an image (such as crops, soil, or farm conditions) along with a user query. "
    "Your task is to carefully analyze the image and provide accurate agricultural insights.\n\n"

    "Identify visible issues such as:\n"
    "- Leaf discoloration\n"
    "- Pest damage\n"
    "- Disease symptoms\n"
    "- Soil condition\n"
    "- Water stress\n\n"

    "If the image is unclear or insufficient, clearly say so instead of guessing.\n\n"

    "1. UNDERSTANDING THE IMAGE / PROBLEM\n"
    "Describe what is visible in the image and the user's concern.\n\n"

    "2. POSSIBLE CAUSES / DIAGNOSIS\n"
    "List likely causes based on visual evidence.\n\n"

    "3. RECOMMENDED ACTIONS\n"
    "Provide practical steps to fix or improve the situation.\n\n"

    "4. RISKS / PRECAUTIONS\n"
    "Mention risks or misdiagnosis possibilities.\n\n"

    "5. NEXT STEPS / MONITORING\n"
    "Explain what to observe going forward.\n\n"

    "6. CONFIDENCE LEVEL\n"
    "State confidence (Low/Medium/High) with reason."
    "Do NOT repeat any section. Avoid duplicate explanations. Keep response concise and structured."
    "reply in the language that user used, if in gujarati then reply in gujarati with same meaning. "
)

        # Send using Responses API with input_image and input_text
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        # input image is in form of a data_url as model only accepts data in url format
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                        
                        # below given is for image from the internet via image's link...
                        #{"type": "input_image", "image_url": "https://media.istockphoto.com/id/618616010/photo/rice-terrace-at-chiangmai-thailand.jpg?s=170667a&w=0&k=20&c=37-1czSgamNAs3KzwgJjryu7M3GhWjiJr-7xNuGAC6o="},
                       
                        {"type": "input_text", "text": question} # question with image...
                    ]

                } 
            ],
            max_output_tokens= 700,
        )

        
        # Extract and print response
        bot_reply = response.output[0].content[0].text
        print(f"\nRAIO Insights:\n{bot_reply}")
        usage = response.usage
        print(f"Usage for this query: \n{usage}\n")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    analyze_image()
