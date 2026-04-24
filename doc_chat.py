import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def analyze_doc():
    print("--- RAIO Document Analysis ---")
    
    file_path = input("Enter PDF file path: ").strip().strip('"').strip("'")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    question = input("Your question about the document: ").strip()
    if not question:
        print("Error: Question cannot be empty.")
        return
    try:
        # Step 1: Check if file already exists in OpenAI
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        print(f"Checking if {file_name} already exists in OpenAI...")
        existing_files = client.files.list(purpose="user_data")
        file_id = None

        for file in existing_files.data:
            if file.filename == file_name and file.bytes == file_size:
                file_id = file.id
                print(f"Reusing existing file (ID: {file_id})")
                break

        if not file_id:
            print(f"Uploading {file_path}...")
            with open(file_path, "rb") as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose="user_data"
                )
            file_id = uploaded_file.id
            print(f"File uploaded successfully (ID: {file_id})")

        # Simple agricultural expert instructions
        instructions = (
    "You are a professional agricultural advisor and report analyst. "
    "You will be given a document (such as a soil report, lab report, or agricultural data report) along with a user query. "
    "Your task is to carefully read and understand the document, interpret its parameters, and answer the user's question clearly.\n\n"

    "If the user asks about specific parameters (e.g., nitrogen, pH, moisture), explain:\n"
    "- What the parameter means\n"
    "- Whether the value is good or bad\n"
    "- Why it matters\n"
    "- What action should be taken\n\n"

    "Always base your answer on the document content. If something is unclear or missing, state that explicitly.\n\n"

    "1. UNDERSTANDING THE DOCUMENT / QUERY\n"
    "Summarize what the document contains and what the user is asking.\n\n"

    "2. KEY INSIGHTS FROM THE DOCUMENT\n"
    "Extract and explain important values and observations.\n\n"

    "3. PARAMETER EXPLANATION (IF APPLICABLE)\n"
    "Explain relevant parameters in simple terms.\n\n"
    "4. RECOMMENDED ACTIONS\n"
    "Suggest actions based on the document analysis.\n\n"

    "5. RISKS / PRECAUTIONS\n"
    "Mention risks or incorrect interpretations to avoid.\n\n"

    "6. CONFIDENCE LEVEL\n"
    "State confidence (Low/Medium/High) with reason."
    "reply in the language that user used, if in gujarati then reply in gujarati with same meaning. "
    "Do NOT repeat any section/line. Avoid duplicate explanations. Keep response concise and structured."

)

        # Send using Responses API with input_file (using file_id) and input_text
        response = client.responses.create(
            model="gpt-4.1-mini",
            instructions=instructions,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "file_id": file_id
                        },
                        {
                            "type": "input_text",
                            "text": question
                        }
                    ]
                }
            ]
        )




        
        # Extract and print response
        bot_reply = response.output[0].content[0].text
        print(f"\nRAIO Analysis:\n{bot_reply}")
        usage = response.usage
        print(f"Usage for this query: \n{usage}\n")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    analyze_doc()
