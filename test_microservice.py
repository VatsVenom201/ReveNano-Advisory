import requests
import json

# Configuration
BASE_URL = "http://127.0.0.1:8000"
USER_ID = 1  # Change this to test different users
THREAD_ID = None  # Set to an existing ID to continue a thread, or None for a new one

def send_chat(message, user_id=USER_ID, thread_id=THREAD_ID):
    url = f"{BASE_URL}/chat"
    payload = {
        "user_id": user_id,
        "thread_id": thread_id,
        "message": message
    }
    
    print(f"\n--- Sending Request to {url} ---")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        print("\n--- Response Received ---")
        if data.get('error'):
            print(f"!!! Error Encountered !!!")
            print(f"Message: {data.get('error')}")
        else:
            print(f"User ID: {data.get('user_id')}")
            print(f"Thread ID: {data.get('thread_id')}")
            print(f"Reply: {data.get('reply')}")
        
        return data.get('thread_id')
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Details: {e.response.text}")
        return None

def test_flow():
    # Define your queries here
    queries = [
        "Hello, I need advice on my wheat crop.",
        "The leaves are turning yellow. What should I do?",
        "Is there any specific fertilizer you recommend for this?"
    ]
    
    current_thread_id = THREAD_ID
    
    for q in queries:
        print(f"\nUser Query: {q}")
        new_id = send_chat(q, thread_id=current_thread_id)
        if new_id:
            current_thread_id = new_id
        
        input("\nPress Enter for next query...")

if __name__ == "__main__":
    # Make sure your backend (main.py) is running before starting this!
    test_flow()
