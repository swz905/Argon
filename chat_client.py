import requests
import json
import random
import string
import os


SERVER_URL = "http://127.0.0.1:3000" 
UPLOAD_ENDPOINT = f"{SERVER_URL}/upload"
CHAT_ENDPOINT = f"{SERVER_URL}/chat"

# --- Helper Functions ---
def generate_id(length=8):
    """Generates a random string for user/session IDs."""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def upload_context(folder_path):
    """Sends the folder path to the server's /upload endpoint."""
    if not folder_path or not (os.path.exists(folder_path)):
         print(f"Error: Invalid path provided: '{folder_path}'")
         print("Please provide a valid path to a file or folder containing your documents.")
         return False

    print(f"\nAttempting to upload/process context from: {folder_path}...")
    print("(This might take a very long time for large files, please wait...)")
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'folder_path': folder_path})

    try:
        
        print("Sending upload request to server...")
        response = requests.post(UPLOAD_ENDPOINT, headers=headers, data=payload) # NO TIMEOUT HERE
        print("Received response from server.")

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        print("✅ Server confirmed: Context processed successfully!")
        print(response.json().get("message", ""))
        return True
   
    except requests.exceptions.RequestException as e:
        print(f"❌ Error during upload/processing request: {e}")
        try:
         
            if 'response' in locals() and response is not None:
                 error_data = response.json()
                 print(f"Server responded with error: {error_data.get('error', 'Unknown error')}")
            else:
                 print("Could not get a detailed error response from the server.")
        except Exception as json_err:
            print(f"Could not parse error response from server: {json_err}")
            pass # Ignore if response is not JSON or doesn't exist
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred during upload: {e}")
        return False


def send_message(query, user_id, session_id):
    """Sends a chat message to the server's /chat endpoint."""
    headers = {
        'Content-Type': 'application/json',
        'x-user-id': user_id,
        'x-session-id': session_id
    }
    payload = json.dumps({'query': query})

    try:
        # Keep timeout for chat responses as they should be quicker
        response = requests.post(CHAT_ENDPOINT, headers=headers, data=payload, timeout=90) # 90 seconds timeout for chat response
        response.raise_for_status()
        data = response.json()
        return data.get('bot_message', 'Error: No bot_message found in response.')
    except requests.exceptions.Timeout:
        return "❌ Error: Chat request timed out. The server might be busy or the LLM is taking too long."
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error sending message: {e}")
        try:
            if 'response' in locals() and response is not None:
                error_data = response.json()
                return f"Server error during chat: {error_data.get('error', 'Unknown error')}"
            else:
                return "Could not get a detailed error response from the server during chat."
        except Exception as json_err:
             print(f"Could not parse error response from server during chat: {json_err}")
             return "Could not connect to the server or decode the chat response."
    except Exception as e:
        print(f"\n❌ An unexpected error occurred during chat: {e}")
        return "An unexpected error occurred."

# --- Main Logic ---
if __name__ == "__main__":
    print("--- Chat Client ---")
    print(f"Connecting to server at {SERVER_URL}")

  
    upload_context_flag = False
    while True:
        choice = input("\nDo you want to upload NEW context (file/folder) now? (y/n): ").lower().strip()
        if choice == 'y':
            upload_context_flag = True
            break
        elif choice == 'n':
            print("\nOkay, proceeding without uploading. Will use existing context on the server (if any).")
            upload_context_flag = False
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    proceed_to_chat = False
    if upload_context_flag:
        while True: 
            doc_path = input("Enter the full path to the document file or folder for context: ")
            if upload_context(doc_path):
                proceed_to_chat = True
                break 
            else:
                print("\nUpload/Processing failed.")
                retry = input("Try uploading again? (y/n): ").lower()
                if retry != 'y':
                    print("Exiting client as upload was not successful.")
                    exit() 
    else:
        proceed_to_chat = True

    if proceed_to_chat:
        user_id = generate_id()
        session_id = generate_id()
        print("\n" + "="*20)
        print("✅ Chat session starting...")
        print(f"   User ID: {user_id}")
        print(f"   Session ID: {session_id}")
        print("   Type 'quit' to exit.")
        print("="*20 + "\n")

        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("\nEnding chat session. Goodbye!")
                break

            if not user_input.strip(): # Ignore empty input
                continue

            bot_response = send_message(user_input, user_id, session_id)
            print(f"Bot: {bot_response}")
    else:
         print("\nCould not start chat session due to upload issues.")

