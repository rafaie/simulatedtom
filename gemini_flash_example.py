import vertexai
from vertexai.generative_models import GenerativeModel, Part
import openai
import google.auth # Add this import
import google.auth.transport.requests # Add this import

# Rename function to be more generic
def generate_text_with_gemini(project_id: str, location: str, prompt: str) -> str:
    """Generates text using a Gemini model."""

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Load the model
    # Trying Gemini 2.5 Pro preview
    # Note: Ensure your project has access to this preview model in the specified location.
    model_name = "gemini-2.5-pro-preview-03-25"
    print(f"Attempting to load model: {model_name} in {location}")
    model = GenerativeModel(model_name)

    # Prepare the prompt
    text_part = Part.from_text(prompt)

    # Generate content
    response = model.generate_content([text_part])

    # Extract and return the text response
    try:
        return response.text
    except ValueError as e:
        # Handle cases where the response might be blocked or contain no text
        print(f"Error processing response: {e}")
        print(f"Full response: {response}")
        return f"Could not extract text from response. Reason: {response.candidates[0].finish_reason.name}"

def generate_text_with_llama2(project_id: str, location: str, prompt: str) -> str:
    MAAS_ENDPOINT = f"{location}-aiplatform.googleapis.com"

    # Get Google Cloud credentials and access token
    try:
        credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        access_token = credentials.token
    except Exception as e:
        print(f"Error getting Google Cloud credentials/token: {e}")
        print("Ensure you are authenticated (e.g., `gcloud auth application-default login`).")
        return "Authentication failed."


    client = openai.OpenAI(
        base_url=f"https://{MAAS_ENDPOINT}/v1beta1/projects/{project_id}/locations/{location}/endpoints/openapi",
        api_key=access_token, # Pass the obtained token here
    )

    MODEL_ID = "meta/llama-3.2-90b-vision-instruct-maas"
    print(f"Attempting to load model: {MODEL_ID} in {location}")
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    # Extract and return the text response
    try:
        return response.choices[0].message.content
    # Update error handling to catch potential OpenAI API errors related to auth
    except openai.AuthenticationError as e:
        print(f"OpenAI Authentication Error: {e}")
        print("Check if the access token is valid or if permissions are correct.")
        return f"Authentication failed for MAAS endpoint. Error: {e}"
    except ValueError as e:
        # Handle cases where the response might be blocked or contain no text
        print(f"Error processing response: {e}")
        # Check if finish_reason exists before accessing it
        finish_reason = "UNKNOWN"
        if response.choices and response.choices[0].finish_reason:
            finish_reason = response.choices[0].finish_reason
        print(f"Full response: {response}")
        return f"Could not extract text from response. Reason: {finish_reason}"
    except Exception as e:
        print(f"An unexpected error occurred during MAAS call: {e}")
        print(f"Full response: {response}")
        return f"MAAS call failed. Error: {e}"

def generate_text_with_llama(project_id: str, location: str, prompt: str) -> str:
    """Generates text using the Llama 3 70B Instruct model via Vertex AI."""

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)

    # Load the Llama model using the standard name format
    # The SDK should resolve the publisher ('meta') based on this name.
    # Ensure your project has access to this specific model in the region.
    model_name = "llama3-70b-instruct" # Changed model name
    print(f"\nAttempting to load model: {model_name} in {location}")
    try:
        # Use GenerativeModel, consistent with Gemini usage
        model = GenerativeModel(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Ensure the model name is correct, the AI Platform API is enabled,")
        print("and your project/service account has permissions for Meta models in this region.")
        return f"Failed to load model {model_name}."


    # Prepare the prompt (using Part, consistent with Gemini)
    text_part = Part.from_text(prompt)

    # Generate content using the standard generate_content method
    try:
        response = model.generate_content([text_part])
        return response.text
    except ValueError as e:
        # Handle cases where the response might be blocked or contain no text
        print(f"Error processing response: {e}")
        try:
            reason = response.candidates[0].finish_reason.name
            print(f"Full response: {response}")
            return f"Could not extract text from response. Reason: {reason}"
        except (AttributeError, IndexError):
            print(f"Could not determine finish reason. Full response: {response}")
            return "Could not extract text from response."
    except Exception as e:
        print(f"An unexpected error occurred during generation with {model_name}: {e}")
        # Include more details from the exception if possible
        return f"Generation failed for model {model_name}. Error: {e}"


if __name__ == "__main__":
    # --- Configuration ---
    PROJECT_ID = "amazing-limiter-458018-a4"  # Replace with your Project ID
    LOCATION = "us-central1"      # Ensure this location supports the Llama model
    USER_PROMPT_GEMINI = "Explain the difference between Gemini 1.5 Pro and Gemini 1.5 Flash in simple terms."
    USER_PROMPT_LLAMA = "Write a short story about a robot learning to paint."
    # --- End Configuration ---

    if PROJECT_ID == "YOUR_PROJECT_ID" or LOCATION == "YOUR_LOCATION":
        print("Please update PROJECT_ID and LOCATION variables in the script.")
    else:
        print(f"Using Project ID: {PROJECT_ID}, Location: {LOCATION}")

        # --- Gemini Call ---
        # print(f"\nPrompt for Gemini: {USER_PROMPT_GEMINI}\n")
        # generated_text_gemini = generate_text_with_gemini(PROJECT_ID, LOCATION, USER_PROMPT_GEMINI)
        # print("--- Gemini Response ---")
        # print(generated_text_gemini)
        # print("-----------------------")

        # --- Llama Call ---
        print(f"\nPrompt for Llama: {USER_PROMPT_LLAMA}\n")
        generated_text_llama = generate_text_with_llama2(PROJECT_ID, LOCATION, USER_PROMPT_LLAMA)
        print("--- Llama Response ---")
        print(generated_text_llama)
        print("----------------------")
