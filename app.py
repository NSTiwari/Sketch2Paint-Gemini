import os
import io
import base64
import PIL.Image
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google.genai import types
from google import genai

load_dotenv('.env')

app = Flask(__name__)

API_KEY = os.environ.get("GOOGLE_API_KEY")

MODEL_ID = 'gemini-2.0-flash-exp-image-generation'

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
else:
    try:
        client = genai.Client(api_key=API_KEY)
        print(f"Gemini Client initialized successfully for model target: {MODEL_ID}")
    except Exception as e:
        print(f"Error configuring Gemini or initializing client: {e}")
        client = None


@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_painting():
    """Receives sketch data and asks Gemini to generate a painting using client.models.generate_content."""
    if not client:
        return jsonify({"error": "Gemini client not initialized. Check API key and configuration."}), 500

    if not request.json or 'image_data' not in request.json:
        return jsonify({"error": "Missing image_data in request"}), 400

    base64_image_data = request.json['image_data']
    user_prompt = request.json.get('prompt', '').strip()

    # Process the canvas sketch: Input image.
    if ',' in base64_image_data:
        header, base64_data = base64_image_data.split(',', 1)
        input_mime_type = header.split(';')[0].split(':')[1]
    else:
        base64_data = base64_image_data
        input_mime_type = 'image/png'

    try:
        image_bytes = base64.b64decode(base64_data)
        sketch_pil_image = PIL.Image.open(io.BytesIO(image_bytes))
        print(f"Received image decoded to PIL: format={sketch_pil_image.format}, size={sketch_pil_image.size}, mime={input_mime_type}")
    except base64.binascii.Error:
        print("Error decoding base64 string")
        return jsonify({"error": "Invalid base64 string received from frontend"}), 400
    except PIL.UnidentifiedImageError:
         return jsonify({"error": "Invalid image data received (cannot be opened by PIL)"}), 400
    except Exception as e:
        print(f"Error processing input image: {e}")
        return jsonify({"error": f"Error processing input image: {e}"}), 500
  


    try:
        
        # Configure input prompt.
        default_prompt = "Convert this input sketch into a beautiful, detailed painting. Maintain the core subject and composition of the sketch."

        if user_prompt:
            prompt_text = f"{default_prompt} {user_prompt}"
            print(f"Using user-provided prompt: '{prompt_text}'")
        else:
            prompt_text = default_prompt
            print(f"Using default prompt: '{prompt_text}'")

        contents_input = [prompt_text, sketch_pil_image]
        generation_config = types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
        )

        print(f"Sending request to Gemini model: {MODEL_ID} using client API...")
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents_input,
            config=generation_config,
        )
        print("Received response from Gemini.")


        # Process response from Gemini.
        generated_image_base64 = None
        error_message = None

        if hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
            error_message = f"Request blocked by Gemini: {response.prompt_feedback.block_reason}"
            if response.prompt_feedback.safety_ratings:
                error_message += f" Safety Ratings: {response.prompt_feedback.safety_ratings}"
            print(error_message)
            return jsonify({"error": error_message}), 500

        if not response.candidates:
             error_message = "Gemini returned no candidates in the response."
             if hasattr(response, 'text') and response.text:
                 error_message += f" Response text: {response.text}"
             print(error_message)
             return jsonify({"error": error_message}), 500
        try:
            candidate = response.candidates[0]
            for part in candidate.content.parts:
                if part.inline_data is not None and part.inline_data.mime_type.startswith('image/'):
                    print(f"Found generated image part (MIME type: {part.inline_data.mime_type}).")
                    image_data_bytes = part.inline_data.data
                    mime_type = part.inline_data.mime_type

                    # Encode to base64 and format for web display.
                    generated_image_base64 = base64.b64encode(image_data_bytes).decode('utf-8')
                    generated_image_base64 = f"data:{mime_type};base64,{generated_image_base64}"
                    break 

        except IndexError:
            error_message = "Error accessing response candidate."
            print(error_message)
            return jsonify({"error": error_message}), 500
        except Exception as e:
             error_message = f"Error processing response parts: {e}"
             print(error_message)
             return jsonify({"error": error_message}), 500


        # Return the image generated by Gemini.
        if generated_image_base64:
             return jsonify({"generated_image": generated_image_base64})
        else:
            # If the loop finished without finding an image.
            error_message = "Gemini did not return an image in the response parts."
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason is not None and finish_reason != types.FinishReason.STOP:
                 error_message += f" Finish Reason: {finish_reason.name}"
            print(error_message)
            return jsonify({"error": error_message}), 500

    except Exception as e:
        print(f"An error occurred during the Gemini API call or response handling: {e}")
        error_detail = getattr(e, 'message', str(e))
        return jsonify({"error": f"Failed to generate painting: {error_detail}"}), 500


if __name__ == '__main__':
    # Use debug=True for development only.
    app.run(debug=True, host='0.0.0.0', port=5000)