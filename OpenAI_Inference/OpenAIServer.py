from flask import Flask, request, jsonify
import json
import hashlib
import time
import random 
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import base64

app = Flask(__name__)

images_path = "./shared_task/images"
load_dotenv()
port = os.getenv("PORT", 5000)  # Default to 5000 if not set in .env
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key,
)

@app.route('/v1/chat/completions', methods=['POST'])
def generate():
    try:
        data = json.loads(request.get_json())
        # Record the request arrival time
        processing_start_time = time.time()
        data["arrival_time"] = processing_start_time

        messages = data.get("conversation", [])
        model = data.get("model", "gpt-4o-mini")
        #print("Received messages:", messages)
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            image_path = item.get("image")
                            try:
                                # Extract the file name and construct the path in the "images" folder
                                image_file = os.path.basename(image_path)
                                image_full_path = os.path.join(images_path, image_file)
                                
                                # Check if the image is already base64 encoded
                                with open(image_full_path, "rb") as image_file:
                                    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                                    item["image"] = encoded_image
                                    item["type"] = "image_url"
                                    item["image_url"] = {"url": f"data:image/png;base64,{encoded_image}"}
                                    del item["image"]
                            except Exception as e:
                                print(f"Error encoding image {image_path}: {e}")

        print("Modified messages:", messages)

        # Generate text
        completion = client.chat.completions.create(
                model=f"{model}",
                messages=messages
            )
        output = completion.choices[0].message.content
        #output = generate_text_from_sample(model, processor, messages)

        # Add the generated response to the incoming JSON
        data["response"] = output
        # Find the message in the messages list that has the role "system"
        system_message = next((msg for msg in messages if msg.get("role") == "system"), None)
        user_message = next((msg for msg in messages if msg.get("role") == "user"), None)

        data["sys_prompt"] = system_message.get("content") if system_message else ""
        data["user_prompt"] = user_message.get("content") if user_message else ""
        # Create a hash of the messages list
        messages_str = json.dumps(messages, sort_keys=True)
        data["prompt_id"] = hashlib.sha256(messages_str.encode()).hexdigest()
        data["request_id"] = "justRandomValue_" + str(hashlib.sha256(str(random.randint(0, int(1e6))).encode()).hexdigest())
        data["conversation_id"] = "justRandomValue_"+str(hashlib.sha256(str(random.randint(0, int(1e6))).encode()).hexdigest())

        # Record the processing end time
        data["processing_time"] = time.time() - processing_start_time

        # Record the request finish time
        data["finished_time"] = time.time()

        # Return the modified JSON as the response
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/v1/health")
def health():
    return "OK", 200


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)