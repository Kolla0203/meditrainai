from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import json

app = Flask(__name__)
CORS(app)

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set pad_token to eos_token for GPT-2, since GPT-2 doesn't have a pad_token by default
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

# Load the dataset (database.json)
with open('dataset.json', 'r') as f:
    medical_data = json.load(f)

# Helper function to generate text using the model
def generate_response(prompt):
    try:
        # Tokenize the input prompt and truncate if it's too long
        tokens = tokenizer.encode(prompt, truncation=True, max_length=512, padding="max_length")

        # Ensure tokens length does not exceed the model's maximum input
        if len(tokens) > 512:
            print(f"Warning: Input too long, truncating to 512 tokens.")
            tokens = tokens[:512]  # Truncate to 512 tokens

        input_ids = torch.tensor([tokens])

        # Generate the response with a limit on the number of new tokens
        output = model.generate(input_ids, max_new_tokens=200)

        # Decode the output and return it
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error: {str(e)}"

# Function to search for medical data based on the query
def search_medical_data(query):
    for entry in medical_data:
        if query.lower() in entry["condition"].lower():
            return entry
    return None

# Endpoint for handling queries
@app.route('/query', methods=['POST'])
def handle_query():
    role = request.json.get('role', 'general')  # 'doctor' or 'patient'
    user_query = request.json.get('query', '')

    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Search the medical data based on the user's query
    medical_info = search_medical_data(user_query)

    if medical_info:
        response = {
            "condition": medical_info["condition"],
            "symptoms": medical_info["symptoms"],
            "medication": medical_info["medication"],
            "instructions": medical_info["instructions"]
        }
    else:
        # If no direct match, use GPT-2 to generate a response
        prompt = f"You are a {role} in a medical setting. The user asked: {user_query}"
        gpt_response = generate_response(prompt)
        response = {
            "role": role,
            "query": user_query,
            "response": gpt_response
        }

    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5001)
