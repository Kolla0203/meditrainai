from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ✅ Load Medical Data
try:
    with open("medicaldata.json", "r", encoding="utf-8") as file:
        medical_data = json.load(file)
except Exception as e:
    print(f"Error loading medical dataset: {e}")
    medical_data = []

# Preprocess medical data to create fast lookup
symptoms_lookup = {}
for idx, condition in enumerate(medical_data):
    for symptom in condition.get('symptoms', []):
        symptoms_lookup[symptom.lower()] = idx  # Map symptom to its condition index

# ✅ Function to Process User Query
def process_query(query):
    query = query.lower()
    matching_conditions = []

    # Search for matching symptoms in the medical data using the preprocessed lookup
    for word in query.split():
        if word in symptoms_lookup:
            matching_conditions.append(symptoms_lookup[word])

    if not matching_conditions:
        return "-> Sorry, I couldn't find any related conditions from your symptoms."

    # Ensure unique matching conditions (some symptoms might appear in multiple conditions)
    matching_conditions = set(matching_conditions)
    
    # Prepare the response
    responses = []
    for idx in matching_conditions:
        condition = medical_data[idx]

        # Basic response
        response = []
        response.append(f"-> Condition: {condition['condition']}")
        response.append(f"-> Symptoms: {', '.join(condition.get('symptoms', []))}")

        # Handle missing medications
        medications = condition.get('medications', [])
        if medications:
            response.append(f"-> Medications: {', '.join(medications)}")
        else:
            response.append("-> Medications: Not available")

        response.append(f"-> Instructions: {condition.get('instructions', 'No instructions provided')}")

        responses.extend(response)

    return "\n".join(responses)


# ✅ Route for Handling Queries
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query parameter is missing"}), 400

    query = data['query']
    response = process_query(query)
    return jsonify({"response": response})


# ✅ Home Route
@app.route('/', methods=['GET'])
def home():
    return "Medical Chatbot API is running! Use POST /query to interact with the chatbot."


# ✅ Run Flask Server
if __name__ == '__main__':
    print("Server is running! Access it at: http://127.0.0.1:5001/")
    app.run(debug=True, host="0.0.0.0", port=5001)