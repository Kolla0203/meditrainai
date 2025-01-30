from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import spacy
from textblob import TextBlob

app = Flask(__name__)
CORS(app)

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# ✅ Load Medical Data
try:
    with open("medicaldata.json", "r", encoding="utf-8") as file:
        medical_data = json.load(file)
except Exception as e:
    print(f"Error loading medical dataset: {e}")
    medical_data = []

# ✅ Find Best Matching Condition Using NLP
def find_best_match(user_symptoms):
    matched_conditions = []
    
    for condition in medical_data:
        condition_symptoms = set(symptom.lower() for symptom in condition.get("symptoms", []))
        user_symptoms_set = set(symptom.lower() for symptom in user_symptoms)

        # Match if at least 50% of symptoms overlap
        common_symptoms = condition_symptoms.intersection(user_symptoms_set)
        match_percentage = len(common_symptoms) / max(len(condition_symptoms), 1)  # Avoid division by zero
        
        if match_percentage >= 0.5:  
            matched_conditions.append((condition, match_percentage))

    # Sort conditions by highest match
    matched_conditions.sort(key=lambda x: x[1], reverse=True)
    
    return matched_conditions[0][0] if matched_conditions else None

# ✅ NLP-based Sentence Generation
def generate_nlp_response(condition_data):
    condition = condition_data["condition"]
    symptoms = ", ".join(condition_data["symptoms"])
    medicines = ", ".join(condition_data["medicines"])

    # Constructing the response with NLP-based sentence structuring
    response_template = (
        f"It looks like you may be experiencing {condition}. "
        f"Common symptoms of this condition include {symptoms}. "
        f"To help manage this, the following medicines are often recommended: {medicines}. "
        f"Maintaining a proper routine and making necessary lifestyle adjustments can help in dealing with {condition}. "
        f"Taking prescribed medicines on time and ensuring proper rest can significantly improve recovery. "
        f"Additionally, it is important to keep track of any changes in symptoms. "
        f"Avoiding stress, staying hydrated, and following a nutritious diet can also support better health. "
        f"Early detection and timely care play a crucial role in managing this condition effectively."
    )

    # Using TextBlob to refine the response for better readability
    refined_response = TextBlob(response_template).correct()

    return str(refined_response)

# ✅ Process User Query with NLP
@app.route('/query', methods=['POST'])
def process_query():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Query parameter is missing"}), 400

    # Extract symptoms from user query using NLP
    user_query = data['query'].lower()
    doc = nlp(user_query)
    user_symptoms = [token.text for token in doc if token.pos_ in ["NOUN", "ADJ"]]

    # Find best matching condition
    matched_condition = find_best_match(user_symptoms)

    if not matched_condition:
        return jsonify({"response": "I'm sorry, I couldn't find any related conditions based on your symptoms."})

    # Generate structured NLP-based response
    response_text = generate_nlp_response(matched_condition)
    
    return jsonify({"response": response_text})

# ✅ Run Flask Server
if __name__ == '__main__':
    print("Server running at: http://127.0.0.1:5001/")
    app.run(debug=True, host="0.0.0.0", port=5001)