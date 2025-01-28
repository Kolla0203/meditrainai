
# MediTrain: A Medical Chatbot

MediTrain is a conversational AI designed to provide medical information and assistance. This chatbot uses the GPT-2 language model to generate human-like responses to user queries.

Features
- Provides medical information on various conditions, symptoms, and medications
- Offers instructions and guidance for patients and doctors
- Supports roles for doctors and patients
- Utilizes GPT-2 for generating responses to user queries

Requirements
- Python 3.7+
- Flask
- Transformers
- torch
- json

Installation
1. Clone the repository: `git clone https://github.com/your-username/MediTrain.git`
2. Install the required libraries: `pip install -r requirements.txt`
3. Run the application: `python app.py`

Usage
1. Send a POST request to `http://127.0.0.1:5001/query` with the following JSON data:
    - `role`: The user's role (doctor or patient)
    - `query`: The user's query
2. The chatbot will respond with a JSON object containing the relevant medical information

API Endpoints
- `/query`: Handles user queries and returns medical information

License
MediTrain is licensed under the MIT License.

Contributing
Contributions are welcome! Please submit a pull request with your changes.

Acknowledgments
- Hugging Face Transformers library for providing the GPT-2 model
- Flask for building the web application

Future Enhancements
- Add support for multilingual queries
- Enhance conversation memory to retain context over extended sessions
- Introduce additional models for region-specific medical advice

#License

This project is licensed under the MIT License.

---

Enjoy using MediTrain and let us know if you have suggestions or encounter issues!