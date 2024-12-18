import os 
from dotenv import load_dotenv
from groq import Groq
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_chatbot_response

load_dotenv()

app = Flask(__name__)
allowed_origin = os.getenv('ALLOWED_ORIGIN')
CORS(app, resources={r"/*": {"origins": allowed_origin}})

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def get_response(text):
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": """You are a Doctor and you are advising a patient on the importance of regular health check-ups."""
        },
        {
            "role": "user",
            "content": text,
        }
    ],
    model="llama3-8b-8192",
)
    return chat_completion.choices[0].message.content


@app.route('/', methods=['GET'])
def checkHealth():
    try:
        return jsonify({"status": "Health check"}), 200
    except Exception as e:
        return jsonify({"error": "An error occurred"}), 500


@app.route('/response', methods=['POST'])
def response():
    try:
        data = request.get_json()
        query = data.get('query')
        response = get_chatbot_response(query)
        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": "An error occurred"}), 500    
    
def get_users():
    url = "https://randomuser.me/api/?results=10"
    response = requests.get(url)
    return response.json()


@app.route("/test_users", methods=["GET"])
def test_users():
    try:
        response = get_users()
        users = response["results"]
        return jsonify(users)

    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500    

port = int(os.getenv('PORT', 5000))
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

# print(chat_completion.choices[0].message.content)
# print(chat_completion)