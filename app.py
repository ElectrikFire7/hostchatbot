import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from assembleModel import download_model_from_s3

load_dotenv()

app = Flask(__name__)
app.config["MONGO_URI"] = os.getenv("DB_URI")
mongo = PyMongo(app)

model = None
tokenizer = None
model_path = "D:/coding_stuff/Chatbot/models/basic_gpt_2"

def test_db_connection():
    try:
        mongo.db.User.estimated_document_count()  # Simple connection test
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise e  # Raise the exception to stop execution if connection fails

# Run the connection test when starting the app
try:
    test_db_connection()
except Exception:
    print("Exiting application due to MongoDB connection failure.")
    exit(1)

def initialize_model(model_path):
    global model, tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    print("successfully loaded model")

def generate_text(sequence, max_length=30):
    try:
        ids = tokenizer.encode(f'{sequence}', return_tensors='pt')
        attention_mask = torch.ones(ids.shape, dtype=torch.long)
        print(ids)
    except Exception as e:
        print(e)
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        attention_mask=attention_mask,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    output = tokenizer.decode(final_outputs[0], skip_special_tokens=True)
    return output

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    if data.get('user_id') == None:
        print("no user id")
        return jsonify({"error": "user_id is required"})
    else:
        userid = data['user_id']

    user_id = ObjectId(userid)  # Convert the string _id to ObjectId
    
    user = mongo.db.users.find_one({"_id": user_id})
    
    if not user:
        print("could not find user")
        return jsonify({"error": "User not found"}), 404
    
    chat_id = user.get('chat_id', None)
    
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "prompt is required"})
    max_length = data.get('max_length', 30)

    response_text = generate_text(prompt, max_length)

    card = {
        "prompt": prompt,
        "response": response_text
    }

    if chat_id:
        mongo.db.Chat.update_one({"_id": chat_id}, {"$push": {"cards": card}})
    else:
        new_chat_id = mongo.db.Chat.insert_one({"user_id": user_id, "cards": [card]}).inserted_id
        new_chat_id = ObjectId(new_chat_id)
        mongo.db.users.update_one({"_id": user_id}, {"$set": {"chat_id": new_chat_id}})

    return jsonify({"response": response_text})

if __name__ == "__main__":

    BUCKET_NAME = "chatbotbucket12"
    S3_MODEL_PATH = "basic_gpt_2"
    LOCAL_MODEL_PATH = "./awsmodel"
    
    download_model_from_s3(BUCKET_NAME, S3_MODEL_PATH, LOCAL_MODEL_PATH)
    initialize_model(LOCAL_MODEL_PATH)

    app.run(port=443)