
from dataIngester import index
from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from googleapiclient.discovery import build
from email.mime.text import MIMEText
import base64
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import json
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import keys

# Load credentials from JSON file
with open('token.json') as json_file:
    credentials = json.load(json_file)

oauth_credentials = Credentials.from_authorized_user_info(credentials)

service = build('gmail', 'v1', credentials=oauth_credentials)

openai.api_key = keys.OPENAI_KEY
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key=keys.PINECONE_KEY, environment='us-east4-gcp')
index = pinecone.Index('langchain-chatbot')

uri = "mongodb+srv://test:test@test-questions-data.efjpcoo.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

database_name = "question_answer_db"
db = client[database_name]

answered_collection = db["answered_questions"]
unanswered_collection = db["unanswered_questions"]
noans_collection = db["noans_questions"]

def add_QA_DB(question, answer, mail, answered=True):
    # Utility function to add data to the collections and return the unique ID
    
    if answered:
        data = {"question": question, "answer": answer}
        collection = answered_collection
    else:
        data = {"question": question, "answer": answer, "mail":mail}
        collection = unanswered_collection

    # Insert the data and retrieve the unique ID assigned to the document
    result = collection.insert_one(data)
    unique_id = result.inserted_id
    return unique_id

def add_QA_DB_NoAns(question):
    # Utility function to add data to the collections and return the unique ID
    
    
    data = {"question": question}
    collection = noans_collection
    
    # Insert the data and retrieve the unique ID assigned to the document
    result = collection.insert_one(data)
    unique_id = result.inserted_id
    return unique_id

def send_mail(mail, query):
    # Compose the email details
    recipient_email = mail
    subject = 'Hello from Gmail API'
    message_text = query

    # Create the Message object
    message = MIMEText(message_text)
    message['to'] = recipient_email
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    send_request = service.users().messages().send(userId='me', body={'raw': raw_message})
    response = send_request.execute()

    print('Email sent successfully.')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=5, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']+result['matches'][2]['metadata']['text']+result['matches'][3]['metadata']['text']+result['matches'][4]['metadata']['text']

def getResponse(query):
    return "hello"+query




# docs = split_docs(documents)
# print(len(docs))