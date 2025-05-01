import os
import json
import sqlite3
import fitz 
import chromadb
import requests
import tempfile
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, HTTPException, Depends, Body, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import uvicorn
import re
from transformers import (
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
import google.generativeai as genai
import jwt  
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

#  Import SentenceTransformer for embeddings.
from sentence_transformers import SentenceTransformer


#  Environment, Gemini & ChromaDB Setup

load_dotenv()
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY") 
SVG_TEMPLATE_PATH="C:/Users/muham/Desktop/medicalapp-backend/ticket.svg"


if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found! Set it as an environment variable.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
class EmbeddingFunctionWrapper:
    def __call__(self, input: list[str]) -> list[list[float]]:
        return embedding_model.encode(input).tolist()

embedding_function = EmbeddingFunctionWrapper()

CHROMA_DB_PATH = "./chroma_db_latest"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(
    "medical_docs", embedding_function=embedding_function
)


# 2. JWT & OAuth2 Configuration

JWT_SECRET = "Shibil@1234"
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if "username" not in payload or "userId" not in payload or "isAdmin" not in payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    return verify_token(token)


# 3. SQLite Database Setup

SQLITE_DB_PATH = "./chat_history.db"
def init_db():
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS upload_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            feedback_label TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            medical_history TEXT,
            allergies TEXT,
            current_medications TEXT
        )
    """)
    conn.commit()
    return conn

db_conn = init_db()

def save_chat_history(user_id: str, role: str, message: str):
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)", (user_id, role, message))
    db_conn.commit()

def save_upload_history(user_id: str, filename: str):
    cursor = db_conn.cursor()
    cursor.execute("INSERT INTO upload_history (user_id, filename) VALUES (?, ?)", (user_id, filename))
    db_conn.commit()

def save_feedback(user_id: str, query: str, response: str, feedback_label: str):
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT INTO feedback (user_id, query, response, feedback_label) VALUES (?, ?, ?, ?)",
        (user_id, query, response, feedback_label)
    )
    db_conn.commit()

def get_all_feedback():
    cursor = db_conn.cursor()
    cursor.execute("SELECT query, feedback_label FROM feedback")
    return cursor.fetchall()

def save_user_profile(username: str, medical_history: str, allergies: str, current_medications: str):
    cursor = db_conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO user_profiles (username, medical_history, allergies, current_medications) VALUES (?, ?, ?, ?)",
        (username, medical_history, allergies, current_medications)
    )
    db_conn.commit()

def get_user_profile(username: str) -> dict:
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT medical_history, allergies, current_medications FROM user_profiles WHERE username=?",
        (username,)
    )
    row = cursor.fetchone()
    if row:
        return {"medical_history": row[0], "allergies": row[1], "current_medications": row[2]}
    return {}

def get_aggregated_feedback(query: str) -> dict:
    cursor = db_conn.cursor()
    cursor.execute(
        "SELECT feedback_label, COUNT(*) FROM feedback WHERE query = ? GROUP BY feedback_label",
        (query,)
    )
    results = cursor.fetchall()
    feedback = {"medical": 0, "non-medical": 0}
    for label, count in results:
        feedback[label] = count
    return feedback


# 4. Utility Functions for PDFs, Classification, and Chroma Storage

UPLOADS_FOLDER = "./uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
distilbert_classifier = None
MODEL_DIR = "./retrained_model"

def load_distilbert_classifier():
    global distilbert_classifier
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        distilbert_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        print("✅ Loaded fine-tuned DistilBERT model from:", MODEL_DIR)
    except Exception as e:
        print("❗ Could not load fine-tuned DistilBERT model. Error:", e)
        distilbert_classifier = None

load_distilbert_classifier()

def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text if text.strip() else None
    except Exception as e:
        print(f"Error reading PDF file '{file_path}': {e}")
        return None

def classify_document(content: str) -> str:
    if not content or not content.strip():
        return "Blank Document"
    snippet = content[:512]
    candidate_labels = ["medical document", "non-medical document"]
    result = zero_shot_classifier(snippet, candidate_labels)
    return "medical document" if result["labels"][0] == "medical document" else "non-medical document"

def store_in_chroma(doc_id: str, text: str, user_id: str = None):
    existing = collection.get(ids=[doc_id])
    if existing.get("documents"):
        print(f"Document with ID {doc_id} already exists. Skipping insertion.")
        return
    metadata = {"source": doc_id, "user_id": user_id or ""}
    collection.add(documents=[text], metadatas=[metadata], ids=[doc_id])

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


#  Query Filtering (Always Generate a Response)

def filter_medical_query(query: str) -> bool:
    """
    If you want to do a quick check before generating a response, you can do it here.
    For now, we always return True.
    """
    if distilbert_classifier:
        result = distilbert_classifier(query)[0]
        label = result["label"]
        conf = result["score"]
        print(f"DistilBERT classified query as {label} with confidence {conf}.")
        # If label is 'non-medical' with high confidence, return False
        # (But we won't do that by default here)
    return True




# 1) Slightly improved extract_location
def extract_location(query: str) -> str:
    """
    Look for “in <place>” or “near <place>”.
    If not found, fallback to the last word (common for "hospitals manjeri").
    """
    q = query.lower().rstrip(" ?!")  # strip trailing punctuation
    if " in " in q:
        return q.split(" in ", 1)[1].strip()
    if " near " in q:
        return q.split(" near ", 1)[1].strip()
    # fallback: assume last word is the city
    tokens = re.findall(r"[a-zA-Z]+", q)
    return tokens[-1] if tokens else ""

# 2) Overpass lookup
def get_hospitals(location: str) -> list[str]:
    if not location:
        return []
    overpass_url = "https://overpass-api.de/api/interpreter"
    # build the area-based query
    query = f"""
[out:json];
area["name"="{location.title()}"]->.searchArea;
(
  node["amenity"="hospital"](area.searchArea);
  way["amenity"="hospital"](area.searchArea);
  relation["amenity"="hospital"](area.searchArea);
);
out center;
"""
    try:
        resp = requests.post(overpass_url, data=query, timeout=10)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        names = []
        for el in elements:
            name = el.get("tags", {}).get("name")
            if name and name not in names:
                names.append(name)
        return names
    except requests.RequestException:
        # you could log the exception here if you like
        return []

def is_hospital_query(query: str) -> bool:
    """
    Returns True if the query is about hospitals or clinics.
    """
    return bool(re.search(r"\bhospital(s)?\b|\bclinic(s)?\b", query.lower()))

# 3) Formatter
def generate_hospital_insights(query: str) -> str:
    city = extract_location(query)
    if not city:
        return "Sorry, I couldn't detect a location in your question."
    hospitals = get_hospitals(city)
    if not hospitals:
        return f"No hospitals found in {city.title()}."
    # return up to 5
    lines = "\n".join(f"- {h}" for h in hospitals[:5])
    return f"Here are some hospitals in {city.title()}:\n{lines}"

#  Response Generation for Chat

def build_conversation_string(history: list) -> str:
    conversation_str = ""
    for msg in history:
        if msg["role"] == "user":
            conversation_str += f"User: {msg['text']}\n"
        else:
            conversation_str += f"Assistant: {msg['text']}\n"
    return conversation_str.strip()

# Load QA model for answering questions about uploaded docs
try:
    qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
except Exception as e:
    print("Error loading QA model:", e)
    raise e

def answer_question(question: str, context: str) -> str:
    inputs = qa_tokenizer(question, context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits)
    answer_tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
    return qa_tokenizer.decode(answer_tokens, skip_special_tokens=True)


# 🧠 Main response generator
def generate_response(query: str, history: list, current_user: dict) -> str:
    # 🚑 Step 0: Handle hospital-related queries first
    if is_hospital_query(query):
        city = extract_location(query)
        hospitals = get_hospitals(city)
        if not city:
            return "Sorry, I couldn't detect a location in your question."
        if not hospitals:
            return f"No hospitals found in {city.title()}."
        hospital_list = "\n".join(f"- {h}" for h in hospitals[:5])
        return f"Here are some hospitals in {city.title()}:\n{hospital_list}"

    # 🔍 Step 1: Optional classification
    if distilbert_classifier is not None:
        classification_result = distilbert_classifier(query)[0]
        predicted_label = classification_result["label"].lower()
        confidence = classification_result["score"]
        print(f"Query classified as {predicted_label} with confidence {confidence:.2f}")
        if predicted_label in ["non-medical", "label_0"] and confidence > 0.5:
            return (
                "I'm sorry, but I only provide medical-related responses. "
                "Please consult another resource for non-medical questions."
            )

    # 📚 Step 2: Retrieve personalized context from ChromaDB
    results = collection.query(query_texts=[query], n_results=5, where={"user_id": current_user["username"]})
    docs = results.get("documents", [])
    flattened_docs = []
    for d in docs:
        if isinstance(d, list):
            flattened_docs.extend(d)
        else:
            flattened_docs.append(str(d))
    selected_context = " ".join(flattened_docs).strip()

    # ✍️ Step 3: Build prompt
    conversation_str = build_conversation_string(history)
    if not selected_context:
        prompt = f"""
You are a helpful medical chatbot. Below is the conversation so far:

{conversation_str}

Now the user has asked: "{query}"

No relevant document context was found.

Instructions:
1. Provide thorough, medically accurate information.
2. Do not prescribe specific medications or give direct diagnoses.
3. Encourage consulting a medical professional for personalized advice.
"""
    else:
        prompt = f"""
You are a helpful medical chatbot. Below is the conversation so far:

{conversation_str}

Now the user has asked: "{query}"

Relevant Document Context (for user {current_user["username"]}):
---
{selected_context}
---

Instructions:
1. Provide thorough, medically accurate information.
2. Do not prescribe specific medications or give direct diagnoses.
3. Encourage consulting a medical professional for personalized advice.
"""

    # 🤖 Step 4: Use Gemini to generate a response
    gemini_response = gemini_model.generate_content(prompt)
    response_text = gemini_response.text.strip() if gemini_response and gemini_response.text else (
        "Sorry, I could not generate a response."
    )
    return response_text

#  Self-Training: Define a Dataset and Training Function

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def retrain_classifier():
    feedback_data = get_all_feedback()
    if not feedback_data:
        print("No feedback data available for training.")
        return "No training data."

    texts = []
    labels = []
    # Map labels: "medical" => 1, "non-medical" => 0
    label_map = {"medical": 1, "non-medical": 0}
    for query, feedback_label in feedback_data:
        if feedback_label in label_map:
            texts.append(query)
            labels.append(label_map[feedback_label])

    if not texts:
        print("No valid training samples.")
        return "No valid training samples."

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = FeedbackDataset(texts, labels, tokenizer)
    try:
        training_args = TrainingArguments(
            output_dir="./retrained_model",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            logging_steps=5,
            save_steps=10,
            learning_rate=2e-5,
            weight_decay=0.01,  # L2 regularization
            logging_dir="./logs",
            report_to="none",
            evaluation_strategy="steps",  # Enable evaluation during training
            eval_steps=10,                # Evaluate every 10 steps
            load_best_model_at_end=True,  # Save the best model based on evaluation loss
        )
    except Exception as e:
        print("Error setting up TrainingArguments:", e)
        return "Error setting up TrainingArguments."

    try:
        from transformers import Trainer, EarlyStoppingCallback
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        model.save_pretrained("./retrained_model")
        tokenizer.save_pretrained("./retrained_model")
        print("Retraining complete. Model updated.")
        return "Retraining complete."
    except Exception as e:
        print("Error during training:", e)
        return f"Training failed: {e}"



#  Symptom Checker

def process_symptoms(symptoms: list, conversation_history: list = None) -> str:
    symptoms_str = ", ".join(symptoms)
    history_str = ""
    if conversation_history:
        history_str = build_conversation_string(conversation_history)
    prompt = f"""
You are a helpful medical assistant. The user has reported the following symptoms: {symptoms_str}.
{("Conversation history: " + history_str) if history_str else ""}
Please ask any necessary follow-up questions to clarify the symptoms and then provide a list of potential health conditions along with any recommendations for next steps.
"""
    gemini_response = gemini_model.generate_content(prompt)
    response_text = gemini_response.text.strip() if gemini_response and gemini_response.text else (
        "Sorry, I could not process the symptoms."
    )
    return response_text


#  Updated Drug Interaction Checker with Fallback

def get_rx_cui(drug_name: str) -> str:
    url = f"https://rxnav.nlm.nih.gov/REST/rxcui.json?name={drug_name}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            rxnorm_ids = data.get("idGroup", {}).get("rxnormId", [])
            if rxnorm_ids:
                return rxnorm_ids[0]
    except Exception as e:
        print(f"Error retrieving RxCUI for {drug_name}: {e}")
    return ""

def fallback_interaction_response(drug_list: list) -> str:
    drugs_str = ", ".join(drug_list)
    prompt = f"""
You are a helpful medical assistant. The user is asking about potential interactions for the following drugs: {drugs_str}.
Because the RxNav service is currently unavailable, provide a general overview of common interactions or considerations for these medications.
Include potential conflicts with herbs, supplements, or foods, severity ratings if known, and any management tips or cautions.
Remind the user to consult a healthcare professional for specific advice.
"""
    gemini_response = gemini_model.generate_content(prompt)
    response_text = gemini_response.text.strip() if gemini_response and gemini_response.text else (
        "Sorry, I could not retrieve drug interaction information at this time."
    )
    return response_text

def check_drug_interactions(drug_list: list) -> str:
    rx_cuis = []
    for drug in drug_list:
        rxcui = get_rx_cui(drug)
        if rxcui:
            rx_cuis.append(rxcui)
        else:
            print(f"RxCUI not found for {drug}.")
    if not rx_cuis:
        return "Could not retrieve standardized identifiers for the provided drugs."
    rx_cuis_str = "+".join(rx_cuis)
    interaction_url = f"https://rxnav.nlm.nih.gov/REST/interaction/list.json?rxcuis={rx_cuis_str}"
    try:
        response = requests.get(interaction_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            interaction_groups = data.get("fullInteractionTypeGroup", [])
            interactions_found = []
            for group in interaction_groups:
                for interaction in group.get("fullInteractionType", []):
                    for pair in interaction.get("interactionPair", []):
                        drug_names = [concept.get("name", "Unknown") for concept in pair.get("minConcept", [])]
                        description = pair.get("description", "No description available.")
                        severity = pair.get("severity", "N/A")
                        interactions_found.append(
                            f"{' vs. '.join(drug_names)}: Severity={severity}. {description}"
                        )
            if interactions_found:
                return "Potential drug interactions found:\n" + "\n".join(interactions_found)
            else:
                return "No significant drug interactions found."
        else:
            print(f"RxNav call returned status {response.status_code}. Using fallback.")
            return fallback_interaction_response(drug_list)
    except Exception as e:
        print(f"Error checking drug interactions with RxNav: {e}")
        return fallback_interaction_response(drug_list)


#  New Feature: Personalized Treatment Recommendations

def generate_personalized_treatment(user_data: dict) -> str:
    medical_history = user_data.get("medical_history", "Not provided")
    allergies = user_data.get("allergies", "Not provided")
    current_medications = user_data.get("current_medications", "Not provided")
    symptoms = user_data.get("symptoms", "Not provided")
    
    # Load the fine-tuned CSV model if it exists, otherwise use the default Gemini model.
    csv_finetuned_model_dir = "./csv_finetuned_model"
    if os.path.exists(csv_finetuned_model_dir):
        try:
            # Load the fine-tuned model for personalized treatment.
            csv_tokenizer = AutoTokenizer.from_pretrained(csv_finetuned_model_dir)
            csv_model = T5ForConditionalGeneration.from_pretrained(csv_finetuned_model_dir)
            prompt = (
                f"Generate personalized treatment recommendations based on the following patient data. "
                f"Medical History: {medical_history}. Allergies: {allergies}. Current Medications: {current_medications}. "
                f"Symptoms: {symptoms}."
            )
            inputs = csv_tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            with torch.no_grad():
                outputs = csv_model.generate(**inputs, max_length=128)
            treatment_recommendation = csv_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return treatment_recommendation
        except Exception as e:
            print("Error generating personalized treatment using CSV fine-tuned model:", e)
    
    # Fallback: Use Gemini model if CSV fine-tuned model is not available.
    prompt = f"""
You are a highly knowledgeable medical assistant. Based on the following patient data, please generate personalized treatment recommendations.
Medical History: {medical_history}
Allergies: {allergies}
Current Medications: {current_medications}
Symptoms: {symptoms}

Provide a detailed treatment plan, including alternative treatments, lifestyle recommendations, and any necessary precautions.
Ensure that the response encourages the patient to consult a medical professional for confirmation.
"""
    gemini_response = gemini_model.generate_content(prompt)
    response_text = gemini_response.text.strip() if gemini_response and gemini_response.text else "Sorry, I could not generate personalized treatment recommendations."
    return response_text



#  Drug Adverse Events Checker using OpenFDA API

def get_adverse_events(drug_name: str):
    url = f"https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:{drug_name}&limit=10"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None


#  Fine-Tune from CSV Data

def fine_tune_from_csv(csv_path: str) -> str:
    import pandas as pd
    from datasets import Dataset
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    df = pd.read_csv(csv_path)
    df["input_text"] = df.apply(
        lambda row: (
            f"Patient_ID: {row['Patient_ID']}. Age: {row['Age']}. Gender: {row['Gender']}. Condition: {row['Condition']}. "
            f"Drug1: {row['drug_1']} ({row['Dosage_1']}, {row['Frequency_1']}, Use: {row['Drug_1_Use']}). "
            f"Drug2: {row['Drug_2']} ({row['Dosage_2']}, {row['Frequency_2']}, Use: {row['Drug_2_Use']}). "
            f"Interaction Severity: {row['Interaction_Severity']}. Interaction Effect: {row['Interaction_Effect']}. "
            f"Side Effect: {row['Side_Effect']}. Allergy: {row['Allergy']}. Lab Test Recommended: {row['Lab_Test_Recommended']}. "
            f"Alternative Drug: {row['Alternative_Drug']}. Personalized Treatment: {row['Personalized_Treatment']}."
        ),
        axis=1
    )
    df["target"] = df["Personalized_Treatment"]

    dataset = Dataset.from_pandas(df[["input_text", "target"]])
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess_function(examples):
        inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target"], max_length=128, truncation=True, padding="max_length")
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    training_args = TrainingArguments(
        output_dir="./csv_finetuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=50,
        evaluation_strategy="no",
        learning_rate=5e-5,
        weight_decay=0.01,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained("./csv_finetuned_model")
    tokenizer.save_pretrained("./csv_finetuned_model")
    return "CSV fine-tuning complete."


#  FastAPI App Setup with CORS

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    payload = {
        "userId": 1,
        "username": username,
        "email": f"{username}@example.com",
        "isAdmin": False,  # Default is false; update as needed in DB
        "sub": username,
    }
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data=payload, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer", "isAdmin": payload["isAdmin"]}




@app.post("/doctact-contact")
async def contact(data: dict):
    try:
        # Parse date and time from preferdate
        prefer_date = data.get("preferdate", "")
        name = data.get("name", "")
        doctor = data.get("doctor", "")
        
        if not prefer_date or not name or not doctor:
            raise ValueError("Missing required data.")
        
        [date, time] = prefer_date.split("T")
        [doct, speciality] = doctor.split("-")

        # Parse the SVG file
        tree = ET.parse(SVG_TEMPLATE_PATH)
        root = tree.getroot()

        # Define the namespace for the SVG
        namespace = {"svg": "http://www.w3.org/2000/svg"}

        # Update the SVG text elements
        def update_text_content(element_id, new_text):
            element = root.find(f".//svg:text[@id='{element_id}']", namespace)
            if element is not None:
                tspan = element.find(".//svg:tspan", namespace)
                if tspan is not None:
                    tspan.text = new_text

        update_text_content("ticket-2-u-appoinmentid", "124232323")
        update_text_content("ticket-2-u-name", name.strip())
        update_text_content("ticket-2-u-date", f"{date.strip()} {time.strip()}")
        update_text_content("ticket-2-u-doctor", doct.strip())
        update_text_content("ticket-2-u-speciality", speciality.strip())

        # Write the modified SVG to a string
        modified_svg = ET.tostring(root, encoding="unicode")

        # Return the modified SVG content
        return Response(content=modified_svg, media_type="image/svg+xml")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="SVG template file not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    



@app.post("/upload")
async def upload_pdf_endpoint(file: UploadFile, current_user: dict = Depends(get_current_user)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDF files are allowed."})
    file_bytes = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(file_bytes)
        temp_pdf.flush()
        temp_filename = temp_pdf.name
    text_content = extract_text_from_pdf(temp_filename)
    os.remove(temp_filename)

    document_type = classify_document(text_content)
    if document_type.lower() == "medical document":
        user_folder = os.path.join(UPLOADS_FOLDER, current_user["username"])
        os.makedirs(user_folder, exist_ok=True)
        file_path = os.path.join(user_folder, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # Chunk, embed, and store with user metadata
        chunks = chunk_text(text_content)
        for i, chunk in enumerate(chunks):
            store_in_chroma(f"{file.filename}_chunk_{i}", chunk, user_id=current_user["username"])
        save_upload_history(current_user["username"], file.filename)
        return {"message": "Medical PDF uploaded and stored successfully!"}
    
    return JSONResponse(
        status_code=400,
        content={"error": f"Uploaded document classified as '{document_type}'. Please upload a valid medical document."},
    )



@app.post("/feedback")
async def feedback_endpoint(
    query: str = Body(...),
    response_text: str = Body(...),
    feedback_label: str = Body(...),
    current_user: dict = Depends(get_current_user)
):
    if feedback_label not in ["medical", "non-medical"]:
        raise HTTPException(status_code=400, detail="Invalid feedback label.")
    save_feedback(current_user["username"], query, response_text, feedback_label)
    return {"message": "Feedback received."}

@app.post("/retrain")
async def retrain_endpoint(current_user: dict = Depends(get_current_user)):
    if not current_user.get("isAdmin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    result = retrain_classifier()
    if "No training data" in result or "No valid training samples" in result:
        return {"message": result}
    load_distilbert_classifier()
    return {"message": result}

@app.post("/train-csv")
async def train_csv_endpoint(current_user: dict = Depends(get_current_user)):
    if not current_user.get("isAdmin"):
        raise HTTPException(status_code=403, detail="Not authorized")
    csv_path = "/workspaces/chatbotv2/medical_ai_training_data.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="CSV file not found.")
    result = fine_tune_from_csv(csv_path)
    return {"message": result}

@app.post("/symptom-checker")
async def symptom_checker_endpoint(
    symptoms: list[str] = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    response_text = process_symptoms(symptoms)
    save_chat_history(current_user["username"], "user", f"Symptoms: {', '.join(symptoms)}")
    save_chat_history(current_user["username"], "bot", response_text)
    return {"message": response_text}

@app.post("/drug-interaction-checker")
async def drug_interaction_checker_endpoint(
    drugs: list[str] = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    response_text = check_drug_interactions(drugs)
    save_chat_history(current_user["username"], "user", f"Drug interaction check for: {', '.join(drugs)}")
    save_chat_history(current_user["username"], "bot", response_text)
    return {"message": response_text}

@app.post("/personalized-treatment")
async def personalized_treatment_endpoint(
    data: dict = Body(...),
    current_user: dict = Depends(get_current_user)
):
    save_user_profile(
        current_user["username"],
        data.get("medical_history", ""),
        data.get("allergies", ""),
        data.get("current_medications", "")
    )
    response_text = generate_personalized_treatment(data)
    save_chat_history(current_user["username"], "user", f"Personalized treatment request: {json.dumps(data)}")
    save_chat_history(current_user["username"], "bot", response_text)
    return {"message": response_text}

@app.post("/drug-adverse-events")
async def drug_adverse_events_endpoint(
    drug: str = Body(..., embed=True),
    current_user: dict = Depends(get_current_user)
):
    adverse_data = get_adverse_events(drug)
    if adverse_data:
        return {"message": adverse_data}
    else:
        raise HTTPException(status_code=500, detail="Failed to fetch adverse events data.")

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008)
        return
    try:
        current_user = verify_token(token)
    except HTTPException:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    conversation_history = []
    try:
        while True:
            user_message = await websocket.receive_text()
            save_chat_history(current_user["username"], "user", user_message)
            conversation_history.append({"role": "user", "text": user_message})

            # Generate the response (which will classify the query if distilbert_classifier is loaded)
            bot_reply = generate_response(user_message, conversation_history, current_user)
            await websocket.send_text(json.dumps({"message": bot_reply}))

            save_chat_history(current_user["username"], "bot", bot_reply)
            conversation_history.append({"role": "assistant", "text": bot_reply})
    except WebSocketDisconnect:
        print(f"User {current_user['username']} disconnected.")
    except Exception as e:
        print(f"WebSocket error for user {current_user['username']}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception as close_error:
            print("Error closing websocket:", close_error)


#   Load and Chunk Medical Dataset

@app.on_event("startup")
def load_medical_dataset():
    dataset_path = "/workspaces/chatbotv2/MAT-TIP_43-MMT_Guidelines2005.pdf"
    if os.path.exists(dataset_path):
        text = extract_text_from_pdf(dataset_path)
        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                store_in_chroma(f"medical_dataset_chunk_{i}", chunk)
            print(" Medical dataset stored in chunks successfully in ChromaDB.")
        else:
            print(" Error: No content found in the dataset.")
    else:
        print(" Error: Medical dataset not found. Skipping dataset loading.")


#  Main: Run FastAPI

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
