from sentence_transformers import SentenceTransformer
import json
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
MODEL_PATH = f"./models/{MODEL_NAME}"

# Load the model once globally
if not os.path.exists(MODEL_PATH):
    print("Embedding model doesn't exist. Downloading the model...")
    model = SentenceTransformer(MODEL_NAME)
    model.save(MODEL_PATH)
    print(f"Model downloaded and saved to {MODEL_PATH}")
else:
    print(f"Model already exists at {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH)
    print("Model loaded successfully from local path.")

def get_embedding(text):
    """Returns the embedding of the given text."""
    return model.encode(text).tolist()

# Computes embeddings for all the questions
def compute_embeddings(input_file="./questions/questions.json", output_file="./questions/embeddings.json"):
    with open(input_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Compute embeddings
    embeddings = {key: get_embedding(question) for key, question in questions.items()}

    # Save embeddings to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)

    print(f"Embeddings saved to {output_file}")
