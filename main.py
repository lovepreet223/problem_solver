from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import requests
import os
import json
import torch
from utils.embeddings import get_embedding
from utils.file_hasher import compute_hash
from sentence_transformers import util
from Levenshtein import ratio
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from utils import question_solver

# Load environment variables from .env file
load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_BASE_URL = os.getenv("AIPROXY_BASE_URL", "https://aiproxy.sanand.workers.dev/openai")
USE_OPENAI_API = os.getenv("USE_OPENAI_API", "false").lower() == "true"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load files once when the app starts
with open("files/hashes.json", "r", encoding="utf-8") as f:
    stored_hashes = json.load(f)

with open("answers/answers.json", "r", encoding="utf-8") as f:
    answers = json.load(f)

# Load embeddings from JSON
with open("questions/embeddings.json", "r", encoding="utf-8") as f:
    saved_embeddings = json.load(f)
# Convert saved embeddings to tensors
saved_tensors = {key: torch.tensor(embedding) for key, embedding in saved_embeddings.items()}

# Load embeddings from JSON
with open("questions/questions.json", "r", encoding="utf-8") as f:
    saved_questions = json.load(f)

# Load embeddings from JSON
with open("utils/question_solver.json", "r", encoding="utf-8") as f:
    question_solver_json = json.load(f)

@app.get("/")
async def root():
    return {"status": "healthy", "message": "API is running"}

@app.post("/api/")
async def get_answer(
        question: str = Form(...),
        file: UploadFile = File(None)
):
    most_similar_question, similarity_score, stored_question = await compare_embeddings(question)
    levenshtein_similarity = ratio(stored_question, question)

    if similarity_score > 0.80 or levenshtein_similarity > 0.80:
        answer = await compare_questions(question, stored_question)
        if answer == 'yes':
            print("Maybe using pre stored answers...")
            if file:
                file_content = await file.read()
                file_hash = await compute_hash(file)

                if stored_hashes.get(most_similar_question) == file_hash:
                    if answers.get(most_similar_question) is not None:
                        return {"answer": answers[most_similar_question]}
            else:
                if answers.get(most_similar_question) is not None:
                    return {"answer": answers[most_similar_question]}

    if (similarity_score > 0.50 or levenshtein_similarity > 0.50) and most_similar_question in question_solver_json:
        print("Calculating answers...")
        if file:
            file_content = await file.read()
            func = getattr(question_solver, most_similar_question)
            result = func(question, file)
            if result is not None:
                    return {"answer": result}
        else:
            func = getattr(question_solver, most_similar_question)
            result = func(question)
            print(result)
            if result is not None:
                return {"answer": result}

    # if nothing works trying luck with prestored answers
    return {"answer": answers[most_similar_question]}

async def compare_questions(userquestion: str, storedquestion: str) -> str:
    """Queries OpenAI or AIProxy LLM to get the answer."""
    api_url = (
        "https://api.openai.com/v1/chat/completions"
        if USE_OPENAI_API
        else f"{AIPROXY_BASE_URL}/v1/chat/completions"
    )
    api_key = OPENAI_API_KEY if USE_OPENAI_API else AIPROXY_TOKEN

    # Prepare prompt
    content0 = ("Compare the two questions and determine if they are likely to have the same answer based on key values"
                " and contextual meaning. Check clearly their variables or their calculation numbers etc. If both"
                " questions have the same expected answer, respond with 'yes'. If they differ, respond with 'no'."
                "CAUTION: based on your answer this will be used for really critical purposes, so even if you have to"
                " compare it word by word, do it. if you think variables for arriving at an answer has changed just"
                " answer 'no'")
    content1 = ("Inputs:"
                "User Question: {userquestion}"
                "Stored Question: {storedquestion}")

    content1 = content1.replace("{userquestion}", userquestion)
    content1 = content1.replace("{storedquestion}", storedquestion)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system",
             "content": "You have no memory of past conversations. Answer the current question only. " + content0},
            {"role": "user", "content": content1},
        ],
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"API Error: {response.text}")

    try:
        llm_response = response.json()["choices"][0]["message"]["content"].strip()

        # Ensure that LLM response is parsed correctly
        parsed_response = json.loads(llm_response) if llm_response.startswith("{") else llm_response

        return parsed_response  # Return parsed JSON correctly

    except (KeyError, IndexError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Unexpected LLM response format.")


async def compare_embeddings(question):
    # Convert list to tensor for comparison
    query_embedding = get_embedding(question)
    query_tensor = torch.tensor(query_embedding)  # Convert to tensor

    # Compute cosine similarity
    similarities = {key: util.pytorch_cos_sim(query_tensor, emb).item() for key, emb in saved_tensors.items()}
    # Sort and get the most similar question
    most_similar = max(similarities, key=similarities.get)
    similarity_score = similarities[most_similar]

    # get question too
    question = saved_questions[most_similar]

    return most_similar, similarity_score, question


'''def query_llm(question: str, file_content: bytes = None) -> str:
    """Queries OpenAI or AIProxy LLM to get the answer."""
    api_url = (
        "https://api.openai.com/v1/chat/completions"
        if USE_OPENAI_API
        else f"{AIPROXY_BASE_URL}/v1/chat/completions"
    )
    api_key = OPENAI_API_KEY if USE_OPENAI_API else AIPROXY_TOKEN

    # Prepare prompt
    content = prompt().replace("{question_text}", question)

    if file_content:
        try:
            file_text = file_content.decode(errors="ignore")  # Safe decoding
        except UnicodeDecodeError:
            file_text = "File content could not be decoded."
        content = content.replace("{file_content}", file_text)
    else:
        content = content.replace("{file_content}", "No file provided.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system",
             "content": "You have no memory of past conversations. Answer the current question only."},
            {"role": "user", "content": content},
        ],
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"API Error: {response.text}")

    try:
        llm_response = response.json()["choices"][0]["message"]["content"].strip()

        # Ensure that LLM response is parsed correctly
        parsed_response = json.loads(llm_response) if llm_response.startswith("{") else {"answer": llm_response}

        return parsed_response  # Return parsed JSON correctly

    except (KeyError, IndexError, json.JSONDecodeError):
        raise HTTPException(status_code=500, detail="Unexpected LLM response format.")


def prompt() -> str:
    """Returns the LLM prompt template."""
    return (
        'You are an AI trained to answer questions from graded assignments. '
        'Your task is to analyze the given question carefully and provide a direct answer in the expected format. '
        'DO NOT include explanations, justifications, or extra details—only the final answer. '
        'If the question involves calculations, return only the final numerical value. '
        'If the question asks for a specific word or phrase, return only that phrase. '
        'If the question requires extracting information from an uploaded file, process it and return only the requested data. '
        'If the question is ambiguous, make the best logical assumption and return the answer accordingly.\n\n'
        'Question: {question_text}\n'
        '(Optional) File Content: {file_content}\n\n'
        'Expected Output:\n'
        '{"answer": "your_final_answer_here"}'
    )
'''

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
