import json
import time
import numpy as np
import pandas as pd
from src.config import client, EMBEDDING_MODEL


# Get embeddings with retry in case of API errors
def get_embeddings_with_retry(texts, model_name=EMBEDDING_MODEL, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.embeddings.create(
                model=model_name,
                input=texts
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


# Calculate cosine similarity between two vectors
def cosine_similarity(vector_a, vector_b):
    vector_a = np.array(vector_a, dtype=float)
    vector_b = np.array(vector_b, dtype=float)

    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 0.0

    return float(np.dot(vector_a, vector_b) / denominator)


def run_semantic_search():
    # Load processed user queries
    query_df = pd.read_csv("data/processed_queries.csv")

    # Load predefined responses
    with open("data/predefined_responses.json", "r", encoding="utf-8") as file:
        response_data = json.load(file)

    # Handle different JSON formats (list or dictionary)
    if isinstance(response_data, dict):
        if "responses" in response_data:
            response_items = response_data["responses"]
        else:
            response_items = list(response_data.values())
    else:
        response_items = response_data

    # Extract response text from JSON
    response_texts = []
    for item in response_items:
        if isinstance(item, str):
            response_texts.append(item)
        elif isinstance(item, dict):
            for key in ["response_text", "response", "text", "content", "answer"]:
                if key in item:
                    response_texts.append(str(item[key]))
                    break

    # Convert queries to list
    query_texts = query_df["query_text"].astype(str).tolist()

    # Generate embeddings for queries and responses
    query_embedding_result = get_embeddings_with_retry(query_texts)
    response_embedding_result = get_embeddings_with_retry(response_texts)

    query_vectors = [item.embedding for item in query_embedding_result.data]
    response_vectors = [item.embedding for item in response_embedding_result.data]

    matched_query_results = []

    # Compare each query with all responses
    for index, row in query_df.iterrows():
        similarity_scores = [
            cosine_similarity(query_vectors[index], response_vector)
            for response_vector in response_vectors
        ]

        # Get top 3 most similar responses
        top_indices = np.argsort(similarity_scores)[::-1][:3]

        top_responses = [response_texts[i] for i in top_indices]
        confidence_scores = [
            round((similarity_scores[i] + 1) / 2, 4)
            for i in top_indices
        ]

        # Store results
        matched_query_results.append({
            "query_id": int(row["query_id"]),
            "query_text": str(row["query_text"]),
            "top_responses": top_responses,
            "confidence_scores": confidence_scores
        })

    # Save results to JSON file
    with open("outputs/query_responses.json", "w", encoding="utf-8") as file:
        json.dump(matched_query_results, file, ensure_ascii=False)

    # Print confirmation
    print("Semantic search results created.")
    print("Saved to outputs/query_responses.json")
