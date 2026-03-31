import json
import time
from datetime import datetime, timezone
import numpy as np
from src.config import client, CHAT_MODEL, EMBEDDING_MODEL


# Get embeddings with retry (skip empty texts)
def get_embeddings_with_retry(texts, model_name=EMBEDDING_MODEL, max_retries=5):
    texts = [str(t) for t in texts if str(t).strip() != ""]
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


# Call chat model with retry in case of errors
def chat_with_retry(messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0
            )
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


# Calculate cosine similarity between vectors
def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_chatbot():
    # Load possible chatbot responses
    with open("data/chatbot_responses.json", "r", encoding="utf-8") as f:
        chatbot_data = json.load(f)

    candidate_responses = []

    # Extract text responses from JSON (handle different formats)
    if isinstance(chatbot_data, list):
        for item in chatbot_data:
            if isinstance(item, str) and item.strip():
                candidate_responses.append(item)
            elif isinstance(item, dict):
                for value in item.values():
                    if isinstance(value, str) and value.strip():
                        candidate_responses.append(value)
                        break

    elif isinstance(chatbot_data, dict):
        for _, value in chatbot_data.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        candidate_responses.append(item)
                    elif isinstance(item, dict):
                        for v in item.values():
                            if isinstance(v, str) and v.strip():
                                candidate_responses.append(v)
                                break
            elif isinstance(value, str) and value.strip():
                candidate_responses.append(value)

    # Remove duplicate responses
    candidate_responses = list(dict.fromkeys(candidate_responses))

    # Generate embeddings for responses
    response_embedding_response = get_embeddings_with_retry(candidate_responses)
    response_vectors = [item.embedding for item in response_embedding_response.data]

    # Example user queries
    test_queries = [
        "How do I get a refund for my order?",
        "When can I talk to someone from support?"
    ]

    # Generate embeddings for queries
    query_embedding_response = get_embeddings_with_retry(test_queries)
    query_vectors = [item.embedding for item in query_embedding_response.data]

    # Initialize conversation history
    conversation_history = [
        {"role": "system", "content": "You are a helpful customer support assistant."}
    ]

    chatbot_results = []
    similarity_threshold = 0.75

    # Process each query
    for query_text, query_vector in zip(test_queries, query_vectors):
        similarities = [
            cosine_similarity(query_vector, rv)
            for rv in response_vectors
        ]

        # Find best matching response
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])

        # Use retrieved response or generate new one
        if best_similarity >= similarity_threshold:
            final_response = candidate_responses[best_index]
        else:
            completion = chat_with_retry(
                conversation_history + [{"role": "user", "content": query_text}]
            )
            final_response = completion.choices[0].message.content

        # Create confidence score and timestamp
        confidence_score = round((best_similarity + 1) / 2, 4)
        timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        # Save result
        chatbot_results.append({
            "query_text": query_text,
            "retrieved_response": final_response,
            "timestamp": timestamp,
            "confidence_score": confidence_score
        })

        # Update conversation history
        conversation_history.append({"role": "user", "content": query_text})
        conversation_history.append({"role": "assistant", "content": final_response})

    # Save chatbot responses
    with open("outputs/sample_chatbot_responses.json", "w", encoding="utf-8") as f:
        json.dump(chatbot_results, f, ensure_ascii=False)

    # Print confirmation
    print("Chatbot responses created successfully.")
    print(chatbot_results)
