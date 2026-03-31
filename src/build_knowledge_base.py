import json
import pandas as pd
from src.config import client, EMBEDDING_MODEL


def build_knowledge_base():
    # Load knowledge base from CSV file
    df = pd.read_csv("data/knowledge_base.csv")

    # Convert text column to a list
    texts = df["document_text"].astype(str).tolist()

    # Generate embeddings for all texts
    embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )

    # Get metadata (use column if available, otherwise combine other columns)
    if "metadata" in df.columns:
        metadata_values = df["metadata"].astype(str).tolist()
    else:
        extra_cols = [c for c in df.columns if c not in ["document_id", "document_text"]]
        if extra_cols:
            metadata_values = df[extra_cols].astype(str).agg(" | ".join, axis=1).tolist()
        else:
            metadata_values = [""] * len(df)

    # Create list with embeddings and metadata
    knowledge_embeddings = []
    for i, row in df.iterrows():
        knowledge_embeddings.append({
            "document_id": int(row["document_id"]),
            "document_text": str(row["document_text"]),
            "embedding_vector": embedding_response.data[i].embedding,
            "metadata": metadata_values[i]
        })

    # Save results to JSON file
    with open("outputs/knowledge_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(knowledge_embeddings, f, ensure_ascii=False)

    # Print small preview
    print("Knowledge base embeddings created.")
    print(knowledge_embeddings[:2])
