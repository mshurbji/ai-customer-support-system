# AI Customer Support System

This project is a simple chatbot that answers customer questions using AI.
It converts text into embeddings, compares queries with stored responses, and returns the most relevant answers. If no good match is found, it generates a response using the AI model.

---

## What this project does

* Converts text data into embeddings
* Finds similar responses using cosine similarity
* Returns top 3 matching answers
* Generates new answers if needed
* Saves results in JSON files

---

## Project structure

* data/ → input files (CSV, JSON)
* src/ → main logic (3 tasks)
* outputs/ → generated results
* main.py → runs everything

---

## How to run

Install requirements:

```
pip install -r requirements.txt
```

Add your API key in `.env`:

```
OPENAI_API_KEY=your_api_key
```

Run:

```
python main.py
```

---

## Notes

* This project was built as part of a practical AI exam
* It shows how to use embeddings + similarity search + chatbot logic
* Sample data is included for testing

