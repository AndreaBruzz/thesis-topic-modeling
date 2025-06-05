from google import genai
import json
import re

def client():
    with open("key.txt", "r") as key_file:
        key = key_file.read().strip()
    
    return genai.Client(api_key=key)

def prompt(doc, query):
    return f"""\
Act like a real user who is evaluating search results. You are given a document and a query.
Return only this JSON:

{{"r": 0 or 1, "n": "max 12 words to explain the choice of relevance"}}

Title: {query['title']}
Narr: {query['narr']}
Desc: {query['desc']}
Doc: {doc}"""

def ask(doc, query, client):
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt(doc, query)
    )

    text = response.text.strip()

    # Strip markdown-style code block if present
    if text.startswith("```json"):
        text = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.DOTALL)

    try:
        body = json.loads(text)
        return body['r'], body['n']
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}\nRaw:\n{text}")
