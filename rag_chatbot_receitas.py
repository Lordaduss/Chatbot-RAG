
import os
import sys
import argparse
import json
import pickle
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from flask import Flask, request, jsonify

# -------- CONFIG --------
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "faiss_metadata.pkl"
CHUNK_TOKEN_SIZE = 500
EMBED_BATCH = 64

SYSTEM_PROMPT = (
    "Você é um assistente da empresa IREKS que responde sobre receitas presentes na base de dados. "
    "Responda apenas com informações encontradas nas receitas importadas. Se a pergunta não puder ser respondida com essas receitas, recuse educadamente dizendo: 'Desculpe, não possuo informação sobre isso nas receitas.'"
    "Caso existam várias receitas similares, pergunte ao usuário qual delas deseja a informação."
    "Saudações como 'Oi', 'Olá' ou 'Tudo Bem?' podem ser respondidas normalmente."
)

# -------- Helpers --------

def ensure_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não encontrado. Defina antes de rodar.")
    return OpenAI(api_key=api_key)


def num_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    return len(enc.encode(text))


def chunk_text(text: str, max_tokens: int = CHUNK_TOKEN_SIZE, model_name: str = "gpt-4o-mini") -> List[str]:
    enc = tiktoken.encoding_for_model(model_name)
    toks = enc.encode(text)
    chunks = []
    for i in range(0, len(toks), max_tokens):
        c = toks[i : i + max_tokens]
        chunks.append(enc.decode(c))
    return chunks

# -------- Dataset ingestion --------

def load_recipes(csv_path: str = "receitas.csv") -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    cols = df.columns.str.lower()

    def find_col(names):
        for n in names:
            for c in df.columns:
                if n in c.lower():
                    return c
        return None

    title_col = find_col(["title", "nome", "nome_receita", "receita"]) or df.columns[0]
    ingredients_col = find_col(["ingred", "ingredientes", "ingredient"]) or None
    steps_col = find_col(["modo", "preparo", "instru", "passo", "steps"]) or None
    url_col = find_col(["url", "link"]) or None
    id_col = find_col(["id"]) or None

    recipes = []
    for _, row in df.iterrows():
        title = str(row.get(title_col, "")).strip()
        ingredients = str(row.get(ingredients_col, "")).strip() if ingredients_col else ""
        steps = str(row.get(steps_col, "")).strip() if steps_col else ""
        url = str(row.get(url_col, "")).strip() if url_col else ""
        rid = str(row.get(id_col, "")) if id_col else None

        full_text = f"Título: {title}\nIngredientes: {ingredients}\nPreparo: {steps}"

        recipes.append({
            "id": rid or title,
            "title": title,
            "ingredients": ingredients,
            "steps": steps,
            "url": url,
            "text": full_text,
        })
    return recipes

# -------- Indexing / Embeddings --------

def build_embeddings_index(recipes: List[Dict[str, Any]]):
    client = ensure_client()

    docs = []
    for r in recipes:
        text = r["text"]
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            docs.append({
                "text": c,
                "meta": {
                    "recipe_id": r["id"],
                    "title": r["title"],
                    "chunk_index": i,
                    "url": r.get("url", ""),
                },
            })

    vectors = []
    metas = []
    for i in range(0, len(docs), EMBED_BATCH):
        batch_texts = [d["text"] for d in docs[i : i + EMBED_BATCH]]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
        for j, emb in enumerate(resp.data):
            vectors.append(np.array(emb.embedding, dtype="float32"))
            metas.append(docs[i + j]["meta"])

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(vectors))

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metas, f)

    print(f"Index criado: {len(vectors)} vetores, dimensão {dim}")


def load_index():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise RuntimeError("Index não encontrado. Rode --build para criar o índice primeiro.")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metas = pickle.load(f)
    return index, metas

# -------- Retrieval/answering --------

def answer_query(query: str, k: int = 6):
    client = ensure_client()
    index, metas = load_index()

    emb = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    qv = np.array(emb.data[0].embedding, dtype="float32")
    D, I = index.search(np.vstack([qv]), k)

    recipes = load_recipes()
    id_to_text = {r["id"]: r["text"] for r in recipes}

    context_pieces = []
    sources = []
    seen = set()
    for idx in I[0]:
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        rid = m["recipe_id"]
        if rid in seen:
            continue
        seen.add(rid)
        txt = id_to_text.get(rid, "")
        if not txt:
            continue
        context_pieces.append(f"Fonte: {m.get('url','')}\n{txt}")
        sources.append(m.get("url") or m.get("title"))

    context = "\n\n---\n\n".join(context_pieces)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Use apenas o conteúdo das receitas fornecidas abaixo para responder. Se a resposta não estiver nas receitas, recuse. "
                f"Contexto:\n{context}\n\nPergunta: {query}"
            ),
        },
    ]

    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=512, temperature=0)
    answer = resp.choices[0].message.content.strip()

    return {"answer": answer, "sources": sources}

# -------- Flask server --------
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def http_query():
    data = request.get_json(force=True)
    q = data.get("q")
    if not q:
        return jsonify({"error": "Campo 'q' obrigatório"}), 400
    res = answer_query(q)
    return jsonify(res)

# -------- CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Construir índice a partir do CSV")
    parser.add_argument("--serve", action="store_true", help="Rodar servidor Flask (assume índice criado)")
    args = parser.parse_args()

    if args.build:
        recipes = load_recipes()
        build_embeddings_index(recipes)
        sys.exit(0)

    if args.serve:
        print("Rodando servidor em http://127.0.0.1:5000")
        app.run()

    parser.print_help()
