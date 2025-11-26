import requests

URL = "http://127.0.0.1:5000/query"  # endpoint do seu servidor Flask

print("🤖 Chatbot de Receitas RAG")
print("Digite 'sair' para encerrar.\n")

while True:
    pergunta = input("Você: ").strip()
    if pergunta.lower() in ["sair", "exit", "quit"]:
        print("👋 Encerrando chat. Até mais!")
        break

    try:
        resp = requests.post(URL, json={"q": pergunta}, timeout=60)
        if resp.status_code == 200:
            resposta = resp.json().get("answer", "(sem resposta)")
            print(f"Bot: {resposta}\n")
        else:
            print(f"⚠️ Erro HTTP {resp.status_code}: {resp.text}\n")
    except Exception as e:
        print(f"❌ Erro: {e}\n")
