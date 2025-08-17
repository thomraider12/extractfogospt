#!/usr/bin/env python3
import requests
import json
from datetime import datetime

# URL da API
API_URL = "https://api-lb.fogos.pt/v1/now/data"

# Ficheiro de saída
OUTPUT_JSON = "json/resumo_total.json"

def fetch_latest_data():
    response = requests.get(API_URL)
    response.raise_for_status()  # levanta erro se houver problema
    data = response.json()

    if not data.get("success"):
        raise Exception("API retornou sucesso = false")

    all_entries = data.get("data", [])
    if not all_entries:
        raise Exception("Nenhum dado disponível na API")

    # Obter o último elemento (mais recente)
    latest = all_entries[-1]

    # Extrair os campos que queremos
    resumo = {
        "man": latest.get("man", 0),
        "terrain": latest.get("terrain", 0),
        "aerial": latest.get("aerial", 0),
        "total_incendios": latest.get("total", 0),
        "ultima_atualizacao": latest.get("label", "")
    }
    return resumo

def save_to_json(resumo: dict):
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=4)
    print(f"Resumo guardado em {OUTPUT_JSON}")

if __name__ == "__main__":
    resumo = fetch_latest_data()
    save_to_json(resumo)