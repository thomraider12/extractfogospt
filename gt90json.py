#!/usr/bin/env python3
"""
Exporta para JSON todos os incêndios com > 90 operacionais e guarda ficheiros KML na pasta `kml/`.

- API: https://api-dev.fogos.pt/new/fires
- Para cada incêndio com man > 90:
    * tenta obter o KML (string ou URL)
    * guarda o KML em kml/<id>_<nome_kml>.kml (ou kml/<id>.kml)
    * tenta extrair o polígono principal e calcular área (km²)
- Guarda resumo em incendios_gt90.json

Alterações: usa headers customizados (User-Agent, Accept, ...) e retry básico para 403/429.
"""

import os
import json
import time
from datetime import datetime, timezone
from typing import List, Tuple, Optional
import requests
import xml.etree.ElementTree as ET

from shapely.geometry import Polygon
from shapely.ops import transform as shapely_transform
import pyproj

API_URL = "https://api-dev.fogos.pt/new/fires"
OUTPUT_JSON = "json/incendios_gt90.json"
KML_DIR = "kml"
TIMEOUT = 15
MIN_OPERACIONAIS = 90  # critério: estritamente > 90

# ---- HEADERS / SESSION -------------------------------------------------------
DEFAULT_HEADERS = {
    # usar um User-Agent plausível de browser reduz a probabilidade de bloqueio
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "pt-PT,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    # não enviar cookies desnecessários
}

SESSION = requests.Session()
SESSION.headers.update(DEFAULT_HEADERS)
SESSION.max_redirects = 5
# Não partilhar cookies previamente; limpa o cookiejar (começa limpa)
SESSION.cookies.clear()

def get_with_retries(url: str, timeout: int = TIMEOUT, headers: Optional[dict] = None, max_retries: int = 3):
    """
    Faz GET com a sessão global, aplica headers opcionais, tenta retry para 429/403 com backoff.
    Devolve o objeto requests.Response ou lança exception.
    """
    attempt = 0
    backoff = 5  # segundos iniciais
    while True:
        attempt += 1
        try:
            if headers:
                r = SESSION.get(url, timeout=timeout, headers=headers)
            else:
                r = SESSION.get(url, timeout=timeout)
        except Exception as e:
            if attempt >= max_retries:
                raise
            wait = backoff * attempt
            print(f"Request error ({e}), retrying in {wait}s (attempt {attempt}/{max_retries})...")
            time.sleep(wait)
            continue

        # sucesso (200, 304, etc.)
        if r.status_code in (200, 201, 202, 203, 204, 304):
            return r

        # lidar com rate limit / bloqueio
        if r.status_code in (429, 403):
            # tenta ler Retry-After
            ra = r.headers.get("Retry-After")
            if ra:
                try:
                    wait = int(ra)
                except Exception:
                    # se o Retry-After for data, usa backoff por segurança
                    wait = backoff * attempt
            else:
                wait = backoff * attempt
            print(f"Recebido {r.status_code} de {url}. Aguardar {wait}s antes de nova tentativa (attempt {attempt}/{max_retries}).")
            if attempt >= max_retries:
                # devolve a resposta para que o chamador possa inspecionar
                r.raise_for_status()
            time.sleep(wait)
            continue

        # outros códigos -> levantar
        r.raise_for_status()

# --- utilitários ---------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fetch_api(url: str, timeout: int = TIMEOUT):
    """
    Faz GET à API e devolve JSON. Usa headers e retries.
    """
    r = get_with_retries(url, timeout=timeout)
    # tenta decodificar JSON (pode lançar)
    return r.json()

def fetch_kml_if_url(maybe_url_or_kml: str, timeout: int = TIMEOUT) -> str:
    """Se for URL, descarrega com headers apropriados; senão devolve a string (pode já ser XML)."""
    if not maybe_url_or_kml:
        return ""
    s = maybe_url_or_kml.strip()
    # detectar provável URL
    if s.lower().startswith(("http://", "https://")):
        # para KML aceitar conteúdo XML
        headers = {
            "Accept": "application/vnd.google-earth.kml+xml, application/xml, text/xml, */*",
            "User-Agent": DEFAULT_HEADERS["User-Agent"],
        }
        try:
            r = get_with_retries(s, timeout=timeout, headers=headers)
            return r.text
        except Exception as e:
            print(f"  ⚠️ Erro ao descarregar KML de {s}: {e}")
            return ""
    # não é URL -> devolve a string (pode já ser KML)
    return s

def extract_polygons_from_kml_string(kml_string: str) -> List[List[Tuple[float, float]]]:
    if not kml_string:
        return []
    try:
        root = ET.fromstring(kml_string)
    except ET.ParseError:
        try:
            root = ET.fromstring(kml_string.strip())
        except Exception:
            return []

    coords_nodes = root.findall('.//{http://www.opengis.net/kml/2.2}coordinates')
    if not coords_nodes:
        coords_nodes = root.findall('.//coordinates')

    polygons = []
    for cn in coords_nodes:
        if cn is None or cn.text is None:
            continue
        tokens = cn.text.strip().split()
        poly = []
        for t in tokens:
            parts = t.split(',')
            if len(parts) >= 2:
                try:
                    lon = float(parts[0]); lat = float(parts[1])
                    poly.append((lon, lat))
                except ValueError:
                    continue
        if len(poly) >= 3:
            if poly[0] != poly[-1]:
                poly.append(poly[0])
            polygons.append(poly)
    return polygons

def polygon_area_km2(coords: List[Tuple[float, float]]) -> float:
    poly = Polygon(coords)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    poly_m = shapely_transform(transformer.transform, poly)
    return poly_m.area / 1e6

def choose_largest_polygon(polygons: List[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
    best = None
    best_area = -1.0
    for p in polygons:
        try:
            a = polygon_area_km2(p)
            if a > best_area:
                best_area = a
                best = p
        except Exception:
            continue
    return best or []

def extract_kml_name(kml_string: str) -> Optional[str]:
    """Tenta extrair o primeiro <name> do KML (respeitando namespace)."""
    if not kml_string:
        return None
    try:
        root = ET.fromstring(kml_string)
    except Exception:
        try:
            root = ET.fromstring(kml_string.strip())
        except Exception:
            return None
    ns_name = root.findall('.//{http://www.opengis.net/kml/2.2}name')
    if ns_name and ns_name[0].text:
        return ns_name[0].text.strip()
    plain_name = root.findall('.//name')
    if plain_name and plain_name[0].text:
        return plain_name[0].text.strip()
    return None

def safe_filename(s: str) -> str:
    """Sanitiza para usar em nome de ficheiro: remove espaços, carateres perigosos."""
    if not s:
        return ""
    keep = []
    for ch in s:
        # permite letras, dígitos, underscore, hífen, ponto
        if ch.isalnum() or ch in ("_", "-", "."):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
        # senão ignora
    name = "".join(keep)
    return name[:200] if len(name) > 200 else name

# --- main ---------------------------------------------------------------------

def main():
    ensure_dir(KML_DIR)

    try:
        resp = fetch_api(API_URL)
    except Exception as e:
        print("Erro ao aceder à API:", e)
        return

    if not resp.get("success"):
        print("API devolveu success=False")
        return

    data = resp.get("data", []) or []
    candidatos = []
    for inc in data:
        try:
            man = int(inc.get("man", 0) or 0)
        except Exception:
            man = 0
        if man > MIN_OPERACIONAIS:
            candidatos.append(inc)

    registos = []
    for inc in candidatos:
        inc_id = inc.get("id")
        unix_ts = inc.get("dateTime", {}).get("sec")
        if unix_ts:
            time_started = datetime.fromtimestamp(unix_ts).strftime("%d-%m-%Y %H:%M")
        else:
            time_started = "null"

        concelho = inc.get("concelho")
        freguesia = inc.get("freguesia")
        distrito = inc.get("district") or inc.get("distrito")
        status = inc.get("status")
        lat = inc.get("lat")
        lng = inc.get("lng")
        try:
            man = int(inc.get("man", 0) or 0)
        except Exception:
            man = 0
        try:
            terrain = int(inc.get("terrain", 0) or 0)
        except Exception:
            terrain = 0
        try:
            aerial = int(inc.get("aerial", 0) or 0)
        except Exception:
            aerial = 0

        # KML handling
        raw_kml = inc.get("kmlVost") or inc.get("kml") or ""
        kml_saved_path = None
        area_km2 = None
        kml_source = None

        if raw_kml:
            # se for URL, descarrega
            kml_string = fetch_kml_if_url(raw_kml)
            if kml_string:
                kml_source = "kmlVost" if inc.get("kmlVost") else ("kml" if inc.get("kml") else None)
                # tenta extrair um nome a partir do KML
                kml_name = extract_kml_name(kml_string)
                if kml_name:
                    fname = f"{inc_id}_{safe_filename(kml_name)}.kml"
                else:
                    fname = f"{inc_id}.kml"
                path = os.path.join(KML_DIR, fname)
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(kml_string)
                    kml_saved_path = path
                    print(f"  ✓ KML guardado: {path}")
                except Exception as e:
                    print(f"  ⚠️ Erro ao gravar KML para {path}: {e}")
                    kml_saved_path = None

                # tenta extrair polígono e calcular área
                try:
                    polys = extract_polygons_from_kml_string(kml_string)
                    main_poly = choose_largest_polygon(polys)
                    if main_poly:
                        area_km2 = polygon_area_km2(main_poly)
                except Exception as e:
                    print(f"  ⚠️ Erro ao processar KML para área: {e}")
                    area_km2 = None

        reg = {
            "id": inc_id,
            "data_inicio": time_started,
            "concelho": concelho,
            "freguesia": freguesia,
            "distrito": distrito,
            "status": status,
            "latitude": lat,
            "longitude": lng,
            "operacionais": man,
            "terrestres": terrain,
            "aereos": aerial,
            "area_km2": round(area_km2, 6) if isinstance(area_km2, (int, float)) else None,
            "tem_kml": bool(raw_kml),
            "kml_source": kml_source,
            "kml_file": kml_saved_path
        }
        registos.append(reg)

    resultado = {
        "atualizado_em": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "criterio": f"> {MIN_OPERACIONAIS} operacionais",
        "total": len(registos),
        "incendios": registos
    }

    ensure_dir(os.path.dirname(OUTPUT_JSON) or ".")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Guardado JSON resumo em {OUTPUT_JSON} (total: {len(registos)})")
    print(f"   KMLs guardados na pasta: {KML_DIR}")

if __name__ == "__main__":
    main()