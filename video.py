#!/usr/bin/env python3
"""
video.py
Gera um vídeo (MP4) com um slide inicial de resumo (resumo_total.json)
seguido dos slides por incêndio (incendios_gt90.json). Usa ícones PNG em emojis/.
Alterações: ordena os incêndios por operacionais (descendente) e aumenta o
espaçamento entre "Resumo Geral" e "Última atualização".
Mostra imagem inc_<id>.png em frame separado quando existir.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# ---------- Config ----------
RESUMO_JSON = "json/resumo_total.json"       # ficheiro com {man, terrain, aerial, total_incendios, ultima_atualizacao}
INPUT_JSON = "json/incendios_gt90.json"      # lista de incêndios
OUTPUT_VIDEO = "video/incendios_gt90_video.mp4"
IMG_SIZE = (1280, 720)            # largura, altura
FPS = 30
DURATION_HOLD = 3.0                # segundos a mostrar o slide sem fades
DURATION_FADE = 0.8                # segundos de fade in/out
BACKGROUND_COLOR = (18, 18, 20)   # fundo escuro (R,G,B)
TEXT_COLOR = (245, 245, 245)      # texto principal
AREA_COLOR = (170, 230, 180)      # cor para área / destaque
FOOTER_COLOR = (150, 150, 150)
OUTPUT_FOURCC = "mp4v"
VERBOSE = True

MAP_IMG_DIR = "images"  # onde estão inc_<id>.png (imagem lateral por incêndio)
EMOJI_DIR = "emojis"    # pasta com fire.png man.png truck.png heli.png tree.png

EMOJI_FILES = {
    "fire": "fire.png",
    "man": "man.png",
    "truck": "truck.png",
    "heli": "heli.png",
    "tree": "tree.png"
}

# Estados -> cor do título
STATE_COLORS = {
    "Despacho": (200, 200, 200),
    "Despacho de 1º Alerta": (200, 170, 80),
    "Chegada ao TO": (255, 200, 80),
    "Em Curso": (230, 90, 80),
    "Em Resolução": (240, 140, 60),
    "Conclusão": (140, 140, 140),
    "Vigilância": (90, 160, 240),
    "Encerrada": (100, 100, 100),
    "Falso Alarme": (150, 100, 100),
    "Falso Alerta": (150, 100, 100),
}
DEFAULT_STATE_COLOR = (230, 90, 80)

# Fonts Windows-first (fallbacks poderão ser ajustadas no runner)
FONT_CANDIDATES_TEXT = [
    "ARLRDBD.TTF",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

def find_font(cands: List[str]) -> Optional[str]:
    for p in cands:
        if os.path.isfile(p):
            return p
    return None

FONT_TEXT_PATH = find_font(FONT_CANDIDATES_TEXT)
if VERBOSE:
    print("FONT_TEXT_PATH:", FONT_TEXT_PATH or "fallback")
    print("EMOJI_DIR:", EMOJI_DIR, "exists=", os.path.isdir(EMOJI_DIR))
    print("MAP_IMG_DIR:", MAP_IMG_DIR, "exists=", os.path.isdir(MAP_IMG_DIR))

# ---------- Utilitários ----------

def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_incidents_from_json(path: str) -> List[Dict[str, Any]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")
    data = load_json_file(path)
    if isinstance(data, dict):
        if "incendios" in data and isinstance(data["incendios"], list):
            return data["incendios"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        # procurar primeira lista de dicts
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        raise ValueError("Estrutura JSON não reconhecida — espera 'incendios' ou 'data' com lista.")
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("JSON tem um tipo inesperado (não dict nem list).")

def safe_get_name(rec: Dict[str, Any]) -> str:
    for k in ("concelho", "local", "nome", "location", "municipio"):
        v = rec.get(k)
        if v:
            return str(v)
    if rec.get("freguesia") and rec.get("concelho"):
        return f"{rec.get('freguesia')} ({rec.get('concelho')})"
    return str(rec.get("id", "Incêndio"))

def safe_get_int(rec: Dict[str, Any], keys: List[str], default: int = 0) -> int:
    for k in keys:
        v = rec.get(k)
        if v is None:
            continue
        try:
            return int(v)
        except Exception:
            try:
                return int(float(v))
            except Exception:
                continue
    return default

def safe_get_area(rec: Dict[str, Any]) -> Optional[float]:
    for k in ("area_km2", "area", "area_ardida"):
        v = rec.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None

def safe_get_time(rec: Dict[str, Any]) -> str:
    # tenta dateTime.sec (unix)
    dt = rec.get("dateTime")
    if isinstance(dt, dict):
        sec = dt.get("sec") or dt.get("seconds")
        if isinstance(sec, (int, float)) and sec > 0:
            try:
                return datetime.fromtimestamp(int(sec)).strftime("%d-%m-%Y %H:%M")
            except Exception:
                pass
    # campos numéricos
    for k in ("started", "time", "timestamp"):
        v = rec.get(k)
        if isinstance(v, (int, float)):
            try:
                return datetime.fromtimestamp(int(v)).strftime("%d-%m-%Y %H:%M")
            except Exception:
                pass
    # campos texto
    for k in ("date", "data_inicio", "data", "datetime", "created_at", "start", "hour"):
        v = rec.get(k)
        if v:
            if k == "hour" and rec.get("date"):
                return f"{rec.get('date')} {str(v)}"
            return str(v)
    return str(rec.get("id", ""))

def status_color(status: str) -> tuple:
    return STATE_COLORS.get(status, DEFAULT_STATE_COLOR)

def safe_get_incident_id(rec: Dict[str, Any]) -> str:
    """
    Tenta várias chaves comuns para obter um id válido para mapear a imagem.
    Se não encontrar nada razoável devolve string vazia.
    """
    candidates = (
        "incident_id", "incidentId", "id", "inc", "codigo", "uid", "gid", "id_inc", "numero", "codigo_incidente"
    )
    for k in candidates:
        v = rec.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    # procurar em propriedades aninhadas (GeoJSON-like)
    props = rec.get("properties")
    if isinstance(props, dict):
        for k in ("incident_id", "incidentId", "id"):
            v = props.get(k)
            if v:
                s = str(v).strip()
                if s:
                    return s
    # fallback: usar a chave 'id' mesmo que seja timestamp — será sanitizado depois
    v = rec.get("id")
    return str(v).strip() if v is not None else ""

def _sanitize_id_for_filename(s: str) -> str:
    # permite só chars alfanuméricos, underscore e hífen
    return "".join(ch for ch in s if ch.isalnum() or ch in ("_", "-"))

def map_image_path_for_id(inc_id: Any) -> Optional[str]:
    """
    Procura imagens em MAP_IMG_DIR com base num id sanitizado.
    - primeiro verifica inc_<sanitizado>.png
    - depois verifica nomes de ficheiro cuja parte base (sem extensão) contenha o sanitizado
    - por fim verifica se a string original aparece no nome do ficheiro (fallback)
    """
    if inc_id is None:
        return None
    s_raw = str(inc_id).strip()
    if not s_raw:
        return None

    s = _sanitize_id_for_filename(s_raw)
    # procura exacta inc_<id>.ext
    if s:
        for ext in (".png", ".jpg", ".jpeg"):
            fname = f"inc_{s}{ext}"
            path = os.path.join(MAP_IMG_DIR, fname)
            if os.path.isfile(path):
                return path

    # procura por substring na versão sanitizada do nome do ficheiro (sem extensão)
    if os.path.isdir(MAP_IMG_DIR):
        for fn in os.listdir(MAP_IMG_DIR):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            base = os.path.splitext(fn)[0]
            base_sanit = _sanitize_id_for_filename(base)
            if s and s in base_sanit:
                return os.path.join(MAP_IMG_DIR, fn)

        # fallback: procura string raw no nome do ficheiro (útil se id tiver espaços ou timestamps)
        for fn in os.listdir(MAP_IMG_DIR):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            if s_raw in fn:
                return os.path.join(MAP_IMG_DIR, fn)

    return None

# ---------- Ícones (cache) ----------
_ICON_CACHE: Dict[str, Optional[Image.Image]] = {}

def load_icon_img(key: str, target_height: int = 48) -> Optional[Image.Image]:
    if key in _ICON_CACHE:
        return _ICON_CACHE[key].copy() if _ICON_CACHE[key] is not None else None
    fname = EMOJI_FILES.get(key)
    if not fname:
        _ICON_CACHE[key] = None
        return None
    path = os.path.join(EMOJI_DIR, fname)
    if not os.path.isfile(path):
        _ICON_CACHE[key] = None
        return None
    try:
        im = Image.open(path).convert("RGBA")
        ratio = target_height / max(1, im.height)
        new_w = max(1, int(im.width * ratio))
        im = im.resize((new_w, target_height), Image.LANCZOS)
        _ICON_CACHE[key] = im.copy()
        return im
    except Exception:
        _ICON_CACHE[key] = None
        return None

# ---------- Slides ----------

def create_summary_slide(summary: Dict[str, Any], size=IMG_SIZE) -> np.ndarray:
    w, h = size
    img = Image.new("RGBA", (w, h), BACKGROUND_COLOR + (255,))
    draw = ImageDraw.Draw(img)

    # fontes
    try:
        if FONT_TEXT_PATH:
            font_title = ImageFont.truetype(FONT_TEXT_PATH, 63)
            font_big = ImageFont.truetype(FONT_TEXT_PATH, 48)
            font_text = ImageFont.truetype(FONT_TEXT_PATH, 36)
            font_small = ImageFont.truetype(FONT_TEXT_PATH, 27)
        else:
            font_title = ImageFont.load_default()
            font_big = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_small = ImageFont.load_default()
    except Exception:
        font_title = ImageFont.load_default()
        font_big = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # extrair valores do resumo (com fallback)
    man = int(summary.get("man", 0))
    terrain = int(summary.get("terrain", 0))
    aerial = int(summary.get("aerial", 0))
    total_inc = int(summary.get("total_incendios", summary.get("total", 0)))
    ultima = summary.get("ultima_atualizacao", summary.get("label", ""))

    # título
    title = "Resumo Geral"
    subtitle = f"Última atualização: {ultima}" if ultima else ""

    # blocos de linhas (cada linha terá ícone + texto)
    infos = [
        ("man", f"Operacionais: {man}"),
        ("truck", f"Meios terrestres: {terrain}"),
        ("heli", f"Meios aéreos: {aerial}"),
        ("tree", f"Total de incêndios: {total_inc}")
    ]

    # layout
    padding_between_icon_text = 20
    subtitle_spacing = 25
    spacing_title_lines = 31
    spacing_between_lines = 28

    t_bbox = draw.textbbox((0,0), title, font=font_title)
    t_w = t_bbox[2]-t_bbox[0]; t_h = t_bbox[3]-t_bbox[1]
    max_line_w = t_w

    line_metrics = []
    for key, text in infos:
        tb = draw.textbbox((0,0), text, font=font_text)
        text_w = tb[2]-tb[0]; text_h = tb[3]-tb[1]
        icon = load_icon_img(key, target_height=64) if key else None
        icon_w = icon.width if icon else 0
        icon_h = icon.height if icon else 0
        total_w = icon_w + (padding_between_icon_text if icon_w>0 else 0) + text_w
        line_h = max(text_h, icon_h)
        line_metrics.append((key, text, text_w, text_h, icon_w, icon_h, total_w, line_h))
        if total_w > max_line_w:
            max_line_w = total_w

    block_w = max_line_w
    lines_total_h = sum(m[7] for m in line_metrics) + spacing_between_lines*(len(line_metrics)-1 if line_metrics else 0)

    sub_h = 0
    if subtitle:
        sub_bbox = draw.textbbox((0,0), subtitle, font=font_small)
        sub_h = sub_bbox[3] - sub_bbox[1]

    block_h = t_h + (subtitle_spacing + sub_h if subtitle else 0) + spacing_title_lines + lines_total_h

    center_x = w//2
    center_y = h//2
    block_x = center_x - block_w//2
    block_y = center_y - block_h//2

    # ícone grande de fogo centrado à esquerda do bloco
    fire_icon = load_icon_img("fire", target_height=220)
    if fire_icon:
        e_w, e_h = fire_icon.width, fire_icon.height
        e_x = block_x - e_w - 40
        e_y = block_y + (block_h - e_h)//2
        try:
            img.paste(fire_icon, (e_x, e_y), fire_icon)
        except Exception:
            img.paste(fire_icon.convert("RGB"), (e_x, e_y))

    # desenhar título e subtitle
    t_x = center_x - t_w//2
    t_y = block_y
    draw.text((t_x, t_y), title, font=font_title, fill=status_color("Em Curso"))
    if subtitle:
        draw.text((center_x - (draw.textbbox((0,0), subtitle, font=font_small)[2] - draw.textbbox((0,0), subtitle, font=font_small)[0])//2,
                   t_y + t_h + subtitle_spacing),
                  subtitle, font=font_small, fill=FOOTER_COLOR)

    # desenhar linhas info
    y = t_y + t_h + (subtitle_spacing + sub_h if subtitle else 0) + spacing_title_lines
    for (key, text, text_w, text_h, icon_w, icon_h, total_w, line_h) in line_metrics:
        line_x = center_x - total_w//2
        if icon_w > 0:
            icon = load_icon_img(key, target_height=icon_h)
            if icon:
                try:
                    img.paste(icon, (line_x, y + (line_h - icon_h)//2), icon)
                except Exception:
                    img.paste(icon.convert("RGB"), (line_x, y + (line_h - icon_h)//2))
            text_x = line_x + icon_w + padding_between_icon_text
        else:
            text_x = line_x
        text_y = y + (line_h - text_h)//2
        color = AREA_COLOR if key == "tree" else TEXT_COLOR
        draw.text((text_x, text_y), text, font=font_text, fill=color)
        y += line_h + spacing_between_lines

    # rodapé com timestamp (UTC +1)
    now_utc = datetime.utcnow()
    now_pt = now_utc + timedelta(hours=1)
    footer = f"Gerado: {now_pt.strftime('%d-%m-%Y %H:%M')}"
    draw.text((40, h - 48), footer, font=font_small, fill=FOOTER_COLOR)

    final = img.convert("RGB")
    return np.asarray(final)

def create_incident_slide(rec: Dict[str, Any], size=IMG_SIZE) -> np.ndarray:
    """
    Cria slide para um registo de incêndio.
    Mostra o nome numa linha e o estado (status) numa linha abaixo, com o status um pouco menor.
    """
    w, h = size
    img = Image.new("RGBA", (w, h), BACKGROUND_COLOR + (255,))
    draw = ImageDraw.Draw(img)

    # fontes
    try:
        if FONT_TEXT_PATH:
            title_size = 64
            status_size = 40   # estado um pouco menor que o título
            font_title = ImageFont.truetype(FONT_TEXT_PATH, title_size)
            font_status = ImageFont.truetype(FONT_TEXT_PATH, status_size)
            font_text = ImageFont.truetype(FONT_TEXT_PATH, 44)
            font_small = ImageFont.truetype(FONT_TEXT_PATH, 28)
        else:
            font_title = ImageFont.load_default()
            font_status = ImageFont.load_default()
            font_text = ImageFont.load_default()
            font_small = ImageFont.load_default()
    except Exception:
        font_title = ImageFont.load_default()
        font_status = ImageFont.load_default()
        font_text = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # campos
    name = safe_get_name(rec)
    status = rec.get("status", "")
    operacionais = safe_get_int(rec, ["operacionais", "man", "oper"], 0)
    terrestres = safe_get_int(rec, ["terrestres", "terrain", "terrestre"], 0)
    aereos = safe_get_int(rec, ["aereos", "aerial", "aerials"], 0)
    area = safe_get_area(rec)
    tempo = safe_get_time(rec)
    inc_id = safe_get_incident_id(rec)

    # linhas de info
    title_name = name
    title_status = status  # será desenhado na 2ª linha, menor
    lines_info = [
        ("man", f"Operacionais: {operacionais}"),
        ("truck", f"Meios terrestres: {terrestres}"),
        ("heli", f"Meios aéreos: {aereos}")
    ]
    if area is not None:
        lines_info.append(("tree", f"Área ≈ {area:.3f} km²"))

    # medir bloco
    padding_between_icon_text = 16
    spacing_title_lines = 20
    spacing_between_lines = 16
    status_line_spacing = 8  # espaço entre nome e estado

    # medir título (duas linhas)
    nb = draw.textbbox((0, 0), title_name, font=font_title)
    name_w = nb[2] - nb[0]; name_h = nb[3] - nb[1]
    sb = draw.textbbox((0, 0), title_status, font=font_status)
    status_w = sb[2] - sb[0]; status_h = sb[3] - sb[1]

    # largura inicial = largura da maior linha do título
    t_w = max(name_w, status_w)
    t_h_total = name_h + status_line_spacing + status_h

    max_line_w = t_w

    # medir linhas de informação (com ícones)
    line_metrics = []
    for key, text in lines_info:
        tb = draw.textbbox((0,0), text, font=font_text)
        text_w = tb[2]-tb[0]; text_h = tb[3]-tb[1]
        icon = load_icon_img(key, target_height=48) if key else None
        icon_w = icon.width if icon else 0
        icon_h = icon.height if icon else 0
        total_w = icon_w + (padding_between_icon_text if icon_w>0 else 0) + text_w
        line_h = max(text_h, icon_h)
        line_metrics.append((key, text, text_w, text_h, icon_w, icon_h, total_w, line_h))
        if total_w > max_line_w:
            max_line_w = total_w

    block_w = max_line_w
    lines_total_h = sum(m[7] for m in line_metrics) + spacing_between_lines * (len(line_metrics)-1 if line_metrics else 0)
    block_h = t_h_total + spacing_title_lines + lines_total_h

    center_x = w // 2
    center_y = h // 2
    block_x = center_x - block_w // 2
    block_y = center_y - block_h // 2

    # ícone fire à esquerda
    fire_icon = load_icon_img("fire", target_height=180)
    if fire_icon:
        e_w, e_h = fire_icon.width, fire_icon.height
        e_x = block_x - e_w - 40
        e_y = block_y + (block_h - e_h) // 2
        try:
            img.paste(fire_icon, (e_x, e_y), fire_icon)
        except Exception:
            img.paste(fire_icon.convert("RGB"), (e_x, e_y))

    # desenhar nome (linha 1) centrado
    t_x_name = center_x - name_w // 2
    t_y = block_y
    draw.text((t_x_name, t_y), title_name, font=font_title, fill=TEXT_COLOR)

    # desenhar estado (linha 2) centrado e menor, com cor do estado
    t_x_status = center_x - status_w // 2
    t_y_status = t_y + name_h + status_line_spacing
    draw.text((t_x_status, t_y_status), title_status, font=font_status, fill=status_color(status))

    # desenhar linhas de estatísticas
    y = t_y_status + status_h + spacing_title_lines
    for (key, text, text_w, text_h, icon_w, icon_h, total_w, line_h) in line_metrics:
        line_x = center_x - total_w // 2
        if icon_w > 0:
            icon = load_icon_img(key, target_height=icon_h)
            if icon:
                try:
                    img.paste(icon, (line_x, y + (line_h - icon_h)//2), icon)
                except Exception:
                    img.paste(icon.convert("RGB"), (line_x, y + (line_h - icon_h)//2))
            text_x = line_x + icon_w + padding_between_icon_text
        else:
            text_x = line_x
        text_y = y + (line_h - text_h) // 2
        draw.text((text_x, text_y), text, font=font_text, fill=AREA_COLOR if key=="tree" else TEXT_COLOR)
        y += line_h + spacing_between_lines

    # rodapé (apenas tempo)
    draw.text((40, h - 48), f"Tempo: {tempo}", font=font_small, fill=FOOTER_COLOR)

    final = img.convert("RGB")
    return np.asarray(final)

def create_image_slide(image_path: str, rec: Dict[str, Any], size=IMG_SIZE) -> np.ndarray:
    """
    Cria um slide separado apenas com a imagem do incêndio.
    """
    w, h = size
    img = Image.new("RGBA", (w, h), BACKGROUND_COLOR + (255,))
    draw = ImageDraw.Draw(img)

    # fontes
    try:
        if FONT_TEXT_PATH:
            font_title = ImageFont.truetype(FONT_TEXT_PATH, 48)
            font_small = ImageFont.truetype(FONT_TEXT_PATH, 28)
        else:
            font_title = ImageFont.load_default()
            font_small = ImageFont.load_default()
    except Exception:
        font_title = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # carregar e redimensionar imagem
    try:
        side = Image.open(image_path).convert("RGBA")
        max_width = int(w * 0.8)
        max_height = int(h * 0.7)
        ratio = min(max_width / side.width, max_height / side.height)
        new_w = int(side.width * ratio)
        new_h = int(side.height * ratio)
        side_resized = side.resize((new_w, new_h), Image.LANCZOS)
    except Exception as e:
        if VERBOSE:
            print(f"  ⚠️ Erro a processar imagem {image_path}: {e}")
        return np.asarray(img.convert("RGB"))

    # centralizar imagem
    img_x = (w - new_w) // 2
    img_y = (h - new_h) // 2

    # adicionar título
    name = safe_get_name(rec)
    title = f"Mapa: {name}"
    t_bbox = draw.textbbox((0, 0), title, font=font_title)
    t_w = t_bbox[2] - t_bbox[0]
    t_x = (w - t_w) // 2
    t_y = img_y - 60
    draw.text((t_x, t_y), title, font=font_title, fill=TEXT_COLOR)

    # colocar imagem
    try:
        img.paste(side_resized, (img_x, img_y), side_resized)
    except Exception:
        img.paste(side_resized.convert("RGB"), (img_x, img_y))

    # rodapé com tempo
    tempo = safe_get_time(rec)
    draw.text((40, h - 48), f"Tempo: {tempo}", font=font_small, fill=FOOTER_COLOR)

    final = img.convert("RGB")
    return np.asarray(final)

# ---------- Fade frames ----------
def make_frames_for_slide(slide_arr: np.ndarray, fps=FPS, dur_hold=DURATION_HOLD, dur_fade=DURATION_FADE) -> List[np.ndarray]:
    hold_frames = int(round(dur_hold * fps))
    fade_frames = int(round(dur_fade * fps))
    total = hold_frames + 2 * fade_frames
    slide_float = slide_arr.astype(np.float32) / 255.0
    frames = []
    for i in range(total):
        if i < fade_frames:
            alpha = (i + 1) / max(1, fade_frames)
        elif i >= (fade_frames + hold_frames):
            alpha = 1.0 - ((i - (fade_frames + hold_frames) + 1) / max(1, fade_frames))
        else:
            alpha = 1.0
        frame = (slide_float * alpha)
        frames.append((frame * 255.0).astype(np.uint8))
    return frames

# ---------- Main ----------
def main():
    # carregar resumo (se existir)
    summary = None
    if os.path.isfile(RESUMO_JSON):
        try:
            summary = load_json_file(RESUMO_JSON)
            if VERBOSE:
                print("Resumo carregado de", RESUMO_JSON, "->", summary)
        except Exception as e:
            print("Erro ao ler resumo:", e)

    # carregar incêndios
    try:
        incidents = load_incidents_from_json(INPUT_JSON)
    except Exception as e:
        print("Erro ao carregar JSON de incêndios:", e)
        return

    # filtrar por operacionais > 90 (mantém a tua lógica)
    filtered = []
    for rec in incidents:
        oper = safe_get_int(rec, ["operacionais", "man", "oper"])
        if oper > 1:
            filtered.append(rec)

    # ordenar por operacionais desc (mais operacionais primeiro)
    incidents = sorted(filtered, key=lambda r: safe_get_int(r, ["operacionais", "man", "oper"]), reverse=True)

    if not incidents and not summary:
        print("Nenhum conteúdo para gerar o vídeo (nenhum incêndio e nenhum resumo).")
        return

    all_frames: List[np.ndarray] = []

    # slide de resumo inicial (se existir)
    if summary:
        if VERBOSE:
            print("Criar slide de resumo inicial...")
        slide = create_summary_slide(summary, size=IMG_SIZE)
        frames = make_frames_for_slide(slide, fps=FPS, dur_hold=DURATION_HOLD, dur_fade=DURATION_FADE)
        all_frames.extend(frames)

    # slides por incêndio
    for idx, rec in enumerate(incidents, start=1):
        inc_id_log = safe_get_incident_id(rec) or rec.get("id", "?")
        if VERBOSE:
            print(f"[{idx}/{len(incidents)}] criar slide para id={inc_id_log}")

        # Slide de informação do incêndio
        slide_img = create_incident_slide(rec, size=IMG_SIZE)
        frames = make_frames_for_slide(slide_img, fps=FPS, dur_hold=DURATION_HOLD, dur_fade=DURATION_FADE)
        all_frames.extend(frames)

        # Slide com imagem (se existir)
        map_img_path = map_image_path_for_id(inc_id_log)
        if map_img_path and os.path.isfile(map_img_path):
            if VERBOSE:
                print(f"  -> criar slide de imagem: {map_img_path}")
            img_slide = create_image_slide(map_img_path, rec, size=IMG_SIZE)
            img_frames = make_frames_for_slide(img_slide, fps=FPS, dur_hold=DURATION_HOLD, dur_fade=DURATION_FADE)
            all_frames.extend(img_frames)

    # escrever vídeo
    if not all_frames:
        print("Sem frames gerados.")
        return

    h, w, _ = all_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)
    vw = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    print(f"Escrever vídeo {OUTPUT_VIDEO} ({len(all_frames)} frames)...")
    for i, fr in enumerate(all_frames):
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        if VERBOSE and (i % 100 == 0):
            print(f" frame {i+1}/{len(all_frames)}")
    vw.release()
    print("Concluído. Vídeo guardado em:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
