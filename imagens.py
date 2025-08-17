#!/usr/bin/env python3
"""
Gerador de imagens de incêndios florestais em Portugal
Baseado na API fogos.pt

Versão ajustada:
 - Selecciona TODOS os incêndios com > 90 operacionais
 - Só gera imagem se existir KML válido (string ou URL) e for possível extrair polígono
 - Mantém o basemap + fallback e o zoom automático
 - Adiciona legenda para a estrela (Início do incêndio) e
   um "pop-up" no canto inferior direito com operacionais/terrestres/aéreos/área
"""

from typing import List, Tuple
import os
import math
import requests
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, Point
from shapely.ops import transform as shapely_transform
import pyproj
import matplotlib.pyplot as plt

# --- Configurações ----------------------------------------------------------------
API_URL = "https://api-dev.fogos.pt/new/fires"
OUTPUT_DIR = "images"
SHOW_PLOTS = False         # True para mostrar janelas matplotlib
EXPORT_DPI = 108
FIGSIZE = (10, 10)  # 10 × 108 = 1080 px por lado
MAX_CITY_DISTANCE_KM = 80  # distância máxima para considerar cidades
# -----------------------------------------------------------------------------


# Pequena base de cidades portuguesas (nome, lat, lon) -- podes estender à vontade
CITIES = [
    ("Lisboa", 38.722252, -9.139337),
    ("Porto", 41.157944, -8.629105),
    ("Braga", 41.545448, -8.426507),
    ("Guimarães", 41.444223, -8.296016),
    ("Viana do Castelo", 41.693579, -8.832419),
    ("Aveiro", 40.640505, -8.653788),
    ("Coimbra", 40.203314, -8.410257),
    ("Leiria", 39.743606, -8.807052),
    ("Viseu", 40.661003, -7.909712),
    ("Castelo Branco", 39.822659, -7.494360),
    ("Guarda", 40.537478, -7.265439),
    ("Faro", 37.019356, -7.930440),
    ("Évora", 38.571, -7.907),
    ("Setúbal", 38.524, -8.892),
    ("Santarém", 39.236, -8.685),
]

# --- Utilitários ------------------------------------------------------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def fetch_api(url: str, timeout: int = 15):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def fetch_kml_if_url(maybe_url_or_kml: str, timeout: int = 15) -> str:
    if not maybe_url_or_kml:
        return ""
    s = maybe_url_or_kml.strip()
    if s.lower().startswith("http://") or s.lower().startswith("https://"):
        try:
            r = requests.get(s, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"   ⚠️ Erro ao descarregar KML de URL: {e}")
            return ""
    else:
        return s  # já é KML em string

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
                    lon = float(parts[0])
                    lat = float(parts[1])
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

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def nearby_cities(lat, lon, max_km=MAX_CITY_DISTANCE_KM):
    res = []
    for name, c_lat, c_lon in CITIES:
        d = haversine_km(lat, lon, c_lat, c_lon)
        if d <= max_km:
            res.append((name, d, c_lat, c_lon))
    res.sort(key=lambda x: x[1])
    return res

# --- Zoom automático ----------------------------------------------------------

def compute_zoom_from_bbox_meters(minx, miny, maxx, maxy) -> int:
    span = max(maxx - minx, maxy - miny)
    if span <= 5000:
        return 15
    if span <= 20000:
        return 14
    if span <= 80000:
        return 13
    if span <= 300000:
        return 12
    if span <= 800000:
        return 11
    return 10

# --- Plot com basemap (geopandas + contextily) --------------------------------

def plot_with_basemap(polygon_coords, start_lat, start_lng, info_text, fname, dpi=EXPORT_DPI, figsize=FIGSIZE):
    import geopandas as gpd
    import contextily as ctx

    poly_geom = Polygon(polygon_coords)
    gdf = gpd.GeoDataFrame([{"geometry": poly_geom}], crs="EPSG:4326")
    gdf_3857 = gdf.to_crs(epsg=3857)

    p = Point(start_lng, start_lat)
    gpt = gpd.GeoDataFrame([{"geometry": p}], crs="EPSG:4326").to_crs(epsg=3857)

    minx, miny, maxx, maxy = gdf_3857.total_bounds
    dx = (maxx - minx) * 0.15 if (maxx - minx) > 0 else 2000
    dy = (maxy - miny) * 0.15 if (maxy - miny) > 0 else 2000
    bbox = (minx - dx, miny - dy, maxx + dx, maxy + dy)

    zoom = compute_zoom_from_bbox_meters(minx - dx, miny - dy, maxx + dx, maxy + dy)

    fig, ax = plt.subplots(figsize=figsize)

    # Desenha polígono (área estimada) e ponto de início (estrela)
    gdf_3857.plot(ax=ax, alpha=0.45, edgecolor="darkred", linewidth=1.6, label="Área estimada")
    gpt.plot(ax=ax, marker="*", markersize=200, label="Início do incêndio", zorder=10)

    # --- construir GeoDataFrame de todas as cidades e filtrar por bbox (em 3857)
    city_geoms = []
    for name, c_lat, c_lon in CITIES:
        city_geoms.append({"name": name, "geometry": Point(c_lon, c_lat)})
    cgdf = gpd.GeoDataFrame(city_geoms, crs="EPSG:4326").to_crs(epsg=3857)

    # Filtrar só cidades cujo ponto caia dentro do bbox (assim não anota fora do mapa)
    cgdf_in = cgdf[(cgdf.geometry.x >= bbox[0]) & (cgdf.geometry.x <= bbox[2]) &
                   (cgdf.geometry.y >= bbox[1]) & (cgdf.geometry.y <= bbox[3])]

    # Anotar apenas cgdf_in (só nomes dentro do mapa)
    if not cgdf_in.empty:
        cgdf_in.plot(ax=ax, color="black", markersize=30)
        for idx, row in cgdf_in.iterrows():
            x, y = row.geometry.x, row.geometry.y
            ax.text(x + (dx * 0.02), y + (dy * 0.02), f"{row['name']}", fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    # adicionar basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.Stamen.TerrainBackground, crs=gdf_3857.crs.to_string(), zoom=zoom)
    except Exception:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=gdf_3857.crs.to_string(), zoom=zoom)
        except Exception as e:
            raise RuntimeError(f"Adição do basemap falhou: {e}")

    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_axis_off()

    # Legenda: assegura que a estrela está identificada como "Início do incêndio"
    ax.legend(loc="upper left", frameon=True)

    # "Popup" no canto inferior direito com informação detalhada (texto multiline)
    ax.text(0.98, 0.02, info_text,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(facecolor="white", alpha=0.85))

    # título com primeira linha do info_text (se houver)
    plt.title(info_text.splitlines()[0] if info_text else "Incêndio")
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    return True

# --- Fallback (matplotlib simples) ------------------------------------------

def plot_fallback(polygon_coords, start_lat, start_lng, info_text, fname, dpi=EXPORT_DPI, figsize=FIGSIZE):
    lon_vals = [p[0] for p in polygon_coords]
    lat_vals = [p[1] for p in polygon_coords]

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill(lon_vals, lat_vals, alpha=0.45, color="red", label="Área estimada")
    ax.plot(lon_vals, lat_vals, linewidth=1.4)

    # Estrela de início do incêndio
    ax.scatter(start_lng, start_lat, color="yellow", s=160, marker="*", label="Início do incêndio", zorder=10)

    # bbox em lon/lat com margem
    min_lon = min(lon_vals)
    max_lon = max(lon_vals)
    min_lat = min(lat_vals)
    max_lat = max(lat_vals)
    lon_margin = (max_lon - min_lon) * 0.15 if (max_lon - min_lon) > 0 else 0.02
    lat_margin = (max_lat - min_lat) * 0.15 if (max_lat - min_lat) > 0 else 0.02
    bbox_lon_min = min_lon - lon_margin
    bbox_lon_max = max_lon + lon_margin
    bbox_lat_min = min_lat - lat_margin
    bbox_lat_max = max_lat + lat_margin

    # anotar só cidades dentro do bbox (sem distância no texto)
    for name, c_lat, c_lon in CITIES:
        if (bbox_lon_min <= c_lon <= bbox_lon_max) and (bbox_lat_min <= c_lat <= bbox_lat_max):
            ax.scatter(c_lon, c_lat, s=20)
            ax.text(c_lon + 0.005, c_lat + 0.005, name, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

    # Caixa (popup) no canto inferior direito com os dados
    ax.text(0.98, 0.02, info_text,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(facecolor='white', alpha=0.85))

    # título e legend
    ax.set_title(info_text.splitlines()[0] if info_text else "Incêndio")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", frameon=True)

    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
    return True

# --- Rotina principal -------------------------------------------------------

def main():
    ensure_dir(OUTPUT_DIR)
    print("🔎 A pedir dados à API:", API_URL)
    try:
        resp = fetch_api(API_URL)
    except Exception as e:
        print("Erro ao aceder à API:", e)
        return

    if not resp.get("success"):
        print("API devolveu success=False")
        return

    data = resp.get("data", [])
    if not data:
        print("Sem dados na API.")
        return

    # Filtrar todos os incêndios com mais de 90 operacionais
    candidatos = []
    for inc in data:
        try:
            man = int(inc.get("man", 0))
        except Exception:
            man = 0
        if man > 90:
            candidatos.append(inc)

    total_candidatos = len(candidatos)
    print(f"Total de incêndios com >90 operacionais: {total_candidatos}")

    imagens_criadas = []
    ignorados = []  # tuples (id, motivo)

    for inc in candidatos:
        try:
            inc_id = inc.get("id", "unknown")
            concelho = inc.get("concelho", "??")
            freguesia = inc.get("freguesia", "")
            status = inc.get("status", "??")
            man = inc.get("man", 0)
            terrain = inc.get("terrain", 0)
            aerial = inc.get("aerial", 0)
            start_lat = inc.get("lat")
            start_lng = inc.get("lng")

            print(f"\n🔥 {inc_id} - {concelho} ({freguesia})")
            print(f"   Estado: {status}")
            print(f"   Operacionais: {man} | Terrestres: {terrain} | Aéreos: {aerial}")

            raw_kml = inc.get("kmlVost") or inc.get("kml") or ""
            if not raw_kml:
                motivo = "Sem KML (kmlVost/kml vazio)"
                print(f"   Ignorado: {motivo}")
                ignorados.append((inc_id, motivo))
                continue

            kml_string = fetch_kml_if_url(raw_kml)
            if not kml_string:
                motivo = "KML presente mas não descarregável/empty"
                print(f"   Ignorado: {motivo}")
                ignorados.append((inc_id, motivo))
                continue

            polygons = extract_polygons_from_kml_string(kml_string)
            if not polygons:
                motivo = "KML sem coordenadas válidas"
                print(f"   Ignorado: {motivo}")
                ignorados.append((inc_id, motivo))
                continue

            main_poly = choose_largest_polygon(polygons)
            if not main_poly:
                motivo = "Não foi possível escolher polígono principal"
                print(f"   Ignorado: {motivo}")
                ignorados.append((inc_id, motivo))
                continue

            try:
                area_km2 = polygon_area_km2(main_poly)
            except Exception as e:
                area_km2 = float('nan')
                print("   Erro ao calcular área:", e)

            info_text = f"{concelho} - {status}\nOperacionais: {man}\nTerrestres: {terrain}\nAéreos: {aerial}\nÁrea ≈ {area_km2:.3f} km²"
            fname = os.path.join(OUTPUT_DIR, f"inc_{inc_id}.png")

            try:
                plot_with_basemap(main_poly, start_lat, start_lng, info_text, fname, dpi=EXPORT_DPI, figsize=FIGSIZE)
            except Exception as e:
                print("   ⚠️ Basemap falhou — a usar fallback simples. Erro:", e)
                try:
                    plot_fallback(main_poly, start_lat, start_lng, info_text, fname, dpi=EXPORT_DPI, figsize=FIGSIZE)
                except Exception as e2:
                    motivo = f"Erro a desenhar imagem: {e2}"
                    print(f"   Ignorado: {motivo}")
                    ignorados.append((inc_id, motivo))
                    continue

            imagens_criadas.append(inc_id)
            print(f"   Imagem guardada: {fname}")

        except Exception as e:
            motivo = f"Erro ao processar: {e}"
            print(f"   Ignorado: {motivo}")
            ignorados.append((inc.get("id", "unknown"), motivo))

    # Resumo final
    print("\n--- Resumo ---")
    print(f"Incêndios com >90 operacionais: {total_candidatos}")
    print(f"Imagens geradas: {len(imagens_criadas)}")
    if imagens_criadas:
        print(" Imagens criadas para IDs:", ", ".join(imagens_criadas))
    if ignorados:
        print(f"Ignorados ({len(ignorados)}):")
        for iid, motivo in ignorados:
            print(f"  - {iid}: {motivo}")

if __name__ == "__main__":
    main()