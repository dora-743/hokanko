from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


STAC_ITEMS = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/collections/{collection}/items"


def fetch_json(url: str, timeout: int = 90) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "codex-emit-search/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def iter_items(collection: str, bbox: tuple[float, float, float, float], limit: int = 100) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"bbox": ",".join(map(str, bbox)), "limit": str(limit)})
    url = f"{STAC_ITEMS.format(collection=collection)}?{query}"
    items: list[dict[str, Any]] = []
    seen = set()
    while url:
        if url in seen:
            break
        seen.add(url)
        data = fetch_json(url)
        items.extend(data.get("features", []))
        next_url = None
        for link in data.get("links", []):
            if link.get("rel") == "next":
                next_url = link.get("href")
                break
        url = next_url
        time.sleep(0.2)
    return items


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        crosses = (y1 > lat) != (y2 > lat)
        if crosses:
            x_at_lat = (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-15) + x1
            if lon < x_at_lat:
                inside = not inside
    return inside


def point_in_geometry(lon: float, lat: float, geometry: dict[str, Any]) -> bool:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon" and coords:
        return point_in_ring(lon, lat, coords[0])
    if geom_type == "MultiPolygon":
        return any(point_in_ring(lon, lat, poly[0]) for poly in coords if poly)
    return False


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return 0.5 * (min_lat + max_lat), 0.5 * (min_lon + max_lon)


def bbox_overlap_area(a: tuple[float, float, float, float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    dy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return dx * dy


def find_asset_url(feature: dict[str, Any], role: str, suffix: str | None = None) -> str:
    for asset in feature.get("assets", {}).values():
        href = asset.get("href", "")
        roles = asset.get("roles", [])
        if role in roles and (suffix is None or href.endswith(suffix)):
            return href
    return ""


def summarize_feature(
    feature: dict[str, Any],
    collection: str,
    target_lat: float,
    target_lon: float,
    target_bbox: tuple[float, float, float, float],
) -> dict[str, Any]:
    bbox = feature.get("bbox", [math.nan] * 4)
    center_lat, center_lon = bbox_center(bbox)
    return {
        "collection": collection,
        "id": feature.get("id", ""),
        "datetime": feature.get("properties", {}).get("datetime", ""),
        "start_datetime": feature.get("properties", {}).get("start_datetime", ""),
        "end_datetime": feature.get("properties", {}).get("end_datetime", ""),
        "cloud_cover": feature.get("properties", {}).get("eo:cloud_cover", math.nan),
        "contains_hisui_center": point_in_geometry(target_lon, target_lat, feature.get("geometry", {})),
        "bbox_overlap_deg2": bbox_overlap_area(target_bbox, bbox),
        "bbox_center_distance_km": haversine_km(target_lat, target_lon, center_lat, center_lon),
        "bbox_min_lon": bbox[0],
        "bbox_min_lat": bbox[1],
        "bbox_max_lon": bbox[2],
        "bbox_max_lat": bbox[3],
        "browse_url": find_asset_url(feature, "browse"),
        "data_url": find_asset_url(feature, "data"),
        "metadata_url": find_asset_url(feature, "metadata"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(r"D:\research\code\outputs_emit_search_permian"))
    parser.add_argument("--target-lat", type=float, default=31.963016)
    parser.add_argument("--target-lon", type=float, default=-103.151902)
    parser.add_argument("--bbox", default="-103.356058,31.797505,-102.976915,32.144274")
    parser.add_argument("--search-margin-deg", type=float, default=0.35)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base_bbox = tuple(float(x) for x in args.bbox.split(","))
    min_lon, min_lat, max_lon, max_lat = base_bbox
    search_bbox = (
        min_lon - args.search_margin_deg,
        min_lat - args.search_margin_deg,
        max_lon + args.search_margin_deg,
        max_lat + args.search_margin_deg,
    )
    collections = [
        "EMITL1BRAD_001",
        "EMITL2ARFL_001",
        "EMITL2BMIN_001",
        "EMITL2BCH4ENH_002",
        "EMITL2BCH4PLM_002",
    ]
    all_rows: list[dict[str, Any]] = []
    raw: dict[str, list[dict[str, Any]]] = {}
    for collection in collections:
        features = iter_items(collection, search_bbox)
        raw[collection] = features
        for feature in features:
            all_rows.append(summarize_feature(feature, collection, args.target_lat, args.target_lon, base_bbox))

    all_rows.sort(
        key=lambda row: (
            row["collection"],
            not row["contains_hisui_center"],
            float(row["cloud_cover"]) if row["cloud_cover"] is not None else 9999.0,
            -float(row["bbox_overlap_deg2"]),
            float(row["bbox_center_distance_km"]),
            row["datetime"],
        )
    )
    csv_path = args.out_dir / "emit_candidates_near_hisui_permian.csv"
    if all_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    (args.out_dir / "emit_candidates_near_hisui_permian_raw_stac.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {csv_path}")
    for collection in collections:
        subset = [row for row in all_rows if row["collection"] == collection]
        print(collection, len(subset))
        for row in subset[:8]:
            print(
                f"  {row['datetime']} cloud={row['cloud_cover']} "
                f"contains={row['contains_hisui_center']} dist_km={row['bbox_center_distance_km']:.1f} "
                f"{row['id']}"
            )


if __name__ == "__main__":
    main()
