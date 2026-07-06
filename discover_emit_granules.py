from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path


CMR_GRANULE_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def cmr_query(short_name: str, version: str, temporal: str | None, page_size: int) -> dict:
    params = {
        "short_name": short_name,
        "version": version,
        "page_size": str(page_size),
    }
    if temporal:
        params["temporal"] = temporal
    url = CMR_GRANULE_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))


def summarize_links(entry: dict) -> dict:
    links = entry.get("links", [])
    out = {
        "title": entry.get("title"),
        "time_start": entry.get("time_start"),
        "time_end": entry.get("time_end"),
        "granule_size_mb": entry.get("granule_size"),
        "protected_data_urls": [],
        "public_browse_urls": [],
        "public_metadata_urls": [],
        "s3_urls": [],
        "s3credentials_url": None,
    }
    for link in links:
        href = link.get("href", "")
        rel = link.get("rel", "")
        if href.startswith("s3://"):
            out["s3_urls"].append(href)
        if "s3credentials" in href:
            out["s3credentials_url"] = href
        if "lp-prod-protected" in href and rel.endswith("/data#"):
            out["protected_data_urls"].append(href)
        if href.startswith("http") and "lp-prod-public" in href and rel.endswith("/browse#"):
            out["public_browse_urls"].append(href)
        if "lp-prod-public" in href and rel.endswith("/metadata#"):
            out["public_metadata_urls"].append(href)
    return out


def download_url(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as res, out_path.open("wb") as f:
        f.write(res.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-name", default="EMITL1BRAD")
    parser.add_argument("--version", default="001")
    parser.add_argument("--temporal", default="2024-01-01T00:00:00Z,2024-01-02T00:00:00Z")
    parser.add_argument("--page-size", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path(r"D:\research\code\outputs_emit_striping_investigation"))
    parser.add_argument("--download-browse", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = cmr_query(args.short_name, args.version, args.temporal, args.page_size)
    entries = data.get("feed", {}).get("entry", [])
    summaries = [summarize_links(entry) for entry in entries]

    (args.output_dir / "emit_cmr_granules.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "emit_granule_link_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.download_browse:
        for summary in summaries:
            for url in summary["public_browse_urls"]:
                name = Path(urllib.parse.urlparse(url).path).name
                download_url(url, args.output_dir / "browse" / name)

    for summary in summaries:
        print(summary["title"])
        print("  protected data:", len(summary["protected_data_urls"]))
        for url in summary["protected_data_urls"][:3]:
            print("   ", url)
        print("  public browse:", len(summary["public_browse_urls"]))
        for url in summary["public_browse_urls"][:3]:
            print("   ", url)
        print("  s3credentials:", summary["s3credentials_url"])


if __name__ == "__main__":
    main()
