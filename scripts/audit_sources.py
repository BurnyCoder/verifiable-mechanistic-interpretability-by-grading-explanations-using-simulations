"""Refresh primary-source metadata and page locators. Network access is only for public publications.

Run when checking citations. The released results-only report uses the committed receipts offline.
PDF page numbers are one-based physical pages. A keyword locator is a reading aid, never a truth verdict.
"""

import hashlib
import json
import re
import shutil
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".pytest_cache/scientific-audit-sources"
OUT = ROOT / "results/scientific-audit/sources.json"
NS = {"a": "http://www.w3.org/2005/Atom"}
# Full-text anchors for the field definitions, closest antecedents and quantitative comparisons in the paper.
ANCHORS = {
    "2301.05217": ["restricted loss", "433", "0.55", "5.27"],
    "2306.17844": ["distance irrelevance", "gradient symmetricity", "hybrid algorithms"],
    "1905.00414": ["invertible linear", "linear regression", "orthogonal"],
    "2507.08802": ["input-injective", "output-surjective", "countable"],
    "1503.02531": ["temperature", "negative logits", "zero-mean"],
    "1411.5279": ["percentile", "under-cover", "skew"],
    "2304.14997": ["KL divergence", "subgraph"],
    "2106.07682": ["stitching", "linear"],
    "2108.01661": ["Procrustes", "nuclear"],
    "1511.07543": ["correlation", "matching"],
    "2305.06329": ["affine", "least squares", "stitching"],
    "2301.04709": ["interchange", "intervention"],
    "2301.05062": ["sort_unique", "compressed"],
    "2412.03773": ["integration", "pizza"],
    "2301.02679": ["quadratic", "ReLU"],
    "2406.11779": ["FLOPs", "proof"],
    "2309.10312": ["0.56", "300", "simulation"],
    "2410.13928": ["fuzzing", "surprisal", "intervention"],
    "2603.20101": ["memorization", "IOI"],
    "2609.08585": ["simulator", "explanations"],
    "2611.18114": ["rebuild", "CKA", "200 words"],
}


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "VMI scientific citation audit (public research repository)"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = (
        subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    )
    identifiers = set()
    for name in paths:
        p = ROOT / name
        if p.suffix in {".md", ".py", ".cff", ".html", ".toml"} and "audit" not in name and "reviews/" not in name:
            identifiers.update(
                re.findall(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", p.read_text(encoding="utf-8"), flags=re.I)
            )
    ids = sorted(identifiers | ANCHORS.keys())
    papers = {}
    for start in range(0, len(ids), 30):
        batch = ids[start : start + 30]
        data = fetch("https://export.arxiv.org/api/query?max_results=30&id_list=" + ",".join(batch))
        for item in ET.fromstring(data).findall("a:entry", NS):
            article = item.findtext("a:id", namespaces=NS).rsplit("/", 1)[1]
            key = re.sub(r"v\d+$", "", article)
            authors = [a.findtext("a:name", namespaces=NS) for a in item.findall("a:author", NS)]
            papers[key] = {
                "title": " ".join(item.findtext("a:title", namespaces=NS).split()),
                "authors": authors,
                "submitted": item.findtext("a:published", namespaces=NS)[:10],
                "version": article,
                "source": "https://arxiv.org/abs/" + article,
            }
        time.sleep(3)
    converter = shutil.which("pdftotext")
    for ident, terms in ANCHORS.items():
        file = CACHE / (ident + ".pdf")
        try:
            if not file.exists():
                file.write_bytes(fetch("https://arxiv.org/pdf/" + ident))
            text = subprocess.check_output([converter, "-enc", "UTF-8", "-layout", str(file), "-"]).decode("utf-8")
            pages = [" ".join(s.lower().split()) for s in text.split("\f") if s.strip()]
            (CACHE / (ident + ".txt")).write_text(text, encoding="utf-8")
            papers.setdefault(ident, {})["full_text_pages"] = len(pages)
            papers[ident]["pdf_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            papers[ident]["locators"] = {
                term: [i + 1 for i, p in enumerate(pages) if term.lower() in p][:12] for term in terms
            }
        except (OSError, subprocess.CalledProcessError) as err:
            papers.setdefault(ident, {})["retrieval_limit"] = type(err).__name__
    settings = (ROOT / ".env.example").read_text(encoding="utf-8")
    rev = re.search(r"^VMI_HF_RUNS_REVISION=(\w+)$", settings, re.M)[1]
    repo = "BurnyCoder/grokking-modular-addition-runs"
    local = json.loads((ROOT / "results/targets_selection.json").read_text(encoding="utf-8"))["key_freqs"]
    targets = {}
    for run, expected in sorted(local.items()):
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{run}/analysis/key_freqs.json"
        record = json.loads(fetch(url))
        if record["key_freqs"] != expected:
            raise ValueError(f"Local selection frequencies differ for {run}")
        targets[run] = {"source": url, **record}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "metadata_scope": "Identity and version checked at arXiv. Page locators describe text searches, not independent replications.",
                "missing_metadata": sorted(i for i in ids if not papers.get(i, {}).get("title")),
                "target_frequency_records": targets,
                "provider_model_details": "https://platform.claude.com/docs/en/models/overview",
                "provider_release": "https://www.anthropic.com/claude-haiku-5-5",
                "papers": dict(sorted(papers.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Primary sources: {len(papers)}. Receipts: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
