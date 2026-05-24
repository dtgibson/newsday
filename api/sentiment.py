"""
Vercel serverless function: fetch NYT headlines and score their sentiment.
GET /api/sentiment → JSON
"""

import json
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler

import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NYT_RSS_URL  = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
NYT_HOME_URL = "https://www.nytimes.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

POSITIVE_THRESHOLD =  0.05
NEGATIVE_THRESHOLD = -0.05


def fetch_via_rss() -> list[str]:
    resp = requests.get(NYT_RSS_URL, headers=HEADERS, timeout=6)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    return [
        t.text.strip()
        for item in root.findall(".//item")
        if (t := item.find("title")) is not None and t.text
    ]


def fetch_via_scrape() -> list[str]:
    resp = requests.get(NYT_HOME_URL, headers=HEADERS, timeout=4)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    seen, titles = set(), []
    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(separator=" ").strip()
        if len(text) > 20 and text not in seen:
            seen.add(text)
            titles.append(text)
            if len(titles) >= 30:
                break
    return titles


def get_headlines() -> tuple[list[str], str]:
    try:
        h = fetch_via_rss()
        if h:
            return h, "rss"
    except Exception:
        pass
    try:
        h = fetch_via_scrape()
        if h:
            return h, "scrape"
    except Exception:
        pass
    return [], "none"


def score_headlines(headlines: list[str]) -> list[dict]:
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in headlines:
        compound = analyzer.polarity_scores(text)["compound"]
        if compound >= POSITIVE_THRESHOLD:
            label = "good"
        elif compound <= NEGATIVE_THRESHOLD:
            label = "bad"
        else:
            label = "meh"
        results.append({"text": text, "compound": round(compound, 4), "label": label})
    return results


def overall_verdict(scored: list[dict]) -> dict:
    if not scored:
        return {"verdict": "UNKNOWN", "avg": 0.0, "good": 0, "bad": 0, "meh": 0}
    avg  = sum(h["compound"] for h in scored) / len(scored)
    good = sum(1 for h in scored if h["label"] == "good")
    bad  = sum(1 for h in scored if h["label"] == "bad")
    meh  = sum(1 for h in scored if h["label"] == "meh")
    if avg >= POSITIVE_THRESHOLD:
        verdict = "GOOD DAY TO READ THE NEWS"
    elif avg <= NEGATIVE_THRESHOLD:
        verdict = "MAYBE SKIP THE NEWS TODAY"
    else:
        verdict = "THE NEWS IS MEH TODAY"
    return {"verdict": verdict, "avg": round(avg, 4),
            "good": good, "bad": bad, "meh": meh}


class handler(BaseHTTPRequestHandler):
    """Vercel Python runtime expects a class named `handler`."""

    def do_GET(self):
        t0 = time.monotonic()
        headlines, source = get_headlines()
        scored = score_headlines(headlines)
        summary = overall_verdict(scored)
        elapsed = round(time.monotonic() - t0, 3)

        payload = {
            "source":    source,
            "elapsed_s": elapsed,
            "summary":   summary,
            "headlines": scored,
        }
        body = json.dumps(payload).encode()

        self.send_response(200)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        # CDN caches for 10 min; revalidates in background for 5 more
        self.send_header("Cache-Control",
                         "public, s-maxage=600, stale-while-revalidate=300")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def log_message(self, *args):
        pass  # suppress default stderr logging in Vercel
