#!/usr/bin/env python3
"""
Newsday: NYT headline sentiment analyzer.
Fetches top headlines from the NYT RSS feed and scores how good or bad the news is.
"""

import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

NYT_RSS_URL = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
NYT_HOME_URL = "https://www.nytimes.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# VADER compound score thresholds
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


@dataclass
class Headline:
    text: str
    compound: float
    label: str


def fetch_via_rss() -> list[str]:
    resp = requests.get(NYT_RSS_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    titles = []
    for item in root.findall(".//item"):
        title_el = item.find("title")
        if title_el is not None and title_el.text:
            titles.append(title_el.text.strip())
    return titles


def fetch_via_scrape() -> list[str]:
    resp = requests.get(NYT_HOME_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    seen = set()
    titles = []
    # NYT uses <h3> for story headlines; filter out nav/footer noise by length
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
        headlines = fetch_via_rss()
        if headlines:
            return headlines, "NYT RSS feed"
    except Exception as e:
        print(f"  RSS fetch failed ({e}), trying homepage scrape...", file=sys.stderr)

    try:
        headlines = fetch_via_scrape()
        if headlines:
            return headlines, "NYT homepage"
    except Exception as e:
        print(f"  Homepage scrape failed ({e})", file=sys.stderr)

    return [], ""


def score_headlines(headlines: list[str]) -> list[Headline]:
    analyzer = SentimentIntensityAnalyzer()
    results = []
    for text in headlines:
        scores = analyzer.polarity_scores(text)
        compound = scores["compound"]
        if compound >= POSITIVE_THRESHOLD:
            label = "GOOD"
        elif compound <= NEGATIVE_THRESHOLD:
            label = "BAD "
        else:
            label = "MEH "
        results.append(Headline(text=text, compound=compound, label=label))
    return results


def overall_verdict(scored: list[Headline]) -> tuple[str, float]:
    if not scored:
        return "UNKNOWN", 0.0
    avg = sum(h.compound for h in scored) / len(scored)
    good = sum(1 for h in scored if h.label == "GOOD")
    bad  = sum(1 for h in scored if h.label == "BAD ")
    meh  = sum(1 for h in scored if h.label == "MEH ")

    if avg >= 0.05:
        verdict = "GOOD DAY TO READ THE NEWS"
    elif avg <= -0.05:
        verdict = "MAYBE SKIP THE NEWS TODAY"
    else:
        verdict = "THE NEWS IS MEH TODAY"

    return verdict, avg, good, bad, meh


def bar(score: float, width: int = 20) -> str:
    """ASCII bar from -1 (all bad) to +1 (all good), centered."""
    pos = int((score + 1) / 2 * width)
    pos = max(0, min(width - 1, pos))
    bar_chars = ["-"] * width
    bar_chars[pos] = "|"
    mid = width // 2
    bar_str = "".join(bar_chars)
    return f"[{bar_str[:mid]}+{bar_str[mid+1:]}]"


def main():
    print("\n=== NEWSDAY: NYT Sentiment Check ===\n")
    print("Fetching headlines...", end=" ", flush=True)
    headlines, source = get_headlines()

    if not headlines:
        print("\nCould not fetch any headlines. Check your network connection.")
        sys.exit(1)

    print(f"got {len(headlines)} from {source}\n")

    scored = score_headlines(headlines)
    verdict, avg, good, bad, meh = overall_verdict(scored)

    # Print each headline with its score
    print(f"{'SCORE':>7}  HEADLINE")
    print("-" * 80)
    for h in scored:
        score_str = f"{h.compound:+.2f}"
        label_prefix = "+" if h.label == "GOOD" else ("-" if h.label == "BAD " else " ")
        print(f"  {label_prefix}{score_str[1:]}  {h.text[:74]}")

    # Summary
    print("\n" + "=" * 80)
    print(f"\nOVERALL SCORE: {avg:+.3f}  {bar(avg)}")
    print(f"  Good: {good}  Bad: {bad}  Neutral: {meh}")
    print(f"\n>>> {verdict} <<<\n")


if __name__ == "__main__":
    main()
