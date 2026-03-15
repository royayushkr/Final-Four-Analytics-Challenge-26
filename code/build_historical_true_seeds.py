#!/usr/bin/env python3
"""Build a checked-in historical true-seed table for 2020-21 through 2024-25.

This script uses CBS 1-68 seed list articles as the operational source for the
full overall seed order. The 2021 article no longer exposes stable HTML in a
way that is easy to parse consistently, so its 1-68 list is kept as a checked
manual fallback sourced from the same article.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "external" / "historical_true_seeds_2021_2025.csv"

HEADERS = {"User-Agent": "Mozilla/5.0"}

MANUAL_2021 = [
    (1, "Gonzaga"),
    (2, "Baylor"),
    (3, "Illinois"),
    (4, "Michigan"),
    (5, "Alabama"),
    (6, "Ohio State"),
    (7, "Iowa"),
    (8, "Houston"),
    (9, "Arkansas"),
    (10, "West Virginia"),
    (11, "Texas"),
    (12, "Kansas"),
    (13, "Florida State"),
    (14, "Purdue"),
    (15, "Oklahoma State"),
    (16, "Virginia"),
    (17, "Creighton"),
    (18, "Villanova"),
    (19, "Tennessee"),
    (20, "Colorado"),
    (21, "USC"),
    (22, "Texas Tech"),
    (23, "BYU"),
    (24, "San Diego State"),
    (25, "Oregon"),
    (26, "UConn"),
    (27, "Clemson"),
    (28, "Florida"),
    (29, "LSU"),
    (30, "Loyola Chicago"),
    (31, "North Carolina"),
    (32, "Oklahoma"),
    (33, "Missouri"),
    (34, "Georgia Tech"),
    (35, "Wisconsin"),
    (36, "Maryland"),
    (37, "Virginia Tech"),
    (38, "VCU"),
    (39, "St. Bonaventure"),
    (40, "Rutgers"),
    (41, "Syracuse"),
    (42, "Utah State"),
    (43, "Michigan State"),
    (44, "UCLA"),
    (45, "Wichita State"),
    (46, "Oregon State"),
    (47, "Georgetown"),
    (48, "Drake"),
    (49, "Winthrop"),
    (50, "UC Santa Barbara"),
    (51, "Ohio"),
    (52, "North Texas"),
    (53, "Liberty"),
    (54, "UNC Greensboro"),
    (55, "Abilene Christian"),
    (56, "Morehead State"),
    (57, "Colgate"),
    (58, "Eastern Washington"),
    (59, "Grand Canyon"),
    (60, "Cleveland State"),
    (61, "Oral Roberts"),
    (62, "Iona"),
    (63, "Drexel"),
    (64, "Hartford"),
    (65, "Mount St. Mary's"),
    (66, "Texas Southern"),
    (67, "Norfolk State"),
    (68, "Appalachian State"),
]

SOURCE_URLS = {
    "2020-21": "https://www.cbssports.com/college-basketball/news/2021-ncaa-tournament-bracket-seed-list-from-1-68-march-madness-printable-bracket-key-dates-times/",
    "2021-22": "https://www.cbssports.com/college-basketball/news/march-madness-2022-committee-reveals-official-ncaa-tournament-bracket-seed-list-from-1-68/",
    "2022-23": "https://www.cbssports.com/college-basketball/news/march-madness-2023-committee-reveals-official-ncaa-tournament-bracket-seed-list-from-1-68/",
    "2023-24": "https://www.cbssports.com/college-basketball/news/march-madness-2024-committee-reveals-official-ncaa-tournament-bracket-seed-list-from-1-68/",
    "2024-25": "https://www.cbssports.com/college-basketball/news/march-madness-2025-committee-reveals-official-ncaa-tournament-bracket-seed-list-from-1-68/",
}

TEAM_CANONICAL = {
    "USC": "Southern California",
    "Ohio State": "Ohio St.",
    "Florida State": "Florida St.",
    "Oklahoma State": "Oklahoma St.",
    "San Diego State": "San Diego St.",
    "Utah State": "Utah St.",
    "Michigan State": "Michigan St.",
    "Oregon State": "Oregon St.",
    "Wichita State": "Wichita St.",
    "Eastern Washington": "Eastern Wash.",
    "Cleveland State": "Cleveland St.",
    "Norfolk State": "Norfolk St.",
    "Appalachian State": "App State",
    "Texas A&M-Corpus Christi": "A&M-Corpus Christi",
    "Saint Mary's": "Saint Mary's (CA)",
    "Miami": "Miami (FL)",
    "Iowa State": "Iowa St.",
    "Kent State": "Kent St.",
    "Montana State": "Montana St.",
    "Northern Kentucky": "Northern Ky.",
    "Charleston": "Col. of Charleston",
    "Western Kentucky": "Western Ky.",
    "Grambling State": "Grambling",
    "Florida Atlantic": "Fla. Atlantic",
    "McNeese State": "McNeese",
    "Long Beach State": "Long Beach St.",
    "Alabama State": "Alabama St.",
    "Morehead State": "Morehead St.",
    "Colorado State": "Colorado St.",
    "Murray State": "Murray St.",
    "Boise State": "Boise St.",
    "New Mexico State": "New Mexico St.",
    "South Dakota State": "South Dakota St.",
    "Jacksonville State": "Jacksonville St.",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Georgia State": "Georgia St.",
    "Wright State": "Wright St.",
    "Washington State": "Washington St.",
    "Mississippi State": "Mississippi St.",
    "Connecticut": "UConn",
    "Michigan St": "Michigan St.",
    "Iowa St": "Iowa St.",
    "Mississippi St": "Mississippi St.",
}

# 2023 source HTML contains a few OCR-ish issues that are safer to override.
MANUAL_SEED_FIXES = {
    "2022-23": {
        21: "Iowa St.",
        32: "Iowa",
        51: "Kent St.",
        52: "Iona",
        56: "UC Santa Barbara",
        57: "Grand Canyon",
        58: "Montana St.",
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def fetch_html(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def parse_seed_strings(url: str) -> list[tuple[int, str]]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    texts = [t.strip() for t in soup.stripped_strings]
    rows: list[tuple[int, str]] = []
    for text in texts:
        match = re.match(r"^(\d{1,2})\.\s+(.+?)\s*\((?:[^)]*)\)\s*$", text)
        if not match:
            continue
        seed = int(match.group(1))
        team = match.group(2).strip()
        rows.append((seed, team))

    dedup: list[tuple[int, str]] = []
    seen: set[int] = set()
    for seed, team in rows:
        if seed in seen:
            continue
        dedup.append((seed, team))
        seen.add(seed)
    return dedup


def parse_2025_ordered_list(url: str) -> list[tuple[int, str]]:
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    rows: list[tuple[int, str]] = []
    for seed, li in enumerate(soup.select("ol li"), start=1):
        text = li.get_text(" ", strip=True)
        team = re.sub(r"\s*\((?:\d+\s*-\s*\d+|\d+-\d+)\)\s*$", "", text).strip()
        team = team.replace(" St .", " St.")
        rows.append((seed, team))
    return rows


def normalize_source_team(team: str) -> str:
    return TEAM_CANONICAL.get(team, team)


def build_rows() -> pd.DataFrame:
    all_rows: list[dict[str, object]] = []

    for seed, team in MANUAL_2021:
        all_rows.append(
            {
                "Season": "2020-21",
                "TeamSource": team,
                "TeamCanonical": normalize_source_team(team),
                "TrueSeed": seed,
                "IsSelected": 1,
                "SourceUrl": SOURCE_URLS["2020-21"],
                "SourceType": "cbssports_manual_fallback",
                "Verified": True,
            }
        )

    for season in ["2021-22", "2022-23", "2023-24"]:
        parsed = parse_seed_strings(SOURCE_URLS[season])
        parsed_by_seed = {seed: team for seed, team in parsed}
        parsed_by_seed.update(MANUAL_SEED_FIXES.get(season, {}))
        if sorted(parsed_by_seed) != list(range(1, 69)):
            raise ValueError(f"Season {season} did not parse to the full 1..68 seed list")
        for seed in range(1, 69):
            team = parsed_by_seed[seed]
            all_rows.append(
                {
                    "Season": season,
                    "TeamSource": team,
                    "TeamCanonical": normalize_source_team(team),
                    "TrueSeed": seed,
                    "IsSelected": 1,
                    "SourceUrl": SOURCE_URLS[season],
                    "SourceType": "cbssports_article",
                    "Verified": True,
                }
            )

    for seed, team in parse_2025_ordered_list(SOURCE_URLS["2024-25"]):
        all_rows.append(
            {
                "Season": "2024-25",
                "TeamSource": team,
                "TeamCanonical": normalize_source_team(team),
                "TrueSeed": seed,
                "IsSelected": 1,
                "SourceUrl": SOURCE_URLS["2024-25"],
                "SourceType": "cbssports_article",
                "Verified": True,
            }
        )

    df = pd.DataFrame(all_rows).sort_values(["Season", "TrueSeed"]).reset_index(drop=True)
    if len(df) != 340:
        raise ValueError(f"Expected 340 selected-team rows, found {len(df)}")
    return df


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_rows()
    df.to_csv(args.output_path, index=False)
    print(f"Wrote {len(df)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
