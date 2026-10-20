"""Data layer for the Fanteam Champions League optimizer (stage 1)."""

from __future__ import annotations

import re
from io import StringIO

import pandas as pd
import requests

PAGE_URL = "https://dailyfantasy.tips/champions-league-projections-odds/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; UCL-optimizer/1.0)"}

P_START = {
    "expected": 1.0,
    "possible": 0.5,
    "unexpected": 0.1,
    "injured": 0.0,
    "unknown": 0.0,
}
POS_MAP = {"goalkeeper": "GK", "defender": "D", "midfielder": "M", "forward": "F"}
NUMERIC_COLS = ["Salary", "Proj", "Ceiling", "Value", "To Score %", "Win %", "CS %"]
STATUS_TAG = r"\s+(Possible|Bench\?|Injured)\s*$"
BUDGET = 120.0


# ---------- projections ----------


def _clean_projections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "Player" in df.columns:
        df["Player"] = (
            df["Player"].astype(str).str.replace(STATUS_TAG, "", regex=True).str.strip()
        )
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Fanteam ID" in df.columns:
        df["Fanteam ID"] = (
            df["Fanteam ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        )
    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].astype(str).str.strip().str.upper()
    return df.drop_duplicates(subset=["Fanteam ID"]).reset_index(drop=True)


def parse_fanteam_html(html: str) -> tuple[pd.DataFrame, str | None]:
    """Find the Fanteam projections table (the one with a Fanteam ID column)."""
    stamp_re = r"Last updated:\s*([0-9/]+,?\s*[0-9:]+)"
    anchor = html.find("Fanteam ID")
    stamps = list(re.finditer(stamp_re, html[:anchor] if anchor > 0 else html))
    updated = stamps[-1].group(1).strip() if stamps else None
    for table in pd.read_html(StringIO(html)):
        if "Fanteam ID" in [str(c).strip() for c in table.columns]:
            return _clean_projections(table), updated
    raise ValueError("No Fanteam projections table found on the page.")


def fetch_projections(
    url: str = PAGE_URL, timeout: int = 15
) -> tuple[pd.DataFrame, str | None]:
    """Live pull of the Fanteam projections. Raises on network or layout errors."""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return parse_fanteam_html(resp.text)


def load_projections_csv(source) -> pd.DataFrame:
    return _clean_projections(pd.read_csv(source, encoding="utf-8-sig"))


# ---------- player list ----------


def load_player_list(source) -> pd.DataFrame:
    df = pd.read_csv(source, encoding="utf-8-sig", dtype={"PlayerID": str})
    df["PlayerID"] = df["PlayerID"].str.strip()
    df["Position"] = df["Position"].str.strip().str.lower().map(POS_MAP)
    df["Lineup"] = df["Lineup"].str.strip().str.lower()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["FullName"] = (df["FName"].fillna("") + " " + df["Name"].fillna("")).str.strip()
    return df.drop_duplicates(subset=["PlayerID"]).reset_index(drop=True)


def merge_projections(
    proj: pd.DataFrame, players: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Join projections to the player list on Fanteam ID and add availability."""
    lst = players.rename(
        columns={"PlayerID": "Fanteam ID", "Position": "PosList", "Price": "PriceList"}
    )
    lst = lst[["Fanteam ID", "FullName", "Club", "PosList", "PriceList", "Lineup"]]
    m = proj.merge(lst, on="Fanteam ID", how="left", indicator=True)
    report = {
        "projections": len(proj),
        "not_in_player_list": int((m["_merge"] != "both").sum()),
        "price_mismatches": int(((m["Salary"] - m["PriceList"]).abs() > 1e-6).sum()),
        "position_mismatches": int((m["Pos"] != m["PosList"]).sum()),
    }
    m = m.drop(columns="_merge")
    m["Proj"] = m["Proj"].fillna(0.0)
    m["Lineup"] = m["Lineup"].fillna("unknown")
    m["p_start"] = m["Lineup"].map(P_START).fillna(0.0)
    return m.reset_index(drop=True), report


# ---------- safety net ----------


def safety_net_chains(
    players: pd.DataFrame, candidate_statuses=("expected",)
) -> dict[str, list[str]]:
    """Replacement candidates per player: same club and position, equal or cheaper
    price, closest price first."""
    pool = players[players["Lineup"].isin(candidate_statuses)]
    chains = {}
    for _, row in players.iterrows():
        same = pool[
            (pool["Club"] == row["Club"])
            & (pool["Position"] == row["Position"])
            & (pool["Price"] <= row["Price"])
            & (pool["PlayerID"] != row["PlayerID"])
        ]
        same = same.assign(gap=row["Price"] - same["Price"]).sort_values("gap")
        chains[row["PlayerID"]] = same["PlayerID"].tolist()
    return chains


def pick_replacement(player_id: str, chains: dict, taken: set[str]) -> str | None:
    for candidate in chains.get(player_id, []):
        if candidate not in taken:
            return candidate
    return None


# ---------- valuation ----------


def lineup_expected_points(
    lineup_ids, values, p_start, chains, captain=None, vice=None
):
    """Expected points for a lineup, with safety-net swaps resolved against the lineup."""
    taken = set(lineup_ids)
    detail, total = {}, 0.0
    
    # Sort by start probability and value so high-risk players reserve replacements first
    ordered_ids = sorted(lineup_ids, key=lambda pid: (p_start.get(pid, 0.0), -values.get(pid, 0.0)))
    
    for pid in ordered_ids:
        p = p_start.get(pid, 0.0)
        rep = pick_replacement(pid, chains, taken) if p < 1 else None
        if rep:
            taken.add(rep)
        rep_pts = values.get(rep, 0.0) if rep else 0.0
        expected = p * values.get(pid, 0.0) + (1 - p) * rep_pts
        detail[pid] = {"expected": round(expected, 2), "replacement": rep}
        total += expected
        
    if captain:
        total += detail[captain]["expected"]
    if vice:
        p_cap = p_start.get(captain, 0.0) if captain else 0.0
        total += (1 - p_cap) * detail[vice]["expected"]
    return total, detail


def static_expected_points(merged: pd.DataFrame, chains: dict) -> dict[str, float]:
    proj = dict(zip(merged["Fanteam ID"], merged["Proj"]))
    p = dict(zip(merged["Fanteam ID"], merged["p_start"]))
    out = {}
    for pid, pts in proj.items():
        chain = chains.get(pid, [])
        rep_pts = proj.get(chain[0], 0.0) if chain else 0.0
        out[pid] = p[pid] * pts + (1 - p[pid]) * rep_pts
    return out


# ---------- Fanteam rules ----------


def check_lineup(merged: pd.DataFrame, lineup_ids, budget: float = BUDGET) -> list[str]:
    rows = merged.set_index("Fanteam ID").loc[list(lineup_ids)]
    problems = []
    if len(lineup_ids) != 11:
        problems.append(f"{len(lineup_ids)} players, need 11")
    counts = rows["Pos"].value_counts().to_dict()
    if counts.get("GK", 0) != 1:
        problems.append("need exactly 1 goalkeeper")
    for pos, minimum in (("D", 3), ("M", 3), ("F", 2)):
        if counts.get(pos, 0) < minimum:
            problems.append(f"need at least {minimum} {pos}")
    for club, n in rows["Club"].value_counts().items():
        if n > 3:
            problems.append(f"{n} players from {club}, max 3")
    cost = rows["Salary"].sum()
    if cost > budget + 1e-9:
        problems.append(f"cost {cost:.1f} is over the {budget:.0f}m budget")
    return problems
