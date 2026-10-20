"""Fanteam Champions League optimizer: Streamlit front end and PuLP lineup builder."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pulp
import streamlit as st

import data_layer as dl

st.set_page_config(page_title="UCL Fanteam Optimizer", page_icon="🏆", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Inter:wght@400;600&display=swap');
.stApp {
  background: radial-gradient(circle at 20% 0%, #1d3f9e 0%, #0a1733 45%, #050b1c 100%);
  color: #e6ecff; font-family: 'Inter', sans-serif;
}
h1, h2, h3 { font-family: 'Oswald', sans-serif; letter-spacing: .08em; text-transform: uppercase; color: #f4c430; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d2153, #070f26); border-right: 1px solid rgba(244,196,48,.35); }
.hero { border: 1px solid rgba(244,196,48,.55); border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
        background: linear-gradient(120deg, rgba(29,63,158,.75), rgba(10,23,51,.92)); box-shadow: 0 0 42px rgba(29,63,158,.55); }
.hero .title { font-family: 'Oswald', sans-serif; font-size: 2.3rem; color: #f4c430; letter-spacing: .1em; text-transform: uppercase; }
.hero .sub { opacity: .85; margin-top: 4px; }
.pitch { background: linear-gradient(180deg, #14613d, #0b3d26); border: 2px solid rgba(255,255,255,.25); border-radius: 18px; padding: 16px; }
.row { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin: 10px 0; }
.card { min-width: 130px; padding: 8px 10px; border-radius: 12px; text-align: center;
        background: linear-gradient(160deg, #1d3f9e, #0a1733); border: 1px solid rgba(244,196,48,.55);
        box-shadow: 0 4px 14px rgba(0,0,0,.45); }
.card .n { font-weight: 700; font-size: .92rem; }
.card .m { font-size: .75rem; opacity: .8; }
.card.cap { border: 2px solid #f4c430; box-shadow: 0 0 18px rgba(244,196,48,.75); }
.card.vc { border: 2px solid #9fb4ff; }
.stButton > button { background: linear-gradient(90deg, #f4c430, #e0a800); color: #0a1733; font-weight: 700; border: none; border-radius: 10px; }
.stButton > button:hover { color: #0a1733; filter: brightness(1.08); }
div[data-testid="stMetricValue"] { color: #f4c430; font-family: 'Oswald', sans-serif; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

VICE_WEIGHT = 0.2  # vice scores double only if captain doesn't play
MAX_PER_CLUB = 3


@st.cache_data(ttl=600, show_spinner="Fetching the latest projections...")
def live_projections():
    return dl.fetch_projections()


def build_lineup(pool, values, cfg, locks, excludes, previous):
    """Solve one lineup with PuLP. Returns (player_ids, captain_id, vice_id) or None."""
    rows = pool.set_index("Fanteam ID")
    ids = list(rows.index)
    pos, club, salary = rows["Pos"], rows["Club"], rows["Salary"]
    prob = pulp.LpProblem("fanteam_ucl", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in ids}
    c = {i: pulp.LpVariable(f"c_{i}", cat="Binary") for i in ids}
    v = {i: pulp.LpVariable(f"v_{i}", cat="Binary") for i in ids}

    prob += pulp.lpSum(
        values[i] * (x[i] + c[i]) + VICE_WEIGHT * values[i] * v[i] for i in ids
    )

    cost = pulp.lpSum(salary[i] * x[i] for i in ids)
    prob += pulp.lpSum(x.values()) == 11
    prob += cost <= cfg["budget"]
    if cfg["min_spend"] > 0:
        prob += cost >= cfg["min_spend"]
    
    # Position constraints
    prob += pulp.lpSum(x[i] for i in ids if pos[i] == "GK") == 1
    
    # Dynamic or standard defensive formations
    min_d, min_m, min_f = cfg["formation"]
    prob += pulp.lpSum(x[i] for i in ids if pos[i] == "D") >= min_d
    prob += pulp.lpSum(x[i] for i in ids if pos[i] == "M") >= min_m
    prob += pulp.lpSum(x[i] for i in ids if pos[i] == "F") >= min_f

    for name in club.unique():
        prob += pulp.lpSum(x[i] for i in ids if club[i] == name) <= MAX_PER_CLUB

    prob += pulp.lpSum(c.values()) == 1
    prob += pulp.lpSum(v.values()) == 1
    for i in ids:
        prob += c[i] <= x[i]
        prob += v[i] <= x[i]
        prob += c[i] + v[i] <= 1
        if i in locks:
            prob += x[i] == 1
        if i in excludes:
            prob += x[i] == 0

    if cfg["stack"] != "None":
        flags = {
            name: pulp.LpVariable(f"s_{name}", cat="Binary") for name in club.unique()
        }
        prob += pulp.lpSum(flags.values()) == 1
        for name, flag in flags.items():
            members = [i for i in ids if club[i] == name]
            prob += pulp.lpSum(x[i] for i in members) >= cfg["stack_size"] * flag
            if cfg["stack"] == "Stack + bring-back":
                opp = cfg["opp"].get(name)
                opp_members = [i for i in ids if club[i] == opp]
                if opp_members:
                    prob += pulp.lpSum(x[i] for i in opp_members) >= flag

    for prev in previous:
        prob += pulp.lpSum(x[i] for i in prev if i in x) <= cfg["max_overlap"]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    chosen = [i for i in ids if x[i].value() > 0.5]
    captain = next(i for i in ids if c[i].value() > 0.5)
    vice = next(i for i in ids if v[i].value() > 0.5)
    return chosen, captain, vice


def pitch_html(lineup_rows, captain, vice):
    def card(pid):
        r = lineup_rows.loc[pid]
        cls, tag = "card", ""
        if pid == captain:
            cls, tag = "card cap", " (C)"
        elif pid == vice:
            cls, tag = "card vc", " (VC)"
        return (
            f'<div class="{cls}"><div class="n">{r["Player"]}{tag}</div>'
            f'<div class="m">{r["Club"]} · {r["Salary"]:.1f}m</div></div>'
        )

    rows_html = []
    for code in ("GK", "D", "M", "F"):
        ids = lineup_rows.index[lineup_rows["Pos"] == code]
        rows_html.append('<div class="row">' + "".join(card(i) for i in ids) + "</div>")
    return '<div class="pitch">' + "".join(rows_html) + "</div>"


# ---------- sidebar: data ----------
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Projections source", ["Live (Daily Fantasy Tips)", "Upload CSV"])
    players_file = st.file_uploader("Fanteam player list (CSV)", type="csv")
    proj_file = None
    if source == "Upload CSV":
        proj_file = st.file_uploader("Projections CSV", type="csv")
    if st.button("Refresh projections now"):
        st.cache_data.clear()

if players_file is None:
    st.info("Upload the Fanteam player list in the sidebar to begin.")
    st.stop()

players = dl.load_player_list(players_file)
proj, updated, error = None, None, None
try:
    if source.startswith("Live"):
        proj, updated = live_projections()
    elif proj_file is not None:
        proj = dl.load_projections_csv(proj_file)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    error = str(exc)
if proj is None:
    st.warning(
        f"Live fetch failed: {error}"
        if error
        else "Upload the projections CSV in the sidebar."
    )
    st.stop()

merged, report = dl.merge_projections(proj, players)
chains = dl.safety_net_chains(players)
static = dl.static_expected_points(merged, chains)
pstart = dict(zip(merged["Fanteam ID"], merged["p_start"]))
proj_by_id = dict(zip(merged["Fanteam ID"], merged["Proj"]))
ceil = merged["Ceiling"].fillna(merged["Proj"])
gap = dict(zip(merged["Fanteam ID"], (ceil - merged["Proj"]).clip(lower=0)))
opp_map = merged.groupby("Club")["Opp"].first().to_dict()
name_to_id = dict(zip(merged["Player"], merged["Fanteam ID"]))

# ---------- sidebar: build controls ----------
with st.sidebar:
    st.header("2 · Build")
    mode = st.radio("Mode", ["Pure Projection", "Cash", "GPP"])
    budget = st.slider("Budget cap (m)", 60.0, 120.0, 120.0, 0.5)
    min_spend = st.slider("Minimum spend (m)", 0.0, 120.0, 0.0, 0.5)
    
    formation_choice = st.selectbox(
        "Minimum Formation (D-M-F)",
        ["Standard (3-3-2)", "Heavy Defense (4-3-2)", "Heavy Midfield (3-4-2)", "Heavy Attack (3-3-3)"]
    )
    form_map = {
        "Standard (3-3-2)": (3, 3, 2),
        "Heavy Defense (4-3-2)": (4, 3, 2),
        "Heavy Midfield (3-4-2)": (3, 4, 2),
        "Heavy Attack (3-3-3)": (3, 3, 3)
    }
    min_formation = form_map[formation_choice]

    stack = st.selectbox(
        "Stacking", ["None", "Stack (no bring-back)", "Stack + bring-back"]
    )
    stack_size = st.radio("Stack size", [2, 3], horizontal=True)
    n_lineups = st.slider("Number of lineups", 1, 15, 1)
    max_overlap = st.slider("Max shared players between lineups", 1, 10, 7)
    upside = st.slider("GPP upside weight", 0.0, 1.0, 0.5, disabled=mode != "GPP")
    allow_possible = st.checkbox("Allow 'possible' starters", value=True)
    
    st.header("3 · Include / exclude")
    search_filter = st.text_input("Search players for lock/fade", "")
    filtered_names = sorted([n for n in name_to_id if search_filter.lower() in n.lower()])
    locks = st.multiselect("Lock in", filtered_names)
    fades = st.multiselect("Exclude", filtered_names)
    
    st.header("4 · Kick-off")
    kick = st.time_input(
        "Kick-off (UK time)", value=datetime.strptime("20:00", "%H:%M").time()
    )

# ---------- hero ----------
now = datetime.now(ZoneInfo("Europe/London"))
target = now.replace(hour=kick.hour, minute=kick.minute, second=0, microsecond=0)
seconds = int((target - now).total_seconds())
countdown = (
    "Kicked off"
    if seconds <= 0
    else f"{seconds // 3600}h {seconds % 3600 // 60:02d}m to kick-off"
)
st.markdown(
    f'<div class="hero"><div class="title">Champions League · Fanteam Optimizer</div>'
    f'<div class="sub">Projections: {updated or "uploaded file"} · {countdown}</div></div>',
    unsafe_allow_html=True,
)

with st.expander("Data check & Diagnostics"):
    st.write(report)
    club_counts = merged["Club"].value_counts()
    st.bar_chart(club_counts)

if min_spend > budget:
    st.error("The minimum spend is above the budget cap.")
    st.stop()

# ---------- player pool for the chosen mode ----------
pool = merged[merged["Lineup"].isin(["expected", "possible", "unexpected"])]
if mode == "Cash":
    pool = pool[pool["Lineup"].isin(["expected", "possible"])]
if not allow_possible:
    pool = pool[pool["Lineup"] != "possible"]
fade_ids = {name_to_id[n] for n in fades}
pool = pool[~pool["Fanteam ID"].isin(fade_ids)].reset_index(drop=True)
pool_ids = set(pool["Fanteam ID"])

lock_ids = {name_to_id[n] for n in locks}
missing = [
    n for n in locks if name_to_id[n] not in pool_ids and name_to_id[n] not in fade_ids
]
if missing:
    st.warning(
        f"Can't lock {', '.join(missing)}: not in this mode's pool (injured, excluded or unknown)."
    )
lock_ids &= pool_ids
if lock_ids & fade_ids:
    st.error("A player is both locked and excluded.")
    st.stop()

if mode == "Pure Projection":
    values = {i: static[i] for i in pool["Fanteam ID"]}
elif mode == "Cash":
    values = {i: static[i] - 0.5 * pstart[i] * gap[i] for i in pool["Fanteam ID"]}
else:
    values = {i: static[i] + upside * pstart[i] * gap[i] for i in pool["Fanteam ID"]}

cfg = {
    "budget": budget,
    "min_spend": min_spend,
    "stack": stack,
    "stack_size": stack_size,
    "opp": opp_map,
    "max_overlap": max_overlap,
    "formation": min_formation,
}

with st.expander(f"Player pool ({len(pool)} players, {mode})"):
    shown = pool[["Player", "Pos", "Club", "Opp", "Salary", "Proj", "Lineup"]].copy()
    shown["Value"] = [round(values[i], 2) for i in pool["Fanteam ID"]]
    st.dataframe(shown.sort_values("Value", ascending=False), hide_index=True)

# ---------- build ----------
if st.button("Build lineups", type="primary"):
    previous, results = [], []
    for _ in range(n_lineups):
        solved = build_lineup(pool, values, cfg, lock_ids, fade_ids, previous)
        if solved is None:
            st.error(
                "No lineup fits these settings. Try a wider budget, fewer locks, or a looser stack."
            )
            break
        chosen, cap, vice = solved
        previous.append(chosen)
        total, detail = dl.lineup_expected_points(
            chosen, proj_by_id, pstart, chains, cap, vice
        )
        results.append((chosen, cap, vice, total, detail))

    names = dict(zip(merged["Fanteam ID"], merged["Player"]))
    exports = []
    for n, (chosen, cap, vice, total, detail) in enumerate(results, 1):
        # Robust sorting by safety-net priority / expectation
        ordered_chosen = sorted(
            chosen, key=lambda i: detail[i]["expected"], reverse=True
        )
        lineup_rows = merged[merged["Fanteam ID"].isin(chosen)].set_index("Fanteam ID")
        table_rows = []
        for i in ordered_chosen:
            rep = detail[i]["replacement"]
            table_rows.append(
                {
                    "Player": lineup_rows.loc[i, "Player"],
                    "Pos": lineup_rows.loc[i, "Pos"],
                    "Club": lineup_rows.loc[i, "Club"],
                    "Opp": lineup_rows.loc[i, "Opp"],
                    "Salary": lineup_rows.loc[i, "Salary"],
                    "Proj": lineup_rows.loc[i, "Proj"],
                    "Status": lineup_rows.loc[i, "Lineup"],
                    "Expected": detail[i]["expected"],
                    "Safety net": names.get(rep, "") if rep else "",
                    "Role": "Captain" if i == cap else "Vice" if i == vice else "",
                }
            )
        table = pd.DataFrame(table_rows)
        exports.append(table.assign(Lineup=n))

        st.subheader(f"Lineup {n}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected points", f"{total:.1f}")
        c2.metric("Cost", f"{lineup_rows['Salary'].sum():.1f}m")
        c3.metric("Captain", lineup_rows.loc[cap, "Player"])
        problems = dl.check_lineup(merged, chosen, budget=budget)
        if problems:
            st.error("Rule check: " + "; ".join(problems))
        st.markdown(pitch_html(lineup_rows, cap, vice), unsafe_allow_html=True)
        st.dataframe(table, hide_index=True)

    if exports:
        combined_export = pd.concat(exports)
        st.download_button(
            "Download all lineups (CSV)",
            combined_export.to_csv(index=False),
            file_name="ucl_lineups_optimized.csv",
            mime="text/csv",
        )
