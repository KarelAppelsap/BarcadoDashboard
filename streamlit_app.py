# streamlit_app.py
# Seconde-nauwkeurig activity dashboard (nieuw logformat)
# - Multi-file upload
# - Timeline per dag (1 rij per dag) of single day
# - Zoom via begin/eindtijd + quick zoom presets
# - Extra interactief zoomen/pannen in de chart (Altair)
# - Counters: plays per game, actieve tijd per state/game, occupancy binnen gekozen tijdvenster

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Dict

import pandas as pd
import streamlit as st
import altair as alt


st.set_page_config(page_title="Arcade Activity Dashboard", layout="wide")
st.title("Arcade Activity Dashboard")
st.caption(
    "Logformat: 'GAME STARTED: DD/MM/YYYY at HH:MM:SS LEVEL NAME: <State>' "
    "en optioneel 'ACTIVITY ENDED'. De duur wordt afgeleid als (event -> volgende event)."
)

# ----------------------------
# Parsing (nieuw format)
# ----------------------------
# Ondersteunt beide varianten:
#   GAME STARTED: 18/12/2025 at 09:41:31 LEVEL NAME: MainMenu P1: true ...
#   STARTED: 18/12/2025 at 09:41:31 LEVEL NAME: MainMenu P1: true ...   (soms zonder "GAME")
RE_STARTED = re.compile(
    r"^(?:GAME\s+)?STARTED:\s*(\d{2}/\d{2}/\d{4})\s*at\s*(\d{2}:\d{2}:\d{2})\s*LEVEL NAME:\s*([A-Za-z0-9_]+)\b(.*)$",
    re.IGNORECASE,
)
RE_ACTIVITY_ENDED = re.compile(r"^ACTIVITY ENDED\s*$", re.IGNORECASE)
RE_PFLAGS = re.compile(r"\bP([1-4]):\s*(true|false)\b", re.IGNORECASE)


@dataclass
class Event:
    ts: datetime
    state: str
    p_active: Optional[int]
    source_file: str


@dataclass
class Segment:
    day: date
    start: datetime
    end: datetime
    state: str
    p_active: Optional[int]
    source_file: str


def parse_events(text: str, filename: str) -> List[Event]:
    events: List[Event] = []
    last_ts: Optional[datetime] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = RE_STARTED.match(line)
        if m:
            d_str, t_str, level, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            try:
                ts = datetime.strptime(f"{d_str} {t_str}", "%d/%m/%Y %H:%M:%S")
            except ValueError:
                continue

            p_active = None
            flags = RE_PFLAGS.findall(tail or "")
            if flags:
                p_active = sum(1 for _, v in flags if v.lower() == "true")

            events.append(Event(ts=ts, state=level.strip(), p_active=p_active, source_file=filename))
            last_ts = ts
            continue

        if RE_ACTIVITY_ENDED.match(line):
            # In je logs heeft deze regel vaak geen timestamp; we gebruiken dan last_ts als start van inactief.
            if last_ts is not None:
                events.append(Event(ts=last_ts, state="INACTIVE", p_active=None, source_file=filename))
            continue

    events.sort(key=lambda e: e.ts)
    return events


def build_segments(events: List[Event]) -> List[Segment]:
    if len(events) < 2:
        return []

    segs: List[Segment] = []
    for i in range(len(events) - 1):
        a, b = events[i], events[i + 1]
        if b.ts <= a.ts:
            continue
        segs.append(
            Segment(
                day=a.ts.date(),
                start=a.ts,
                end=b.ts,
                state=a.state,
                p_active=a.p_active,
                source_file=a.source_file,
            )
        )
    return segs


def stable_domain(states: List[str]) -> List[str]:
    uniq = sorted(set(states))
    ordered: List[str] = []
    for first in ["MainMenu", "Screensaver"]:
        if first in uniq:
            uniq.remove(first)
            ordered.append(first)
    # INACTIVE graag als laatste
    if "INACTIVE" in uniq:
        uniq.remove("INACTIVE")
        ordered.extend(uniq)
        ordered.append("INACTIVE")
    else:
        ordered.extend(uniq)
    return ordered


def is_occupied_state(state: str, screensaver_name: str, count_mainmenu: bool) -> bool:
    if state == "INACTIVE":
        return False
    if state == screensaver_name:
        return False
    if (not count_mainmenu) and state == "MainMenu":
        return False
    return True


# ----------------------------
# UI: Upload
# ----------------------------
files = st.file_uploader("Upload .txt logbestanden", type=["txt"], accept_multiple_files=True)
if not files:
    st.info("Upload een of meerdere .txt bestanden om te starten.")
    st.stop()

all_events: List[Event] = []
for f in files:
    txt = f.read().decode("utf-8", errors="ignore")
    all_events.extend(parse_events(txt, f.name))

if len(all_events) < 2:
    st.warning("Te weinig events gevonden om een tijdlijn te bouwen.")
    st.stop()

segments = build_segments(all_events)
df = pd.DataFrame(
    [
        {
            "day": s.day.isoformat(),
            "start": s.start,
            "end": s.end,
            "state": s.state,
            "p_active": s.p_active,
            "source_file": s.source_file,
        }
        for s in segments
    ]
)

days_all = sorted(df["day"].unique().tolist())
states_all = sorted(df["state"].unique().tolist())

# ----------------------------
# Sidebar: controls (A + B gecombineerd)
# ----------------------------
with st.sidebar:
    st.header("Weergave")
    mode = st.radio("Mode", ["All days", "Single day"], index=0)

    st.header("Zoom (tijdvenster)")
    quick = st.selectbox("Quick zoom", ["Custom", "30 min", "1 uur", "2 uur", "4 uur", "Hele dag"], index=0)

    zoom_start = st.time_input("Starttijd", value=time(0, 0, 0))

    if quick == "Hele dag":
        zoom_end = time(23, 59, 59)
        st.caption("Eindtijd: 23:59:59")
    elif quick == "Custom":
        zoom_end = st.time_input("Eindtijd", value=time(23, 59, 59))
    else:
        minutes = {"30 min": 30, "1 uur": 60, "2 uur": 120, "4 uur": 240}[quick]
        dt_s = datetime.combine(date.today(), zoom_start)
        dt_e = dt_s + timedelta(minutes=minutes)
        dt_e = min(dt_e, datetime.combine(date.today(), time(23, 59, 59)))
        zoom_end = dt_e.time()
        st.caption(f"Eindtijd automatisch: {zoom_end.strftime('%H:%M:%S')}")

    if zoom_end <= zoom_start:
        st.warning("Eindtijd moet later zijn dan starttijd (binnen dezelfde dag).")

    st.header("Timeline gedrag")
    enable_interactive = st.checkbox("Interactief zoomen/pannen", value=True, help="Slepen = pannen, scroll = zoomen, dubbelklik = reset")
    show_inactive = st.checkbox("Toon INACTIVE", value=False)
    show_mainmenu = st.checkbox("Toon MainMenu", value=True)
    show_screensaver = st.checkbox("Toon Screensaver", value=True)

    st.header("Bezetting")
    count_mainmenu_as_occupied = st.checkbox("MainMenu telt mee als bezet", value=True)
    screensaver_state_name = st.text_input("Screensaver state naam", value="Screensaver")

# ----------------------------
# Main filters
# ----------------------------
c1, c2, c3 = st.columns([1.2, 2.0, 1.2])

with c1:
    if mode == "Single day":
        selected_day = st.selectbox("Dag", options=days_all, index=len(days_all) - 1)
        day_list = [selected_day]
    else:
        day_list = st.multiselect("Dagen", options=days_all, default=days_all)

with c2:
    # Filter states (behalve INACTIVE) - user kan games + menu/saver toggles via sidebar sturen
    states_pickable = sorted([s for s in states_all if s != "INACTIVE"])
    selected_states = st.multiselect("State filter", options=states_pickable, default=states_pickable)

with c3:
    highlight = st.selectbox("Highlight", options=["None"] + states_pickable, index=0)

df_f = df[df["day"].isin(day_list)].copy()
if selected_states:
    df_f = df_f[df_f["state"].isin(selected_states + (["INACTIVE"] if show_inactive else []))]

if not show_inactive:
    df_f = df_f[df_f["state"] != "INACTIVE"]
if not show_mainmenu:
    df_f = df_f[df_f["state"] != "MainMenu"]
if not show_screensaver:
    df_f = df_f[df_f["state"] != screensaver_state_name]

if df_f.empty:
    st.warning("Geen segmenten na filters.")
    st.stop()

# ----------------------------
# Clip segments to zoom window (B)
# ----------------------------
df_f["day_date"] = pd.to_datetime(df_f["day"]).dt.date
df_f["win_start_dt"] = df_f["day_date"].apply(lambda d: datetime.combine(d, zoom_start))
df_f["win_end_dt"] = df_f["day_date"].apply(lambda d: datetime.combine(d, zoom_end))

df_f["start_clipped"] = df_f[["start", "win_start_dt"]].max(axis=1)
df_f["end_clipped"] = df_f[["end", "win_end_dt"]].min(axis=1)
df_f = df_f[df_f["end_clipped"] > df_f["start_clipped"]].copy()
df_f["duration_s"] = (df_f["end_clipped"] - df_f["start_clipped"]).dt.total_seconds()

if df_f.empty:
    st.warning("Geen segmenten binnen het gekozen tijdvenster.")
    st.stop()

# Maak "tijd-only" kolommen op een dummy datum zodat alle dagen exact dezelfde X-as schaal delen.
dummy_day = date(2000, 1, 1)
df_f["x_start"] = df_f["start_clipped"].apply(lambda dt: datetime.combine(dummy_day, dt.time()))
df_f["x_end"] = df_f["end_clipped"].apply(lambda dt: datetime.combine(dummy_day, dt.time()))

win_s = datetime.combine(dummy_day, zoom_start)
win_e = datetime.combine(dummy_day, zoom_end)

# ----------------------------
# Counters + occupancy (binnen zoom window)
# ----------------------------
# Occupancy per dag: som(occupied duration) / window length
window_seconds = {}
for d in day_list:
    d_date = datetime.strptime(d, "%Y-%m-%d").date()
    ws = datetime.combine(d_date, zoom_start)
    we = datetime.combine(d_date, zoom_end)
    window_seconds[d] = max(0.0, (we - ws).total_seconds())

df_occ = df_f.copy()
df_occ["occupied"] = df_occ["state"].apply(lambda s: is_occupied_state(s, screensaver_state_name, count_mainmenu_as_occupied))

occ_day = (
    df_occ[df_occ["occupied"]]
    .groupby("day")["duration_s"]
    .sum()
    .rename("occupied_s")
    .reset_index()
)
occ_day["window_s"] = occ_day["day"].map(window_seconds).astype(float)
occ_day["occupied_min"] = (occ_day["occupied_s"] / 60.0).round(1)
occ_day["occupancy_pct"] = ((occ_day["occupied_s"] / occ_day["window_s"]).fillna(0.0) * 100.0).round(1)
occ_day = occ_day.sort_values("day")

# Active time per state/game
active_time = (
    df_f.groupby("state")["duration_s"]
    .sum()
    .rename("active_s")
    .reset_index()
)
active_time["active_min"] = (active_time["active_s"] / 60.0).round(1)
active_time = active_time.sort_values("active_s", ascending=False)

# Plays per game: count segments that start in window, excluding MainMenu/Screensaver/INACTIVE
exclude_for_plays = {"INACTIVE", screensaver_state_name, "MainMenu"}
plays = (
    df_f[~df_f["state"].isin(exclude_for_plays)]
    .groupby("state")
    .size()
    .rename("plays")
    .reset_index()
    .sort_values("plays", ascending=False)
)

summary = active_time.merge(plays, on="state", how="left").fillna({"plays": 0})
summary["plays"] = summary["plays"].astype(int)

# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Dagen", len(day_list))
k2.metric("Tijdvenster", f"{zoom_start.strftime('%H:%M:%S')} - {zoom_end.strftime('%H:%M:%S')}")
k3.metric("Unieke states", df_f["state"].nunique())
k4.metric("Gem. bezetting", f"{occ_day['occupancy_pct'].mean():.1f}%" if not occ_day.empty else "n/a")

st.subheader("Bezetting per dag (binnen tijdvenster)")
if occ_day.empty:
    st.info("Geen bezette tijd gevonden binnen het gekozen tijdvenster (volgens je bezettingsdefinitie).")
else:
    st.dataframe(occ_day[["day", "occupied_min", "occupancy_pct"]], use_container_width=True, hide_index=True)

st.subheader("Counters")
left, right = st.columns([1.4, 1.0])

with left:
    st.caption("Plays telt alleen echte games: MainMenu/Screensaver/INACTIVE worden uitgesloten.")
    st.dataframe(
        summary.rename(columns={"state": "state_or_game"}),
        use_container_width=True,
        hide_index=True
    )

with right:
    top = summary[summary["plays"] > 0].sort_values("plays", ascending=False).head(15)
    if top.empty:
        st.info("Geen plays gevonden (mogelijk door filters of window).")
    else:
        chart = alt.Chart(top.rename(columns={"state": "game"})).mark_bar().encode(
            x=alt.X("plays:Q", title="Plays"),
            y=alt.Y("game:N", sort="-x", title=None),
            tooltip=["game:N", "plays:Q", "active_min:Q"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)

# ----------------------------
# Timeline (A + B gecombineerd)
# ----------------------------
st.subheader("Tijdlijn")

plot = df_f.copy()
plot["row"] = plot["day"]
plot["state_norm"] = plot["state"]

domain = stable_domain(plot["state_norm"].tolist())

base = alt.Chart(plot).encode(
    x=alt.X(
        "x_start:T",
        title="Tijd",
        axis=alt.Axis(format="%H:%M:%S"),
        scale=alt.Scale(domain=[win_s, win_e]),
    ),
    x2="x_end:T",
    y=alt.Y("row:N", title=None, sort=days_all),
    color=alt.Color("state_norm:N", title="State", scale=alt.Scale(domain=domain)),
    tooltip=[
        alt.Tooltip("day:N", title="Dag"),
        alt.Tooltip("state_norm:N", title="State"),
        alt.Tooltip("start_clipped:T", title="Start", format="%H:%M:%S"),
        alt.Tooltip("end_clipped:T", title="Eind", format="%H:%M:%S"),
        alt.Tooltip("duration_s:Q", title="Duur (s)"),
        alt.Tooltip("p_active:Q", title="Players (true)"),
        alt.Tooltip("source_file:N", title="Bestand"),
    ],
)

if highlight != "None":
    opacity = alt.condition(alt.datum.state_norm == highlight, alt.value(1.0), alt.value(0.18))
else:
    opacity = alt.value(1.0)

height = 70 if mode == "Single day" else min(900, 60 * max(1, len(day_list)))
timeline = base.mark_bar(size=18).encode(opacity=opacity).properties(height=height)

# A: Interactief zoomen/pannen bovenop het gekozen window
# Let op: we binden alleen X (tijd). Y blijft vast (bind_y=False).
if enable_interactive:
    timeline = timeline.interactive(bind_y=False)

st.altair_chart(timeline, use_container_width=True)

with st.expander("Debug: geclipped segmenten"):
    st.dataframe(
        plot[["day", "start", "end", "start_clipped", "end_clipped", "state", "duration_s", "p_active", "source_file"]]
        .sort_values(["day", "start_clipped"]),
        use_container_width=True,
        hide_index=True
    )
