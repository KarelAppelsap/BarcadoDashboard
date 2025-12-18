# streamlit_app.py
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from typing import List, Optional, Dict, Tuple

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Arcadetafel Activity Dashboard", page_icon="🎮", layout="wide")
st.title("🎮 Arcadetafel Activity Dashboard (seconde-nauwkeurig)")

st.caption(
    "Dit dashboard gebruikt het nieuwe logformat met `GAME STARTED: ... LEVEL NAME: ...` "
    "en bouwt segmenten (start→volgende event) om echte tijdlijnen te tekenen."
)

# --- Parsing ---
# Ondersteunt:
#   GAME STARTED: 18/12/2025 at 09:41:31 LEVEL NAME: MainMenu P1: true ...
# En in sommige logs lijkt een regel te beginnen met "STARTED:" zonder "GAME" (komt voor in je voorbeeld) :contentReference[oaicite:3]{index=3}
RE_STARTED = re.compile(
    r"^(?:GAME\s+)?STARTED:\s*(\d{2}/\d{2}/\d{4})\s*at\s*(\d{2}:\d{2}:\d{2})\s*LEVEL NAME:\s*([A-Za-z0-9_]+)\b(.*)$"
)
RE_ACTIVITY_ENDED = re.compile(r"^ACTIVITY ENDED\s*$", re.IGNORECASE)

RE_P = re.compile(r"\bP([1-4]):\s*(true|false)\b", re.IGNORECASE)

@dataclass
class Event:
    ts: datetime
    state: str  # level name or "INACTIVE"
    p_active: Optional[int] = None
    source_file: str = ""

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
            d, t, level, tail = m.group(1), m.group(2), m.group(3), m.group(4)
            try:
                ts = datetime.strptime(f"{d} {t}", "%d/%m/%Y %H:%M:%S")
            except ValueError:
                continue

            # Player flags (optioneel)
            p = RE_P.findall(tail or "")
            p_active = None
            if p:
                # count "true"
                p_active = sum(1 for _, v in p if v.lower() == "true")

            events.append(Event(ts=ts, state=level, p_active=p_active, source_file=filename))
            last_ts = ts
            continue

        if RE_ACTIVITY_ENDED.match(line):
            # In je logs heeft ACTIVITY ENDED geen timestamp, dus: "inactive start op last_ts" :contentReference[oaicite:4]{index=4}
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
        segs.append(Segment(
            day=a.ts.date(),
            start=a.ts,
            end=b.ts,
            state=a.state,
            p_active=a.p_active,
            source_file=a.source_file
        ))
    return segs

def clip_segments_to_window(segs: pd.DataFrame, win_start: datetime, win_end: datetime) -> pd.DataFrame:
    df = segs.copy()
    df["start_clipped"] = df["start"].apply(lambda x: max(x, win_start))
    df["end_clipped"] = df["end"].apply(lambda x: min(x, win_end))
    df = df[df["end_clipped"] > df["start_clipped"]].copy()
    df["duration_s"] = (df["end_clipped"] - df["start_clipped"]).dt.total_seconds()
    return df

def stable_domain(states: List[str]) -> List[str]:
    # nette ordering: MainMenu en Screensaver vooraan, dan rest, INACTIVE achteraan
    uniq = sorted(set(states))
    def take(name):
        if name in uniq:
            uniq.remove(name)
            return [name]
        return []
    ordered = []
    ordered += take("MainMenu")
    ordered += take("Screensaver")
    ordered += [s for s in uniq if s != "INACTIVE"]
    if "INACTIVE" in states:
        ordered += ["INACTIVE"]
    return ordered

# --- UI: upload ---
files = st.file_uploader("⬆️ Upload één of meerdere .txt logbestanden", type=["txt"], accept_multiple_files=True)
if not files:
    st.info("Upload logbestanden om te starten.")
    st.stop()

all_events: List[Event] = []
for f in files:
    txt = f.read().decode("utf-8", errors="ignore")
    all_events.extend(parse_events(txt, f.name))

if len(all_events) < 2:
    st.warning("Te weinig events gevonden om segmenten te bouwen.")
    st.stop()

segments = build_segments(all_events)
df = pd.DataFrame([{
    "day": s.day.isoformat(),
    "start": s.start,
    "end": s.end,
    "state": s.state,
    "p_active": s.p_active,
    "source_file": s.source_file
} for s in segments])

days = sorted(df["day"].unique().tolist())
states = sorted(df["state"].unique().tolist())

# --- Sidebar controls ---
with st.sidebar:
    st.header("Weergave")
    mode = st.radio("Mode", ["All days", "Single day"], index=0)

    st.header("Zoom (tijdvenster)")
    # venster geldt per dag (dus 10:00–14:00 op elke dag)
    zoom_start = st.time_input("Starttijd", value=time(0, 0))
    zoom_end = st.time_input("Eindtijd", value=time(23, 59))

    st.header("Bezetting definitie")
    count_mainmenu_as_occupied = st.checkbox("MainMenu telt mee als bezet", value=True)
    screensaver_state_name = st.text_input("Screensaver state naam", value="Screensaver")
    inactive_state_name = "INACTIVE"

    st.header("Filters")
    show_inactive = st.checkbox("Toon INACTIVE in timeline", value=False)
    show_mainmenu = st.checkbox("Toon MainMenu in timeline", value=True)
    show_screensaver = st.checkbox("Toon Screensaver in timeline", value=True)

# Day selection
c1, c2, c3 = st.columns([1.2, 2.0, 1.2])
with c1:
    if mode == "Single day":
        day_sel = st.selectbox("📅 Dag", options=days, index=len(days)-1)
        day_list = [day_sel]
    else:
        day_list = st.multiselect("📅 Dagen", options=days, default=days)
with c2:
    # game filter (exclude INACTIVE)
    all_levels = sorted([s for s in states if s != inactive_state_name])
    game_filter = st.multiselect("🎯 Filter states", options=all_levels, default=all_levels)
with c3:
    highlight = st.selectbox("🔍 Highlight", options=["— geen —"] + all_levels, index=0)

df_f = df[df["day"].isin(day_list)].copy()
if game_filter:
    df_f = df_f[df_f["state"].isin(game_filter + ([inactive_state_name] if show_inactive else []))]

if not show_inactive:
    df_f = df_f[df_f["state"] != inactive_state_name]
if not show_mainmenu:
    df_f = df_f[df_f["state"] != "MainMenu"]
if not show_screensaver:
    df_f = df_f[df_f["state"] != screensaver_state_name]

if df_f.empty:
    st.warning("Geen segmenten na filters.")
    st.stop()

# --- Build per-day zoom window and clip ---
# Clip per row using day-specific datetime window
df_f["day_date"] = pd.to_datetime(df_f["day"]).dt.date
df_f["win_start"] = df_f["day_date"].apply(lambda d: datetime.combine(d, zoom_start))
df_f["win_end"] = df_f["day_date"].apply(lambda d: datetime.combine(d, zoom_end))

# Clip row-wise
df_f["start_clipped"] = df_f.apply(lambda r: max(r["start"], r["win_start"]), axis=1)
df_f["end_clipped"] = df_f.apply(lambda r: min(r["end"], r["win_end"]), axis=1)
df_f = df_f[df_f["end_clipped"] > df_f["start_clipped"]].copy()
df_f["duration_s"] = (df_f["end_clipped"] - df_f["start_clipped"]).dt.total_seconds()

if df_f.empty:
    st.warning("Geen segmenten binnen het gekozen zoom-venster.")
    st.stop()

# --- Occupancy + counters (binnen zoom) ---
# occupied = alles behalve Screensaver/INACTIVE (en optioneel MainMenu)
def is_occupied(state: str) -> bool:
    if state == inactive_state_name:
        return False
    if state == screensaver_state_name:
        return False
    if (not count_mainmenu_as_occupied) and state == "MainMenu":
        return False
    return True

df_occ = df_f.copy()
df_occ["occupied"] = df_occ["state"].apply(is_occupied)

# timeframe length per day (seconds)
window_seconds = {}
for d in day_list:
    d_date = datetime.strptime(d, "%Y-%m-%d").date()
    ws = datetime.combine(d_date, zoom_start)
    we = datetime.combine(d_date, zoom_end)
    window_seconds[d] = max(0.0, (we - ws).total_seconds())

occ_day = (df_occ[df_occ["occupied"]]
           .groupby("day")["duration_s"].sum()
           .rename("occupied_s")
           .reset_index())

occ_day["window_s"] = occ_day["day"].map(window_seconds).astype(float)
occ_day["occupancy_%"] = (occ_day["occupied_s"] / occ_day["window_s"]).replace([float("inf")], 0.0).fillna(0.0) * 100.0
occ_day["occupied_min"] = (occ_day["occupied_s"] / 60.0).round(1)
occ_day["occupancy_%"] = occ_day["occupancy_%"].round(1)
occ_day = occ_day.sort_values("day")

# plays per game = count of segments that start within window, excluding MainMenu/Screensaver/INACTIVE by default
df_plays = df_f.copy()
exclude_for_plays = {inactive_state_name, screensaver_state_name, "MainMenu"}
plays = (df_plays[~df_plays["state"].isin(exclude_for_plays)]
         .groupby("state").size().rename("plays").reset_index()
         .sort_values("plays", ascending=False))

# active time per state
active_time = (df_f.groupby("state")["duration_s"].sum()
               .rename("active_s").reset_index())
active_time["active_min"] = (active_time["active_s"] / 60.0).round(1)
active_time = active_time.sort_values("active_s", ascending=False)

summary = active_time.merge(plays, on="state", how="left").fillna({"plays": 0})
summary["plays"] = summary["plays"].astype(int)

# --- KPIs ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Dagen geselecteerd", len(day_list))
k2.metric("Zoom venster", f"{zoom_start.strftime('%H:%M')}–{zoom_end.strftime('%H:%M')}")
k3.metric("Unieke states", df_f["state"].nunique())
k4.metric("Gem. bezetting", f"{occ_day['occupancy_%'].mean():.1f}%" if not occ_day.empty else "—")

st.subheader("📌 Bezetting per dag (binnen zoom-venster)")
st.dataframe(occ_day[["day", "occupied_min", "occupancy_%"]], use_container_width=True, hide_index=True)

st.subheader("📊 Counters")
cL, cR = st.columns([1.4, 1.0])
with cL:
    st.write("**Per state: actieve tijd + plays** (plays telt alleen echte games; MainMenu/Screensaver/INACTIVE uitgesloten)")
    st.dataframe(summary.rename(columns={"state": "state_or_game"}), use_container_width=True, hide_index=True)
with cR:
    top = summary[summary["plays"] > 0].sort_values("plays", ascending=False).head(15)
    if not top.empty:
        chart = alt.Chart(top.rename(columns={"state": "game"})).mark_bar().encode(
            x=alt.X("plays:Q", title="Plays"),
            y=alt.Y("game:N", sort='-x', title=None),
            tooltip=["game:N", "plays:Q", "active_min:Q"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Geen plays gevonden (mogelijk door filters / korte window).")

# --- Timeline chart ---
st.subheader("🧵 Tijdlijn")

plot = df_f.copy()
plot["row"] = plot["day"]  # 1 row per dag
plot["state_norm"] = plot["state"]

domain = stable_domain(plot["state_norm"].tolist())

base = alt.Chart(plot).encode(
    x=alt.X("start_clipped:T", title="Tijd", axis=alt.Axis(format="%H:%M:%S")),
    x2="end_clipped:T",
    y=alt.Y("row:N", title=None, sort=days),
    color=alt.Color("state_norm:N", title="State", scale=alt.Scale(domain=domain)),
    tooltip=[
        alt.Tooltip("day:N", title="Dag"),
        alt.Tooltip("state_norm:N", title="State"),
        alt.Tooltip("start_clipped:T", title="Start", format="%H:%M:%S"),
        alt.Tooltip("end_clipped:T", title="End", format="%H:%M:%S"),
        alt.Tooltip("duration_s:Q", title="Duur (s)"),
        alt.Tooltip("p_active:Q", title="Players (true)"),
        alt.Tooltip("source_file:N", title="Bestand"),
    ]
)

if highlight != "— geen —":
    opacity = alt.condition(alt.datum.state_norm == highlight, alt.value(1.0), alt.value(0.18))
else:
    opacity = alt.value(1.0)

height = 70 if mode == "Single day" else min(900, 60 * max(1, len(day_list)))
timeline = base.mark_bar(size=18).encode(opacity=opacity).properties(height=height)

st.altair_chart(timeline, use_container_width=True)

with st.expander("🔎 Debug: segmenten (geclipped op zoom)"):
    st.dataframe(
        plot[["day", "start", "end", "start_clipped", "end_clipped", "state", "duration_s", "p_active", "source_file"]]
        .sort_values(["day", "start_clipped"]),
        use_container_width=True,
        hide_index=True
    )
