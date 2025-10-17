# streamlit_app.py
# -*- coding: utf-8 -*-
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(page_title="Arcadetafel Dashboard", page_icon="🎮", layout="wide")

st.title("🎮 Arcadetafel Activiteit — Dashboard")
st.caption(
    "Elke logregel betekent: in de **afgelopen minuut** was er activiteit, en op dat moment stond de genoemde game open. "
    "We tellen die afgelopen minuut als **actieve minuut** (0/1) voor die game. Dubbele logs binnen dezelfde minuut voor dezelfde game worden 1x geteld."
)

# ============== Upload & parsing ==============
uploaded = st.file_uploader("⬆️ Upload het .txt logbestand", type=["txt"])

if not uploaded:
    st.info("Upload een .txt bestand om te starten.")
    st.stop()

text = uploaded.read().decode("utf-8", errors="ignore")

# Patroon: DD/MM/YYYY at H:MM in  Game
pat = re.compile(
    r"ACTIVITY REGISTERED:\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*at\s*(\d{1,2}):(\d{1,2})\s*in\s*(.+)$",
    re.IGNORECASE
)

rows = []
for line in text.splitlines():
    m = pat.search(line.strip())
    if not m:
        continue
    dd, mm, yyyy, hh, minute, game = m.groups()
    dd = int(dd); mm = int(mm); yyyy = int(yyyy); hh = int(hh); minute = int(minute)
    game = game.strip()
    try:
        ts = datetime(yyyy, mm, dd, hh, minute)
    except ValueError:
        continue
    activity_minute = (ts - timedelta(minutes=1)).replace(second=0, microsecond=0)
    rows.append({"timestamp": ts, "game": game, "activity_minute": activity_minute})

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

if df.empty:
    st.warning("Geen geldige regels gevonden in het logbestand.")
    st.stop()

# ============== Basisfeatures & filters ==============
df["date"] = df["activity_minute"].dt.date

dates = sorted(df["date"].unique())
left, mid, right = st.columns([1.2, 1, 1.5])

with left:
    selected_date = st.selectbox("📅 Kies een datum", options=dates, index=len(dates)-1)
with mid:
    game_list = sorted(df["game"].unique())
    selected_games = st.multiselect("🎯 Filter op games", options=game_list, default=game_list)
with right:
    highlight_game = st.selectbox("🔍 Highlight game (optie)", options=["— geen —"] + game_list)

df_day = df[df["date"] == selected_date].copy()
if selected_games:
    df_day = df_day[df_day["game"].isin(selected_games)]

# ============== Tijdlijn + crop-bereik ==============
# Unieke (game, minute) -> dedupe
if not df_day.empty:
    gm = (df_day[["game", "activity_minute"]]
          .drop_duplicates()
          .rename(columns={"activity_minute": "time"}))
else:
    gm = pd.DataFrame(columns=["game", "time"])

if gm.empty:
    st.warning("Geen activiteit op deze dag/filters.")
    st.stop()

# Crop naar eerste/laatste activiteit (met 2 min marge aan beide kanten, binnen de dag)
first_ts = gm["time"].min()
last_ts = gm["time"].max()
margin = timedelta(minutes=2)
crop_start = max(datetime.combine(selected_date, datetime.min.time()), first_ts - margin)
crop_end = min(datetime.combine(selected_date, datetime.max.time()), last_ts + margin)

minute_index = pd.date_range(crop_start, end=crop_end, freq="1min")

# Totaal-gebruik 0/1 over cropped range
usage = pd.Series(0, index=minute_index, dtype=int)
usage.loc[gm["time"].unique()] = 1
usage_df = usage.rename("gebruik").reset_index().rename(columns={"index": "time"})

# Per-game 0/1 over alle minuten (zoals totaalgebruik, maar kleur per game)
# Bouw voor elke game een 0/1 serie over minute_index en concat naar lang formaat
frames = []
for g, sub in gm.groupby("game"):
    s = pd.Series(0, index=minute_index, dtype=int)
    s.loc[sub["time"].values] = 1
    frames.append(pd.DataFrame({"time": s.index, "game": g, "active": s.values}))
per_game_active_full = pd.concat(frames, ignore_index=True)

# ============== Charts ==============
st.subheader("📈 Tijdlijn: totaal gebruik (0/1)")
chart_total = alt.Chart(usage_df).mark_line(point=False, interpolate="monotone").encode(
    x=alt.X("time:T", title="Tijd", axis=alt.Axis(format="%H:%M")),
    y=alt.Y("gebruik:Q", title="Gebruik (0/1)", scale=alt.Scale(domain=[-0.05, 1.05])),
    tooltip=[alt.Tooltip("time:T", title="Tijd", format="%H:%M"),
             alt.Tooltip("gebruik:Q", title="Gebruik")]
).properties(height=180)
st.altair_chart(chart_total, use_container_width=True)

st.subheader("🎨 Gebruik per game (0/1 per minuut)")
base = alt.Chart(per_game_active_full).encode(
    x=alt.X("time:T", title="Tijd", axis=alt.Axis(format="%H:%M")),
    y=alt.Y("active:Q", title="Actief (0/1)", scale=alt.Scale(domain=[-0.05, 1.05])),
    color=alt.Color("game:N", title="Game"),
    tooltip=[alt.Tooltip("time:T", title="Tijd", format="%H:%M"),
             alt.Tooltip("game:N", title="Game"),
             alt.Tooltip("active:Q", title="Actief (0/1)")]
)

if highlight_game != "— geen —":
    opacity = alt.condition(alt.datum.game == highlight_game, alt.value(1), alt.value(0.2))
    size = alt.condition(alt.datum.game == highlight_game, alt.value(2.6), alt.value(1.2))
else:
    opacity = alt.value(1)
    size = alt.value(1.6)

per_game_chart = base.mark_line(interpolate="monotone").encode(opacity=opacity, size=size).properties(height=260)
st.altair_chart(per_game_chart, use_container_width=True)

# ============== Samenvatting (alleen actieve minuten per game) ==============
st.subheader("📊 Actieve minuten per game")
active_minutes = (per_game_active_full[per_game_active_full["active"] == 1]
                  .groupby("game")["time"].nunique()
                  .rename("actieve_minuten")
                  .sort_values(ascending=False))

# Dataframe links wat compacter en zonder indexkolom
st.dataframe(active_minutes.reset_index(), use_container_width=True)

# Bar: dichter bij levelnaam (minder padding) en start exact bij 0
bar = alt.Chart(active_minutes.reset_index()).mark_bar().encode(
    x=alt.X("actieve_minuten:Q", title="Actieve minuten",
            scale=alt.Scale(domain=[0, float(active_minutes.max()) * 1.05], nice=False, zero=True),
            axis=alt.Axis(labelPadding=4)),
    y=alt.Y("game:N", sort='-x', title=None, axis=alt.Axis(labelPadding=4))
).properties(height=300, padding={"left": 8, "right": 10, "top": 10, "bottom": 10})
st.altair_chart(bar, use_container_width=True)

st.caption("Tijdlijnen zijn gecropt naar het eerste/laatste event (±2 min marge). Per-game lijn = 0/1 per minuut, samengevoegd met dubbele logs binnen dezelfde minuut.")
