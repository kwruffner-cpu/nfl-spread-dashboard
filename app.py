import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="NFL Spread Model",
    page_icon="🏈",
    layout="wide",
)


@st.cache_data
def load_data():
    df = pd.read_csv("data/production_backtest.csv")

    numeric_cols = [
        "season",
        "week",
        "model_margin",
        "model_spread",
        "market_margin",
        "model_edge",
        "abs_edge",
        "actual_margin",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def format_line(team, margin):
    """
    Convert a home-perspective margin into a conventional betting line.
    Positive margin = home team favored.
    """
    if pd.isna(margin):
        return "—"

    if abs(margin) < 0.05:
        return "PK"

    if margin > 0:
        return f"{team} -{abs(margin):.1f}"

    return f"{team} +{abs(margin):.1f}"


def model_pick(row):
    if pd.isna(row["model_edge"]):
        return "—"

    if row["model_edge"] > 0:
        return row["home_team"]

    if row["model_edge"] < 0:
        return row["away_team"]

    return "—"


def final_score(row):
    if pd.isna(row["actual_margin"]):
        return "—"

    # We only store actual margin in the backtest table,
    # so show the result as a home-margin value.
    if row["actual_margin"] > 0:
        return f"{row['home_team']} +{row['actual_margin']:.0f}"

    if row["actual_margin"] < 0:
        return f"{row['away_team']} +{abs(row['actual_margin']):.0f}"

    return "Tie"


df = load_data()

st.title("🏈 NFL Spread Model")
st.caption(
    "Historical walk-forward backtest using only information "
    "available before each game."
)

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.subheader("Filters")

filter_col1, filter_col2 = st.columns(2)

seasons = sorted(df["season"].dropna().astype(int).unique())

with filter_col1:
    season = st.selectbox(
        "Season",
        seasons,
        index=len(seasons) - 1,
    )

season_df = df[df["season"] == season].copy()

weeks = sorted(
    season_df["week"].dropna().astype(int).unique()
)

week_options = ["All"] + weeks

with filter_col2:
    week = st.selectbox(
        "Week",
        week_options,
    )

min_edge = st.slider(
    "Minimum model edge",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
)

if week == "All":
    filtered = season_df.copy()
else:
    filtered = season_df[
        season_df["week"] == week
    ].copy()

filtered = filtered[
    filtered["abs_edge"] >= min_edge
].copy()

# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

graded = filtered[
    filtered["ats_result"].isin(["W", "L", "P"])
].copy()

wins = (graded["ats_result"] == "W").sum()
losses = (graded["ats_result"] == "L").sum()
pushes = (graded["ats_result"] == "P").sum()

decisions = wins + losses

ats_pct = (
    wins / decisions
    if decisions > 0
    else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Games",
    len(filtered),
)

c2.metric(
    "ATS Record",
    f"{wins}-{losses}-{pushes}",
)

c3.metric(
    "ATS Win %",
    f"{ats_pct:.1%}" if decisions else "—",
)

c4.metric(
    "Avg Edge",
    (
        f"{filtered['abs_edge'].mean():.2f}"
        if len(filtered)
        else "—"
    ),
)

# ---------------------------------------------------------
# GAME TABLE
# ---------------------------------------------------------

st.subheader("Games")

if filtered.empty:

    st.info("No games match the selected filters.")

else:

    display = filtered.copy()

    display["Matchup"] = (
        display["away_team"]
        + " @ "
        + display["home_team"]
    )

    display["Model Line"] = display.apply(
        lambda row: format_line(
            row["home_team"],
            row["model_margin"],
        ),
        axis=1,
    )

    display["Market Line"] = display.apply(
        lambda row: format_line(
            row["home_team"],
            row["market_margin"],
        ),
        axis=1,
    )

    display["Edge"] = display["abs_edge"].round(1)

    display["Model Pick"] = display.apply(
        model_pick,
        axis=1,
    )

    display["Final Margin"] = display.apply(
        final_score,
        axis=1,
    )

    display["ATS"] = display["ats_result"]

    display = display.sort_values(
        ["week", "abs_edge"],
        ascending=[True, False],
    )

    table = display[
        [
            "week",
            "Matchup",
            "Model Line",
            "Market Line",
            "Edge",
            "Model Pick",
            "Final Margin",
            "ATS",
        ]
    ].rename(
        columns={
            "week": "Week",
        }
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )

# ---------------------------------------------------------
# ATS BY WEEK
# ---------------------------------------------------------

st.subheader("ATS Performance by Week")

chart_df = season_df[
    (season_df["abs_edge"] >= min_edge)
    & season_df["ats_result"].isin(["W", "L"])
].copy()

if chart_df.empty:

    st.info("No graded games available.")

else:

    weekly = (
        chart_df.groupby("week")["ats_result"]
        .agg(
            Games="count",
            Wins=lambda x: (x == "W").sum(),
        )
        .reset_index()
    )

    weekly["ATS Win %"] = (
        weekly["Wins"]
        / weekly["Games"]
        * 100
    )

    st.line_chart(
        weekly,
        x="week",
        y="ATS Win %",
    )

# ---------------------------------------------------------
# MODEL INFO
# ---------------------------------------------------------

with st.expander("About the model"):

    st.write(
        """
        **Base model**

        Predicted home margin:

        `1.744 + 0.516 × (Home Rating - Away Rating)`

        Ratings are opponent-adjusted EPA-based team strength.

        Early-season ratings blend the previous season's final
        rating with current-season performance using a fixed
        week-by-week decay schedule.

        This backtest is walk-forward: each prediction uses only
        information that would have been available before that game.

        Closing market lines are used only as a benchmark and are
        not inputs to the model.
        """
    )