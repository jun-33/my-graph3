import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 연도별 평균기온을 분석하고 회귀선을 이용해 "
    "연도별 예상 기온을 확인합니다."
)


# =========================================================
# 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# =========================================================
# 연도별 평균기온 계산
# =========================================================

# 2025년까지의 데이터만 사용
df = df[df["연도"] <= 2025].copy()

# 평균기온이 있는 데이터만 사용
df_valid = df.dropna(subset=["평균기온"]).copy()

# 연도별 평균기온 + 관측일수
yearly = (
    df_valid
    .groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일수가 300일 이상인 연도만 사용
yearly = yearly[
    yearly["관측일수"] >= 300
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# =========================================================
# 전체 기간 회귀분석
# =========================================================

# 1908년부터 지난 연수
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["평균기온"].to_numpy()

# 1차 선형 회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선 예상값
yearly["회귀예측기온"] = (
    slope * yearly["지난연수"] + intercept
)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 1년당 변화량 → 100년당 변화량
slope_100 = slope * 100


# =========================================================
# 최근 20년 회귀분석
# =========================================================

# 2006~2025년
recent = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()

# 최근 20년 데이터가 충분할 경우 계산
if len(recent) >= 2:

    recent_x = (
        recent["연도"] - 2006
    ).to_numpy()

    recent_y = recent["평균기온"].to_numpy()

    recent_slope, recent_intercept = np.polyfit(
        recent_x,
        recent_y,
        1
    )

    # 100년당 변화량
    recent_slope_100 = recent_slope * 100

else:
    recent_slope_100 = np.nan


# =========================================================
# 회귀선 정보
# =========================================================

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# =========================================================
# 100년당 기온 변화 크게 표시
# =========================================================

st.subheader("🌡️ 기온 변화 속도")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{slope_100:+.2f}℃ / 100년"
    )

with col2:
    if not np.isnan(recent_slope_100):
        st.metric(
            "최근 20년",
            f"{recent_slope_100:+.2f}℃ / 100년"
        )
    else:
        st.metric(
            "최근 20년",
            "계산할 수 없음"
        )


# =========================================================
# 전체 기간 vs 최근 20년 비교 설명
# =========================================================

st.write(
    f"전체 기간의 회귀선은 1년에 약 "
    f"**{slope:+.4f}℃**씩 변하는 것으로 계산되며, "
    f"이를 100년 기준으로 환산하면 "
    f"**{slope_100:+.2f}℃ / 100년**입니다."
)

if not np.isnan(recent_slope_100):
    st.write(
        f"최근 20년(2006~2025년)의 회귀선은 "
        f"100년 기준 **{recent_slope_100:+.2f}℃ / 100년**입니다."
    )


# =========================================================
# 전체 기간 회귀선 정보
# =========================================================

st.subheader("📊 분석에 사용된 데이터")

info1, info2, info3, info4 = st.columns(4)

with info1:
    st.metric(
        "회귀선에 사용된 연도 수",
        f"{data_count}개"
    )

with info2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with info3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )

with info4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )


# =========================================================
# 산점도 + 전체 회귀선 + 최근 20년 회귀선
# =========================================================

st.subheader("📈 서울 연평균기온과 회귀선 비교")

fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "연평균기온: %{y:.2f}℃<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)


# 전체 기간 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "전체 기간 회귀선: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 최근 20년 회귀선
if len(recent) >= 2:

    recent_line_x = np.array([
        recent["연도"].min(),
        recent["연도"].max()
    ])

    recent_line_y = (
        recent_slope *
        (recent_line_x - 2006)
        + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_x,
            y=recent_line_y,
            mode="lines",
            name="최근 20년 회귀선",
            line=dict(
                width=4,
                dash="dash"
            ),
            hovertemplate=(
                "<b>%{x}년</b><br>"
                "최근 20년 회귀선: %{y:.2f}℃"
                "<extra></extra>"
            )
        )
    )


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=600,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 연도 선택 슬라이더
# =========================================================

st.subheader("🔮 원하는 연도의 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 전체 기간 회귀식으로 예상
selected_x = selected_year - 1908

predicted_temp = (
    slope * selected_x + intercept
)


# =========================================================
# 예측 결과
# =========================================================

st.markdown("---")

st.subheader(f"🔮 {selected_year}년 예상 기온")

st.metric(
    label="예상 연평균기온",
    value=f"{predicted_temp:.2f}℃"
)
# =========================================================
# 선택한 연도의 회귀선 위치
# =========================================================

st.subheader("📍 선택한 연도의 회귀선 위치")

prediction_fig = go.Figure()

# 전체 회귀선
prediction_fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(
            width=3
        )
    )
)

# 선택한 연도 예상값
prediction_fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=14
        ),
        hovertemplate=(
            f"<b>{selected_year}년</b><br>"
            f"예상 연평균기온: {predicted_temp:.2f}℃"
            "<extra></extra>"
        )
    )
)

prediction_fig.update_layout(
    xaxis_title="연도",
    yaxis_title="예상 평균기온 (℃)",
    height=450,
    xaxis=dict(
        tickmode="linear",
        dtick=10
    )
)

st.plotly_chart(
    prediction_fig,
    use_container_width=True
)


# =========================================================
# 분석 기준
# =========================================================

st.caption(
    "분석 기준: 2025년까지의 데이터 중 연간 관측일수가 300일 이상인 연도만 사용했습니다. "
    "전체 회귀분석의 독립 변수는 1908년부터 지난 연수이며, "
    "최근 20년 분석은 2006~2025년 데이터를 사용했습니다."
)
