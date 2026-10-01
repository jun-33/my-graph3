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
    "서울의 연도별 평균기온 데이터를 이용해 회귀선을 만들고 "
    "선택한 연도의 예상 기온을 확인합니다."
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
        encoding="utf-8-sig"
    )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# =========================================================
# 연도별 데이터 만들기
# =========================================================

# 2025년까지의 데이터만 사용
df = df[df["연도"] <= 2025].copy()

# 평균기온이 실제로 존재하는 데이터만 사용
df_valid = df.dropna(subset=["평균기온"]).copy()

# 연도별 관측일 수와 평균기온 계산
yearly = (
    df_valid
    .groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 관측일이 300일 이상인 연도만 사용
yearly = yearly[yearly["관측일수"] >= 300].copy()

# 연도순 정렬
yearly = yearly.sort_values("연도").reset_index(drop=True)


# =========================================================
# 회귀분석
# =========================================================

# 독립 변수:
# 1908년부터 지난 연수
# 예) 1908년 -> 0
#     1909년 -> 1
#     2008년 -> 100
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["평균기온"].to_numpy()

# 1차 선형 회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선에서 예측한 값
yearly["회귀예측기온"] = slope * yearly["지난연수"] + intercept

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# =========================================================
# 회귀선 정보
# =========================================================

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
data_count = len(yearly)


# =========================================================
# 화면에 회귀분석 정보 표시
# =========================================================

st.subheader("📊 회귀분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀선에 사용된 연도 수",
        f"{data_count}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )


st.write(
    f"회귀식: **예상 기온 = {slope:.4f} × (연도 - 1908) "
    f"+ {intercept:.2f}**"
)


# =========================================================
# 산점도 + 회귀선
# =========================================================

st.subheader("📈 서울 연평균기온과 회귀선")

fig = go.Figure()

# 실제 연평균기온 산점도
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

# 회귀선
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "회귀선 예상기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (℃)",
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

# 선택한 연도의 회귀선 예상기온 계산
selected_x = selected_year - 1908
predicted_temp = slope * selected_x + intercept


# =========================================================
# 예측 결과
# =========================================================

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:30px;
        border-radius:15px;
        background-color:#f0f2f6;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <div style="font-size:24px; font-weight:bold;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="
            font-size:52px;
            font-weight:bold;
            margin-top:10px;
        ">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 선택한 연도의 위치를 회귀선 위에 표시
# =========================================================

st.subheader("📍 선택한 연도의 회귀선 위치")

prediction_fig = go.Figure()

# 회귀선
prediction_fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["회귀예측기온"],
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        )
    )
)

# 선택한 연도
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
# 데이터 설명
# =========================================================

st.caption(
    "분석 기준: 2025년까지의 데이터 중 연간 관측일수가 300일 이상인 연도만 사용했습니다. "
    "회귀의 독립 변수는 1908년을 0으로 하는 '1908년부터 지난 연수'입니다."
)
