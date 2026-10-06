import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="서울 기온 선형회귀 평가",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 선형회귀 모델 평가")

st.write(
    "서울의 연평균기온을 이용하여 선형회귀 모델을 만들고, "
    "과거 데이터를 학습한 모델이 최근 20년의 기온을 얼마나 잘 예측하는지 평가합니다."
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
# 2025년까지의 데이터만 사용
# =========================================================

df = df[
    df["연도"] <= 2025
].copy()


# 평균기온이 존재하는 데이터만 사용
df_valid = df.dropna(
    subset=["평균기온"]
).copy()


# =========================================================
# 연도별 평균기온 계산
# =========================================================

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


yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# =========================================================
# 선형회귀 함수
# =========================================================

def make_regression(data):
    """
    연도를 독립변수,
    연평균기온을 종속변수로 하는
    1차 선형회귀를 계산한다.
    """

    x = data["연도"].to_numpy()
    y = data["평균기온"].to_numpy()

    # y = slope * x + intercept
    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    # 예측값
    prediction = slope * x + intercept

    return slope, intercept, prediction


# =========================================================
# 평가 함수
# =========================================================

def evaluate_model(actual, predicted):

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # MAE
    mae = np.mean(
        np.abs(actual - predicted)
    )

    # MSE
    mse = np.mean(
        (actual - predicted) ** 2
    )

    # R²
    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - (ss_res / ss_tot)

    return mae, mse, r2


# =========================================================
# 데이터 구분
# =========================================================

# 전체 데이터
all_data = yearly.copy()


# 최근 50년 훈련 데이터
train_50 = yearly[
    (yearly["연도"] >= 1956) &
    (yearly["연도"] <= 2005)
].copy()


# 최근 100년 훈련 데이터
train_100 = yearly[
    (yearly["연도"] >= 1906) &
    (yearly["연도"] <= 2005)
].copy()


# 공통 테스트 데이터
test = yearly[
    (yearly["연도"] >= 2006) &
    (yearly["연도"] <= 2025)
].copy()


# =========================================================
# 회귀모델 만들기
# =========================================================

# 전체 기간
all_slope, all_intercept, all_prediction = make_regression(
    all_data
)


# 최근 50년
slope_50, intercept_50, prediction_50_train = make_regression(
    train_50
)


# 최근 100년
slope_100, intercept_100, prediction_100_train = make_regression(
    train_100
)


# =========================================================
# 전체 데이터에 대한 평가
# =========================================================

all_mae, all_mse, all_r2 = evaluate_model(
    all_data["평균기온"],
    all_prediction
)


# =========================================================
# 테스트 데이터 예측
# =========================================================

test_years = test["연도"].to_numpy()
test_actual = test["평균기온"].to_numpy()


# 50년 모델의 테스트 예측
test_prediction_50 = (
    slope_50 * test_years
    + intercept_50
)


# 100년 모델의 테스트 예측
test_prediction_100 = (
    slope_100 * test_years
    + intercept_100
)


# =========================================================
# 테스트 성능 평가
# =========================================================

mae_50, mse_50, r2_50 = evaluate_model(
    test_actual,
    test_prediction_50
)


mae_100, mse_100, r2_100 = evaluate_model(
    test_actual,
    test_prediction_100
)


# =========================================================
# 100년에 따른 기온 변화량
# =========================================================

slope_50_100 = slope_50 * 100
slope_100_100 = slope_100 * 100
all_slope_100 = all_slope * 100


# =========================================================
# 데이터 개수
# =========================================================

all_count = len(all_data)
train_50_count = len(train_50)
train_100_count = len(train_100)
test_count = len(test)


# =========================================================
# 분석 기간 표시
# =========================================================

st.subheader("📅 데이터 구분")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 훈련",
        f"1956~2005년"
    )
    st.caption(
        f"실제 사용 연도: {train_50_count}개"
    )

with col2:
    st.metric(
        "최근 100년 훈련",
        f"1906~2005년"
    )
    st.caption(
        f"실제 사용 연도: {train_100_count}개"
    )

with col3:
    st.metric(
        "공통 테스트",
        f"2006~2025년"
    )
    st.caption(
        f"실제 사용 연도: {test_count}개"
    )


st.info(
    "모든 연도는 연간 관측일수가 300일 이상인 경우만 사용했습니다. "
    "따라서 실제 데이터가 없는 1906년은 자동으로 제외됩니다."
)


# =========================================================
# 전체 데이터 회귀선 평가
# =========================================================

st.subheader("📊 전체 데이터로 만든 회귀선")

st.write(
    "전체 기간의 연평균기온을 모두 이용하여 회귀선을 만들고, "
    "그 회귀선이 같은 전체 데이터를 얼마나 잘 설명하는지 평가합니다."
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "기울기",
        f"{all_slope_100:+.2f}℃ / 100년"
    )

with col2:
    st.metric(
        "MAE",
        f"{all_mae:.3f}℃"
    )

with col3:
    st.metric(
        "MSE",
        f"{all_mse:.3f}"
    )

with col4:
    st.metric(
        "R²",
        f"{all_r2:.3f}"
    )


# =========================================================
# 50년 vs 100년 기울기 비교
# =========================================================

st.subheader("📈 최근 50년 vs 최근 100년 기울기 비교")

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 🟦 최근 50년 모델")

    st.metric(
        "100년에 따른 기온 변화",
        f"{slope_50_100:+.2f}℃"
    )

    st.write(
        f"훈련 기간: 1956~2005년"
    )

    st.write(
        f"1년당 기울기: {slope_50:+.4f}℃"
    )


with col2:

    st.markdown("### 🟥 최근 100년 모델")

    st.metric(
        "100년에 따른 기온 변화",
        f"{slope_100_100:+.2f}℃"
    )

    st.write(
        f"훈련 기간: 1906~2005년"
    )

    st.write(
        f"1년당 기울기: {slope_100:+.4f}℃"
    )


# =========================================================
# 테스트 데이터 성능 비교
# =========================================================

st.subheader("🎯 최근 20년 테스트 성능 비교")

st.write(
    "1956~2005년 또는 1906~2005년으로 회귀선을 학습한 뒤, "
    "학습에 사용하지 않은 2006~2025년을 예측했습니다."
)


comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습",
        "최근 100년 학습"
    ],
    "훈련기간": [
        "1956~2005",
        "1906~2005"
    ],
    "기울기(℃/100년)": [
        slope_50_100,
        slope_100_100
    ],
    "MAE(℃)": [
        mae_50,
        mae_100
    ],
    "MSE": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})


st.dataframe(
    comparison.style.format({
        "기울기(℃/100년)": "{:+.2f}",
        "MAE(℃)": "{:.3f}",
        "MSE": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 어떤 모델이 더 좋은지
# =========================================================

st.subheader("🏆 테스트 성능 비교")

if mae_50 < mae_100:
    mae_result = "MAE는 최근 50년 모델이 더 낮아 더 좋습니다."
elif mae_50 > mae_100:
    mae_result = "MAE는 최근 100년 모델이 더 낮아 더 좋습니다."
else:
    mae_result = "MAE는 두 모델이 같습니다."


if mse_50 < mse_100:
    mse_result = "MSE는 최근 50년 모델이 더 낮아 더 좋습니다."
elif mse_50 > mse_100:
    mse_result = "MSE는 최근 100년 모델이 더 낮아 더 좋습니다."
else:
    mse_result = "MSE는 두 모델이 같습니다."


if r2_50 > r2_100:
    r2_result = "R²는 최근 50년 모델이 더 높아 더 좋습니다."
elif r2_50 < r2_100:
    r2_result = "R²는 최근 100년 모델이 더 높아 더 좋습니다."
else:
    r2_result = "R²는 두 모델이 같습니다."


st.write(f"• **MAE:** {mae_result}")
st.write(f"• **MSE:** {mse_result}")
st.write(f"• **R²:** {r2_result}")


# =========================================================
# 테스트 데이터 실제값 vs 예측값
# =========================================================

st.subheader("📉 2006~2025년 실제 기온과 예측값")

fig = go.Figure()


# 실제값
fig.add_trace(
    go.Scatter(
        x=test_years,
        y=test_actual,
        mode="markers+lines",
        name="실제 연평균기온",
        marker=dict(
            size=8
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "실제 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 모델 예측
fig.add_trace(
    go.Scatter(
        x=test_years,
        y=test_prediction_50,
        mode="lines",
        name="최근 50년 학습 모델",
        line=dict(
            width=3,
            dash="dash"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "50년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 100년 모델 예측
fig.add_trace(
    go.Scatter(
        x=test_years,
        y=test_prediction_100,
        mode="lines",
        name="최근 100년 학습 모델",
        line=dict(
            width=3,
            dash="dot"
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "100년 모델: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    xaxis=dict(
        tickmode="linear",
        dtick=2
    ),
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# 회귀선 비교
# =========================================================

st.subheader("📐 훈련 데이터와 회귀선 비교")

fig2 = go.Figure()


# 실제 전체 연평균기온
fig2.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=6
        ),
        hovertemplate=(
            "<b>%{x}년</b><br>"
            "평균기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    )
)


# 50년 회귀선
line_years_50 = np.array([
    1956,
    2025
])

line_temp_50 = (
    slope_50 * line_years_50
    + intercept_50
)


fig2.add_trace(
    go.Scatter(
        x=line_years_50,
        y=line_temp_50,
        mode="lines",
        name="최근 50년 회귀선",
        line=dict(
            width=4,
            dash="dash"
        )
    )
)


# 100년 회귀선
line_years_100 = np.array([
    1906,
    2025
])

line_temp_100 = (
    slope_100 * line_years_100
    + intercept_100
)


fig2.add_trace(
    go.Scatter(
        x=line_years_100,
        y=line_temp_100,
        mode="lines",
        name="최근 100년 회귀선",
        line=dict(
            width=4,
            dash="dot"
        )
    )
)


fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    height=600,
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified"
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================================
# 최종 설명
# =========================================================

st.subheader("📝 해석하기")

st.write(
    "MAE는 실제 기온과 예측 기온의 차이를 절댓값으로 평균낸 값으로, "
    "작을수록 예측이 좋습니다."
)

st.write(
    "MSE는 예측 오차를 제곱하여 평균낸 값으로, "
    "작을수록 좋습니다. 큰 오차에 더 민감합니다."
)

st.write(
    "R²는 회귀모델이 실제 기온의 변동을 얼마나 설명하는지를 나타내며, "
    "일반적으로 1에 가까울수록 좋습니다. 테스트 데이터에서는 음수가 나올 수도 있습니다."
)

st.write(
    "따라서 최근 50년 모델과 최근 100년 모델 중 어떤 것이 최근 20년을 "
    "더 잘 예측했는지는 MAE와 MSE가 더 작은지, R²가 더 큰지를 비교하면 됩니다."
)


# =========================================================
# 분석 기준
# =========================================================

st.caption(
    "분석 기준: 2025년까지의 데이터 중 연간 관측일수가 300일 이상인 연도만 사용. "
    "훈련 데이터는 1956~2005년(최근 50년), 1906~2005년(최근 100년), "
    "테스트 데이터는 두 모델 모두 2006~2025년으로 동일하게 사용."
)
