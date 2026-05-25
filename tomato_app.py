import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# 모델 로드 (tomato_model.pkl 로딩)
# ---------------------------------------------------------
@st.cache_resource
def load_tomato_model():
    import os
    model_path = 'tomato_model.pkl'
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception as e:
            st.error(f"🚨 모델 로딩 중 오류 발생: {e}")
            return None
    else:
        # 모델 파일이 없을 경우를 위한 가상 모델 (테스트용)
        class DummyModel:
            def predict(self, data):
                return [75.0 + data['내부온도'][0]*0.1 + data['내부습도'][0]*0.05 + data['지온'][0]*0.2]
        return DummyModel()

rf_model = load_tomato_model()


# ---------------------------------------------------------
# 1. 페이지 설정 및 사진 없는 오리지널 CSS 테마 정의
# ---------------------------------------------------------
st.set_page_config(page_title="깜찍한 스마트팜", page_icon="🍅", layout="wide")

# 크리스마스 방지용 토마토 전용 색상 조합
BG_TOMATO_CREAM = "#FFF5F5"   # 전체 배경: 따뜻하고 연한 토마토 크림색
CARD_WHITE = "#FFFFFF"        # 콘텐츠 상자: 깔끔한 화이트
TOMATO_RED = "#E74C3C"        # 메인 포인트: 잘 익은 토마토 레드
LEAF_GREEN = "#27AE60"        # 서브 포인트: 싱싱한 토마토 잎사귀 그린
DEEP_STEM_GREEN = "#1E5E3A"   # 메인 글씨: 아주 진한 그리니시 블랙 (가독성 확보)
LIGHT_GREEN_BORDER = "#C8E6C9" # 입력창 테두리: 연한 그린

css = f"""
<style>
    /* 전체 배경색 변경 */
    .stApp {{
        background-color: {BG_TOMATO_CREAM};
    }}

    /* 메인 콘텐츠 컨테이너 (중앙 카드박스) */
    .block-container {{
        background-color: {CARD_WHITE};
        padding: 3rem;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(231, 76, 60, 0.08); /* 토마토빛 은은한 그림자 */
        max-width: 950px;
        margin: auto;
    }}

    /* 메인 제목 글씨 색상 */
    h1 {{
        color: {TOMATO_RED} !important;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        font-size: 2.5rem;
        margin-bottom: 5px;
    }}
    
    /* 상단 서브 타이틀 설명 문구 */
    .subtitle-text {{
        color: {DEEP_STEM_GREEN};
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }}

    /* 소제목 (환경 데이터 입력) 글씨 색상 */
    h3 {{
        color: {LEAF_GREEN} !important;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        border-left: 5px solid {TOMATO_RED}; /* 왼쪽에 깜찍한 레드 포인트 바 */
        padding-left: 10px;
    }}

    /* 입력 필드 스타일링 (글씨색 및 테두리) */
    .stNumberInput label {{
        color: {DEEP_STEM_GREEN} !important; /* 입력창 위에 뜨는 글씨 색 */
        font-weight: bold;
    }}
    .stNumberInput div[data-baseweb="input"] {{
        border-radius: 12px;
        border: 2px solid {LIGHT_GREEN_BORDER} !important;
        background-color: #FAFAFA;
    }}
    
    /* 버튼 스타일 (토마토 레드) */
    .stButton button {{
        background-color: {TOMATO_RED} !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 14px 28px !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        border: none !important;
        width: 100%;
        box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
        transition: all 0.2s ease;
    }}
    .stButton button:hover {{
        background-color: #C0392B !important; /* 마우스 올렸을 때 더 진한 레드 */
        transform: translateY(-2px);
    }}

    /* 결과 메트릭 영역 디자인 */
    div[data-testid="stMetricValue"] {{
        color: {TOMATO_RED} !important; /* 결과 숫자 색상 */
        font-size: 3rem !important;
        font-weight: 900;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {LEAF_GREEN} !important; /* 결과 이름 색상 */
        font-size: 1.2rem !important;
        font-weight: bold;
    }}

    /* 성공 알림창 커스텀 */
    [data-testid="stNotificationAlert"] .stAlert {{
        background-color: #E8F5E9 !important; /* 연초록 배경 */
        border: 1px solid {LEAF_GREEN} !important;
        color: {DEEP_STEM_GREEN} !important;
        border-radius: 12px;
    }}
    
    /* 구분선 컬러 변경 */
    hr {{
        border: 0;
        height: 2px;
        background: linear-gradient(to right, rgba(39, 174, 96, 0.1), rgba(39, 174, 96, 0.6), rgba(39, 174, 96, 0.1));
    }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. UI 구성 및 기존 파이썬 로직 연결
# ---------------------------------------------------------
# 웹 페이지 제목 및 설명
st.title("🌱 방울토마토 착과율 예측기 🍅")
st.markdown('<p class="subtitle-text">내부 온도, 습도, 지온을 입력하여 예측 착과율을 확인하세요.</p>', unsafe_allow_html=True)

st.divider()

# 1. 사용자 입력 받기
st.subheader("📊 환경 데이터 입력")

col1, col2, col3 = st.columns(3)

with col1:
    temp = st.number_input("내부온도 (°C)", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)

with col2:
    humidity = st.number_input("내부습도 (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

with col3:
    soil_temp = st.number_input("지온 (°C)", min_value=-10.0, max_value=50.0, value=18.0, step=0.1)

# 2. DataFrame으로 변환
input_data = pd.DataFrame([[temp, humidity, soil_temp]], columns=['내부온도', '내부습도', '지온'])

st.divider()

# 3. 예측 및 결과 출력
if st.button("🔮 착과율 예측하기 🍅"):
    if rf_model:
        try:
            predicted = rf_model.predict(input_data)
            
            # 성공 메시지
            st.success("예측이 완료되었습니다! 싱싱한 토마토가 자라나는 중 👏")
            
            # 예측 결과를 강조된 대형 메트릭으로 출력
            st.metric(label="🍅 오늘의 예측 착과율", value=f"{predicted[0]:.1f}%")
            
        except Exception as e:
            st.error(f"🚨 예측 중 오류가 발생했습니다: {e}")
    else:
        st.error("🚨 모델이 로드되지 않았습니다.")