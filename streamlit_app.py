import streamlit as st
import pickle
import gdown
from fastai.tabular.all import load_learner

# Google Drive 파일 ID
model_files = {
    "선형 회귀": {"file_id": "1-N1PEizrUY1knoryezvrVoWKoN--xl0j", "type": "fastai"},
    "랜덤 포레스트": {"file_id": "1-E3o2qvk0j0jtbTdUvd1OhomBovRVvK3", "type": "sklearn"},
    "인공 신경망": {"file_id": "1-CUb9h3fdoIfHCIE0pX0B1J1qjUDgkMM", "type": "fastai"}
}

# 모델 로드 함수
@st.cache(allow_output_mutation=True)
def load_model(file_id, model_type):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = f'{file_id}.pkl'
    gdown.download(url, output, quiet=False)
    
    if model_type == "fastai":
        # Fastai 모델 로드
        model = load_learner(output)
    elif model_type == "sklearn":
        # Scikit-learn 모델 로드
        with open(output, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")
    return model

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
models = {
    name: load_model(info["file_id"], info["type"])
    for name, info in model_files.items()
}
st.success("모델이 성공적으로 로드되었습니다!")

# 모델 정보 (Fastai 모델 기준)
independent_vars = ["변수1", "변수2", "변수3"]  # 예시로 입력값을 사용
categorical_vars = ["변수2"]  # 예시로 범주형 변수를 설정
numeric_vars = [var for var in independent_vars if var not in categorical_vars]

# 사용자 입력 UI 생성
st.sidebar.write("### 입력값 설정")
inputs = {}
for var in independent_vars:
    if var in categorical_vars:
        inputs[var] = st.sidebar.selectbox(f"{var} (범주형 선택)", options=["옵션1", "옵션2", "옵션3"])
    else:
        inputs[var] = st.sidebar.number_input(f"{var} (숫자형 입력)", value=0.0, step=0.1)

# 예측 버튼
if st.sidebar.button("예측 실행"):
    input_data = pd.DataFrame([inputs])  # 입력값을 데이터프레임으로 변환
    st.write("### 예측 결과")
    for model_name, model in models.items():
        if isinstance(model, load_learner):
            # Fastai 모델 예측
            prediction = model.predict(input_data)[0]
        else:
            # Scikit-learn 모델 예측
            prediction = model.predict(input_data)[0]
        st.write(f"{model_name} 예측값: {prediction:.4f}")

    # 모델 선택 및 추가 정보 표시
    selected_model = st.selectbox("모델 선택", options=list(models.keys()))
    if st.button("추가 정보 표시"):
        model = models[selected_model]
        if isinstance(model, load_learner):
            prediction = model.predict(input_data)[0]
        else:
            prediction = model.predict(input_data)[0]
        
        # 추가 정보 조건 (예시)
        if prediction > 50:
            st.write("### 결과: 예측값이 50을 초과합니다.")
            st.image("https://via.placeholder.com/300?text=High+Prediction")
            st.video("https://www.youtube.com/watch?v=3JZ_D3ELwOQ")
            st.write("관련 텍스트: 높은 예측값에 대한 설명.")
        else:
            st.write("### 결과: 예측값이 50 이하입니다.")
            st.image("https://via.placeholder.com/300?text=Low+Prediction")
            st.video("https://www.youtube.com/watch?v=2Vv-BfVoq4g")
            st.write("관련 텍스트: 낮은 예측값에 대한 설명.")
