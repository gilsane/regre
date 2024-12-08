#파일 이름 streamlit_app.py
#세 개의 모델 결과값과 모델 선택 후 결과값 if에 따른 다른 출력결과 보여주기

import streamlit as st
import pickle
import gdown

# Google Drive 파일 ID
model_files = {
    "선형 회귀": "file_id_1",
    "랜덤 포레스트": "file_id_2",
    "인공 신경망": "file_id_3"
}

# 모델 로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = f'{file_id}.pkl'
    gdown.download(url, output, quiet=False)
    
    # 모델 로드
    with open(output, 'rb') as f:
        model = pickle.load(f)
    return model

# 선형 회귀 모델에서 독립변수 및 종속변수 정보 추출
@st.cache(allow_output_mutation=True)
def extract_model_metadata(model):
    if hasattr(model, 'dls'):  # Fastai 모델일 경우
        independent_vars = model.dls.cat_names + model.dls.cont_names
        categorical_vars = model.dls.cat_names
        dependent_var = model.dls.y_names[0]
    elif hasattr(model, 'independent_vars') and hasattr(model, 'dependent_var'):  # Scikit-learn 모델에 저장된 메타정보
        independent_vars = model.independent_vars
        categorical_vars = [var for var in independent_vars if var in model.categorical_vars]
        dependent_var = model.dependent_var
    else:
        raise ValueError("모델에서 독립변수 및 종속변수 정보를 추출할 수 없습니다.")
    return independent_vars, categorical_vars, dependent_var

# 모델 로드 및 메타정보 추출
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
models = {name: load_model_from_drive(file_id) for name, file_id in model_files.items()}
independent_vars, categorical_vars, dependent_var = extract_model_metadata(models["선형 회귀"])
st.success("모델이 성공적으로 로드되었습니다!")

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
        prediction = model.predict(input_data)[0]
        st.write(f"{model_name} 예측값: {prediction:.4f}")

    # 모델 선택 및 추가 정보 표시
    selected_model = st.selectbox("모델 선택", options=list(models.keys()))
    if st.button("추가 정보 표시"):
        prediction = models[selected_model].predict(input_data)[0]
        
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
