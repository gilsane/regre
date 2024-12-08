import streamlit as st
import pickle
from fastai.tabular.all import load_learner
import gdown

# Google Drive 파일 ID 설정
model_files = {
    "선형 회귀": {"file_id": "1-N1PEizrUY1knoryezvrVoWKoN--xl0j", "type": "fastai"},
    "랜덤 포레스트": {"file_id": "1-CUb9h3fdoIfHCIE0pX0B1J1qjUDgkMM", "type": "sklearn"},
    "인공 신경망": {"file_id": "1-E3o2qvk0j0jtbTdUvd1OhomBovRVvK3", "type": "fastai"}
}

# 모델 로드 함수
#@st.cache(allow_output_mutation=True)
def load_model(file_id, model_type):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"{file_id}.pkl"
    gdown.download(url, output, quiet=False)

    if model_type == "fastai":
        try:
            model = load_learner(output)
        except Exception as e:
            raise ValueError(f"Fastai 모델 로드 실패: {e}")
    elif model_type == "sklearn":
        try:
            with open(output, 'rb') as f:
                model_metadata = pickle.load(f)
            model = model_metadata
        except Exception as e:
            raise ValueError(f"Scikit-learn 모델 로드 실패: {e}")
    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")

    return model

# 모델 로드 및 출력
st.write("모델을 Google Drive에서 로드 중입니다. 잠시만 기다려주세요...")
loaded_models = {}
for name, info in model_files.items():
    st.write(f"{name} 로드 중...")
    try:
        loaded_models[name] = load_model(info["file_id"], info["type"])
        st.success(f"{name} 로드 성공!")
    except Exception as e:
        st.error(f"{name} 로드 실패: {e}")

# 로드된 모델 출력
st.write("### 로드된 모델 정보:")
for name, model in loaded_models.items():
    st.write(f"#### {name}")
    if isinstance(model, dict):
        st.write("Scikit-learn 모델 메타데이터:")
        st.write("독립변수:", model.get("independent_vars"))
        st.write("종속변수:", model.get("dependent_var"))
        st.write("모델 객체 타입:", type(model.get("model")))
    else:
        st.write("Fastai 모델:")
        st.write("모델 구조:", model.model)
        st.write("독립변수 (범주형):", model.dls.cat_names)
        st.write("독립변수 (수치형):", model.dls.cont_names)
        st.write("종속변수:", model.dls.y_names)
