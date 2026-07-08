import os
os.environ.setdefault('PYCOX_DATA_DIR', '/tmp/pycox_data')  # 클라우드 환경에서 pycox가 site-packages 안에 쓰기 시도하다 PermissionError 나는 것 방지

import os
os.environ.setdefault('PYCOX_DATA_DIR', '/tmp/pycox_data')  # 클라우드 환경에서 pycox가 site-packages 안에 쓰기 시도하다 PermissionError 나는 것 방지

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchtuples as tt
import pickle
from lifelines import CoxPHFitter
from pycox.models import DeepHitSingle
from supabase import create_client

st.set_page_config(page_title="결측치·중도절단 자동 처리 시스템", layout="wide")

IS_LOGGED_IN = st.user.is_logged_in

# ── Supabase 클라이언트 (로그인 여부와 무관하게 항상 준비) ──
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

supabase = get_supabase()

if "local_history" not in st.session_state:
    st.session_state.local_history = []  # 로그인 안 한 경우, 이 세션 동안만 유지되는 임시 기록

# ── 사이드바: 로그인 상태 + 최근 분석 기록 ──
with st.sidebar:
    if IS_LOGGED_IN:
        st.write(f"👤 {st.user.name}")
        st.caption(st.user.email)
        st.button("로그아웃", on_click=st.logout)
    else:
        st.info("로그인하면 분석 기록이 영구 저장됩니다.")
        st.button("Google로 로그인", on_click=st.login)

    st.divider()
    st.subheader("내 최근 분석 기록")

    if IS_LOGGED_IN:
        try:
            history = (
                supabase.table("analysis_history")
                .select("*")
                .eq("user_email", st.user.email)
                .order("created_at", desc=True)
                .limit(20)
                .execute()
            )
            if history.data:
                st.dataframe(pd.DataFrame(history.data), use_container_width=True, hide_index=True)
            else:
                st.caption("아직 분석 기록이 없습니다.")
        except Exception as e:
            st.caption(f"기록 조회 실패: {e}")
    else:
        if st.session_state.local_history:
            st.dataframe(pd.DataFrame(st.session_state.local_history), use_container_width=True, hide_index=True)
            st.caption("⚠ 로그인하지 않아 새로고침하면 이 기록은 사라집니다.")
        else:
            st.caption("아직 분석 기록이 없습니다.")

st.title("결측치·중도절단 자동 처리 시스템 (프로토타입)")
st.caption("METABRIC 임상변수 16개 스키마 전용 · Engine1(신경망)/Engine2(Cox) 자동 라우팅")

# ── 저장된 메타데이터/모델 로드 ──
@st.cache_resource
def load_metadata():
    with open('app_metadata.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_models():
    with open('phase2_preprocessors.pkl', 'rb') as f:
        prep = pickle.load(f)
    meta = load_metadata()
    n_features = len(meta['CORE_COLS'])

    class MaskedAutoencoder(nn.Module):
        def __init__(self, in_features, latent_dim=8):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(in_features, 32), nn.ReLU(), nn.Linear(32, latent_dim), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, in_features))
        def forward(self, x):
            z = self.encoder(x)
            return z, self.decoder(z)

    ae_model = MaskedAutoencoder(n_features, 8)
    ae_model.load_state_dict(torch.load('ae_model.pt', map_location='cpu'))
    ae_model.eval()

    deephit_models = {}
    for cause in [1, 2]:
        with open(f'deephit_cause{cause}_duration_index.pkl', 'rb') as f:
            duration_index = pickle.load(f)
        # 주의: output_bias=False 를 반드시 학습 때와 동일하게 지정해야 state_dict가 일치함
        net = tt.practical.MLPVanilla(8, [32, 32], len(duration_index), batch_norm=False,
                                       dropout=0.2, output_bias=False)
        net.load_state_dict(torch.load(f'deephit_cause{cause}_net.pt', map_location='cpu'))
        deephit_models[cause] = DeepHitSingle(net, duration_index=duration_index)

    return ae_model, deephit_models, prep

meta = load_metadata()
CORE_COLS = meta['CORE_COLS']
CORE_CATEGORICAL = meta['CORE_CATEGORICAL']
DUR, EVT = meta['DUR'], meta['EVT']
encoders = meta['encoders']

uploaded = st.file_uploader("임상 데이터 CSV 업로드", type='csv')

if uploaded is not None:
    user_df = pd.read_csv(uploaded)

    # ── P3-5 간이 스키마 체크 ──
    missing_cols = [c for c in CORE_COLS + [DUR, EVT] if c not in user_df.columns]
    if missing_cols:
        st.error(f"다음 컬럼이 없어 이 데이터는 처리할 수 없습니다: {missing_cols}\n"
                 f"(이 시스템은 METABRIC 임상변수 16개 스키마 전용입니다)")
        st.stop()

    st.success(f"스키마 확인 완료 — {len(user_df)}명 데이터 로드됨")

    # ── 범주형 인코딩 (학습 시 매핑 그대로 재사용) ──
    encoded_df = user_df.copy()
    unseen_flag = False
    for col in CORE_CATEGORICAL:
        encoded_df[col] = encoded_df[col].map(encoders[col])
        if encoded_df[col].isna().any():
            unseen_flag = True
    if unseen_flag:
        st.warning("일부 범주형 값이 학습 데이터에 없던 값입니다 — 해당 값은 결측으로 처리됩니다.")

    purpose = st.radio("분석 목적을 선택하세요", ["개별 위험도 예측", "치료 효과 유의성 비교"])

    if purpose == "개별 위험도 예측":
        st.subheader("Engine1 (통합 신경망) — 개별 위험도 예측")
        time_horizon = st.select_slider("예측 시점(개월)", options=[12, 36, 60, 120], value=60)

        ae_model, deephit_models, prep = load_models()
        fill_stats, ae_scaler, latent_scaler = prep['ae_fillstats'], prep['ae_scaler'], prep['latent_scaler']

        X_filled = encoded_df[CORE_COLS].fillna(fill_stats).values.astype('float32')
        X_scaled = ae_scaler.transform(X_filled).astype('float32')
        with torch.no_grad():
            z, _ = ae_model(torch.tensor(X_scaled))
        z_scaled = latent_scaler.transform(z.numpy()).astype('float32')

        result_df = user_df.copy()
        for cause, cause_name in [(1, '질병사망_위험도'), (2, '타원인사망_위험도')]:
            surv = deephit_models[cause].predict_surv_df(z_scaled)
            risk = 1 - surv.reindex(surv.index.union([time_horizon])).sort_index()\
                          .interpolate(method='index').loc[time_horizon]
            result_df[cause_name] = risk.values

        st.dataframe(result_df[[c for c in result_df.columns if '위험도' in c or c in CORE_COLS[:3]]])

        avg_risk = result_df['질병사망_위험도'].mean()
        if avg_risk < 0.2:
            interp = "양호 — 평균적으로 낮은 위험도군입니다."
        elif avg_risk < 0.4:
            interp = "중간 — 주의 관찰이 필요한 수준입니다."
        else:
            interp = "높음 — 위험도가 높은 환자 비중이 큽니다."
        st.info(f"[해석] {time_horizon}개월 시점 평균 질병사망 위험도 {avg_risk:.1%} → {interp}")

        record = {
            "user_email": st.user.email if IS_LOGGED_IN else "(로그인 안 함)",
            "filename": uploaded.name,
            "purpose": "예측",
            "n_patients": len(user_df),
            "avg_risk": float(avg_risk),
        }
        if IS_LOGGED_IN:
            try:
                supabase.table("analysis_history").insert(record).execute()
            except Exception as e:
                st.caption(f"(기록 저장 실패: {e})")
        else:
            st.session_state.local_history.insert(0, record)

    else:
        st.subheader("Engine2 (Cox 회귀) — 치료 효과 유의성 비교")
        compare_col = st.selectbox("비교할 치료 변수", meta['CORE_BINARY'])

        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(encoded_df[CORE_COLS + [DUR, EVT]], duration_col=DUR, event_col=EVT)

        row = cph.summary.loc[compare_col]
        hr, p = row['exp(coef)'], row['p']
        st.metric(f"{compare_col} 위험비(HR)", f"{hr:.3f}", f"p = {p:.4f}")

        if p < 0.05:
            direction = "위험을 유의하게 낮춥니다" if hr < 1 else "위험을 유의하게 높입니다"
            interp = f"{compare_col}는 통계적으로 유의미하게 {direction} (p<0.05)."
        else:
            interp = f"{compare_col}의 효과는 통계적으로 유의하지 않습니다 (p≥0.05) — 표본 크기나 데이터 특성상 추가 검증이 필요합니다."
        st.info(f"[해석] {interp}")

        st.write("전체 Cox 회귀 결과:")
        st.dataframe(cph.summary[['coef', 'exp(coef)', 'p']])

        record = {
            "user_email": st.user.email if IS_LOGGED_IN else "(로그인 안 함)",
            "filename": uploaded.name,
            "purpose": "효과비교",
            "n_patients": len(user_df),
            "hr_variable": compare_col,
            "hr_value": float(hr),
            "p_value": float(p),
        }
        if IS_LOGGED_IN:
            try:
                supabase.table("analysis_history").insert(record).execute()
            except Exception as e:
                st.caption(f"(기록 저장 실패: {e})")
        else:
            st.session_state.local_history.insert(0, record)
else:
    st.info("CSV 파일을 업로드하면 분석이 시작됩니다.")
