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

st.set_page_config(page_title="SURVFLOW", layout="wide")

IS_LOGGED_IN = st.user.is_logged_in

# ── Supabase 클라이언트 (로그인 여부와 무관하게 항상 준비) ──
@st.cache_resource
def get_supabase():
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])

supabase = get_supabase()

@st.cache_resource
def load_preprocessors():
    with open('phase2_preprocessors.pkl', 'rb') as f:
        return pickle.load(f)

if "local_history" not in st.session_state:
    st.session_state.local_history = []  # 로그인 안 한 경우, 이 세션 동안만 유지되는 임시 기록

def go_home():
    """업로드 파일까지 포함해 완전히 초기 상태로 되돌림 (파일 업로더도 새 위젯으로 교체)"""
    st.session_state['uploader_key_version'] = st.session_state.get('uploader_key_version', 0) + 1
    for k in ['confirmed_file_id', 'uploaded_file_id', 'uploaded_bytes', 'uploaded_name', 'last_result']:
        st.session_state.pop(k, None)

# ── 사이드바 여백 미세조정용 CSS (key 기반 정밀 타겟팅) ──
st.html("""
<style>
section[data-testid="stSidebar"] .stMainBlockContainer {
    padding-top: 1.2rem !important;
}
[class*="st-key-histrow_"] div[data-testid="stHorizontalBlock"] {
    gap: 0.2rem !important;
}
[class*="st-key-histrow_"] [data-testid="stColumn"]:first-of-type {
    padding-left: 0 !important;
}
[class*="st-key-histrow_"] .stButton button {
    padding-left: 0 !important;
    justify-content: flex-start !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
}
[class*="st-key-new_analysis_wrap"], [class*="st-key-search_wrap"] {
    margin-top: -0.6rem !important;
    margin-bottom: -1.1rem !important;
}
/* 팝업(⋮ 메뉴) 자체의 여백을 큰 폭으로 축소 */
div[data-testid="stPopoverBody"] {
    padding: 0.3rem !important;
    min-width: 0 !important;
    width: 160px !important;
}
div[data-testid="stPopoverBody"] .stButton {
    margin: 0 !important;
}
div[data-testid="stPopoverBody"] .stButton button {
    padding: 0.2rem 0.4rem !important;
    margin: 0 !important;
    min-height: 0 !important;
}
div[data-testid="stPopoverBody"] div[data-testid="stVerticalBlock"] {
    gap: 0.1rem !important;
}
</style>
""")

# ── 이름 변경 모달창 ──
@st.dialog("이름 변경")
def rename_dialog(rid, current_title):
    new_title = st.text_input("새 이름", value=current_title, key=f"dialog_rename_{rid}",
                                label_visibility="collapsed")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("저장", key=f"dialog_save_{rid}", use_container_width=True, type="primary"):
            supabase.table("analysis_history").update({"title": new_title}).eq("id", rid).execute()
            st.session_state['history_ui_version'] = st.session_state.get('history_ui_version', 0) + 1
            st.rerun()

# ── 사이드바: 로고+이름, 새 분석/검색, 로그인 정보, 최근 분석 기록 ──
with st.sidebar:
    with st.container(key="logo_row"):
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; gap:0.3rem; margin-bottom:0.1rem;">
            <svg width="26" height="26" viewBox="0 0 170 170" style="flex-shrink:0;">
                <path fill="#CC785C" d="M 98.0 24.5 L 130.9 43.5 Q 143.9 51.0 143.9 66.0 L 143.9 104.0 Q 143.9 119.0 130.9 126.5 L 98.0 145.5 Q 85.0 153.0 72.0 145.5 L 39.1 126.5 Q 26.1 119.0 26.1 104.0 L 26.1 66.0 Q 26.1 51.0 39.1 43.5 L 72.0 24.5 Q 85.0 17.0 98.0 24.5 Z"/>
                <g transform="translate(38,60)" fill="none" stroke-linecap="round">
                  <path d="M 0 50 Q 35.0 4.5 74.3 25.07" stroke="#FAF9F5" stroke-width="7"/>
                  <path d="M 74.3 25.07 Q 85.0 30.5 100 50" stroke="#FAF9F5" stroke-width="6" stroke-dasharray="2.2 10.51" stroke-dashoffset="2.2"/>
                </g>
            </svg>
            <div style="font-family:sans-serif; font-weight:700; font-size:1.2rem; line-height:1;">
                <span style="color:#CC785C;">surv</span><span style="color:#9B9B9B;">flow</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with st.container(key="new_analysis_wrap"):
        if st.button("새 분석", key="new_analysis_btn", icon=":material/add:",
                      use_container_width=True, type="tertiary"):
            go_home()
            st.rerun()

    with st.container(key="search_wrap"):
        show_search = st.session_state.get("show_history_search", False)
        if st.button("분석 검색", key="toggle_search_btn", icon=":material/search:",
                      use_container_width=True, type="tertiary"):
            st.session_state["show_history_search"] = not show_search
            st.rerun()
    search_query = ""
    if st.session_state.get("show_history_search", False):
        search_query = st.text_input("분석 검색", key="history_search_input",
                                      placeholder="파일명으로 검색...", label_visibility="collapsed")

    st.markdown("<hr style='border:none; border-top:1px solid #E7E3D8; margin:0.3rem 0;'>", unsafe_allow_html=True)

    if IS_LOGGED_IN:
        st.markdown(
            f"<div style='font-size:1.3rem; font-weight:700; color:#3D3929; margin-bottom:0.1rem;'>{st.user.name}</div>",
            unsafe_allow_html=True,
        )
        st.html(
            f'<div style="display:flex; align-items:baseline; gap:0.35rem; margin-bottom:0.1rem;">'
            f'<span style="position:relative; top:2px; font-size:0.95rem;">&#9993;</span>'
            f'<span>{st.user.email}</span></div>'
        )
        st.button("로그아웃", on_click=st.logout)
    else:
        st.info("로그인하면 분석 기록이 영구 저장됩니다.")
        st.button("Google로 로그인", on_click=st.login)

    st.markdown("<hr style='border:none; border-top:1px solid #E7E3D8; margin:0.7rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.1rem; font-weight:700; text-align:left;'>최근 분석 기록</div>",
                unsafe_allow_html=True)

    def render_history_item(record, editable_db=False):
        rid = record.get('id')
        display_title = record.get('title') or record.get('filename') or '(제목 없음)'

        row = st.container(key=f"histrow_{rid}")
        col_title, col_menu = row.columns([5, 1])
        with col_title:
            if st.button(display_title, key=f"open_{rid}", use_container_width=True, type="tertiary"):
                st.session_state[f"show_{rid}"] = not st.session_state.get(f"show_{rid}", False)
        with col_menu:
            if editable_db:
                ui_v = st.session_state.get('history_ui_version', 0)
                with st.popover("⋮", type="tertiary", key=f"popover_{rid}_{ui_v}"):
                    if st.session_state.get(f"confirming_delete_{rid}", False):
                        st.write("정말 삭제하시겠어요?")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("예, 삭제", key=f"confirmyes_{rid}", type="tertiary"):
                                supabase.table("analysis_history").delete().eq("id", rid).execute()
                                st.session_state['history_ui_version'] = ui_v + 1
                                st.session_state['confirmed_file_id'] = None  # 진행중이던 분석 프롬프트도 리셋
                                st.session_state.pop('last_result', None)
                                st.rerun()
                        with col_no:
                            if st.button("아니오", key=f"confirmno_{rid}", type="tertiary"):
                                st.session_state[f"confirming_delete_{rid}"] = False
                                st.rerun()
                    else:
                        if st.button("✏️ 이름 변경", key=f"renamebtn_{rid}", use_container_width=True, type="tertiary"):
                            rename_dialog(rid, display_title)
                        if st.button("🗑 삭제", key=f"delbtn_{rid}", use_container_width=True, type="tertiary"):
                            st.session_state[f"confirming_delete_{rid}"] = True
                            st.rerun()

        if st.session_state.get(f"show_{rid}", False):
            st.write(f"**목적**: {record.get('purpose')}")
            st.write(f"**환자 수**: {record.get('n_patients')}")
            if record.get('purpose') == '예측':
                avg_risk = record.get('avg_risk')
                if avg_risk is not None:
                    st.write(f"**평균 질병사망 위험도**: {avg_risk:.1%}")
            else:
                st.write(f"**비교 변수**: {record.get('hr_variable')}")
                if record.get('hr_value') is not None:
                    st.write(f"**HR**: {record.get('hr_value'):.3f}, **p-value**: {record.get('p_value'):.4f}")
            st.divider()

    def matches_search(record):
        if not search_query:
            return True
        title = (record.get('title') or record.get('filename') or '')
        return search_query.lower() in title.lower()

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
            filtered = [r for r in history.data if matches_search(r)] if history.data else []
            if filtered:
                for record in filtered:
                    render_history_item(record, editable_db=True)
            elif search_query:
                st.caption("검색 결과가 없습니다.")
            else:
                st.caption("아직 분석 기록이 없습니다.")
        except Exception as e:
            st.caption(f"기록 조회 실패: {e}")
    else:
        filtered_local = [r for r in st.session_state.local_history if matches_search(r)]
        if filtered_local:
            for i, record in enumerate(filtered_local):
                record.setdefault('id', f'local_{i}')
                render_history_item(record, editable_db=False)
            st.caption("⚠ 로그인하지 않아 새로고침하면 이 기록은 사라집니다.")
        elif search_query:
            st.caption("검색 결과가 없습니다.")
        else:
            st.caption("아직 분석 기록이 없습니다.")

st.markdown("""
<div style="font-family:sans-serif; font-weight:700; font-size:2.6rem; line-height:1.1; margin-bottom:0.2rem;">
  <span style="color:#CC785C;">surv</span><span style="color:#9B9B9B;">flow</span>
</div>
""", unsafe_allow_html=True)
st.caption("결측치와 중도절단, 자동으로 분석합니다")

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

st.html('<link rel="stylesheet" as="style" crossorigin '
        'href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css">')

greeting_name = f"{st.user.name}님" if IS_LOGGED_IN else "반가워요"
st.markdown(
    f"<div style=\"font-family:'Pretendard', sans-serif; font-size:2.3rem; font-weight:700; "
    f"color:#6B6862; margin:0.8rem 0 1rem 0;\">{greeting_name}, 환영해요</div>",
    unsafe_allow_html=True,
)

with st.chat_message("assistant"):
    st.write("임상 데이터 CSV를 업로드해주세요.")
    new_upload = st.file_uploader("임상 데이터 CSV 업로드", type='csv', label_visibility="collapsed",
                                   key=f"uploader_{st.session_state.get('uploader_key_version', 0)}")
    if new_upload is not None and st.session_state.get('uploaded_file_id') != new_upload.file_id:
        # 새 파일이 들어온 시점에 내용을 세션에 통째로 복사 -> 이후엔 위젯이 아니라 이 값만 참조
        st.session_state['uploaded_bytes'] = new_upload.getvalue()
        st.session_state['uploaded_name'] = new_upload.name
        st.session_state['uploaded_file_id'] = new_upload.file_id
        st.session_state['confirmed_file_id'] = None  # 새 파일이면 확인 상태 초기화

    up_file_id = st.session_state.get('uploaded_file_id')
    up_name = st.session_state.get('uploaded_name')

    if up_file_id:
        if st.session_state.get('confirmed_file_id') != up_file_id:
            if st.button("다음 →", key=f"proceed_{up_file_id}"):
                st.session_state['confirmed_file_id'] = up_file_id
                st.rerun()
        else:
            if st.button("🔄 같은 데이터로 새 분석 시작", key=f"restart_{up_file_id}"):
                st.session_state['confirmed_file_id'] = None
                st.session_state.pop('last_result', None)
                st.rerun()

if up_file_id and st.session_state.get('confirmed_file_id') == up_file_id:
    import io
    user_df = pd.read_csv(io.BytesIO(st.session_state['uploaded_bytes']))

    # ── P3-5 간이 스키마 체크 (예측에는 실제 결과 데이터가 필요 없으므로 임상변수 16개만 필수) ──
    missing_cols = [c for c in CORE_COLS if c not in user_df.columns]
    if missing_cols:
        with st.chat_message("assistant"):
            st.error(f"다음 컬럼이 없어 이 데이터는 처리할 수 없습니다: {missing_cols}\n"
                     f"(이 시스템은 METABRIC 임상변수 16개 스키마 전용입니다)")
        st.stop()

    # ── 범주형 인코딩 (학습 시 매핑 그대로 재사용) ──
    encoded_df = user_df.copy()
    unseen_flag = False
    for col in CORE_CATEGORICAL:
        encoded_df[col] = encoded_df[col].map(encoders[col])
        if encoded_df[col].isna().any():
            unseen_flag = True

    submitted = False
    with st.chat_message("assistant"):
        st.write(f"스키마 확인 완료 — {len(user_df)}명 데이터 로드됐어요.")
        if unseen_flag:
            st.write("⚠️ 일부 범주형 값이 학습 데이터에 없던 값이라 결측으로 처리했어요.")
        st.write("어떤 걸 도와드릴까요?")
        purpose = st.radio("분석 목적을 선택하세요", ["개별 위험도 예측", "치료 효과 유의성 비교"],
                            label_visibility="collapsed")

        if purpose == "개별 위험도 예측":
            with st.form("predict_form"):
                time_horizon = st.select_slider("예측 시점(개월)", options=[12, 36, 60, 120], value=60)
                submitted = st.form_submit_button("분석 실행")
        else:
            missing_outcome = [c for c in [DUR, EVT] if c not in user_df.columns]
            if missing_outcome:
                st.error(f"효과비교(Cox 회귀)는 실제 생존기간·사망여부 데이터가 있어야 합니다. "
                         f"다음 컬럼이 없습니다: {missing_outcome}\n"
                         f"(신규 환자 예측만 하시려면 '개별 위험도 예측'을 선택하세요 — 그건 결과 데이터 없이도 됩니다)")
                st.stop()
            with st.form("cox_form"):
                compare_col = st.selectbox("비교할 치료 변수", meta['CORE_BINARY'])
                submitted = st.form_submit_button("분석 실행")

    if submitted and purpose == "개별 위험도 예측":
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

        avg_risk = result_df['질병사망_위험도'].mean()
        st.session_state['last_result'] = {
            'type': '예측', 'file_id': up_file_id,
            'result_df': result_df, 'avg_risk': avg_risk, 'time_horizon': time_horizon,
        }

        save_key = f"saved_{up_file_id}_예측_{time_horizon}"
        if not st.session_state.get(save_key, False):
            record = {
                "user_email": st.user.email if IS_LOGGED_IN else "(로그인 안 함)",
                "filename": up_name, "title": up_name, "purpose": "예측",
                "n_patients": len(user_df), "avg_risk": float(avg_risk),
            }
            if IS_LOGGED_IN:
                try:
                    supabase.table("analysis_history").insert(record).execute()
                except Exception as e:
                    st.caption(f"(기록 저장 실패: {e})")
            else:
                st.session_state.local_history.insert(0, record)
            st.session_state[save_key] = True
        st.rerun()  # 사이드바 기록을 즉시 갱신하기 위해 한 번 더 실행 (가드로 중복저장 없음)

    if submitted and purpose == "치료 효과 유의성 비교":
        cph = CoxPHFitter(penalizer=0.1)
        prep_for_cox = load_preprocessors()
        cox_input_df = encoded_df.copy()
        n_missing_before = cox_input_df[CORE_COLS + [DUR, EVT]].isna().sum().sum()
        missing_note = None
        if n_missing_before > 0:
            cox_input_df[CORE_COLS] = cox_input_df[CORE_COLS].fillna(prep_for_cox['ae_fillstats'])
            missing_note = f"※ 결측값 {n_missing_before}개를 Engine1과 동일한 학습 데이터 평균값으로 채운 뒤 분석했습니다."
        cph.fit(cox_input_df[CORE_COLS + [DUR, EVT]], duration_col=DUR, event_col=EVT)

        row = cph.summary.loc[compare_col]
        hr, p = row['exp(coef)'], row['p']
        st.session_state['last_result'] = {
            'type': '효과비교', 'file_id': up_file_id, 'compare_col': compare_col,
            'hr': hr, 'p': p, 'summary': cph.summary[['coef', 'exp(coef)', 'p']],
            'missing_note': missing_note,
        }

        save_key = f"saved_{up_file_id}_효과비교_{compare_col}"
        if not st.session_state.get(save_key, False):
            record = {
                "user_email": st.user.email if IS_LOGGED_IN else "(로그인 안 함)",
                "filename": up_name, "title": up_name, "purpose": "효과비교",
                "n_patients": len(user_df), "hr_variable": compare_col,
                "hr_value": float(hr), "p_value": float(p),
            }
            if IS_LOGGED_IN:
                try:
                    supabase.table("analysis_history").insert(record).execute()
                except Exception as e:
                    st.caption(f"(기록 저장 실패: {e})")
            else:
                st.session_state.local_history.insert(0, record)
            st.session_state[save_key] = True
        st.rerun()

    # ── 결과 표시 (제출 직후든, 그 다음 재실행이든 항상 세션에 저장된 최신 결과를 보여줌) ──
    lr = st.session_state.get('last_result')
    if lr and lr.get('file_id') == up_file_id:
        with st.chat_message("assistant"):
            if lr['type'] == '예측':
                st.write("**Engine1 (통합 신경망)** 으로 개별 위험도를 예측했어요.")
                result_df, avg_risk, time_horizon = lr['result_df'], lr['avg_risk'], lr['time_horizon']
                st.dataframe(result_df[[c for c in result_df.columns if '위험도' in c or c in CORE_COLS[:3]]])
                if avg_risk < 0.2:
                    interp = "양호 — 평균적으로 낮은 위험도군입니다."
                elif avg_risk < 0.4:
                    interp = "중간 — 주의 관찰이 필요한 수준입니다."
                else:
                    interp = "높음 — 위험도가 높은 환자 비중이 큽니다."
                st.info(f"[해석] {time_horizon}개월 시점 평균 질병사망 위험도 {avg_risk:.1%} → {interp}")
            else:
                st.write("**Engine2 (Cox 회귀)** 로 효과 비교를 진행했어요.")
                if lr.get('missing_note'):
                    st.caption(lr['missing_note'])
                hr, p, compare_col = lr['hr'], lr['p'], lr['compare_col']
                st.metric(f"{compare_col} 위험비(HR)", f"{hr:.3f}", f"p = {p:.4f}")
                if p < 0.05:
                    direction = "위험을 유의하게 낮춥니다" if hr < 1 else "위험을 유의하게 높입니다"
                    interp = f"{compare_col}는 통계적으로 유의미하게 {direction} (p<0.05)."
                else:
                    interp = f"{compare_col}의 효과는 통계적으로 유의하지 않습니다 (p≥0.05) — 표본 크기나 데이터 특성상 추가 검증이 필요합니다."
                st.info(f"[해석] {interp}")
                st.write("전체 Cox 회귀 결과:")
                st.dataframe(lr['summary'])
