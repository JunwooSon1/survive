import os
os.environ.setdefault('PYCOX_DATA_DIR', '/tmp/pycox_data')  # 클라우드 환경에서 pycox가 site-packages 안에 쓰기 시도하다 PermissionError 나는 것 방지

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchtuples as tt
import pickle
import json
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
    for k in ['confirmed_file_id', 'uploaded_file_id', 'uploaded_bytes', 'uploaded_name', 'last_result', 'viewing_history_record']:
        st.session_state.pop(k, None)

# ── 사이드바 여백 미세조정용 CSS (key 기반 정밀 타겟팅) ──
st.html("""
<style>
section[data-testid="stSidebar"] .stMainBlockContainer {
    padding-top: 0.3rem !important;
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
[class*="st-key-histrow_"] {
    margin-bottom: -0.9rem !important;
}
[class*="st-key-histrowactive_"] {
    margin-bottom: -0.9rem !important;
    background-color: #EFE9DC !important;
    border-radius: 8px !important;
}
[class*="st-key-histrowactive_"] div[data-testid="stHorizontalBlock"] {
    gap: 0.2rem !important;
}
[class*="st-key-histrowactive_"] [data-testid="stColumn"]:first-of-type {
    padding-left: 0 !important;
}
[class*="st-key-histrowactive_"] .stButton button {
    padding-left: 0 !important;
    justify-content: flex-start !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
}
[class*="st-key-new_analysis_wrap"], [class*="st-key-search_wrap"] {
    margin-top: -0.6rem !important;
    margin-bottom: -1.1rem !important;
}
[class*="st-key-logo_row"] {
    margin-bottom: 0.3rem !important;
    position: relative !important;
}
[class*="st-key-logo_row"] [data-testid="stButton"] {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 10 !important;
    margin: 0 !important;
}
[class*="st-key-logo_row"] [data-testid="stButton"] button {
    width: 100% !important;
    height: 100% !important;
    background: transparent !important;
    border: none !important;
    opacity: 0 !important;
    cursor: pointer !important;
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
[class*="st-key-upload_card"] {
    background-color: #F0EEE5 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[class*="st-key-upload_card"] div[data-testid="stFileUploaderDropzone"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}
/* 파일 첨부 후 나오는 아이콘 교체 시도 (선택자 확정 안 됨 - 실험적) */
[data-testid="stFileUploaderFile"] svg {
    display: none !important;
}
[data-testid="stFileUploaderFile"]::before {
    content: "📄";
    margin-right: 0.5rem;
    font-size: 1.3rem;
}
[class*="st-key-upload_card"] div[data-testid="stVerticalBlock"] {
    gap: 0.4rem !important;
}
[class*="st-key-upload_card"] div[data-testid="stElementContainer"] {
    margin-bottom: -0.5rem !important;
}
/* 비활성 버튼일 때 확실히 회색으로 */
[class*="st-key-upload_card"] button:disabled {
    background-color: #D8D4C8 !important;
    color: #A9A296 !important;
    border-color: #D8D4C8 !important;
}
</style>
""")

# ── 삭제 확인 모달창 ──
@st.dialog("삭제 확인")
def delete_dialog(rid, title):
    st.write(f"**{title}** 을(를) 정말 삭제하시겠어요?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("예, 삭제", key=f"dialog_delyes_{rid}", use_container_width=True, type="primary"):
            supabase.table("analysis_history").delete().eq("id", rid).execute()
            st.session_state['history_ui_version'] = st.session_state.get('history_ui_version', 0) + 1
            st.session_state['confirmed_file_id'] = None  # 진행중이던 분석 프롬프트도 리셋
            st.session_state.pop('last_result', None)
            st.rerun()
    with col2:
        if st.button("아니오", key=f"dialog_delno_{rid}", use_container_width=True):
            st.rerun()

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

# ── 분석 검색 모달창 ──
@st.dialog("분석 검색")
def search_dialog():
    query = st.text_input("검색어", key="search_dialog_query",
                           placeholder="파일명으로 검색...", label_visibility="collapsed")
    if query:
        all_records = []
        if IS_LOGGED_IN:
            try:
                res = (
                    supabase.table("analysis_history")
                    .select("*")
                    .eq("user_email", st.user.email)
                    .order("created_at", desc=True)
                    .limit(50)
                    .execute()
                )
                all_records = res.data or []
            except Exception:
                all_records = []
        else:
            all_records = st.session_state.get('local_history', [])

        matches = [r for r in all_records
                   if query.lower() in (r.get('title') or r.get('filename') or '').lower()]

        if matches:
            st.caption(f"{len(matches)}건 찾았어요. 원하는 항목을 선택하세요.")
            for i, r in enumerate(matches):
                label = r.get('title') or r.get('filename') or '(제목 없음)'
                if st.button(label, key=f"searchresult_{r.get('id', i)}", use_container_width=True, type="tertiary"):
                    st.session_state['viewing_history_record'] = r
                    st.rerun()
        else:
            st.caption("검색 결과가 없습니다.")

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
        if st.button("\u200b", key="logo_home_btn"):
            go_home()
            st.rerun()

    with st.container(key="new_analysis_wrap"):
        if st.button("새 분석", key="new_analysis_btn", icon=":material/add:",
                      use_container_width=True, type="tertiary"):
            go_home()
            st.rerun()

    with st.container(key="search_wrap"):
        if st.button("분석 검색", key="toggle_search_btn", icon=":material/search:",
                      use_container_width=True, type="tertiary"):
            search_dialog()

    st.markdown("<hr style='border:none; border-top:1px solid #E7E3D8; margin:0.3rem 0;'>", unsafe_allow_html=True)

    if IS_LOGGED_IN:
        st.markdown(
            f"<div style='font-size:1.3rem; font-weight:700; color:#3D3929; margin-bottom:0rem;'>{st.user.name}</div>",
            unsafe_allow_html=True,
        )
        st.html(
            f'<div style="display:flex; align-items:baseline; gap:0.35rem; margin-bottom:0.15rem;">'
            f'<span style="position:relative; top:2px; font-size:0.95rem;">&#9993;</span>'
            f'<span>{st.user.email}</span></div>'
        )
        with st.container(key="logout_wrap"):
            st.button("로그아웃", on_click=st.logout)
    else:
        st.info("로그인하면 분석 기록이 영구 저장됩니다.")
        st.button("Google로 로그인", on_click=st.login)

    st.markdown("<hr style='border:none; border-top:1px solid #E7E3D8; margin:0.7rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.1rem; font-weight:700; text-align:left; margin-bottom:0.1rem;'>최근 분석 기록</div>",
                unsafe_allow_html=True)

    def render_history_item(record, editable_db=False):
        rid = record.get('id')
        display_title = record.get('title') or record.get('filename') or '(제목 없음)'
        is_active = st.session_state.get('viewing_history_record', {}).get('id') == rid
        row_key = f"histrowactive_{rid}" if is_active else f"histrow_{rid}"

        row = st.container(key=row_key)
        col_title, col_menu = row.columns([5, 1])
        with col_title:
            if st.button(display_title, key=f"open_{rid}", use_container_width=True, type="tertiary"):
                st.session_state['viewing_history_record'] = record
                st.rerun()
        with col_menu:
            if editable_db:
                ui_v = st.session_state.get('history_ui_version', 0)
                with st.popover("⋮", type="tertiary", key=f"popover_{rid}_{ui_v}"):
                    if st.button("✏️ 이름 변경", key=f"renamebtn_{rid}", use_container_width=True, type="tertiary"):
                        rename_dialog(rid, display_title)
                    if st.button("🗑 삭제", key=f"delbtn_{rid}", use_container_width=True, type="tertiary"):
                        delete_dialog(rid, display_title)


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
                for record in history.data:
                    render_history_item(record, editable_db=True)
            else:
                st.caption("아직 분석 기록이 없습니다.")
        except Exception as e:
            st.caption(f"기록 조회 실패: {e}")
    else:
        if st.session_state.local_history:
            for i, record in enumerate(st.session_state.local_history):
                record.setdefault('id', f'local_{i}')
                render_history_item(record, editable_db=False)
            st.caption("⚠ 로그인하지 않아 새로고침하면 이 기록은 사라집니다.")
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

with st.container(key="upload_card"):
    st.write("임상 데이터 CSV를 업로드해주세요.")
    new_upload = st.file_uploader("임상 데이터 CSV 업로드", type='csv', label_visibility="collapsed",
                                   key=f"uploader_{st.session_state.get('uploader_key_version', 0)}")
    if new_upload is not None and st.session_state.get('uploaded_file_id') != new_upload.file_id:
        # 새 파일이 들어온 시점에 내용을 세션에 통째로 복사 -> 이후엔 위젯이 아니라 이 값만 참조
        st.session_state['uploaded_bytes'] = new_upload.getvalue()
        st.session_state['uploaded_name'] = new_upload.name
        st.session_state['uploaded_file_id'] = new_upload.file_id
        st.session_state['confirmed_file_id'] = None  # 새 파일이면 확인 상태 초기화
        st.session_state.pop('viewing_history_record', None)

    up_file_id = st.session_state.get('uploaded_file_id')
    up_name = st.session_state.get('uploaded_name')

    if st.session_state.get('confirmed_file_id') != up_file_id:
        if st.button("분석 시작", key="proceed_btn", icon=":material/arrow_forward:",
                      use_container_width=True, type="primary", disabled=(up_file_id is None)):
            st.session_state['confirmed_file_id'] = up_file_id
            st.rerun()

if up_file_id and st.session_state.get('confirmed_file_id') == up_file_id:
    import io
    user_df = pd.read_csv(io.BytesIO(st.session_state['uploaded_bytes']))

    # ── P3-5 간이 스키마 체크 (예측에는 실제 결과 데이터가 필요 없으므로 임상변수 16개만 필수) ──
    missing_cols = [c for c in CORE_COLS if c not in user_df.columns]
    if missing_cols:
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
    lines = [f"스키마 확인 완료 — {len(user_df)}명 데이터 로드됐어요."]
    if unseen_flag:
        lines.append("⚠️ 일부 범주형 값이 학습 데이터에 없던 값이라 결측으로 처리했어요.")
    if st.session_state.get('last_result', {}).get('file_id') == up_file_id:
        lines.append("💡 아래에서 조건을 바꾸고 '분석 실행'을 다시 누르면 같은 데이터로 재분석할 수 있어요.")
    lines.append("어떤 걸 도와드릴까요?")
    st.markdown(
        "<div style='line-height:1.7; margin-bottom:0.3rem;'>" + "<br>".join(lines) + "</div>",
        unsafe_allow_html=True,
    )
    purpose = st.radio("분석 목적을 선택하세요", ["개별 위험도 예측", "치료 효과 유의성 검정"],
                        label_visibility="collapsed")

    if purpose == "개별 위험도 예측":
        with st.form("predict_form", border=False):
            time_horizon = st.select_slider("예측 시점(개월)", options=[12, 36, 60, 120], value=60)
            submitted = st.form_submit_button("분석 실행")
    else:
        missing_outcome = [c for c in [DUR, EVT] if c not in user_df.columns]
        if missing_outcome:
            st.error(f"효과 유의성 검정은 실제 생존기간·사망여부 데이터가 있어야 합니다. "
                     f"다음 컬럼이 없습니다: {missing_outcome}\n"
                     f"(신규 환자 예측만 하시려면 '개별 위험도 예측'을 선택하세요 — 그건 결과 데이터 없이도 됩니다)")
            st.stop()
        with st.form("cox_form", border=False):
            compare_col = st.selectbox("비교할 치료 변수", meta['CORE_BINARY'])
            submitted = st.form_submit_button("분석 실행")

    if submitted and purpose == "개별 위험도 예측":
        st.session_state.pop('viewing_history_record', None)
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
            detail_cols = [c for c in result_df.columns if '위험도' in c or c in CORE_COLS[:3]]
            detail_json = json.loads(result_df[detail_cols].to_json(orient='records'))
            record = {
                "user_email": st.user.email if IS_LOGGED_IN else "(로그인 안 함)",
                "filename": up_name, "title": up_name, "purpose": "예측",
                "n_patients": len(user_df), "avg_risk": float(avg_risk),
                "detail_json": detail_json,
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

    if submitted and purpose == "치료 효과 유의성 검정":
        st.session_state.pop('viewing_history_record', None)
        cph = CoxPHFitter(penalizer=0.1)
        prep_for_cox = load_preprocessors()
        cox_input_df = encoded_df.copy()
        n_missing_before = cox_input_df[CORE_COLS + [DUR, EVT]].isna().sum().sum()
        missing_note = None
        if n_missing_before > 0:
            cox_input_df[CORE_COLS] = cox_input_df[CORE_COLS].fillna(prep_for_cox['ae_fillstats'])
            missing_note = f"※ 결측값 {n_missing_before}개를 예측 모델 학습 시 사용했던 평균값으로 채운 뒤 분석했습니다."
        cph.fit(cox_input_df[CORE_COLS + [DUR, EVT]], duration_col=DUR, event_col=EVT)

        row = cph.summary.loc[compare_col]
        hr, p = row['exp(coef)'], row['p']
        st.session_state['last_result'] = {
            'type': '효과비교', 'file_id': up_file_id, 'compare_col': compare_col,
            'hr': hr, 'p': p, 'summary': cph.summary[['coef', 'exp(coef)', 'p']],
            'missing_note': missing_note, 'n_patients': len(user_df),
        }

        save_key = f"saved_{up_file_id}_효과비교_{compare_col}"
        if not st.session_state.get(save_key, False):
            detail_json = json.loads(
                cph.summary[['coef', 'exp(coef)', 'p']].reset_index().to_json(orient='records')
            )
            record = {
                "user_email": st.user.email if IS_LOGGED_IN else "(로그인 안 함)",
                "filename": up_name, "title": up_name, "purpose": "효과비교",
                "n_patients": len(user_df), "hr_variable": compare_col,
                "hr_value": float(hr), "p_value": float(p),
                "detail_json": detail_json,
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
    # 과거 기록을 보는 중일 땐 라이브 결과를 숨겨서 둘이 섞여 보이지 않게 함
    lr = st.session_state.get('last_result')
    if lr and lr.get('file_id') == up_file_id and not st.session_state.get('viewing_history_record'):
        if lr['type'] == '예측':
            st.write("개별 위험도를 예측했어요.")
            result_df, avg_risk, time_horizon = lr['result_df'], lr['avg_risk'], lr['time_horizon']
            display_cols = [c for c in result_df.columns if '위험도' in c or c in CORE_COLS[:3]]
            min_risk = st.slider("질병사망 위험도 이 값 이상만 보기", 0.0, 1.0, 0.0, 0.05, key="risk_filter_slider")
            filtered_result_df = result_df[result_df['질병사망_위험도'] >= min_risk]
            st.caption(f"{len(filtered_result_df)} / {len(result_df)}명 표시 중")
            st.dataframe(filtered_result_df[display_cols])
            if avg_risk < 0.2:
                interp = "양호 — 평균적으로 낮은 위험도군입니다."
            elif avg_risk < 0.4:
                interp = "중간 — 주의 관찰이 필요한 수준입니다."
            else:
                interp = "높음 — 위험도가 높은 환자 비중이 큽니다."
            st.info(f"[해석] {time_horizon}개월 시점 평균 질병사망 위험도 {avg_risk:.1%} → {interp}")
        else:
            st.write("효과 유의성 검정을 진행했어요.")
            if lr.get('missing_note'):
                st.caption(lr['missing_note'])
            hr, p, compare_col = lr['hr'], lr['p'], lr['compare_col']
            st.metric(f"{compare_col} 위험비(HR)", f"{hr:.3f}", f"p = {p:.4f}")
            if p < 0.05:
                if hr < 1:
                    interp = (f"이 데이터에서는 **{compare_col}을(를) 받은 환자군이 안 받은 환자군보다 사망 위험이 "
                               f"더 낮게** 나타났고, 이 차이가 우연으로 보기엔 통계적으로 뚜렷했어요 (p<0.05).")
                else:
                    interp = (f"이 데이터에서는 **{compare_col}을(를) 받은 환자군이 안 받은 환자군보다 사망 위험이 "
                               f"더 높게** 나타났고, 이 차이가 우연으로 보기엔 통계적으로 뚜렷했어요 (p<0.05).")
            else:
                n_patients = lr.get('n_patients')
                small_sample_note = ""
                if n_patients is not None and n_patients < 30:  # 표본크기 30명 미만이면 실제로 작은 표본으로 판단(규칙 기반)
                    small_sample_note = (f" 특히 지금 표본이 **{n_patients}명**으로 적은 편이라, "
                                          f"이 결과가 우연히 그렇게 나왔을 가능성도 같이 감안해야 해요.")
                interp = (f"**{compare_col}을(를) 받은 환자군과 안 받은 환자군 사이에 뚜렷한 생존 차이가 "
                           f"확인되지 않았어요** (p≥0.05). 즉 이 표본만으로는 '{compare_col}이(가) 실제로 효과가 있다'고 "
                           f"확신하기 어렵다는 뜻이에요 (효과가 없다고 확정하는 것도 아니에요).{small_sample_note}")
            st.info(f"[해석] {interp}")
            st.write("전체 Cox 회귀 결과:")
            st.dataframe(lr['summary'])

# ── 사이드바에서 과거 기록을 클릭해서 볼 때 (상세 표 포함) ──
if st.session_state.get('viewing_history_record'):
    record = st.session_state['viewing_history_record']
    title = record.get('title') or record.get('filename') or '(제목 없음)'
    st.write(f"**{title}** 분석 결과예요.")
    if record.get('purpose') == '예측':
        st.write("**목적**: 개별 위험도 예측")
        st.write(f"**환자 수**: {record.get('n_patients')}")
        avg_risk = record.get('avg_risk')
        if avg_risk is not None:
            st.write(f"**평균 질병사망 위험도**: {avg_risk:.1%}")
            if avg_risk < 0.2:
                hist_interp = "양호 — 평균적으로 낮은 위험도군입니다."
            elif avg_risk < 0.4:
                hist_interp = "중간 — 주의 관찰이 필요한 수준입니다."
            else:
                hist_interp = "높음 — 위험도가 높은 환자 비중이 큽니다."
            st.info(f"[해석] 평균 질병사망 위험도 {avg_risk:.1%} → {hist_interp}")
        if record.get('detail_json'):
            hist_df = pd.DataFrame(record['detail_json'])
            if '질병사망_위험도' in hist_df.columns:
                hist_min_risk = st.slider("질병사망 위험도 이 값 이상만 보기", 0.0, 1.0, 0.0, 0.05,
                                           key=f"hist_risk_filter_{record.get('id')}")
                hist_df = hist_df[hist_df['질병사망_위험도'] >= hist_min_risk]
                st.caption(f"{len(hist_df)} / {len(record['detail_json'])}명 표시 중")
            st.dataframe(hist_df)
    else:
        st.write("**목적**: 치료 효과 유의성 검정")
        st.write(f"**환자 수**: {record.get('n_patients')}")
        compare_col = record.get('hr_variable')
        st.write(f"**비교 변수**: {compare_col}")
        hr, p = record.get('hr_value'), record.get('p_value')
        if hr is not None and p is not None:
            st.write(f"**HR**: {hr:.3f}, **p-value**: {p:.4f}")
            if p < 0.05:
                direction = "더 낮게" if hr < 1 else "더 높게"
                hist_interp = (f"이 데이터에서는 **{compare_col}을(를) 받은 환자군이 안 받은 환자군보다 사망 위험이 "
                               f"{direction}** 나타났고, 이 차이가 우연으로 보기엔 통계적으로 뚜렷했어요 (p<0.05).")
            else:
                n_patients = record.get('n_patients')
                small_sample_note = ""
                if n_patients is not None and n_patients < 30:
                    small_sample_note = (f" 특히 지금 표본이 **{n_patients}명**으로 적은 편이라, "
                                          f"이 결과가 우연히 그렇게 나왔을 가능성도 같이 감안해야 해요.")
                hist_interp = (f"**{compare_col}을(를) 받은 환자군과 안 받은 환자군 사이에 뚜렷한 생존 차이가 "
                               f"확인되지 않았어요** (p≥0.05). 즉 이 표본만으로는 '{compare_col}이(가) 실제로 효과가 있다'고 "
                               f"확신하기 어렵다는 뜻이에요 (효과가 없다고 확정하는 것도 아니에요).{small_sample_note}")
            st.info(f"[해석] {hist_interp}")
        if record.get('detail_json'):
            st.write("전체 Cox 회귀 결과:")
            st.dataframe(pd.DataFrame(record['detail_json']))
    if st.button("닫기", key="close_history_view", type="tertiary"):
        st.session_state.pop('viewing_history_record', None)
        st.rerun()
