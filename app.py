import streamlit as st
import tempfile
import pandas as pd
import sys
import concurrent.futures
import requests
import hashlib

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional dependencies for Office docs (installed as needed)
# - Word: pip install python-docx
# - PowerPoint: pip install python-pptx
try:
    from docx import Document  # python-docx
except Exception:
    Document = None

try:
    from pptx import Presentation  # python-pptx
except Exception:
    Presentation = None

# Optional dependencies for OCR (installed as needed)
# - pip install pytesseract pillow pymupdf
# - plus install Tesseract OCR binary on Windows and ensure it's in PATH
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# =========================
# 0) 설정값 (모델명 등)
# =========================
LLM_MODEL_NAME = "llama3.1"
EMBEDDING_MODEL_NAME = "nomic-embed-text"

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TIMEOUT_SECONDS = 120

# OCR 설정 (스캔 PDF 대비)
ENABLE_OCR_FOR_PDF = True
OCR_MAX_PAGES = 5          # 너무 오래 걸리지 않게 처음 N페이지만 OCR
OCR_LANGUAGE = "kor+eng"   # Tesseract 언어팩 필요


# ==========================================
# 1) 세션 상태 초기화 (Streamlit은 매번 재실행되므로 필수)
# ==========================================
st.session_state.setdefault("messages", [])
st.session_state.setdefault("llm", None)
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("engine_error", "")
st.session_state.setdefault("files_signature", "")


def get_llm():
    return st.session_state.get("llm", None)


def get_retriever():
    return st.session_state.get("retriever", None)


def build_files_signature(files) -> str:
    if not files:
        return ""
    signatures = []
    for file in files:
        data = file.getvalue()
        digest = hashlib.md5(data).hexdigest()
        signatures.append(f"{file.name}:{len(data)}:{digest}")
    return "|".join(signatures)


def model_exists(models: list[str], base_name: str) -> bool:
    """
    Ollama는 'llama3.1:latest'처럼 태그가 붙어 올 수 있음.
    base_name이 'llama3.1'이어도 ':latest'까지 True로 인식.
    """
    return any(m == base_name or m.startswith(base_name + ":") for m in models)


# ==========================================
# 2) 페이지 UI
# ==========================================
st.set_page_config(page_title="서연이화 AI 시스템", layout="wide", page_icon="🏭")
st.title("🏭서연이화 AI 시스템")


# ==========================================
# 3) 사이드바
# ==========================================
with st.sidebar:
    st.header("🏭 공장 맞춤 설정")
    factory_filter = st.selectbox(
        "검색할 공장/라인", ["전체 공장", "울산 공장 (A)", "아산 공장 (B)", "해외 법인"]
    )

    st.header("📁 실무 문서 업로드")
    st.caption("지원 포맷: PDF, TXT, CSV, 엑셀(XLSX/XLS), Word(DOCX), PowerPoint(PPTX)")
    uploaded_files = st.file_uploader(
        "매뉴얼, PFMEA, MES 불량 데이터 등 업로드",
        type=["pdf", "txt", "csv", "xlsx", "xls", "docx", "ppt", "pptx"],
        accept_multiple_files=True,
    )

    st.header("💾 대화 관리")
    with st.container():
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.download_button(
            "📤 대화 저장",
            data=str(st.session_state.get("messages", [])),
            file_name="sy_smt_history.json",
            use_container_width=True,
        )

        if st.button("🔄 AI 엔진 초기화 / 문서 재분석", use_container_width=True):
            st.session_state["llm"] = None
            st.session_state["retriever"] = None
            st.session_state["engine_error"] = ""
            st.session_state["files_signature"] = ""
            cached_loader = globals().get("load_documents")
            if cached_loader is not None:
                cached_loader.clear()
            st.rerun()

    st.divider()
    st.caption(f"🤖 LLM: {LLM_MODEL_NAME} | Embedding: {EMBEDDING_MODEL_NAME}")

    # Ollama 연결/모델 상태 표시
    st.caption("🧪 Ollama 상태 체크")
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        if r.ok:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            st.caption(f"✅ Ollama 연결됨 (모델 {len(models)}개)")
            st.caption(f"- LLM 존재: {model_exists(models, LLM_MODEL_NAME)}")
            st.caption(f"- Embedding 존재: {model_exists(models, EMBEDDING_MODEL_NAME)}")
        else:
            st.caption(f"⚠️ Ollama 응답 오류: HTTP {r.status_code}")
    except Exception as e:
        st.caption(f"❌ Ollama 연결 실패: {e}")
        st.caption("→ `ollama serve` 실행 + `ollama pull`로 모델 설치 확인")

    with st.expander("⚙️ 실행 환경(디버그)"):
        st.write(
            {
                "python_executable": sys.executable,
                "python_version": sys.version,
                "pandas": getattr(pd, "__version__", "unknown"),
                "ocr": {
                    "enabled": ENABLE_OCR_FOR_PDF,
                    "fitz_pymupdf_installed": fitz is not None,
                    "pytesseract_installed": pytesseract is not None,
                    "PIL_installed": Image is not None,
                    "lang": OCR_LANGUAGE,
                    "max_pages": OCR_MAX_PAGES,
                },
            }
        )


def ocr_pdf_first_pages(pdf_path: str, max_pages: int = OCR_MAX_PAGES) -> str:
    """
    스캔 PDF처럼 텍스트가 없는 경우 OCR로 텍스트 추출.
    필요:
      - pip install pytesseract pillow pymupdf
      - Windows: Tesseract OCR 프로그램 설치 + PATH 등록
    """
    if fitz is None or pytesseract is None or Image is None:
        raise RuntimeError(
            "OCR 실행에 필요한 패키지가 없습니다.\n"
            "1) pip install pytesseract pillow pymupdf\n"
            "2) Windows: Tesseract OCR 프로그램 설치(UB Mannheim 빌드 권장) 후 PATH 등록\n"
        )

    doc = fitz.open(pdf_path)
    pages = min(len(doc), max_pages)
    out_lines: list[str] = []

    for i in range(pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        text = pytesseract.image_to_string(img, lang=OCR_LANGUAGE)
        text = (text or "").strip()
        if text:
            out_lines.append(f"[OCR_PAGE_{i+1}]\n{text}")

    doc.close()
    return "\n\n".join(out_lines)


# ==========================================
# 4) 문서 로딩/인덱싱
# ==========================================
@st.cache_resource
def load_documents(_uploaded_files, _signature):
    if not _uploaded_files:
        return None, None

    docs = []
    temp_files = []

    # 업로드 파일을 임시 저장
    for f in _uploaded_files:
        parts = f.name.rsplit(".", 1)
        ext = parts[-1].lower() if len(parts) == 2 else ""

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(f.getvalue())
            temp_files.append((tmp.name, ext, f.name))

    # 확장자별 로드
    for path, ext, orig_name in temp_files:
        try:
            if ext == "pdf":
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = orig_name
                docs.extend(loaded_docs)

                # 텍스트가 거의 없으면 OCR 추가 시도
                if ENABLE_OCR_FOR_PDF:
                    total_chars = sum(len((d.page_content or "").strip()) for d in loaded_docs)
                    if total_chars < 50:
                        try:
                            ocr_text = ocr_pdf_first_pages(path)
                            if ocr_text.strip():
                                from langchain_core.documents import Document as LCDocument

                                docs.append(
                                    LCDocument(
                                        page_content=ocr_text,
                                        metadata={"source": f"[OCR]{orig_name}"},
                                    )
                                )
                            else:
                                st.warning(f"OCR 시도했지만 텍스트 추출이 거의 없습니다: {orig_name}")
                        except Exception as e:
                            st.warning(f"OCR 시도 실패 ({orig_name}): {e}")

            elif ext == "txt":
                loader = TextLoader(path, encoding="utf-8")
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = orig_name
                docs.extend(loaded_docs)

            elif ext == "csv":
                loader = CSVLoader(path, encoding="utf-8")
                loaded_docs = loader.load()
                for d in loaded_docs:
                    d.metadata["source"] = orig_name
                docs.extend(loaded_docs)

            elif ext in ["xls", "xlsx"]:
                try:
                    df = pd.read_excel(path)
                except ImportError as ie:
                    st.error(
                        "엑셀(.xlsx/.xls) 파일을 읽으려면 openpyxl이 필요합니다.\n"
                        "CMD: pip install openpyxl"
                    )
                    st.caption(f"상세 오류: {ie}")
                    continue
                except Exception as e:
                    st.error(f"엑셀 로드 실패 ({orig_name}): {e}")
                    continue

                text_data = df.to_string()
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as txt_tmp:
                    txt_tmp.write(text_data)
                    loader = TextLoader(txt_tmp.name, encoding="utf-8")
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        d.metadata["source"] = f"[엑셀데이터] {orig_name}"
                    docs.extend(loaded_docs)

            elif ext == "docx":
                if Document is None:
                    st.error("Word(.docx) 처리: pip install python-docx 필요")
                    continue

                try:
                    doc = Document(path)
                    text_data = "\n".join([p.text for p in doc.paragraphs if p.text])
                except Exception as e:
                    st.error(f"Word 로드 실패 ({orig_name}): {e}")
                    continue

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as txt_tmp:
                    txt_tmp.write(text_data)
                    loader = TextLoader(txt_tmp.name, encoding="utf-8")
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        d.metadata["source"] = f"[Word] {orig_name}"
                    docs.extend(loaded_docs)

            elif ext in ["ppt", "pptx"]:
                if ext == "ppt":
                    st.error("PPT(.ppt)는 파싱이 어렵습니다. .pptx로 저장 후 업로드하세요.")
                    continue

                if Presentation is None:
                    st.error("PPTX 처리: pip install python-pptx 필요")
                    continue

                try:
                    pres = Presentation(path)
                    lines = []
                    for slide in pres.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text") and shape.text:
                                lines.append(shape.text)
                    text_data = "\n".join(lines)
                except Exception as e:
                    st.error(f"PPTX 로드 실패 ({orig_name}): {e}")
                    continue

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as txt_tmp:
                    txt_tmp.write(text_data)
                    loader = TextLoader(txt_tmp.name, encoding="utf-8")
                    loaded_docs = loader.load()
                    for d in loaded_docs:
                        d.metadata["source"] = f"[PPTX] {orig_name}"
                    docs.extend(loaded_docs)

            else:
                st.warning(f"지원하지 않는 확장자입니다: {orig_name}")
                continue

        except Exception as e:
            st.error(f"파일 로드 실패 ({orig_name}): {e}")
            continue

    # chunk로 분할
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    splits = splitter.split_documents(docs)

    if len(splits) == 0:
        st.warning(
            "인덱싱할 텍스트를 추출하지 못했습니다.\n"
            "- PDF가 스캔본이면 OCR 설치 필요\n"
            "- 파일이 비어있을 수도 있음\n"
        )
        return None, None

    # 임베딩 + 벡터DB
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(splits, embeddings)

    llm = OllamaLLM(model=LLM_MODEL_NAME, temperature=0.0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return llm, retriever


# ==========================================
# 5) 업로드 파일이 있으면 인덱싱 실행
# ==========================================
current_signature = build_files_signature(uploaded_files)
if current_signature and current_signature != st.session_state.get("files_signature"):
    st.session_state["llm"] = None
    st.session_state["retriever"] = None
    st.session_state["engine_error"] = ""
    st.session_state["files_signature"] = current_signature

if uploaded_files and (get_llm() is None or get_retriever() is None):
    with st.spinner("전문 데이터 정밀 분석 및 학습 중..."):
        llm, retriever = load_documents(uploaded_files, current_signature)

    if llm is None or retriever is None:
        st.session_state["engine_error"] = (
            "문서 분석에 실패했습니다. 파일을 다시 업로드하거나 엔진 초기화를 눌러주세요."
        )
    else:
        st.session_state["engine_error"] = ""
        st.session_state["llm"] = llm
        st.session_state["retriever"] = retriever
        st.success("✅ 고정밀 문서/데이터 분석 완료!")


# ==========================================
# 6) 누적 대화 표시
# ==========================================
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ==========================================
# 7) 메인 챗봇
# ==========================================
if st.session_state.get("engine_error"):
    st.error(st.session_state.get("engine_error"))

if get_llm() and get_retriever():

    # 액션 아이템 추출
    if st.button("📋 현재까지의 회의/대화 액션 아이템 추출"):
        if len(st.session_state.get("messages", [])) > 0:
            with st.spinner("핵심 업무 요약 중..."):
                history_text = "\n".join(
                    [f"{m['role']}: {m['content']}" for m in st.session_state.get("messages", [])]
                )
                summary_prompt = (
                    "다음 대화를 읽고, 엔지니어가 현장에서 즉시 처리해야 할 Action Item을 추출해.\n"
                    + history_text
                )
                llm = get_llm()
                if llm is None:
                    st.error("LLM이 초기화되지 않았습니다. 파일을 다시 업로드하거나 엔진 초기화를 눌러주세요.")
                    st.stop()
                summary = llm.invoke(summary_prompt)
                st.info(f"**⚡ 핵심 액션 아이템:**\n{summary}")

    # 사용자 입력
    prompt = st.chat_input("공정 기준, 불량 원인, MES 데이터 분석 등을 질문하세요...")

    if prompt:
        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        factory_context = (
            f"반드시 [{factory_filter}] 기준에 맞춰서 "
            if factory_filter != "전체 공장"
            else ""
        )

        with st.chat_message("assistant"):
            with st.spinner("업로드된 문서 교차 검증 중..."):
                try:
                    retriever = get_retriever()
                    if retriever is None:
                        st.error("Retriever가 초기화되지 않았습니다. 파일 업로드 후 다시 시도하세요.")
                        st.stop()

                    docs = retriever.invoke(prompt)

                    if not docs:
                        answer = (
                            "⚠️ **업로드하신 문서에서 관련된 정보를 찾을 수 없습니다.** "
                            "다른 말로 질문하시거나 관련 문서를 추가로 업로드해 주세요."
                        )
                        st.warning(answer)
                    else:
                        context_list = []
                        for i, d in enumerate(docs):
                            source = d.metadata.get("source", "알 수 없는 문서")
                            page = d.metadata.get("page", "")
                            page_info = f"(페이지: {page})" if page != "" else ""
                            context_list.append(
                                f"[출처 {i+1}: {source} {page_info}]\n{d.page_content}"
                            )
                        context_text = "\n\n".join(context_list)

                        full_prompt = f"""너는 서연이화의 엄격한 생산기술 엔지니어입니다.
아래 제공된 [현장 데이터]만을 근거로 사용자의 [질문]에 답변하십시오.

[엄격한 규칙]
1. {factory_context}
2. 제공된 [현장 데이터]에 없는 내용은 절대 지어내지 마십시오 (No Hallucination).
3. 데이터에 답이 없다면 "업로드된 문서에 해당 내용이 없습니다"라고 명확히 답변하십시오.
4. 답변 시 반드시 근거가 된 [출처 X: 문서명]을 함께 명시하여 신뢰성을 높이십시오.
5. 질문에 대한 답을 먼저 제시한 뒤, [현장 데이터]에 근거해 추가로 확인하면 좋은 관련 항목을 1~3개 소개하십시오.

[현장 데이터]
{context_text}

[질문]
{prompt}

정확하고 검증된 답변:"""

                        current_llm = get_llm()
                        if current_llm is None:
                            answer = (
                                "LLM이 초기화되지 않았습니다. 파일을 다시 업로드하거나 "
                                "엔진 초기화를 눌러주세요."
                            )
                            st.error(answer)
                        else:
                            def _run_llm():
                                return current_llm.invoke(full_prompt)

                            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                                future = ex.submit(_run_llm)
                                try:
                                    answer = future.result(timeout=LLM_TIMEOUT_SECONDS)
                                except concurrent.futures.TimeoutError:
                                    answer = (
                                        f"⚠️ LLM 응답이 지연되고 있습니다({LLM_TIMEOUT_SECONDS}초 타임아웃).\n\n"
                                        "확인사항:\n"
                                        "1) Ollama가 실행 중인지 (ollama serve)\n"
                                        "2) 모델이 설치되어 있는지 (ollama list / ollama pull)\n"
                                        "3) PC 사양 대비 모델이 무거운지\n"
                                    )
                                except Exception as e:
                                    answer = f"LLM 호출 중 오류: {e}"

                            st.markdown(answer)

                            with st.expander("🔍 AI가 참고한 실제 문서 원문 보기 (팩트 체크)"):
                                st.text(context_text)

                except Exception as e:
                    answer = f"검색 중 오류가 발생했습니다: {e}"
                    st.error(answer)

        st.session_state.setdefault("messages", [])
        st.session_state["messages"].append({"role": "assistant", "content": answer})

else:
    st.info("👈 왼쪽 사이드바에서 파일을 업로드하세요.")
    st.markdown(
        """
### 🛡️ 고정밀 팩트 체크 모드
- 업로드한 문서 근거로만 답변
- 문서에 없으면 '없다'고 답변

### 📌 스캔 PDF(OCR) 지원 안내
- PDF에서 텍스트가 거의 없으면 자동 OCR을 시도합니다.
- OCR 사용 전 설치 필요:
  1) python -m pip install pytesseract pillow pymupdf
  2) Windows: Tesseract OCR 프로그램 설치 + PATH 등록
"""
    )
