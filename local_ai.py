import os
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
import faiss
import requests
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# ========== 설정 ==========
BASE_DIR = Path(__file__).resolve().parent
APP_DIR_ENV = os.environ.get("LOCAL_AI_APP_DIR")
APP_DIR = Path(APP_DIR_ENV).expanduser() if APP_DIR_ENV else (BASE_DIR / ".local_llm_gui")
MODELS_DIR = APP_DIR / "models"
INDICES_DIR = APP_DIR / "indices"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_TOP_K = 5
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 512
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

st.set_page_config(
    page_title="로컬 AI 모델 & PDF RAG 시스템 (Ollama)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========= 유틸리티 함수 =========

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """벡터 정규화 (코사인 유사도용)"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """텍스트를 청크로 분할"""
    text = text.strip().replace("\r", " ")
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def clean_extracted_text(text: str) -> str:
    """추출된 텍스트 정리"""
    import re

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def load_pdf_text(file_bytes: bytes, filename: str) -> Tuple[str, Dict]:
    """PyPDFLoader로 PDF 텍스트 추출"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text_parts = [d.page_content for d in docs if d.page_content]
        full_text = clean_extracted_text("\n\n".join(text_parts))

        metadata = {
            "title": filename,
            "author": "",
            "pages": len(docs),
            "encrypted": False,
        }
        return full_text, metadata
    except Exception as e:
        st.error(f"PDF 파싱 오류: {e}")
        return "", {"title": filename, "author": "", "pages": 0, "encrypted": False}


# ========= RAG 인덱스 관리 =========

def get_index_dir(name: str) -> Path:
    return INDICES_DIR / name


def save_faiss_index(dirpath: Path, index: faiss.Index, metadatas: List[Dict], texts: List[str], dim: int):
    """FAISS 인덱스와 메타데이터 저장"""
    dirpath.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(dirpath / "index.faiss"))

    with open(dirpath / "meta.pkl", "wb") as f:
        pickle.dump({
            "metadatas": metadatas,
            "texts": texts,
            "dim": dim,
        }, f)


def load_faiss_index(dirpath: Path) -> Tuple[faiss.Index, List[Dict], List[str], int]:
    """FAISS 인덱스와 메타데이터 로드"""
    index = faiss.read_index(str(dirpath / "index.faiss"))

    with open(dirpath / "meta.pkl", "rb") as f:
        data = pickle.load(f)

    return index, data["metadatas"], data["texts"], data["dim"]


def build_index_from_pdfs(
    index_name: str,
    uploaded_files: List,
    chunk_size: int,
    overlap: int,
    embedder: OllamaEmbeddings,
):
    """PDF 파일들로부터 FAISS 인덱스 구축"""
    texts = []
    metadatas = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(uploaded_files):
        status_text.text(f"처리 중: {file.name} ({i+1}/{len(uploaded_files)})")

        try:
            file_bytes = file.read()
            text, pdf_metadata = load_pdf_text(file_bytes, file.name)

            if not text.strip():
                st.warning(f"'{file.name}'에서 텍스트를 추출할 수 없습니다.")
                continue

            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

            st.info(f"📄 {file.name}: {pdf_metadata['pages']}페이지, {len(chunks)}개 청크 생성")

            for j, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    "source": file.name,
                    "chunk": j,
                    "chars": len(chunk),
                    "total_chunks": len(chunks),
                    "pdf_pages": pdf_metadata["pages"],
                    "pdf_title": pdf_metadata.get("title", ""),
                    "pdf_author": pdf_metadata.get("author", ""),
                })

        except Exception as e:
            st.error(f"'{file.name}' 처리 중 오류: {e}")
            continue

        progress_bar.progress((i + 1) / len(uploaded_files))

    if not texts:
        raise ValueError("처리할 수 있는 텍스트가 없습니다.")

    status_text.text(f"임베딩 생성 중... (총 {len(texts)}개 청크)")
    embeddings = embedder.embed_documents(texts)
    embeddings = np.array(embeddings, dtype=np.float32)
    embeddings = normalize_vectors(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    save_faiss_index(get_index_dir(index_name), index, metadatas, texts, dim)

    progress_bar.empty()
    status_text.empty()

    st.success(f"✅ 색인 생성 완료: {len(texts)}개 청크, {dim}차원 벡터")


def search_index(index_name: str, query: str, top_k: int, embedder: OllamaEmbeddings) -> List[Dict]:
    """인덱스에서 관련 문서 검색"""
    try:
        dirpath = get_index_dir(index_name)
        index, metadatas, texts, _ = load_faiss_index(dirpath)

        q_embedding = embedder.embed_query(query)
        q_embedding = normalize_vectors(np.array([q_embedding], dtype=np.float32))

        scores, indices = index.search(q_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            result = metadatas[idx].copy()
            result["score"] = float(score)
            result["text"] = texts[idx]
            results.append(result)

        return results
    except Exception as e:
        st.error(f"검색 중 오류: {e}")
        return []


def get_index_stats(index_name: str) -> Dict:
    """색인 통계 정보 반환"""
    try:
        dirpath = get_index_dir(index_name)
        if not (dirpath / "meta.pkl").exists():
            return {}

        with open(dirpath / "meta.pkl", "rb") as f:
            data = pickle.load(f)

        metadatas = data["metadatas"]
        texts = data["texts"]

        sources = set(meta["source"] for meta in metadatas)
        total_chars = sum(len(text) for text in texts)
        avg_chunk_size = total_chars / len(texts) if texts else 0

        return {
            "total_chunks": len(texts),
            "total_sources": len(sources),
            "total_chars": total_chars,
            "avg_chunk_size": avg_chunk_size,
            "sources": list(sources),
            "dimension": data["dim"],
        }
    except Exception:
        return {}


# ========= Ollama LLM / Embedding =========

def create_ollama_llm(model_name: str, temperature: float, max_tokens: int, base_url: str) -> OllamaLLM:
    return OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_predict=max_tokens,
        base_url=base_url,
    )


def create_ollama_embeddings(model_name: str, base_url: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model_name, base_url=base_url)


def get_ollama_models(base_url: str) -> List[str]:
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [model.get("name") for model in data.get("models", []) if model.get("name")]
    except Exception:
        return []


# ========= 채팅 및 프롬프트 =========

def build_context_from_results(results: List[Dict]) -> str:
    """검색 결과를 컨텍스트 텍스트로 변환"""
    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results, 1):
        source = result.get("source", "문서")
        chunk_id = result.get("chunk", 0)
        total_chunks = result.get("total_chunks", 0)
        text = result.get("text", "")
        score = result.get("score", 0)
        pdf_title = result.get("pdf_title", "")

        title_info = f" - {pdf_title}" if pdf_title else ""
        chunk_info = f"청크 {chunk_id+1}/{total_chunks}" if total_chunks > 0 else f"청크 {chunk_id}"

        context_parts.append(
            f"[참조 {i}] {source}{title_info} ({chunk_info}, 유사도: {score:.3f})\n{text}"
        )

    return "\n\n".join(context_parts)


def build_prompt(system_prompt: str, context: str, history: List[Dict], user_prompt: str) -> str:
    prompt_parts = [f"System: {system_prompt}"]
    if context:
        prompt_parts.append(f"Context:\n{context}")

    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        prompt_parts.append(f"{role}: {msg['content']}")

    prompt_parts.append(f"User: {user_prompt}\nAssistant:")
    return "\n\n".join(prompt_parts)


def generate_llm_response(llm: OllamaLLM, prompt: str) -> str:
    return llm.invoke(prompt)


# ========= Streamlit UI =========

def init_session_state():
    """세션 상태 초기화"""
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "embedder" not in st.session_state:
        st.session_state.embedder = None
    if "active_index" not in st.session_state:
        st.session_state.active_index = None
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {}
    if "embedder_config" not in st.session_state:
        st.session_state.embedder_config = {}


def ensure_llm(model_name: str, temperature: float, max_tokens: int, base_url: str):
    config = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "base_url": base_url,
    }
    if st.session_state.llm is None or st.session_state.llm_config != config:
        st.session_state.llm = create_ollama_llm(model_name, temperature, max_tokens, base_url)
        st.session_state.llm_config = config


def ensure_embedder(model_name: str, base_url: str):
    config = {"model_name": model_name, "base_url": base_url}
    if st.session_state.embedder is None or st.session_state.embedder_config != config:
        st.session_state.embedder = create_ollama_embeddings(model_name, base_url)
        st.session_state.embedder_config = config


def main():
    init_session_state()

    st.title("🤖 로컬 AI 모델 & PDF RAG 시스템 (Ollama)")
    st.markdown("*PyPDFLoader 기반 PDF 텍스트 추출, Ollama LLM/임베딩 사용*")
    st.caption(f"저장 경로: {APP_DIR}")
    st.markdown("---")

    # 사이드바: 모델 관리
    with st.sidebar:
        st.header("🔧 Ollama 설정")

        base_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_BASE_URL)
        llm_model = st.text_input("LLM 모델명", value=DEFAULT_LLM_MODEL)
        embed_model = st.text_input("임베딩 모델명", value=DEFAULT_EMBED_MODEL)

        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
        with col2:
            max_tokens = st.number_input("최대 토큰", 64, 4096, DEFAULT_MAX_TOKENS, 64)

        if st.button("🔄 LLM/임베딩 적용"):
            ensure_llm(llm_model, temperature, max_tokens, base_url)
            ensure_embedder(embed_model, base_url)
            st.success("설정이 적용되었습니다.")

        with st.expander("🔎 Ollama 상태 확인", expanded=False):
            if st.button("연결 확인"):
                available_models = get_ollama_models(base_url)
                if available_models:
                    st.success("Ollama 연결 성공")
                    st.write("사용 가능한 모델:")
                    st.write(available_models)
                else:
                    st.warning("Ollama에 연결했지만 모델 목록을 가져오지 못했습니다.")

        st.header("📚 RAG 설정")
        with st.expander("📄 PDF 색인", expanded=False):
            index_name = st.text_input("색인 이름", "default")

            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.number_input("청크 크기", 200, 2000, 800, 100)
            with col2:
                overlap = st.number_input("오버랩", 0, 500, 200, 50)

            uploaded_files = st.file_uploader(
                "PDF 파일 업로드",
                type=["pdf"],
                accept_multiple_files=True,
                help="PDF 문서를 업로드하여 색인을 생성합니다.",
            )

            if st.button("🔨 색인 생성") and uploaded_files:
                if not index_name.strip():
                    st.warning("색인 이름을 입력해주세요.")
                else:
                    try:
                        ensure_embedder(embed_model, base_url)
                        with st.spinner("PDF 처리 및 색인 생성 중..."):
                            build_index_from_pdfs(
                                index_name,
                                uploaded_files,
                                chunk_size,
                                overlap,
                                st.session_state.embedder,
                            )
                        st.success(f"✅ 색인 생성 완료: {index_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 색인 생성 실패: {e}")

        existing_indices = [
            p.name
            for p in INDICES_DIR.iterdir()
            if p.is_dir() and (p / "index.faiss").exists()
        ]

        if existing_indices:
            current_active = st.session_state.get("active_index")
            if current_active not in existing_indices:
                st.session_state.active_index = existing_indices[0]

            active_index_idx = (
                existing_indices.index(st.session_state.active_index)
                if st.session_state.active_index in existing_indices
                else 0
            )
            active_index = st.selectbox("활성 색인", existing_indices, index=active_index_idx)
            st.session_state.active_index = active_index

            stats = get_index_stats(active_index)
            if stats:
                st.info(
                    f"""
                📊 **색인 정보: {active_index}**
                - 총 청크: {stats.get('total_chunks', 0):,}개
                - 문서 수: {stats.get('total_sources', 0)}개
                - 평균 청크 크기: {stats.get('avg_chunk_size', 0):.0f}자
                - 벡터 차원: {stats.get('dimension', 0)}차원
                """
                )
        else:
            st.info("생성된 색인이 없습니다.")
            st.session_state.active_index = None

    # 메인 영역: 채팅
    st.header("💬 AI 채팅")

    ensure_llm(llm_model, temperature, max_tokens, base_url)

    use_rag_default = bool(st.session_state.active_index)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        system_prompt = st.text_input(
            "시스템 프롬프트",
            "당신은 도움이 되는 AI 어시스턴트입니다. 제공된 문서를 참조하여 한국어로 정확하고 친절하게 답변해주세요.",
        )
    with col2:
        use_rag = st.checkbox("RAG 사용", value=use_rag_default, disabled=not st.session_state.active_index)
    with col3:
        top_k = st.number_input("검색 개수", 1, 10, DEFAULT_TOP_K)

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("메시지를 입력하세요..."):
        if not st.session_state.llm:
            st.error("LLM이 초기화되지 않았습니다. 사이드바에서 설정을 확인하세요.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            context = ""
            if use_rag and st.session_state.active_index:
                ensure_embedder(embed_model, base_url)
                with st.spinner("관련 문서 검색 중..."):
                    search_results = search_index(
                        st.session_state.active_index,
                        prompt,
                        top_k,
                        st.session_state.embedder,
                    )
                    if search_results:
                        context = build_context_from_results(search_results)
                        with st.expander("참조된 문서", expanded=False):
                            st.info(context)

            prompt_text = build_prompt(
                system_prompt=system_prompt,
                context=context,
                history=st.session_state.messages[-10:-1],
                user_prompt=prompt,
            )

            try:
                with st.spinner("응답 생성 중..."):
                    full_response = generate_llm_response(st.session_state.llm, prompt_text)
                response_placeholder.markdown(full_response)
            except Exception as e:
                error_msg = f"응답 생성 중 오류: {e}"
                response_placeholder.error(error_msg)
                full_response = error_msg

            st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🗑️ 대화 기록 삭제"):
            st.session_state.messages = []
            st.rerun()

    with col2:
        model_name = st.session_state.llm_config.get("model_name", "없음")
        st.write(f"💾 LLM: {model_name}")

    with col3:
        rag_status = (
            st.session_state.active_index if use_rag and st.session_state.active_index else "비활성"
        )
        st.write(f"📚 RAG: {rag_status}")


if __name__ == "__main__":
    main()
