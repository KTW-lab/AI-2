import os
import io
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import streamlit as st
import faiss
import tempfile
from huggingface_hub import HfApi, hf_hub_download, list_models
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from langchain_community.document_loaders import PyPDFLoader

# ========== 설정 ==========
BASE_DIR = Path(__file__).resolve().parent
APP_DIR_ENV = os.environ.get("LOCAL_AI_APP_DIR")
APP_DIR = Path(APP_DIR_ENV).expanduser() if APP_DIR_ENV else (BASE_DIR / ".local_llm_gui")
MODELS_DIR = APP_DIR / "models"
INDICES_DIR = APP_DIR / "indices"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
INDICES_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CTX = 4096
DEFAULT_TOP_K = 5

st.set_page_config(
    page_title="로컬 AI 모델 & RAG 시스템", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========= 유틸리티 함수 =========
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str = DEFAULT_EMBED_MODEL):
    """임베딩 모델 로드 (캐싱됨)"""
    return SentenceTransformer(model_name)

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
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
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
            "dim": dim
        }, f)

def load_faiss_index(dirpath: Path) -> Tuple[faiss.Index, List[Dict], List[str], int]:
    """FAISS 인덱스와 메타데이터 로드"""
    index = faiss.read_index(str(dirpath / "index.faiss"))
    
    with open(dirpath / "meta.pkl", "rb") as f:
        data = pickle.load(f)
    
    return index, data["metadatas"], data["texts"], data["dim"]

def build_index_from_pdfs(index_name: str, uploaded_files: List, chunk_size: int, overlap: int, embedder):
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
                    "pdf_pages": pdf_metadata['pages'],
                    "pdf_title": pdf_metadata.get('title', ''),
                    "pdf_author": pdf_metadata.get('author', ''),
                })
                
        except Exception as e:
            st.error(f"'{file.name}' 처리 중 오류: {e}")
            continue
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    if not texts:
        raise ValueError("처리할 수 있는 텍스트가 없습니다.")
    
    status_text.text(f"임베딩 생성 중... (총 {len(texts)}개 청크)")
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    embeddings = embeddings.astype(np.float32)
    embeddings = normalize_vectors(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    save_faiss_index(get_index_dir(index_name), index, metadatas, texts, dim)
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ 색인 생성 완료: {len(texts)}개 청크, {dim}차원 벡터")

def search_index(index_name: str, query: str, top_k: int, embedder) -> List[Dict]:
    """인덱스에서 관련 문서 검색"""
    try:
        dirpath = get_index_dir(index_name)
        index, metadatas, texts, dim = load_faiss_index(dirpath)
        
        q_embedding = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        q_embedding = normalize_vectors(q_embedding)
        
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
            "dimension": data["dim"]
        }
    except Exception:
        return {}

# ========= 모델 관리 =========
@st.cache_data(show_spinner=False)
def search_hf_models(query: str, limit: int = 30):
    """허깅페이스에서 GGUF 모델 검색"""
    try:
        models = list_models(
            search=query,
            task="text-generation",
            sort="downloads",
            direction=-1,
            limit=limit * 3,
            full=True
        )
        
        filtered_models = []
        for model in models:
            if hasattr(model, 'siblings') and model.siblings:
                gguf_files = [s.rfilename for s in model.siblings if s.rfilename.lower().endswith('.gguf')]
                if gguf_files:
                    filtered_models.append({
                        "id": model.modelId,
                        "downloads": getattr(model, 'downloads', 0) or 0,
                        "gguf_files": gguf_files[:5]
                    })
            
            if len(filtered_models) >= limit:
                break
        
        return filtered_models
    except Exception as e:
        st.error(f"모델 검색 오류: {e}")
        return []

def download_gguf_model(repo_id: str, filename: str) -> Path:
    """GGUF 모델 다운로드"""
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR / repo_id.replace("/", "__")),
            local_dir_use_symlinks=False
        )
        return Path(local_path)
    except Exception as e:
        st.error(f"다운로드 오류: {e}")
        raise

def get_local_models() -> List[Path]:
    """로컬 GGUF 모델 목록"""
    return sorted(MODELS_DIR.rglob("*.gguf"))

@st.cache_resource(show_spinner=False)
def load_llama_model(model_path: Path, **kwargs):
    """Llama 모델 로드 (캐싱됨)"""
    return Llama(model_path=str(model_path), **kwargs)

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

def generate_llm_response(llm, messages: List[Dict], stream: bool = True, **kwargs):
    """LLM 응답 생성"""
    return llm.create_chat_completion(messages=messages, stream=stream, **kwargs)

# ========= Streamlit UI =========
def init_session_state():
    """세션 상태 초기화"""
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "embedder" not in st.session_state:
        st.session_state.embedder = load_embedder()
    if "active_index" not in st.session_state:
        st.session_state.active_index = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []


def main():
    init_session_state()
    
    st.title("🤖 로컬 AI 모델 & PDF RAG 시스템")
    st.markdown("*pdfplumber를 사용한 고급 PDF 텍스트 추출 지원*")
    st.caption(f"저장 경로: {APP_DIR}")
    st.markdown("---")
    
    # 사이드바: 모델 관리
    with st.sidebar:
        st.header("🔧 모델 관리")
        
        # 모델 검색 및 다운로드
        with st.expander("📥 모델 검색 & 다운로드", expanded=False):
            search_query = st.text_input(
                "검색어", 
                value="llama",
                help="예: llama, mistral, qwen, phi"
            )
            
            if st.button("🔍 검색"):
                with st.spinner("모델 검색 중..."):
                    results = search_hf_models(search_query)
                
                if results:
                    st.session_state.search_results = results
                    st.success(f"{len(results)}개 모델 발견")
                else:
                    st.session_state.search_results = []
                    st.warning("GGUF 모델을 찾을 수 없습니다.")
            
            # 검색 결과 표시
            if st.session_state.search_results:
                results = st.session_state.search_results
                
                model_options = {r["id"]: f"{r['id']} ({r.get('downloads', 0):,} 다운로드)" for r in results}
                
                selected_model = st.selectbox(
                    "모델 선택",
                    options=list(model_options.keys()),
                    format_func=lambda x: model_options.get(x, x)
                )
                
                if selected_model:
                    model_info = next((r for r in results if r["id"] == selected_model), None)
                    if model_info:
                        selected_file = st.selectbox("GGUF 파일 선택", model_info["gguf_files"])
                        
                        if st.button("⬇️ 다운로드"):
                            with st.spinner(f"다운로드 중: {selected_file}"):
                                try:
                                    path = download_gguf_model(selected_model, selected_file)
                                    st.success(f"✅ 다운로드 완료: {path.name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ 다운로드 실패: {e}")
        
        # 로컬 모델 로드
        with st.expander("🚀 모델 로드", expanded=True):
            local_models = get_local_models()
            
            if not local_models:
                st.info("로컬에 GGUF 모델이 없습니다.")
            else:
                model_path_str = st.selectbox(
                    "모델 파일",
                    [str(p) for p in local_models],
                    format_func=lambda p: Path(p).name
                )
                model_path = Path(model_path_str) if model_path_str else None

                col1, col2 = st.columns(2)
                with col1:
                    n_ctx = st.number_input("컨텍스트 길이", 1024, 8192, DEFAULT_CTX, 512)
                    n_gpu_layers = st.number_input("GPU 레이어", -1, 100, 0, help="-1은 자동")
                
                with col2:
                    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
                    max_tokens = st.number_input("최대 토큰", 64, 4096, 512, 64)
                
                if st.button("🔄 모델 로드") and model_path:
                    with st.spinner("모델 로딩 중..."):
                        try:
                            load_llama_model.clear()
                            st.session_state.llm = load_llama_model(
                                model_path,
                                n_ctx=n_ctx,
                                n_gpu_layers=n_gpu_layers,
                                verbose=False
                            )
                            st.success(f"✅ 모델 로드 완료: {model_path.name}")
                        except Exception as e:
                            st.error(f"❌ 모델 로드 실패: {e}")
                
                if st.session_state.llm and st.button("🗑️ 모델 언로드"):
                    st.session_state.llm = None
                    load_llama_model.clear()
                    st.success("모델 언로드 완료")
                    st.rerun()

        # RAG 설정
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
                help="테이블과 복잡한 레이아웃이 포함된 PDF도 처리 가능"
            )
            
            if st.button("🔨 색인 생성") and uploaded_files:
                if not index_name.strip():
                    st.warning("색인 이름을 입력해주세요.")
                else:
                    try:
                        with st.spinner("PDF 처리 및 색인 생성 중..."):
                            build_index_from_pdfs(index_name, uploaded_files, chunk_size, overlap, st.session_state.embedder)
                        st.success(f"✅ 색인 생성 완료: {index_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 색인 생성 실패: {e}")
        
        # 활성 색인 선택 및 정보
        existing_indices = [p.name for p in INDICES_DIR.iterdir() if p.is_dir() and (p / "index.faiss").exists()]
        
        if existing_indices:
            current_active = st.session_state.get('active_index')
            if current_active not in existing_indices:
                st.session_state.active_index = existing_indices[0]

            active_index_idx = existing_indices.index(st.session_state.active_index) if st.session_state.active_index in existing_indices else 0
            active_index = st.selectbox("활성 색인", existing_indices, index=active_index_idx)
            st.session_state.active_index = active_index
            
            # 색인 통계 표시
            stats = get_index_stats(active_index)
            if stats:
                st.info(f"""
                📊 **색인 정보: {active_index}**
                - 총 청크: {stats.get('total_chunks', 0):,}개
                - 문서 수: {stats.get('total_sources', 0)}개
                - 평균 청크 크기: {stats.get('avg_chunk_size', 0):.0f}자
                - 벡터 차원: {stats.get('dimension', 0)}차원
                """)
        else:
            st.info("생성된 색인이 없습니다.")
            st.session_state.active_index = None

    # 메인 영역: 채팅
    st.header("💬 AI 채팅")
    
    # 설정 패널
    use_rag_default = bool(st.session_state.active_index)
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        system_prompt = st.text_input(
            "시스템 프롬프트",
            "당신은 도움이 되는 AI 어시스턴트입니다. 제공된 문서를 참조하여 한국어로 정확하고 친절하게 답변해주세요."
        )
    with col2:
        use_rag = st.checkbox("RAG 사용", value=use_rag_default, disabled=not st.session_state.active_index)
    with col3:
        top_k = st.number_input("검색 개수", 1, 10, DEFAULT_TOP_K)
    
    # 채팅 기록 표시
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        if not st.session_state.llm:
            st.error("먼저 모델을 로드해주세요.")
            st.stop()
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            context = ""
            if use_rag and st.session_state.active_index:
                with st.spinner("관련 문서 검색 중..."):
                    search_results = search_index(st.session_state.active_index, prompt, top_k, st.session_state.embedder)
                    if search_results:
                        context = build_context_from_results(search_results)
                        with st.expander("참조된 문서", expanded=False):
                            st.info(context)

            messages = [{"role": "system", "content": system_prompt + (f"\n\n다음 문서들을 참조하여 답변하세요:\n\n{context}" if context else "")}]
            messages.extend(st.session_state.messages[-10:-1])
            messages.append({"role": "user", "content": prompt})

            full_response = ""
            try:
                # 스트리밍 시도
                try:
                    stream = generate_llm_response(
                        st.session_state.llm,
                        messages,
                        stream=True,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9
                    )
                    
                    for chunk in stream:
                        if isinstance(chunk, dict):
                            choices = chunk.get('choices', [])
                            if choices and len(choices) > 0:
                                delta = choices[0].get('delta', {})
                                text_chunk = delta.get('content', '')
                                if text_chunk:
                                    full_response += text_chunk
                                    response_placeholder.markdown(full_response + "▌")
                        else:
                            # 객체 형태인 경우
                            try:
                                text_chunk = chunk.choices[0].delta.content
                                if text_chunk:
                                    full_response += text_chunk
                                    response_placeholder.markdown(full_response + "▌")
                            except (AttributeError, IndexError):
                                continue
                
                except (AttributeError, KeyError, TypeError) as e:
                    # 스트리밍 실패 시 비스트리밍으로 폴백
                    st.warning("스트리밍 모드 실패, 일반 모드로 전환...")
                    response = generate_llm_response(
                        st.session_state.llm,
                        messages,
                        stream=False,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=0.9
                    )
                    
                    if isinstance(response, dict):
                        full_response = response.get('choices', [{}])[0].get('message', {}).get('content', '')
                    else:
                        full_response = response.choices[0].message.content
                
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
        model_name = Path(st.session_state.llm.model_path).name if st.session_state.llm else '없음'
        st.write(f"💾 모델: {model_name}")
    
    with col3:
        rag_status = st.session_state.active_index if use_rag and st.session_state.active_index else '비활성'
        st.write(f"📚 RAG: {rag_status}")


if __name__ == "__main__":
    main()
