# RAG + Co-RAG Chatbot (Flask + Streamlit)

Do an nay la chatbot hoi dap tren tai lieu noi bo, su dung:

- RAG (Retrieval-Augmented Generation)
- Co-RAG (vong tinh chinh cau tra loi)
- Multi-document RAG voi metadata filtering (AND)
- Ollama + Qwen chay local (khong can API key)

README nay duoc viet lai de nguoi moi co the clone repo va chay ngay.

## 1. Tong quan he thong

- Backend: Flask API (`main.py`)
- Frontend: Streamlit UI (`app.py`)
- Vector store: FAISS luu tai `storage/vectorstore`
- Lich su hoi thoai: JSON luu tai `storage/conversations`
- Dinh dang file ho tro: `.pdf`, `.doc`, `.docx`

## 2. Cau truc thu muc chinh

- `main.py`: entry point Flask
- `app.py`: giao dien Streamlit
- `app/api/routes.py`: cac endpoint API
- `app/rag/loader.py`: doc file upload PDF/DOC/DOCX
- `app/rag/chunker.py`: chia chunk
- `app/rag/embeddings.py`: tao embeddings
- `app/rag/vectorstore.py`: luu/tai FAISS
- `app/rag/retriever.py`: tim top-k context
- `hybrid_search.py`: hybrid FAISS + BM25 retriever
- `app/llm/llm_service.py`: prompt + goi Ollama
- `app/corag/pipeline.py`: pipeline Co-RAG
- `app/config.py`: config tu `.env`

## 3. Yeu cau moi truong

- Python 3.10+
- Ollama da cai tren may
- Model Qwen trong Ollama

## 4. Cai dat thu vien

### 4.1 Tao virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 4.2 Cai dependencies

```bash
pip install -r requirements.txt
```

### 4.3 Tai model Ollama

```bash
ollama pull qwen2.5:5b
```

### 4.4 Cai spaCy model (bat buoc cho parser DOC/DOCX trong mot so truong hop)

```bash
python -m spacy download en_core_web_sm
```

Neu gap loi SSL tren macOS, cai truc tiep bang lenh:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl --trusted-host github.com
```

## 5. Cau hinh `.env`

Tao file `.env` o root project:

```env
FLASK_DEBUG=true
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

TOP_K=3
CHUNK_SIZE=500
CHUNK_OVERLAP=50
USE_HYBRID_SEARCH=true
```

## 6. Chay du an

Mo 2 terminal rieng, cung o root project va da `source venv/bin/activate`.

### Terminal 1: chay Flask API

```bash
python main.py
```

Mac dinh API: `http://127.0.0.1:5000`

### Terminal 2: chay Streamlit UI

```bash
streamlit run app.py
```

Mac dinh UI: `http://localhost:8501` (co the la 8502 neu 8501 dang duoc su dung)

## 7. Cach su dung nhanh

1. Mo Streamlit UI.
2. Tai len file `.pdf`, `.doc` hoac `.docx`.

- Co the tai nhieu file cung luc.
- Co the append vao hoi thoai dang chon.

3. Dat cau hoi trong o input.
4. Chon che do hoi dap trong sidebar:

- `RAG`: chi tra loi bang luong RAG
- `Co-RAG`: chi tra loi bang luong Co-RAG
- `Compare`: tra ve ca 2 ket qua de doi chieu

5. Xem lich su o tab History.
6. Bat bo loc metadata o sidebar neu muon chi truy hoi theo mot nhom tai lieu.

## 8. Hybrid Search (FAISS + BM25)

He thong ho tro hybrid search de ket hop:

- FAISS vector search cho y nghia ngu canh
- BM25 keyword search cho tu khoa va cau hoi ngan

Ban co the bat/tat che do nay theo 2 cach:

- Qua `.env`: `USE_HYBRID_SEARCH=true` hoac `false`
- Qua Streamlit sidebar: checkbox `Hybrid search`

Khi muon so sanh cac che do tim kiem, dung helper trong `hybrid_search.py`:

```python
from hybrid_search import HybridSearchRetriever

retriever = HybridSearchRetriever(top_k=5)
retriever.build()
result = retriever.compare_performance(
  query="Product service la gi?",
  ground_truth_docs=[],
  k=5,
)
```

Ham `compare_performance()` se in bang so sanh cho 3 che do:

- `vector`
- `bm25`
- `ensemble`

Chi so bao gom:

- `Precision@k`
- `Recall@k`
- `MRR`
- `latency_ms`

Neu ban muon test nhanh bang sample document, chay truc tiep:

```bash
python hybrid_search.py
```

## 9. API contract (hien tai)

Tat ca response deu theo envelope:

```json
{
  "success": true,
  "message": "...",
  "data": {},
  "error": null
}
```

### `GET /api/health`

Kiem tra service con song.

### `POST /api/upload`

- Content-Type: `multipart/form-data`
- Field: `file` hoac `files` (`.pdf/.doc/.docx`)
- Optional field: `chunk_size` (`500 | 1000 | 1500 | 2000`)
- Optional field: `chunk_overlap` (`50 | 100 | 200`)
- Optional field: `conversation_id` (neu muon append vao hoi thoai cu)
- Tac vu: parse file(s) -> chunk -> embedding -> save/merge FAISS -> cap nhat metadata document

Metadata luu cho moi document:

- `document_id`
- `source` (ten file)
- `upload_date` (UTC ISO)
- `file_type` (`pdf`, `doc`, `docx`)

### `POST /api/rag-ask`

Request body:

```json
{
  "question": "Noi dung cau hoi",
  "conversation_id": "optional",
  "use_hybrid_search": true,
  "metadata_filters": {
    "sources": ["sales_q1.pdf"],
    "file_types": ["pdf"],
    "document_ids": [],
    "upload_date_from": "2026-04-01T00:00:00+00:00",
    "upload_date_to": "2026-04-30T23:59:59+00:00"
  }
}
```

Data tra ve:

- `rag_answer`
- `context`
- `source_summary` (tong hop dong gop theo document)
- `turn_id`, `conversation_id`

### `POST /api/corag-ask`

Request body:

```json
{
  "question": "Noi dung cau hoi",
  "conversation_id": "optional",
  "corag_rounds": 2,
  "use_hybrid_search": true,
  "metadata_filters": {
    "sources": ["sales_q1.pdf"],
    "file_types": ["pdf"],
    "document_ids": [],
    "upload_date_from": "2026-04-01T00:00:00+00:00",
    "upload_date_to": "2026-04-30T23:59:59+00:00"
  }
}
```

Data tra ve:

- `corag_answer`
- `corag_trace`
- `context`
- `source_summary` (tong hop dong gop theo document)
- `turn_id`, `conversation_id`

### `POST /api/ask` (backward-compatible compare endpoint)

Request body:

```json
{
  "question": "Noi dung cau hoi",
  "conversation_id": "optional",
  "corag_rounds": 2,
  "use_hybrid_search": true,
  "metadata_filters": {
    "sources": [],
    "file_types": [],
    "document_ids": [],
    "upload_date_from": null,
    "upload_date_to": null
  }
}
```

Data tra ve:

- `rag_answer`
- `corag_answer`
- `corag_trace`
- `source_summary`
- `turn_id`, `conversation_id`

### `GET /api/conversations`

Lay danh sach hoi thoai.

### `GET /api/conversations/<conversation_id>`

Lay chi tiet 1 hoi thoai.

### `DELETE /api/conversations/<conversation_id>`

Xoa 1 hoi thoai.

### `DELETE /api/conversations`

Xoa toan bo hoi thoai.

## 10. Su co thuong gap

### Port 5000 bi chiem

```bash
lsof -i :5000
kill -9 <PID>
```

### Streamlit khong chay

Kiem tra dung ten file:

```bash
streamlit run app.py
```

Khong dung `straemlit_app.py` (sai chinh ta).

### Loi parse DOC/DOCX

- Dam bao da cai `unstructured[docx]`
- Da cai `en_core_web_sm`
- Neu la file scan/anh: can OCR truoc

## 11. Ghi chu cho nguoi phat trien

- Co-RAG da duoc noi voi base RAG answer de tranh sinh 2 ket qua giong het nhau.
- Lich su da luu ca `answer` (RAG) va `corag_answer`.
- Input question tren Streamlit da auto reset sau moi lan gui thanh cong.

## 12. Benchmark chunk parameters

Project da co script benchmark de thu 12 to hop:

- chunk_size: `500, 1000, 1500, 2000`
- chunk_overlap: `50, 100, 200`

Chay benchmark:

```bash
python scripts/run_chunk_experiments.py --file /duong_dan/toi/tai_lieu.pdf --max-questions 20 --top-k 3
```

Ket qua se duoc ghi vao:

- `storage/experiments/chunk_grid_report_<timestamp>.json`

Report xep hang theo `avg_accuracy_proxy` (answer-based proxy), dong thoi hien latency va retrieval score de so sanh.
