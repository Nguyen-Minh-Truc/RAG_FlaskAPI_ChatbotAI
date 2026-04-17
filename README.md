# RAG + Co-RAG Chatbot (Flask + Streamlit)

Do an nay la chatbot hoi dap tren tai lieu noi bo, su dung:

- RAG (Retrieval-Augmented Generation)
- Co-RAG (vong tinh chinh cau tra loi)
- Ollama + Qwen chay local (khong can API key)

README nay duoc viet lai de nguoi moi co the clone repo va chay ngay.

## 1. Tong quan he thong

- Backend: Flask API (`main.py`)
- Frontend: Streamlit UI (`streamlit_app.py`)
- Vector store: FAISS luu tai `storage/vectorstore`
- Lich su hoi thoai: JSON luu tai `storage/conversations`
- Dinh dang file ho tro: `.pdf`, `.doc`, `.docx`

## 2. Cau truc thu muc chinh

- `main.py`: entry point Flask
- `streamlit_app.py`: giao dien Streamlit
- `app/api/routes.py`: cac endpoint API
- `app/rag/loader.py`: doc file upload PDF/DOC/DOCX
- `app/rag/chunker.py`: chia chunk
- `app/rag/embeddings.py`: tao embeddings
- `app/rag/vectorstore.py`: luu/tai FAISS
- `app/rag/retriever.py`: tim top-k context
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
streamlit run streamlit_app.py
```

Mac dinh UI: `http://localhost:8501` (co the la 8502 neu 8501 dang duoc su dung)

## 7. Cach su dung nhanh

1. Mo Streamlit UI.
2. Tai len file `.pdf`, `.doc` hoac `.docx`.
3. Dat cau hoi trong o input.
4. He thong tra ve 2 ket qua:
   - RAG tieu chuan
   - Co-RAG (da tinh chinh qua nhieu vong)
5. Xem lich su o tab History.

## 8. API contract (hien tai)

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
- Field: `file` (`.pdf/.doc/.docx`)
- Tac vu: parse file -> chunk -> embedding -> save FAISS -> tao conversation

### `POST /api/ask`

Request body:

```json
{
  "question": "Noi dung cau hoi",
  "conversation_id": "optional",
  "corag_rounds": 3
}
```

Data tra ve:

- `rag_answer`
- `corag_answer`
- `corag_trace`
- `turn_id`, `conversation_id`

### `GET /api/conversations`

Lay danh sach hoi thoai.

### `GET /api/conversations/<conversation_id>`

Lay chi tiet 1 hoi thoai.

### `DELETE /api/conversations/<conversation_id>`

Xoa 1 hoi thoai.

### `DELETE /api/conversations`

Xoa toan bo hoi thoai.

## 9. Su co thuong gap

### Port 5000 bi chiem

```bash
lsof -i :5000
kill -9 <PID>
```

### Streamlit khong chay

Kiem tra dung ten file:

```bash
streamlit run streamlit_app.py
```

Khong dung `straemlit_app.py` (sai chinh ta).

### Loi parse DOC/DOCX

- Dam bao da cai `unstructured[docx]`
- Da cai `en_core_web_sm`
- Neu la file scan/anh: can OCR truoc

## 10. Ghi chu cho nguoi phat trien

- Co-RAG da duoc noi voi base RAG answer de tranh sinh 2 ket qua giong het nhau.
- Lich su da luu ca `answer` (RAG) va `corag_answer`.
- Input question tren Streamlit da auto reset sau moi lan gui thanh cong.
