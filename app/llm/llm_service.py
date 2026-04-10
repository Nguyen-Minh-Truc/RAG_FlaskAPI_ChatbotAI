"""LLM abstraction layer (skeleton only)."""

from langchain_ollama import ChatOllama

from app import config


SYSTEM_PROMPT = """Bạn là trợ lý AI trả lời dựa trên ngữ cảnh được cung cấp.
Mục tiêu: trả lời chính xác, ngắn gọn, dễ hiểu, không bịa thông tin.

Quy tắc trả lời:
1. Chỉ dùng thông tin có trong Ngữ cảnh.
2. Nếu Ngữ cảnh không đủ, trả lời đúng mẫu:
Không tìm thấy thông tin trong tài liệu đã nạp.
3. Không suy đoán, không thêm kiến thức ngoài.
4. Trình bày rõ ràng, ưu tiên gạch đầu dòng khi phù hợp.
5. Nếu có nhiều ý, nhóm theo từng mục ngắn.

Định dạng đầu ra:
- Trả lời chính: <nội dung trả lời>
- Nguồn ngữ cảnh đã dùng: <liệt kê ngắn các đoạn/chunk liên quan nếu có>
"""


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build a grounded RAG prompt from retrieved chunks and user question."""
    context_lines: list[str] = []

    for idx, item in enumerate(context_chunks, start=1):
        chunk_text = item.get("text") or item.get("chunk") or ""
        if not chunk_text.strip():
            continue
        context_lines.append(f"[{idx}] {chunk_text.strip()}")

    context_text = "\n\n".join(context_lines) if context_lines else "(không có ngữ cảnh)"

    return (
        f"{SYSTEM_PROMPT}\n"
        f"Ngữ cảnh:\n{context_text}\n\n"
        f"Câu hỏi người dùng:\n{question.strip()}"
    )


def _get_qwen_client() -> ChatOllama:
    """Create the Ollama chat client configured for Qwen."""
    if config.LLM_PROVIDER.lower() != "ollama":
        raise ValueError("LLM_PROVIDER must be 'ollama' to use local Qwen")

    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Final RAG flow step:
    retrieve top-k chunks -> send context + question to local Qwen -> return answer
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    prompt = build_prompt(question=question, context_chunks=context_chunks)
    llm = _get_qwen_client()
    response = llm.invoke(prompt)

    # LangChain can return AIMessage; keep API output strictly as plain string.
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(part) for part in content).strip()

    return str(content).strip()
