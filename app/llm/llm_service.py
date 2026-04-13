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

CORAG_SYSTEM_PROMPT = """Bạn là trợ lý AI đang tinh chỉnh câu trả lời qua nhiều vòng truy hồi.
Mục tiêu: dựa trên ngữ cảnh mới để sửa lỗi, bổ sung chi tiết còn thiếu và làm câu trả lời đầy đủ hơn.

Quy tắc trả lời:
1. Ưu tiên dùng ngữ cảnh mới nhất và ghi nhận nếu ngữ cảnh bổ sung làm thay đổi câu trả lời.
2. Giữ lại phần đúng của bản nháp trước, nhưng chủ động sửa hoặc thay thế phần chưa chắc chắn.
3. Nếu bản nháp trước còn thiếu ý, hãy bổ sung ngắn gọn và rõ ràng.
4. Nếu ngữ cảnh vẫn không đủ, vẫn trả lời đúng mẫu:
Không tìm thấy thông tin trong tài liệu đã nạp.
5. Không lặp lại máy móc bản nháp trước; hãy viết lại như một bản trả lời đã được hiệu chỉnh.

Định dạng đầu ra:
- Trả lời chính: <nội dung trả lời>
- Nguồn ngữ cảnh đã dùng: <liệt kê ngắn các đoạn/chunk liên quan nếu có>
"""


def _render_context(context_chunks: list[dict]) -> str:
    context_lines: list[str] = []

    for idx, item in enumerate(context_chunks, start=1):
        chunk_text = item.get("text") or item.get("chunk") or ""
        if not chunk_text.strip():
            continue
        context_lines.append(f"[{idx}] {chunk_text.strip()}")

    return "\n\n".join(context_lines) if context_lines else "(không có ngữ cảnh)"


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build a grounded RAG prompt from retrieved chunks and user question."""
    context_text = _render_context(context_chunks)

    return (
        f"{SYSTEM_PROMPT}\n"
        f"Ngữ cảnh:\n{context_text}\n\n"
        f"Câu hỏi người dùng:\n{question.strip()}"
    )


def build_corag_prompt(
    question: str,
    context_chunks: list[dict],
    previous_answer: str = "",
    round_no: int | None = None,
    rounds: int | None = None,
) -> str:
    """Build a refinement prompt for the iterative Co-RAG loop."""
    context_text = _render_context(context_chunks)
    previous_text = previous_answer.strip() or "(chưa có bản nháp trước đó)"
    round_text = f"Vòng hiện tại: {round_no}/{rounds}" if round_no and rounds else ""

    return (
        f"{CORAG_SYSTEM_PROMPT}\n"
        f"{round_text}\n"
        f"Bản nháp trước đó:\n{previous_text}\n\n"
        f"Ngữ cảnh mới:\n{context_text}\n\n"
        f"Câu hỏi người dùng:\n{question.strip()}"
    ).strip()


def _get_qwen_client() -> ChatOllama:
    """Create the Ollama chat client configured for Qwen."""
    if config.LLM_PROVIDER.lower() != "ollama":
        raise ValueError("LLM_PROVIDER must be 'ollama' to use local Qwen")

    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0,
    )


def _invoke_llm(prompt: str, temperature: float = 0.0) -> str:
    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
    )
    response = llm.invoke(prompt)

    # LangChain can return AIMessage; keep API output strictly as plain string.
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(part) for part in content).strip()

    return str(content).strip()


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Final RAG flow step:
    retrieve top-k chunks -> send context + question to local Qwen -> return answer
    """
    if not question.strip():
        raise ValueError("question cannot be empty")

    prompt = build_prompt(question=question, context_chunks=context_chunks)
    return _invoke_llm(prompt, temperature=0.0)


def generate_corag_refined_answer(
    question: str,
    context_chunks: list[dict],
    previous_answer: str = "",
    round_no: int | None = None,
    rounds: int | None = None,
) -> str:
    """Generate a refinement pass for Co-RAG using an explicit correction prompt."""
    if not question.strip():
        raise ValueError("question cannot be empty")

    prompt = build_corag_prompt(
        question=question,
        context_chunks=context_chunks,
        previous_answer=previous_answer,
        round_no=round_no,
        rounds=rounds,
    )
    return _invoke_llm(prompt, temperature=0.2)
