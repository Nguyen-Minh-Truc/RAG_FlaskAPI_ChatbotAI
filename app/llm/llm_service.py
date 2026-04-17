"""LLM abstraction layer (skeleton only)."""

from functools import lru_cache

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

FOLLOW_UP_REWRITE_PROMPT = """Bạn là bộ chuẩn hoá câu hỏi cho hệ thống hỏi đáp theo hội thoại.
Nhiệm vụ: biến câu hỏi hiện tại thành một câu hỏi độc lập, rõ nghĩa để truy hồi tài liệu.

Quy tắc:
1. Dùng lịch sử hỏi/đáp gần nhất để hiểu các đại từ, từ chỉ định, hoặc câu hỏi tiếp nối.
2. Nếu câu hỏi đã độc lập rồi, giữ nguyên ý nghĩa.
3. Nếu là follow-up, viết lại thành câu hỏi đầy đủ, không nhắc đến "câu hỏi trước", "nó", "đó" một cách mơ hồ.
4. Không trả lời câu hỏi, chỉ viết lại câu hỏi.

Đầu ra:
- Câu hỏi đã chuẩn hoá: <câu hỏi độc lập>
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


def _render_memory(memory_turns: list[dict] | None) -> str:
    if not memory_turns:
        return "(không có lịch sử hội thoại gần đây)"

    memory_lines: list[str] = []
    for idx, turn in enumerate(memory_turns, start=1):
        question = (turn.get("question") or "").strip()
        answer = (turn.get("corag_answer") or turn.get("answer") or "").strip()
        if not question and not answer:
            continue
        memory_lines.append(f"[{idx}] Q: {question}\nA: {answer}")

    return "\n\n".join(memory_lines) if memory_lines else "(không có lịch sử hội thoại gần đây)"


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """Build a grounded RAG prompt from retrieved chunks and user question."""
    context_text = _render_context(context_chunks)

    return (
        f"{SYSTEM_PROMPT}\n"
        f"Ngữ cảnh:\n{context_text}\n\n"
        f"Câu hỏi người dùng:\n{question.strip()}"
    )


def build_memory_aware_prompt(
    question: str,
    context_chunks: list[dict],
    memory_turns: list[dict] | None = None,
    original_question: str | None = None,
) -> str:
    context_text = _render_context(context_chunks)
    memory_text = _render_memory(memory_turns)
    original_text = (original_question or question).strip()

    return (
        f"{SYSTEM_PROMPT}\n"
        f"Lịch sử hội thoại gần đây:\n{memory_text}\n\n"
        f"Ngữ cảnh:\n{context_text}\n\n"
        f"Câu hỏi gốc của người dùng:\n{original_text}\n\n"
        f"Câu hỏi đã chuẩn hoá để trả lời:\n{question.strip()}"
    )


def build_corag_prompt(
    question: str,
    context_chunks: list[dict],
    previous_answer: str = "",
    round_no: int | None = None,
    rounds: int | None = None,
    memory_turns: list[dict] | None = None,
    original_question: str | None = None,
) -> str:
    """Build a refinement prompt for the iterative Co-RAG loop."""
    context_text = _render_context(context_chunks)
    memory_text = _render_memory(memory_turns)
    previous_text = previous_answer.strip() or "(chưa có bản nháp trước đó)"
    round_text = f"Vòng hiện tại: {round_no}/{rounds}" if round_no and rounds else ""
    original_text = (original_question or question).strip()

    return (
        f"{CORAG_SYSTEM_PROMPT}\n"
        f"Lịch sử hội thoại gần đây:\n{memory_text}\n\n"
        f"{round_text}\n"
        f"Bản nháp trước đó:\n{previous_text}\n\n"
        f"Ngữ cảnh mới:\n{context_text}\n\n"
        f"Câu hỏi gốc của người dùng:\n{original_text}\n\n"
        f"Câu hỏi đã chuẩn hoá để tinh chỉnh:\n{question.strip()}"
    ).strip()


def rewrite_followup_question(question: str, memory_turns: list[dict] | None = None) -> str:
    """Rewrite a follow-up question into a standalone retrieval query."""
    if not question.strip():
        raise ValueError("question cannot be empty")

    if not memory_turns:
        return question.strip()

    prompt = (
        f"{FOLLOW_UP_REWRITE_PROMPT}\n"
        f"Lịch sử hội thoại gần đây:\n{_render_memory(memory_turns)}\n\n"
        f"Câu hỏi hiện tại:\n{question.strip()}"
    )
    rewritten = _invoke_llm(prompt, temperature=0.0)

    prefix = "Câu hỏi đã chuẩn hoá:"
    if prefix in rewritten:
        rewritten = rewritten.split(prefix, 1)[-1].strip()

    return rewritten or question.strip()


@lru_cache(maxsize=4)
def _get_qwen_client(temperature: float = 0.0) -> ChatOllama:
    """Create and cache Ollama chat clients configured for Qwen."""
    if config.LLM_PROVIDER.lower() != "ollama":
        raise ValueError("LLM_PROVIDER must be 'ollama' to use local Qwen")

    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
    )


def _invoke_llm(prompt: str, temperature: float = 0.0) -> str:
    llm = _get_qwen_client(temperature=temperature)
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


def generate_memory_aware_answer(
    question: str,
    context_chunks: list[dict],
    memory_turns: list[dict] | None = None,
    original_question: str | None = None,
) -> str:
    if not question.strip():
        raise ValueError("question cannot be empty")

    prompt = build_memory_aware_prompt(
        question=question,
        context_chunks=context_chunks,
        memory_turns=memory_turns,
        original_question=original_question,
    )
    return _invoke_llm(prompt, temperature=0.0)


def generate_corag_refined_answer(
    question: str,
    context_chunks: list[dict],
    previous_answer: str = "",
    round_no: int | None = None,
    rounds: int | None = None,
    memory_turns: list[dict] | None = None,
    original_question: str | None = None,
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
        memory_turns=memory_turns,
        original_question=original_question,
    )
    return _invoke_llm(prompt, temperature=0.2)
