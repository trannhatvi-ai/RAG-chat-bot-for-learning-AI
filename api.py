import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json

from retrieval import build_answer_style_instruction, get_rag_components


# Load environment at startup
def _load_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    
    for line in env_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#") or "=" not in row:
            continue
        key, value = row.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not os.getenv(key):
            os.environ[key] = value


_load_env()

# Global RAG components
retriever = None
prompt_template = None
llm = None
format_docs = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, prompt_template, llm, format_docs
    # Load on startup
    print("Initializing RAG components...")
    retriever, prompt_template, llm, format_docs = get_rag_components()
    print("RAG components initialized.")
    yield
    # Cleanup on shutdown
    print("Shutting down...")


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def get_home():
    """Serve the main HTML page."""
    return FileResponse(Path(__file__).resolve().parent / "index.html")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream chat response with RAG pipeline."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    prompt = request.message.strip()

    async def event_generator():
        try:
            # Step 1: Retrieve
            yield f"data: {json.dumps({'type': 'step', 'text': 'B1: Đang truy xuất hybrid (vector + keyword) và rerank ngữ cảnh...'})}\n\n"
            docs = retriever.invoke(prompt)

            sources = []
            for d in docs[:5]:
                src = d.metadata.get("source", "N/A")
                page = d.metadata.get("page")
                if page is not None:
                    sources.append(f"- {src} (page {page + 1})")
                else:
                    sources.append(f"- {src}")

            # Step 2: Prepare prompt
            yield f"data: {json.dumps({'type': 'step', 'text': f'B2: Đang chuẩn bị prompt từ {len(docs)} đoạn ngữ cảnh...'})}\n\n"
            context = format_docs(docs)
            answer_style_instruction = build_answer_style_instruction(prompt)
            prompt_value = prompt_template.invoke(
                {
                    "context": context,
                    "question": prompt,
                    "answer_style_instruction": answer_style_instruction,
                }
            )

            # Step 3: Generate answer
            yield f"data: {json.dumps({'type': 'step', 'text': 'B3: Đang tạo câu trả lời...'})}\n\n"

            full_response = ""
            for chunk in llm.stream(prompt_value):
                text = getattr(chunk, "content", "") or ""
                if text:
                    full_response += text
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

            # Add sources
            if sources:
                sources_text = "\n\nNguồn tham khảo:\n" + "\n".join(sources)
                full_response += sources_text
                yield f"data: {json.dumps({'type': 'sources', 'text': sources_text})}\n\n"

            # Done
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
