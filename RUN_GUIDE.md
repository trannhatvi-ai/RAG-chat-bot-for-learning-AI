# RAG Chatbot - Hướng dẫn chạy

## Yêu cầu môi trường

### 1. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 2. Cài đặt Poppler (cho OCR fallback)
- Tải từ: https://github.com/oschwartz10612/poppler-windows/releases
- Giải nén vào `D:\poppler`
- Thêm `D:\poppler\Library\bin` vào PATH

### 3. Cài đặt Tesseract OCR (cho OCR fallback)
- Tải từ: https://github.com/UB-Mannheim/tesseract/wiki
- Cài đặt vào `C:\Program Files\Tesseract-OCR`

## Cấu hình

### .env file
```dotenv
# OCR fallback
OCR_FALLBACK_ENABLED=true

# LangSmith Tracing
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT="your-project-name"
```

## Chạy ứng dụng

### 1. Khởi tạo dữ liệu (lần đầu hoặc khi cập nhật PDF)
```bash
python pre.py
```

### 2. Chạy API server
```bash
python api.py
```

Hoặc với debug mode:
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

### 3. Mở browser
Truy cập: `http://127.0.0.1:8000`

## Cấu trúc project

```
RAGChatbot/
├── api.py                 # FastAPI backend
├── index.html             # Frontend HTML/CSS/JS
├── pre.py                 # PDF ingestion pipeline
├── retrieval.py           # RAG logic (hybrid retrieval + rerank)
├── app.py                 # [Deprecated] Old Streamlit version
├── .env                   # Environment config
├── requirements.txt       # Python dependencies
├── papers/
│   └── ai_thucchien/      # PDF documents directory
└── chroma_db/             # Vector database
```

## Tính năng

✅ Hybrid retrieval (Vector + BM25)
✅ Reranking với Flashrank
✅ Real-time streaming response
✅ LangSmith tracing & debugging
✅ OCR fallback cho PDF rỗng
✅ Summary mode detection
✅ Vietnamese language output
✅ Modern HTML/CSS UI

## Troubleshooting

### Lỗi "No such file or directory: 'chroma_db'"
→ Chạy `python pre.py` lần đầu để khởi tạo dữ liệu

### Lỗi "LANGSMITH_API_KEY not found"
→ Thêm API key vào file `.env` hoặc set biến môi trường

### Response bằng tiếng Trung
→ Đã fix bằng cách thêm instruction rõ ràng về tiếng Việt trong prompt

### OCR không hoạt động
→ Kiểm tra Poppler và Tesseract đã cài đặt và có trong PATH chưa
