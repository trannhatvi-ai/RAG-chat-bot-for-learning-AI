import os
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm # Thêm thư viện hiển thị thanh tiến trình

# 1. CẤU HÌNH POPPLER
poppler_path = r"D:\poppler\Library\bin"
os.environ["PATH"] += os.pathsep + poppler_path

def main():
    print("1. Đang tải tài liệu PDF...")
    loader = DirectoryLoader(
        path="./papers",
        glob="**/test/*.pdf",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
        use_multithreading=True,
    )
    docs = loader.load()
    print(f"Đã tải {len(docs)} tài liệu.")

    # 2. CHIA NHỎ VĂN BẢN (CHUNKING)
    MARKDOWN_SEPARATOR = [
        "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
    ]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        add_start_index=True, 
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATOR
    )
    
    splits = text_splitter.split_documents(docs)
    print(f"2. Đã chia nhỏ thành {len(splits)} chunks.")

    # 3. KHỞI TẠO OLLAMA EMBEDDINGS
    print("3. Đang khởi tạo Ollama Embeddings...")
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text" 
    )

    # 4. LƯU VÀO VECTOR DATABASE CÓ THANH TIẾN TRÌNH (TDM)
    print("4. Đang nhúng dữ liệu và lưu vào ChromaDB...")
    persist_directory = "./chroma_db"
    
    # Khởi tạo database rỗng trước
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    
    # Kích thước mỗi lô (batch). Có thể tăng/giảm tùy sức mạnh máy tính
    batch_size = 100 
    
    # Vòng lặp kết hợp tqdm để tạo progress bar
    for i in tqdm(range(0, len(splits), batch_size), desc="Tiến trình Embedding", unit="batch"):
        batch = splits[i : i + batch_size]
        vectorstore.add_documents(batch)
    
    print(f"\n✅ Hoàn tất! Toàn bộ dữ liệu đã được nhúng và lưu tại thư mục: {persist_directory}")

if __name__ == "__main__":
    main()