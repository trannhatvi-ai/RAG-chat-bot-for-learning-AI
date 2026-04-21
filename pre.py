import shutil
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


def load_pdf_documents(data_dir: Path):
    """Load all PDFs recursively from papers directory."""
    loader = DirectoryLoader(
        path=str(data_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def chunk_documents(documents):
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
    )
    return splitter.split_documents(documents)


def rebuild_chroma(chunks, persist_dir: Path, batch_size: int = 100):
    """Embed chunks with Ollama and persist vectors into ChromaDB."""
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=str(persist_dir))

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding", unit="batch"):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)


def main():
    project_root = Path(__file__).resolve().parent
    papers_dir = project_root / "papers" / "ai_thucchien"
    chroma_dir = project_root / "chroma_db"

    if not papers_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc du lieu: {papers_dir}")

    print("1) Dang tai PDF trong papers/ai_thucchien...")
    docs = load_pdf_documents(papers_dir)
    if not docs:
        raise ValueError("Khong tim thay file PDF nao trong thu muc papers.")
    print(f"   Da tai {len(docs)} trang tai lieu.")

    print("2) Dang chunking...")
    chunks = chunk_documents(docs)
    print(f"   Da tao {len(chunks)} chunks.")

    print("3) Dang embedding va luu vao ChromaDB...")
    rebuild_chroma(chunks, chroma_dir, batch_size=100)
    print(f"Hoan tat. Du lieu da duoc luu tai: {chroma_dir}")


if __name__ == "__main__":
    main()
