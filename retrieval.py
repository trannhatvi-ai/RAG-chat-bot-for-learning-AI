from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

def get_rag_chain():
    template = (
        "You are a strict, citation-focused assistant for a private knowledge base. \n"
        "RULES:\n"
        "1. Only use information from the provided context.\n"
        "2. If you don't know the answer, say so.\n"
        "3. Always include citations for any information you provide.\n"
        "4. Use the following format for citations: [source_name](source_url)\n"
        "5. Do not include any information that is not in the provided context.\n"
        "6. Be concise and to the point.\n"
        "7. Do not include any personal opinions or assumptions.\n"
        "8. Always prioritize accuracy and relevance over completeness.\n"
        "9. If the context contains conflicting information, present all sides and cite the sources.\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
    )

    # 1. Khởi tạo Embeddings & VectorStore
    print("Đang khởi tạo Ollama Embeddings và ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    # 2. Cấu hình Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.2} 
    ) 

    # 3. Thiết lập Prompt Template
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Hàm Format Docs
    def format_docs(docs):
        return "\n\n".join([f"Source: {d.metadata.get('source', 'N/A')}\nContent: {d.page_content}" for d in docs])

    # 5. Khởi tạo LLM (Qwen 2.5)
    llm = ChatOllama(
        model="qwen2.5", 
        temperature=0
    )

    # 6. Tạo Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Xóa bỏ phần test bằng input() ở cuối file cũ đi để tránh bị chạy lúc import