import streamlit as st
import os
import base64

# Import hàm khởi tạo từ file retrieval.py của bạn
from retrieval import get_rag_chain

# ================= 1. CẤU HÌNH TRANG =================
st.set_page_config(page_title="RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= 2. KHỞI TẠO AI TỪ FILE RETRIEVAL.PY =================
@st.cache_resource(show_spinner="Đang khởi tạo hệ thống AI...")
def load_chain():
    # Gọi hàm từ file retrieval.py
    return get_rag_chain()

rag_chain = load_chain()

# ================= 3. HÀM XỬ LÝ POPUP TÀI LIỆU =================
@st.dialog("📄 Chi tiết tài liệu", width="large")
def show_pdf_popup(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi mở file: {e}")

# ================= 4. GIAO DIỆN NAVBAR BÊN TRÁI =================
with st.sidebar:
    st.title("⚙️ Điều khiển")
    
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("### 🕒 Lịch sử Chat")
    st.button("💬 Tổng quan dự án", use_container_width=True)
    st.button("💬 Hỏi đáp quy trình", use_container_width=True)

    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🎧 Hỗ trợ")
    st.markdown("📧 support@yourdomain.com")
    st.markdown("---")

    st.markdown("### 📂 Nguồn tài liệu")
    paper_dir = "./papers/test"
    
    if os.path.exists(paper_dir):
        files = [f for f in os.listdir(paper_dir) if f.endswith('.pdf')]
        if files:
            for file in files:
                if st.button(f"📄 {file}", use_container_width=True, help="Click để xem nội dung"):
                    file_path = os.path.join(paper_dir, file)
                    show_pdf_popup(file_path)
        else:
            st.warning("Thư mục trống.")
    else:
        st.error("Chưa tạo thư mục ./papers")

# ================= 5. GIAO DIỆN CHATBOT CHÍNH =================
st.title("🤖 RAG Knowledge Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hỏi tôi bất cứ điều gì về tài liệu..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # GỌI STREAMING TỪ RAG CHAIN ĐÃ IMPORT
            response_stream = rag_chain.stream(prompt)
            full_response = st.write_stream(response_stream)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Lỗi khi gọi mô hình: {e}")