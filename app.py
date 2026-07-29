import streamlit as st
from rag import RAGChatbot

# Cấu hình trang
st.set_page_config(page_title="ChefAI - Món Ngon Việt Nam", page_icon="🤖")
st.title("🤖 ChefAI Assistant")

#  Khởi tạo RAGChatbot trong session_state
if "bot" not in st.session_state:
    try:
        st.session_state.bot = RAGChatbot()
    except Exception as e:
        st.error(f"Khởi tạo Bot thất bại: {e}")
        st.session_state.bot = None

#  Khởi tạo lịch sử tin nhắn
if "messages" not in st.session_state:
    st.session_state.messages = []

#  Hiển thị lại toàn bộ lịch sử chat cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Xử lý khi người dùng nhập câu hỏi mới
question = st.chat_input("Hỏi ChefAI điều gì đó...")

if question:
    # Hiển thị câu hỏi của user lên giao diện
    with st.chat_message("user"):
        st.write(question)

    # Lưu câu hỏi vào lịch sử
    st.session_state.messages.append({"role": "user", "content": question})

    # Gọi Bot trả lời
    if st.session_state.bot is not None:
        with st.chat_message("assistant"):
            with st.spinner("ChefAI đang suy nghĩ..."):
                try:


                    answer = st.session_state.bot.ask(str(question))
                    st.write(answer)

                    # Lưu câu trả lời vào lịch sử
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Lỗi khi xử lý câu hỏi: {e}")
    else:
        st.error("Chưa khởi tạo được Bot! Vui lòng kiểm tra lại cấu hình trong rag.py.")