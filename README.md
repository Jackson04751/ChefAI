# 🍳 ChefAI - RAG Culinary Chatbot Application

ChefAI là một trợ lý nấu ăn thông minh được xây dựng trên kiến trúc **RAG (Retrieval-Augmented Generation)**. Ứng dụng hỗ trợ tìm kiếm công thức nấu ăn, gợi ý kỹ thuật chế biến và lên thực đơn cá nhân hóa dựa trên dữ liệu ẩm thực tùy chỉnh.

---

## 🚀 Tính năng chính

- 📖 **Truy xuất công thức chính xác:** Sử dụng RAG để tìm kiếm ngữ nghĩa trong tập dữ liệu món ăn tùy chỉnh.
- 🥗 **Gợi ý thực đơn cá nhân hóa:** Đưa ra thực đơn dựa trên yêu cầu, sở thích hoặc nguyên liệu người dùng nhập vào.
- ⚡ **Tìm kiếm ngữ nghĩa hiệu quả:** Lưu trữ vector embedding với ChromaDB giúp truy xuất ngữ cảnh nhanh chóng.
- 🎯 **Prompt được tối ưu:** Hệ thống prompt đảm bảo câu trả lời an toàn, đúng trọng tâm và mượt mà.

---

## 🛠️ Công nghệ sử dụng

- **Ngôn ngữ:** Python 3.12.10
- **Framework AI:** LangChain
- **Vector Database:** ChromaDB
- **LLM & Embeddings:** OpenAI API / Hugging Face Embeddings
- **Công cụ phát triển:** PyCharm, Git/GitHub

---

## 📦 Hướng dẫn cài đặt & Chạy ứng dụng
Bước 1: Clone (tải) project về máy
git clone https://github.com/<username-cua-ban>/ChefAI.git
cd ChefAI
Bước 2: Cài đặt các thư viện cần thiết
pip install -r requirements.txt
Bước 3: Cấu hình API Key (Biến môi trường)
GOOGLE_API_KEY=your_openai_api_key_here
Bước 4: Chạy dự án
streamlit run app.py
```bash
git clone [https://github.com/](https://github.com/)<username-cua-ban>/ChefAI.git
cd ChefAI
