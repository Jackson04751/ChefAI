import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy


class RAGChatbot:
    def __init__(self):
        # 1. Đường dẫn file PDF
        pdf_path = os.path.join("data", "VietnameseCooking.pdf")

        # 2. Load file PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # 3. Cắt nhỏ văn bản (Text Splitting)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # --- BƯỚC DEBUG: In ra Terminal để kiểm tra ---
        print(f"DEBUG: Số trang đọc được = {len(documents)}")
        print(f"DEBUG: Số chunks tạo ra = {len(splits)}")

        # 4. Lọc bỏ các chunk rỗng (chỉ giữ lại chunk có nội dung chữ)
        splits = [doc for doc in splits if doc.page_content.strip()]

        if not splits:
            raise ValueError("Không tìm thấy nội dung văn bản hợp lệ trong file PDF!")

        # 5. Khởi tạo Embeddings (Đảm bảo bạn đã khai báo API Key / Model đúng)
        from langchain_huggingface import HuggingFaceEmbeddings

        # Khởi tạo mô hình Embedding miễn phí (chạy trực tiếp trên máy của bạn)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # --- BƯỚC TẠO VECTORSTORE AN TOÀN ---
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )

    def ask(self, question: str) -> str:
        # Tìm 3 đoạn liên quan nhất trong file PDF
        docs = self.vectorstore.similarity_search(question, k=3)

        if not docs:
            return "Không tìm thấy thông tin liên quan trong tài liệu."

        context = "\n\n".join([doc.page_content for doc in docs])
        return f"Dữ liệu tìm được từ sách nấu ăn:\n\n{context}"