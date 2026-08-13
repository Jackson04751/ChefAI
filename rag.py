import os
import json

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


class RAGChatbot:

    # =====================================================
    # KHỞI TẠO CHATBOT
    # =====================================================

    def __init__(self):

        # -------------------------------------------------
        # 1. Đường dẫn PDF
        # -------------------------------------------------

        pdf_path = os.path.join(
            "data",
            "VietnameseCooking.pdf"
        )

        print("DEBUG: Đang đọc PDF...")

        # -------------------------------------------------
        # 2. Đọc PDF
        # -------------------------------------------------

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        print(
            f"DEBUG: Số trang đọc được = {len(documents)}"
        )

        # -------------------------------------------------
        # 3. Chia nhỏ văn bản
        # -------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ]
        )

        splits = text_splitter.split_documents(
            documents
        )

        # Loại bỏ chunk rỗng
        splits = [
            doc
            for doc in splits
            if doc.page_content.strip()
        ]

        print(
            f"DEBUG: Số chunks tạo ra = {len(splits)}"
        )

        if not splits:
            raise ValueError(
                "Không tìm thấy nội dung trong PDF!"
            )

        # -------------------------------------------------
        # 4. Load Embedding Model
        # -------------------------------------------------

        print(
            "DEBUG: Bắt đầu load embedding model..."
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",

            # Ép chạy CPU
            model_kwargs={
                "device": "cpu"
            },

            # Chuẩn hóa vector
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        print(
            "DEBUG: Embedding model đã load xong!"
        )

        # -------------------------------------------------
        # 5. Tạo FAISS Vector Database
        # -------------------------------------------------

        print(
            "DEBUG: Bắt đầu tạo FAISS index..."
        )

        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )

        print(
            "DEBUG: FAISS index đã tạo xong!"
        )

        # -------------------------------------------------
        # 6. Lấy GOOGLE API KEY
        # -------------------------------------------------

        google_api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not google_api_key:

            raise ValueError(
                "Không tìm thấy GOOGLE_API_KEY trong file .env"
            )

        # -------------------------------------------------
        # 7. Khởi tạo Gemini
        # -------------------------------------------------

        print(
            "DEBUG: Đang khởi tạo Gemini..."
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            google_api_key=google_api_key,
            temperature=0.3,
            max_retries=6
        )

        # -------------------------------------------------
        # 8. Gemini dự phòng
        # -------------------------------------------------

        self.fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-pro",
            google_api_key=google_api_key,
            temperature=0.3,
            max_retries=3
        )

        # -------------------------------------------------
        # 9. Thiết lập fallback
        # -------------------------------------------------

        self.llm = self.llm.with_fallbacks(
            [self.fallback_llm]
        )

        print(
            "DEBUG: Gemini đã khởi tạo xong!"
        )

        print(
            "DEBUG: ChefAI đã khởi động thành công!"
        )

    # =====================================================
    # HÀM ASK
    # =====================================================

    def ask(self, question: str) -> str:

        # -------------------------------------------------
        # 1. Tìm tài liệu liên quan bằng FAISS
        # -------------------------------------------------

        docs = self.vectorstore.similarity_search(
            question,
            k=3
        )

        if not docs:

            return (
                "Xin lỗi, mình không tìm thấy "
                "thông tin trong tài liệu."
            )

        # -------------------------------------------------
        # 2. Tạo context
        # -------------------------------------------------

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        # -------------------------------------------------
        # 3. Tạo prompt
        # -------------------------------------------------

        prompt = f"""
Bạn là ChefAI - trợ lý nấu ăn thân thiện.

Hãy trả lời tự nhiên bằng tiếng Việt.

QUY TẮC:

- Chỉ sử dụng thông tin trong tài liệu.
- Không tự bịa thông tin.
- Nếu tài liệu không có câu trả lời thì hãy nói:
"Xin lỗi, mình không tìm thấy thông tin đó trong tài liệu."
- Trả lời rõ ràng và dễ hiểu.
- Nếu người dùng hỏi về công thức,
  hãy trình bày nguyên liệu và cách làm
  nếu tài liệu có.

=====================
TÀI LIỆU
=====================

{context}

=====================
CÂU HỎI
=====================

{question}
"""

        # -------------------------------------------------
        # 4. Gọi Gemini
        # -------------------------------------------------

        response = self.llm.invoke(
            prompt
        )

        # -------------------------------------------------
        # 5. Xử lý response
        # -------------------------------------------------

        if hasattr(response, "content"):

            content = response.content

            # Gemini có thể trả về list
            if isinstance(content, list):

                content_str = "".join(
                    [
                        item.get("text", "")
                        if isinstance(item, dict)
                        else str(item)
                        for item in content
                    ]
                )

            else:

                content_str = str(content)

            # -------------------------------------------------
            # 6. Kiểm tra JSON
            # -------------------------------------------------

            try:

                data = json.loads(
                    content_str
                )

                if (
                    isinstance(data, dict)
                    and "text" in data
                ):

                    return str(
                        data["text"]
                    )

            except (
                json.JSONDecodeError,
                TypeError
            ):

                pass

            return content_str

        # -------------------------------------------------
        # 7. Response dạng dictionary
        # -------------------------------------------------

        if isinstance(response, dict):

            if "text" in response:

                return str(
                    response["text"]
                )

            if "content" in response:

                return str(
                    response["content"]
                )

        # -------------------------------------------------
        # 8. Trường hợp khác
        # -------------------------------------------------

        return str(response)