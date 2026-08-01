import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


class RAGChatbot:
    def __init__(self):
        #  Đường dẫn file PDF
        pdf_path = os.path.join("data", "VietnameseCooking.pdf")

        # Đọc PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Chia nhỏ văn bản
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        splits = text_splitter.split_documents(documents)

        print(f"DEBUG: Số trang đọc được = {len(documents)}")
        print(f"DEBUG: Số chunks tạo ra = {len(splits)}")

        splits = [doc for doc in splits if doc.page_content.strip()]

        if not splits:
            raise ValueError("Không tìm thấy nội dung trong PDF!")


        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )


        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=embeddings,
            distance_strategy=DistanceStrategy.COSINE,
        )

        self.llm = ChatGoogleGenerativeAI(
            model='gemini-3.5-flash',
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            max_retries=6,
        )
        self.fallback_llm = ChatGoogleGenerativeAI(
            model='gemini-3.1-pro',
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
            max_retries=3,
        )
        self.llm = self.llm.with_fallbacks([self.fallback_llm])

    def ask(self, question: str) -> str:

        docs = self.vectorstore.similarity_search(question, k=3)

        if not docs:
            return "Xin lỗi, mình không tìm thấy thông tin trong tài liệu."

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
        Bạn là ChefAI - trợ lý nấu ăn thân thiện.

        Hãy trả lời tự nhiên bằng tiếng Việt.

        Chỉ sử dụng thông tin trong tài liệu dưới đây.

        Nếu tài liệu không có câu trả lời thì hãy nói:

        "Xin lỗi, mình không tìm thấy thông tin đó trong tài liệu."

        =====================
        TÀI LIỆU

        {context}

        =====================

        CÂU HỎI

        {question}
        """


        response = self.llm.invoke(prompt)
        import json

        
        if hasattr(response, 'content'):
            content = response.content
            # Nếu content là kiểu list (multimodal output), nối các phần text lại
            if isinstance(content, list):
                content_str = "".join(
                    [item.get('text', '') if isinstance(item, dict) else str(item) for item in content]
                )
            else:
                content_str = str(content)

            # Thử giải mã nếu content là một chuỗi JSON
            try:
                data = json.loads(content_str)
                if isinstance(data, dict) and 'text' in data:
                    return data['text']
            except (json.JSONDecodeError, TypeError):
                pass

            return content_str

        # Trường hợp response là một Dictionary
        if isinstance(response, dict):
            if 'text' in response:
                return response['text']
            if 'content' in response:
                return response['content']

        return str(response)

