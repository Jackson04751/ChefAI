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
            model ="gemini-3.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,

        )

    def ask(self, question: str) -> str:

        docs = self.vectorstore.similarity_search(question, k=3)

        if not docs:
            return "Xin lỗi, mình không tìm thấy thông tin trong tài liệu."

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
Here is the English version of your prompt:

You are ChefAI – a friendly cooking assistant.

Please reply in natural English as if you are conversing with the user.

Only use the information provided in the document below.

If the document does not contain the answer, please say:
"Sorry, I couldn't find that information in the book."

=====================
TÀI LIỆU

{context}

=====================

CÂU HỎI

{question}
"""

        response = self.llm.invoke(prompt)

        return str(response.content)