from rag import RAGChatbot

bot = RAGChatbot()

while True:

    question = input("Bạn: ")

    if question.lower() == "exit":
        break

    answer = bot.ask(question)

    print()
    print("AI:", answer)
    print()