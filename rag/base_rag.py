from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import InMemoryVectorStore
from dotenv import load_dotenv

from rag.utils import hash_text, load_documents, split_documents

load_dotenv()


class BaseRag:
    def __init__(self, folder: str = "documents"):
        self.folder = folder
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
        self.vectordb = InMemoryVectorStore(self.embeddings)
        self.qa_chain = None

    def sync_documents(self):
        documents = load_documents(self.folder)
        chunks = split_documents(documents=documents)

        newly_added_chunk = []
        existing_chunk_hashes = {
            m.metadata.get("chunk_id") for m in self.vectordb.store.values()
        }
        for chunk in chunks:
            chunk_hash = hash_text(text=chunk.page_content)
            if chunk_hash not in existing_chunk_hashes:
                chunk.metadata.update(
                    {
                        "chunk_id": chunk_hash,
                        "source": chunk.metadata.get("source", "unknown"),
                    }
                )
                newly_added_chunk.append(chunk)

        if newly_added_chunk is not None:
            self.vectordb.add_documents(newly_added_chunk)
            print(f"Added {len(newly_added_chunk)} new chunks to the vector store.")

    def chat(self):
        if not self.qa_chain:
            self.qa_chain = self._build_chain()

        chat_history = []
        print(
            "You can start chatting with the assistant now! Type 'exit' or 'quit' to stop."
        )
        while True:
            query = input("User: ")
            if query.lower() in ["exit", "quit"]:
                break
            if query.lower() in ["clear", "cls"]:
                chat_history = []
                print("Chat history cleared.")
                continue
            result = self.qa_chain.invoke(
                {"input": query, "chat_history": chat_history}
            )
            answer = result["answer"]
            print("Assistant:", answer)

            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=answer))

    def _build_chain(self):
        retriever = self.get_vectordb_as_retriever()

        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "Given the user query and the conversation history, rephrase the query to be more specific and relevant to the context. Do NOT answer the query, just rephrase it.",
                    ),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            ),
        )
        stuff_doc_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a helpful assistant, Please answer the user's query based on the context. Keep the answer concise and if the answer cannot be found in the context, Just say that you don't have enough information to answer that question \n\nContext: {context}",
                    ),
                    ("human", "{input}"),
                ]
            ),
        )
        return create_retrieval_chain(history_aware_retriever, stuff_doc_chain)

    def get_vectordb_as_retriever(self):
        return self.vectordb.as_retriever()


if __name__ == "__main__":
    rag = BaseRag()

    rag.sync_documents()

    rag.chat()
