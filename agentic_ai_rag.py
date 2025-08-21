import os
import glob
from langdetect import detect
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, LLM, Process
from crewai_tools import SerperDevTool

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

#  Web Crew Setup 
llm_for_agent = LLM(model="gemini/gemini-1.5-flash", temperature=0.7, api_key=GEMINI_API_KEY)
web_search_tool = SerperDevTool(api_key=SERPER_API_KEY)

def run_web_agent(query):
    print("Local answer insufficient")
    print("Creating Web search crew agent...")
    agent = Agent(
        role="Information Fetcher",
        goal="Answer questions using web search if internal data is not sufficient.",
        backstory="You specialize in online searches when internal documents are not enough.",
        tools=[web_search_tool],
        llm=llm_for_agent,
        verbose=True
    )

    task = Task(
        description="Use the web to answer this question: {inputs}",
        expected_output="A detailed and accurate answer to the question.",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff(inputs={"inputs": query})
    return str(result)

#  Document QA Agent Setup 
class QnAAgent:
    def __init__(self, file_paths):
        self.vectorstore = self._create_vector_store(file_paths)
        self.local_llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.7)

    def _create_vector_store(self, file_paths):
        docs = []
        for file in file_paths:
            if file.endswith(".pdf"):
                docs.extend(PyPDFLoader(file).load())
            elif file.endswith(".docx"):
                docs.extend(Docx2txtLoader(file).load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_documents(chunks, embeddings)

    def answer_query(self, query):
        lang = detect(query)
        retriever = self.vectorstore.as_retriever()

        chain = RetrievalQA.from_chain_type(
            llm=self.local_llm,
            retriever=retriever,
            return_source_documents=True
        )

        result_dict = chain.invoke({"query": query})
        result = result_dict.get("result", "").strip()

        if result and len(result) > 20:
            if lang == "es":
                try:
                    translated = self.local_llm.invoke(f"Traduce esto al español:\n\n{result}")
                    return translated.strip()
                except Exception as e:
                    return f"[ERROR AL TRADUCIR]: {e}\n\n{result}"
            return result

        # Fallback to web search using CrewAI
        web_result = run_web_agent(query)

        if lang == "es":
            try:
                translated_web_result = self.local_llm.invoke(f"Traduce esto al español:\n\n{web_result}")
                return translated_web_result.content.strip()
            except Exception as e:
                return f"[ERROR AL TRADUCIR RESULTADO WEB]: {e}\n\n{web_result}"

        return web_result.strip()

#  Main Execution 
if __name__ == "__main__":
    INPUT_FILES = glob.glob("docs/*.pdf") + glob.glob("docs/*.docx")
    qna = QnAAgent(INPUT_FILES)

    print("Type 'quit' to exit.")
    while True:
        query = input("\nAsk your question (English or Spanish): ").strip()
        if not query:
            print("Please enter a valid question.")
            continue
        if query.lower() == "quit":
            print("Goodbye!")
            break

        answer = qna.answer_query(query)
        print("\nAnswer:", answer)
