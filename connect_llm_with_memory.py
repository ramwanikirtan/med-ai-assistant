from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# MultiQueryRetriever not available in this environment; provide a local implementation
from typing import List


class LocalMultiQueryRetriever:
    """Simple local MultiQueryRetriever: uses the LLM to rewrite the query
    into multiple related queries, runs the retriever for each, and
    aggregates unique documents ordered by first-seen relevance.
    """

    def __init__(self, llm, retriever, n_queries: int = 3, rewrite_template: str = None):
        self.llm = llm
        self.retriever = retriever
        self.n_queries = n_queries
        self.rewrite_template = (
            rewrite_template
            or "Generate {n} concise query variations for the question: {question}"
        )

    @classmethod
    def from_llm(cls, llm, retriever, n_queries: int = 3):
        return cls(llm=llm, retriever=retriever, n_queries=n_queries)

    def _generate_rewrites(self, question: str) -> List[str]:
        prompt = self.rewrite_template.format(n=self.n_queries, question=question)
        # ask the LLM to generate n queries separated by newlines
        resp = self.llm.invoke(prompt)
        text = resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
        # split lines and keep first n non-empty lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) >= self.n_queries:
            return lines[: self.n_queries]
        # fallback: naive token-based splits
        joined = " ".join(lines)
        parts = [joined[i:: self.n_queries] for i in range(self.n_queries)]
        return [p.strip() for p in parts if p.strip()][: self.n_queries]

    def get_relevant_documents(self, question: str):
        rewrites = self._generate_rewrites(question)
        seen = set()
        results = []
        for q in rewrites:
            # Support multiple retriever APIs across LangChain versions
            retr = self.retriever
            if hasattr(retr, "get_relevant_documents"):
                docs = retr.get_relevant_documents(q)
            elif hasattr(retr, "_get_relevant_documents"):
                try:
                    docs = retr._get_relevant_documents(q, run_manager=None)
                except TypeError:
                    docs = retr._get_relevant_documents(q)
            else:
                # fallback to underlying vectorstore similarity_search
                vs = getattr(retr, "vectorstore", None)
                if vs and hasattr(vs, "similarity_search"):
                    docs = vs.similarity_search(q, k=self.n_queries)
                else:
                    raise RuntimeError("Retriever has no compatible retrieval method")
            for d in docs:
                key = getattr(d, "id", None) or getattr(d, "metadata", {}).get("source") or d.page_content[:200]
                if key not in seen:
                    seen.add(key)
                    results.append(d)
        return results

# 1. Setup LLM
load_dotenv()

def load_llm():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, max_tokens=1000)
    return llm

llm = load_llm()

# 2. Connect LLM with Memory and create a pipeline
DB_FAISS_PATH = "vectorstore/db_faiss"

prompt_template = """
You are a retrieval-augmented medical information assistant operating in strict evidence-based mode.

You must answer using ONLY the information provided below.

==============================
CONTEXT:
{context}
==============================

==============================
QUESTION:
{question}
==============================

MANDATORY RULES:
- Use ONLY the provided CONTEXT.
- Do NOT use prior knowledge.
- Do NOT guess or infer beyond what is written.
- If the answer is not explicitly supported, say:
  "The provided documents do not contain sufficient information to answer this."
- Do NOT provide diagnosis or personalized treatment plans.
- Provide general educational information only.

SAFETY:
If the question involves severe symptoms, emergencies, medication dosage, pregnancy, or pediatric cases, include:
"This information is for educational purposes only and is not a substitute for professional medical advice."

RESPONSE REQUIREMENTS:
1. Extract relevant facts strictly from the CONTEXT.
2. Construct a clear, structured answer.
3. Quote short supporting phrases when helpful.
4. State limitations if the context is incomplete.
5. Provide a confidence level: High | Moderate | Low.

OUTPUT FORMAT:

Clinical Answer:
<grounded answer>

Supporting Evidence:
- "<quoted phrase>"
- "<quoted phrase>"

Limitations:
<state missing information if applicable>

Confidence:
High | Moderate | Low
"""

custom_prompt = PromptTemplate.from_template(prompt_template)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
# use local implementation
multiquery_retriever = LocalMultiQueryRetriever.from_llm(llm=llm, retriever=db.as_retriever(search_kwargs={"k": 3}))

def answer_question(question):
    docs = multiquery_retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = custom_prompt.format(context=context, question=question)
    response = llm.invoke(prompt)
    # Extract the content from the AIMessage object
    if hasattr(response, 'content'):
        answer = response.content
    else:
        answer = str(response)

    # Add detailed source information to the output
    sources = "\n".join([
        f"- Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}"
        for doc in docs
    ])

    # Format the final output
    formatted_response = f"{answer}\n\nSources:\n{sources}"
    return formatted_response

if __name__ == "__main__":
    query = input("Enter your question: ")
    result = answer_question(query)
    print(result)