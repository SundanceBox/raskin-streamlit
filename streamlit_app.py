import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ─── Set your OpenAI key directly ───────────────────────────────────────────
os.environ["OPENAI_API_KEY"] = "sk‑proj‑AFKzFPYBje7EnWChyGsMT3BlbkFJ9zUKVY5reYvG8P8xtzjK"

# 1) Vector store
emb = OpenAIEmbeddings(model="text-embedding-3-small")
db  = Chroma(persist_directory="raskin_chroma", embedding_function=emb)
retriever = db.as_retriever(search_kwargs={"k": 4})

# 2) Summarizer
summ_llm   = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, max_tokens=1024)
splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
summ_chain = load_summarize_chain(summ_llm, chain_type="map_reduce")

def summarize(text: str) -> str:
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    return summ_chain.run(docs).strip()

# 3) RASCAN rewrite
ras_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, max_tokens=400)
TEMPLATE = """You are RASCAN, Rep. Jamie Raskin’s rhetorical cognition module.
Reference style:
{style_examples}

Task: Rewrite the INPUT in Raskin voice as a {output_format} of ≤{char_limit} characters.
=====
INPUT:
{input_text}
=====
Rewrite now:
"""
prompt = PromptTemplate(
    template=TEMPLATE,
    input_variables=["style_examples","output_format","char_limit","input_text"]
)

def rascan_rewrite(raw: str, fmt: str, limit: int) -> str:
    exemplars = "\n---\n".join(
        d.page_content for d in retriever.get_relevant_documents(raw)
    )
    chain = LLMChain(llm=ras_llm, prompt=prompt)
    out   = chain.run(
        style_examples=exemplars,
        output_format=fmt,
        char_limit=limit,
        input_text=raw
    ).strip()
    while len(out) > limit:
        out = chain.run(
            style_examples=exemplars,
            output_format=fmt,
            char_limit=limit,
            input_text=out
        ).strip()
    return out

# ─── Streamlit UI ─────────────────────────────────────────────────────────
st.title("RASCAN Content Generator")

mode = st.radio("Choose output type", ("270‑char tweet", "2‑sentence caption"))
raw  = st.text_area("Paste your article excerpt or transcript here")

if st.button("Generate"):
    if mode == "270‑char tweet":
        summary = summarize(raw)
        result  = rascan_rewrite(summary, fmt="tweet", limit=270)
    else:
        summary = summarize(raw)
        result  = rascan_rewrite(summary, fmt="caption", limit=300)
    st.markdown("**Result:**")
    st.write(result)
