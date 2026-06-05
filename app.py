import os
import streamlit as st
import pypdf
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Interview Trainer Agent", layout="wide")

WATSONX_APIKEY = os.environ.get("WATSONX_APIKEY", "YOUR_IBM_WATSONX_APIKEY")
PROJECT_ID = os.environ.get("PROJECT_ID", "YOUR_IBM_PROJECT_ID")

@st.cache_resource
def load_granite_model():
    if WATSONX_APIKEY == "YOUR_IBM_WATSONX_APIKEY" or PROJECT_ID == "YOUR_IBM_PROJECT_ID":
        return None
    try:
        credentials = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": WATSONX_APIKEY
        }
        params = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 600,
            GenParams.TEMPERATURE: 0.2
        }
        return Model(
            model_id="ibm/granite-4-0-8b-instruct",
            credentials=credentials,
            params=params,
            project_id=PROJECT_ID
        )
    except:
        return None

granite_model = load_granite_model()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

@st.cache_resource
def setup_rag_knowledge_base():
    documents = [
        "DSA includes arrays, stacks, trees, graphs, DP",
        "System design includes scalability, microservices, load balancing",
        "HR interviews use STAR method",
        "DBMS and OS include scheduling, SQL, deadlocks"
    ]
    embeddings = embedding_model.encode(documents)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, documents

rag_index, rag_docs = setup_rag_knowledge_base()

def query_rag(query_text):
    query_vector = embedding_model.encode([query_text]).astype("float32")
    _, indices = rag_index.search(query_vector, k=1)
    return rag_docs[indices[0][0]]

def call_granite_agent(prompt):
    if granite_model:
        try:
            return granite_model.generate_text(prompt=prompt)
        except Exception as e:
            return f"Error: {str(e)}"
    if "Resume Analysis" in prompt:
        return "Skills: Python, Java, React. Missing: System Design, DevOps, Cloud."
    if "Question Generator" in prompt:
        return "1. DSA question 2. System design question 3. DBMS question"
    if "Evaluation Agent" in prompt:
        return "Score: 8/10. Improve threading and synchronization."
    return "Focus on DSA, System Design, and Mock Interviews"

st.title("AI Interview Trainer Agent")
st.subheader("Multi-Agent Placement Preparation System")
st.markdown("---")

if not granite_model:
    st.warning("Running in local mode. Add IBM credentials for full functionality.")

col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.header("Configuration")
    target_comp = st.selectbox(
        "Target Company",
        ["General Tech Core", "Google", "Amazon", "TCS", "Accenture", "Infosys"]
    )
    interview_track = st.radio(
        "Interview Track",
        ["Technical", "HR"]
    )
    st.markdown("---")
    st.checkbox("IBM Granite Model", value=True, disabled=True)
    st.checkbox("LangFlow System", value=True, disabled=True)
    st.checkbox("FAISS RAG", value=True, disabled=True)
    st.checkbox("Multi-Agent System", value=True, disabled=True)

with col_main:
    tab1, tab2, tab3, tab4 = st.tabs([
        "Resume Analysis",
        "Question Generator",
        "Evaluation",
        "Recommendation"
    ])

    with tab1:
        st.markdown("Resume Analysis")
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])
        fallback = st.text_area("Or paste resume", value="Python, Java, ML")
        if st.button("Run Resume Agent"):
            if uploaded_file:
                reader = pypdf.PdfReader(uploaded_file)
                text = "".join([p.extract_text() for p in reader.pages])
            else:
                text = fallback
            prompt = f"Resume Analysis: {text}"
            result = call_granite_agent(prompt)
            st.session_state["resume"] = result
            st.markdown(result)

    with tab2:
        st.markdown("Question Generator")
        skills = st.text_area(
            "Skills",
            value=st.session_state.get("resume", "Python, Java, ML")
        )
        if st.button("Run Question Agent"):
            rag_context = query_rag(skills)
            prompt = f"Question Generator\nContext: {rag_context}\nSkills: {skills}\nCompany: {target_comp}\nTrack: {interview_track}"
            result = call_granite_agent(prompt)
            st.session_state["questions"] = result
            st.markdown(result)

    with tab3:
        st.markdown("Evaluation")
        question = st.text_area("Question")
        answer = st.text_area("Answer")
        if st.button("Run Evaluation Agent"):
            if answer:
                prompt = f"Evaluation Agent\nQuestion: {question}\nAnswer: {answer}"
                result = call_granite_agent(prompt)
                st.session_state["evaluation"] = result
                st.markdown(result)
            else:
                st.error("Enter answer")

    with tab4:
        st.markdown("Recommendation")
        eval_text = st.text_area(
            "Evaluation",
            value=st.session_state.get("evaluation", "Needs improvement")
        )
        if st.button("Run Recommendation Agent"):
            prompt = f"Recommendation: {eval_text}"
            result = call_granite_agent(prompt)
            st.markdown(result)