
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import warnings
warnings.filterwarnings("ignore")

CHROMA_PATH = "chroma_vector"
embedding_function = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

PROMPT_TEMPLATE_RAG = """
Answer the question based only on the following context: {context}
---
Answer the question based on the above context: {question}
"""
PROMPT_TEMPLATE_LLM = """
Answer the question based on trained model: {user_question}
"""

def get_reponse(query_text, chat_history):
    try:

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        model = Ollama(model="llama2")

        # if no answer found in RAG DB then answered by native llm
        if len(results) == 0 or results[0][1] < 0.4:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_LLM)
            chain = prompt_template | model | StrOutputParser()
            return chain.invoke({"user_question": query_text})
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_RAG)
            prompt = prompt_template.format(context=context_text, question=query_text)

            response_text = model.invoke(prompt)
            sources = [doc.metadata.get("source", None) for doc, _score in results]

            formatted_response = f"Response: {response_text} RAG Context - Sources: {sources}"

            return (formatted_response)
    except Exception as e:
        print("Error: ",e)

def main():
    load_dotenv()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    st.set_page_config(page_title="GenAI RAG", page_icon=':coffee:')
    st.title ("GenAI RAG")
    try:
        #Chatting
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            else:
                 with st.chat_message("AI"):
                    st.markdown(message.content)

        query_text = st.chat_input("Type your question here...")
        if query_text is not None and query_text !="":
            st.session_state.chat_history.append(HumanMessage(query_text))

            with st.chat_message("Human"):
                st.markdown(query_text)

            with st.chat_message("AI"):
                ai_text = get_reponse(query_text,st.session_state.chat_history)
                st.markdown(ai_text)
            st.session_state.chat_history.append(AIMessage(ai_text))
    except Exception as e:
        print("Error: ",e)

if __name__ == "__main__":
    main()
