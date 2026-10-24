import streamlit as st
from hr_assistant.pipeline import ask, build_hr_assistant

st.title("HR Policy Assistant")


@st.cache_resource
def get_agent():
    return build_hr_assistant()


agent = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

if question := st.chat_input("Ask an HR policy question"):
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask(agent, question)
        st.write(answer)
    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
