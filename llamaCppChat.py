import streamlit as st
from llama_cpp import Llama

st.set_page_config(page_title="Prag 🦙💬 Llama 2 Chatbot DEMO")

with st.sidebar:
    st.title('Prag 🦙💬 Llama 2 Chatbot DEMO')
    st.write('This chatbot uses quantized (4bit) models run on a local machine')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Model Name 1', 'Model Name 2'], key='selected_model')    
    user_temperature = st.sidebar.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    user_top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('max_length', min_value=32, max_value=128, value=120, step=8)
    if selected_model == 'Model Name 1':
        llm = Llama(model_path="Model Path 1",
            n_ctx=512,
            n_batch=128)
    else:
        llm = Llama(model_path="Model Path 2",
            n_ctx=512,
            n_batch=128)
    
if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hello I am Alfred🤵🏻! How may I assist you today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def generateLLama2Output(llamaModel, user_prompt):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'.'"
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = llm(string_dialogue,
             max_tokens=max_length,
             echo=False,
             temperature=user_temperature,
             top_p=user_top_p)
    return output['choices'][0]['text']


if prompt := st.chat_input("Type your message here:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generateLLama2Output(llm, prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
