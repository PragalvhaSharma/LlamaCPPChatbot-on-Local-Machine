# Import necessary libraries
import streamlit as st
from ctransformers import AutoModelForCausalLM

# Set the configuration for the Streamlit page, including the title
st.set_page_config(page_title="Prags 🦙💬 Llama 2 Chatbot")

# Define a function to load the chat model. It is cached to improve performance.
@st.cache_resource()
def ChatModel(temperature, top_p):
    # Load a specific model based on user selection
    if selected_model == 'llama2-7b-chat (4bit)':
        return AutoModelForCausalLM.from_pretrained(
            '/Users/pragalvhasharma/Downloads/Prag GO to Documents/Ulink ProtoType/i8/Final Chatbot/llama-2-7b-chat.Q4_K_M.gguf', 
            model_type='llama',
            temperature=temperature, 
            top_p=top_p)
    else:
        return AutoModelForCausalLM.from_pretrained(
            '/Users/pragalvhasharma/Downloads/Prag GO to Documents/Ulink ProtoType/i8/Final Chatbot/llama-pro-8b-instruct.Q4_K_M.gguf', 
            model_type='llama',
            temperature=temperature, 
            top_p=top_p)

# Set up the sidebar for user inputs
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.subheader('Models and parameters')
    # Sliders for model parameters
    temperature = st.sidebar.slider('temperature', 0.01, 2.0, 0.1, 0.01)
    top_p = st.sidebar.slider('top_p', 0.01, 1.0, 0.9, 0.01)
    max_length = st.sidebar.slider('max_length', 64, 4096, 512, 8)
    # Dropdown to select the model
    selected_model = st.sidebar.selectbox(
        'Choose a Llama2 model', 
        ['llama2-7b-chat (4bit)', 'llama-pro-8b-instruct (4bit)'], 
        key='selected_model')
    # Load the selected chat model
    chat_model = ChatModel(temperature, top_p)

# Initialize session state for storing chat messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Hello I am Alfred! How may I assist you today?"}]

# Display the chat messages in the main window
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Define a function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": " How may I assist you today?"}]
# Button in sidebar to clear chat history
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function to generate a response from LLaMA2 model
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Alfred'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
    output = chat_model(f"prompt {string_dialogue} {prompt_input} Assistant: ")
    return output

# Capture user input through chat and append it to session state
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate and display a response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
