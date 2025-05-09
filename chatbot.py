import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Streamlit app
st.title('OpenAI Chatbot')

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Accept user input
if prompt := st.chat_input('What is up?'):
    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Get assistant response
    with st.spinner('Thinking...'):
        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[{'role': m['role'], 'content': m['content']} for m in st.session_state.messages]
        )
        assistant_response = response.choices[0].message.content

    # Add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})
    with st.chat_message('assistant'):
        st.markdown(assistant_response)