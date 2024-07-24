import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama


# Load environment variables from .env file
load_dotenv()

# Langsmith tracking
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Simple Q&A ChatBot with Ollama"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries"),
        ("user", "Question:{question}")
    ]
)

#Generate response from user question based on API, engine
# Temperature is level of creativity, 0 is less creative
def generate_response(question,  engine, temperature, max_tokens):
    llm = Ollama(model=engine)
    # Output parser
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot with Ollama")

# Sidebar for settings
st.sidebar.title("Settings")

# Dropdown to select the models
llm = st.sidebar.selectbox("Select an OpenAI model", ["gemma:latest"])

# Adjust the response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask me anything")
user_input = st.text_input("You:")

if user_input:
        response = generate_response(user_input,  llm, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide a query")
