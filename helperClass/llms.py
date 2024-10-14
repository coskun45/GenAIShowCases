import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import yaml
from langchain_openai import AzureChatOpenAI
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
import credentials
def get_azure_openAI():
    azure_llm = AzureChatOpenAI(
            deployment_name=credentials.MODEL_NAME,
            temperature=0,
            max_tokens=None,
            api_version=credentials.API_VERSION,
            azure_endpoint =credentials.AZURE_OPENAI_ENDPOINT,
            api_key=credentials.OPENAI_API_KEY,

        )
    return azure_llm

def google_gemini():
    client = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=credentials.GOOGLE_API_KEY,
        )
    return client

def llama3_405B():
    TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
    client = ChatTogether(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key=credentials.TOGETHER_API_KEY,
    )
    return client
