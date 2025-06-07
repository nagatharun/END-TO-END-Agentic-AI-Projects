import streamlit as st
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os

# Optional: Load from .env
from dotenv import load_dotenv
load_dotenv()

# ✅ Set page configuration
st.set_page_config(page_title="Product Assistant", page_icon="🛒")

# 🔐 Sidebar for Groq API Login
st.sidebar.header("🔐 API Login")
user_groq_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if user_groq_key:
    os.environ["GROQ_API_KEY"] = user_groq_key

    # Validate the key
    try:
        test_llm = ChatGroq(
            temperature=1.0,
            model="qwen-qwq-32b",
            api_key=user_groq_key
        )
        _ = test_llm.invoke("Hello")
        st.sidebar.success("✅ Groq API Key is valid.")
    except Exception as e:
        st.sidebar.error("❌ Invalid Groq API Key.")
        st.stop()
else:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Load other environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_v2"] = "true"
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

# Step 1: Define Pydantic schema
class ProductAssistant(BaseModel):
    product_name: str = Field(description="Name of the product")
    product_details: str = Field(description="Details about the product")
    price_usd: int = Field(description="Tentative price in USD")

# Step 2: Initialize parser and LLM
parser = JsonOutputParser(pydantic_object=ProductAssistant)

llm = ChatGroq(
    temperature=0.7,
    model="qwen-qwq-32b",  # Or "gemma2-9b-it" if supported
    api_key=os.environ["GROQ_API_KEY"]
)

# Step 3: Create ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant that provides product details in a structured JSON format."),
    ("user", "{question}"),
    ("system", "{format_instructions}")
])

# Step 4: Inject formatting instructions
prompt_with_instructions = chat_prompt.partial(
    format_instructions=parser.get_format_instructions()
)

# Step 5: Create the chain
chain = prompt_with_instructions | llm | parser

# Step 6: Main App UI
st.title("🛒 AI Product Info Assistant")

query = st.text_input("Ask about any product:")

if st.button("Get Product Info") and query:
    with st.spinner("Fetching product information..."):
        try:
            response = chain.invoke({"question": query})
            st.success("✅ Product Details:")
            st.json(response)
        except Exception as e:
            st.error("❌ Failed to fetch or parse product info.")
            st.code(str(e))
