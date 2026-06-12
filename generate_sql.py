import streamlit as st  # Added to access cloud secrets
from groq import Groq   # Swapped from ollama to groq

# Initialize the Groq Client safely using Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_sql_from_llm(user_question, error_feedback=None):
    database_schema = """
    Table: customers
    Columns:
      - customer_id (INTEGER, Primary Key)
      - name (TEXT)
      - email (TEXT)
      - join_date (DATE)

    Table: orders
    Columns:
      - order_id (INTEGER, Primary Key)
      - customer_id (INTEGER, Foreign Key pointing to customers.customer_id)
      - product_name (TEXT)
      - order_date (DATE)
      - total_amount (REAL)
    """

    system_prompt = f"""
    You are an expert AI Data Assistant. Your job is to convert a user's English question into a valid SQLite SQL query based on the database schema provided below.

    Database Schema:
    {database_schema}

    CRITICAL RULES:
    1. Respond ONLY with the raw SQL query. 
    2. Do NOT wrap the query in markdown code blocks like ```sql ... ```.
    3. Do NOT include any explanations, greetings, or text other than the SQL query itself.
    4. If the user's request is too vague to write a query, reply with the exact word: PLEASE CLARIFY
    """

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question}
    ]

    if error_feedback:
        healing_context = f"""
        ⚠️ Your previous query failed with this error:
        {error_feedback}
        
        Fix the query syntax error. Output ONLY the corrected raw SQL query.
        """
        messages.append({'role': 'user', 'content': healing_context})

    # Swapped from ollama.chat to cloud-hosted Groq
    response = client.chat.completions.create(
        model='llama3-8b-8192', 
        messages=messages
    )
    return response.choices.message.content.strip()


def get_english_explanation(user_question, db_results):
    """Translates raw database rows into a clear, natural English sentence."""
    system_prompt = """
    You are a precise Data Analyst Assistant. 
    Read the raw database output and answer the user's question directly in a single, clear, friendly English sentence.
    
    CRITICAL RULE: Treat the database numbers literally. If you see (5,), say '5 orders' or '5 products' based on the question.
    """
    
    user_content = f"User Question: {user_question}\nRaw Database Output: {str(db_results)}"

    # Swapped from ollama.chat to cloud-hosted Groq
    response = client.chat.completions.create(
        model='llama3-8b-8192',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]
    )
    return response.choices.message.content.strip()
