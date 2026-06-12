import streamlit as st  
from groq import Groq   

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def get_sql_from_llm(user_question, error_feedback=None):
    database_schema = """
    Table: customers (customer_id, name, email, age, gender, profession, join_date)
    Table: orders (order_id, customer_id, shoe_model, order_year, unit_price, quantity, total_amount)
    """

    system_prompt = f"""
    You are an expert AI Data Assistant. Convert the question to SQLite SQL.
    Schema: {database_schema}
    Rules: Respond ONLY with valid SQL. No markdown.
    """

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_question}
    ]

    if error_feedback:
        messages.append({'role': 'user', 'content': f"Fix error: {error_feedback}"})

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant', 
        messages=messages
    )
    return response.choices[0].message.content.strip()


def get_english_explanation(user_question, db_results):
    system_prompt = "You are a helpful Retail Analyst. Explain the database results in clear English."
    user_content = f"Question: {user_question}\nData: {str(db_results)}"

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]
    )
    return response.choices[0].message.content.strip()
