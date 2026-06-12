import sqlite3
import re
import streamlit as st  
from groq import Groq   
from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ==========================================
# 1. CORE LLM FUNCTIONS
# ==========================================
def get_sql_from_llm(user_question, error_feedback=None, history=None):
    # UPDATED SCHEMA TO MATCH NIKE RETAIL DATA
    database_schema = """
    Table: customers
    Columns:
      - customer_id (INTEGER PK)
      - name (TEXT)
      - email (TEXT)
      - age (INTEGER)
      - gender (TEXT)
      - profession (TEXT) -- e.g., 'Pro Cricketer', 'Doctor'
      - join_date (DATE)

    Table: orders
    Columns:
      - order_id (INTEGER PK)
      - customer_id (INTEGER FK)
      - shoe_model (TEXT) -- e.g., 'Nike Air Jordan'
      - order_year (INTEGER)
      - unit_price (REAL)
      - quantity (INTEGER)
      - total_amount (REAL)
    """

    system_prompt = f"""
    You are an expert SQLite Data Assistant for a Nike Retail Store. 
    Convert the user's question into valid SQL based on the schema below.
    
    Schema:
    {database_schema}
    
    RULES:
    1. Return ONLY raw SQL. No markdown.
    2. For "Golden Customer" or "Top Customer", usually ORDER BY total_amount DESC LIMIT 1.
    3. If asked about specific years, use WHERE order_year = 2025 etc.
    4. To join customers and orders, use: JOIN orders ON customers.customer_id = orders.customer_id.
    """

    messages = [{'role': 'system', 'content': system_prompt}]
    
    # Add history for context if available
    if history:
        for turn in history:
            messages.append({'role': 'user', 'content': turn['question']})
            messages.append({'role': 'assistant', 'content': turn['sql']})

    messages.append({'role': 'user', 'content': user_question})

    if error_feedback:
        messages.append({'role': 'user', 'content': f"Fix this SQL error: {error_feedback}"})

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant', 
        messages=messages
    )
    return response.choices[0].message.content.strip()


def get_english_explanation(user_question, db_results):
    # UPDATED PROMPT FOR INTERACTIVE, STORYTELLING RESPONSES
    system_prompt = """
    You are a Senior Retail Analytics Consultant for Nike. 
    Analyze the raw database results and answer the user's question in an engaging, professional, and interactive way.
    
    GUIDELINES:
    1. Don't just list numbers. Tell a story.
    2. If the data includes a customer (like Rohit), mention their **Profession**, **Age**, and **Spending Habits** to give a complete picture.
    3. Example: "Rohit is clearly our VIP! Being a Pro Cricketer, he invests heavily in performance gear..."
    4. If the result is empty, politely say no data was found for that specific criteria.
    """
    
    user_content = f"User Question: {user_question}\nRaw Database Output: {str(db_results)}"
    
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]
    )
    return response.choices[0].message.content.strip()


# ==========================================
# 2. STATE & GRAPH DEFINITIONS
# ==========================================
class AgentState(TypedDict):
    user_question: str
    generated_sql: Optional[str]
    error_feedback: Optional[str]
    attempt_count: int
    db_results: Optional[List[Any]]
    final_explanation: Optional[str]
    chat_history: List[dict]

def generate_query_node(state: AgentState) -> dict:
    attempt = state.get("attempt_count", 0) + 1
    sql = get_sql_from_llm(
        state["user_question"], 
        state.get("error_feedback"), 
        state.get("chat_history", [])
    )
    return {"generated_sql": sql, "attempt_count": attempt}

def execute_query_node(state: AgentState) -> dict:
    raw_sql = state.get("generated_sql", "")
    # Basic cleanup
    cleaned_sql = re.sub(r"```sql|```", "", raw_sql).strip()
    
    try:
        conn = sqlite3.connect('ecommerce.db')
        cursor = conn.cursor()
        cursor.execute(cleaned_sql)
        results = cursor.fetchall()
        conn.close()
        return {"db_results": results, "error_feedback": None, "generated_sql": cleaned_sql}
    except Exception as e:
        return {"error_feedback": str(e), "db_results": None}

def explain_results_node(state: AgentState) -> dict:
    explanation = get_english_explanation(state["user_question"], state["db_results"])
    return {"final_explanation": explanation}

# ==========================================
# 3. GRAPH ROUTING
# ==========================================
builder = StateGraph(AgentState)
builder.add_node("generate_query", generate_query_node)
builder.add_node("execute_query", execute_query_node)
builder.add_node("explain_results", explain_results_node)

builder.set_entry_point("generate_query")
builder.add_edge("generate_query", "execute_query")

def route_after_execution(state: AgentState) -> str:
    if state.get("error_feedback"):
        if state.get("attempt_count", 0) < 3:
            return "retry"
        return "fail"
    return "success"

builder.add_conditional_edges(
    "execute_query", 
    route_after_execution, 
    {"retry": "generate_query", "fail": END, "success": "explain_results"}
)
builder.add_edge("explain_results", END)

sql_agent = builder.compile(checkpointer=MemorySaver())
