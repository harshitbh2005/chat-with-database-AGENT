# run_agent.py
import sqlite3
import streamlit as st  
from groq import Groq   
from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# DATABASE SCHEMA DEFINITION USED BY ALL MODEL NODES
DATABASE_SCHEMA = """
Table: customers
  - customer_id (INTEGER PRIMARY KEY)
  - name (TEXT)
  - email (TEXT)
  - age (INTEGER)
  - gender (TEXT)
  - profession (TEXT) -- e.g., 'Pro Cricketer', 'Tech CEO', 'Gym Trainer', 'Fashion Designer'
  - join_date (DATE)
  
Table: orders
  - order_id (INTEGER PRIMARY KEY)
  - customer_id (INTEGER FOREIGN KEY -> customers.customer_id)
  - shoe_model (TEXT)
  - order_year (INTEGER)
  - unit_price (REAL)
  - quantity (INTEGER)
  - total_amount (REAL)
"""

# ==========================================
# 1. CORE LLM FUNCTIONS
# ==========================================
def classify_user_intent(user_question: str) -> str:
    system_prompt = f"""You are a Database Security Guardrail. Inspect the user's question against the exact available schema:
    {DATABASE_SCHEMA}
    
    If the question asks to filter or report on structural columns or concepts that do not exist (such as locations, cities, countries, tracking status, shipping, warehouse codes), respond with exactly 'REJECTED'.
    
    EXCEPTION: If the user asks about "where" or "demographics" to advertise but no city/state exists, allow it if it can be answered using customer 'profession' fields instead."""
    
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_question}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def get_sql_from_llm(user_question: str, error_feedback: str = None, history: list = None) -> str:
    """
    Generates deterministic SQLite code based on conversation context, enforcing strict
    relational database rules to prevent data blind spots and column hallucinations.
    """
    system_prompt = f"""You are an expert SQLite Data Assistant for a Nike Retail Store.
    Convert the user's natural language question into valid, executable SQLite syntax based on the schema below.
    
    ### DATABASE SCHEMA Matrix:
    {DATABASE_SCHEMA}
    
    ### CRITICAL COMPLIANCE RULES:
    1. STRICT COLUMN BOUNDS: Only query columns that explicitly exist in the schema. Do NOT invent fields like 'city', 'state', 'location', 'country', 'tracking_status', 'shipping_code', or 'reviews'.
    2. INTERPRETING LOCATION/TARGETS: If asked "where" or "which place" to promote/advertise, interpret this as finding the highest-spending or highest-ordering customer 'profession' or 'gender' segments. 
    3. MANDATORY AGGREGATION COLUMNS: When using aggregate functions (SUM, COUNT, AVG), you MUST include the calculated metric column in your SELECT statement alongside the grouping attribute. Never hide calculations solely inside an ORDER BY clause.
    4. NO LIMIT BLIND SPOTS ON BREAKDOWNS: For any question asking for a breakdown, comparison, distribution, or percentage (e.g., "by year", "which gender", "compare professions"), do NOT use a LIMIT clause. Return all active rows/categories so the analyst node can see the full dataset to compute accurate breakdowns.
    5. TRANSACTION INTEGRITY: For questions regarding "purchases", "sales", "spending", or "items bought", you MUST use an INNER JOIN to link the 'customers' and 'orders' tables on customer_id to evaluate metrics based on actual orders rather than profile registries.
    6. DETERMINISTIC RESPONSE: Output ONLY the raw executable SQL query string. Do NOT wrap output inside markdown backticks (no ```sql) and do NOT add conversational text or notes.
    """
    
    messages = [{'role': 'system', 'content': system_prompt}]
    
    if history:
        for turn in history:
            messages.append({'role': 'user', 'content': turn.get('question', '')})
            messages.append({'role': 'assistant', 'content': turn.get('sql', '')})
            
    messages.append({'role': 'user', 'content': user_question})

    if error_feedback:
        messages.append({
            'role': 'user', 
            'content': f"CRITICAL REWRITE REQUIRED: Your previous query failed validation or syntax execution. Fix instruction: {error_feedback}"
        })

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant', 
        messages=messages, 
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def verify_sql_logic(user_question: str, generated_sql: str, db_results: list) -> str:
    system_prompt = f"""You are a QA Database Auditor. Check if the generated SQL statement safely and comprehensively answers the user's question using ONLY valid columns.
    
    AVAILABLE SCHEMA MATRIX:
    {DATABASE_SCHEMA}
    
    CRITICAL CONSTRAINT: Do NOT recommend adding columns like 'city', 'state', or 'location'. They do not exist. 
    If the question seeks demographic targets, ensure the SQL groups by 'profession', 'gender', or 'age'.
    
    Respond with exactly 'PASSED' if the query logic is safe and execution can complete.
    If it is wrong, respond with a direct description of what to fix without hallucinating new columns."""
    
    user_content = f"Question: {user_question}\nGenerated SQL: {generated_sql}\nReturned Data Matrix: {str(db_results)}"
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[
            {'role': 'system', 'content': system_prompt}, 
            {'role': 'user', 'content': user_content}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def get_english_explanation(user_question: str, db_results: list) -> str:
    system_prompt = """You are a direct Nike Retail Analytics Consultant.
    1. Give the exact metric answers in your very first sentence. No storytelling.
    2. Use short sentences and punchy markdown bullet points.
    3. If answering target advertisement groups based on profession data, explain clearly."""
    
    user_content = f"User Question: {user_question}\nData Matrix: {str(db_results)}"
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}],
        temperature=0.1
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

def guardrail_node(state: AgentState) -> dict:
    status = classify_user_intent(state["user_question"])
    if "REJECTED" in status:
        return {"error_feedback": "OUT_OF_SCOPE"}
    return {"error_feedback": None}

def generate_query_node(state: AgentState) -> dict:
    attempt = state.get("attempt_count", 0) + 1
    sql = get_sql_from_llm(state["user_question"], state.get("error_feedback"), state.get("chat_history", []))
    return {"generated_sql": sql, "attempt_count": attempt, "error_feedback": None}

def execute_query_node(state: AgentState) -> dict:
    raw_sql = state.get("generated_sql", "")
    cleaned_sql = raw_sql.replace("```sql", "").replace("```SQL", "").replace("```", "").strip()
    try:
        conn = sqlite3.connect('ecommerce.db')
        cursor = conn.cursor()
        results = cursor.execute(cleaned_sql).fetchall()
        conn.close()
        return {"db_results": results, "error_feedback": None, "generated_sql": cleaned_sql}
    except Exception as e:
        return {"error_feedback": f"SQLite Syntax Error: {str(e)}", "db_results": None}

def verify_results_node(state: AgentState) -> dict:
    if state.get("error_feedback"): 
        return {}
    validation = verify_sql_logic(state["user_question"], state["generated_sql"], state["db_results"])
    if "PASSED" not in validation:
        return {"error_feedback": validation}
    return {"error_feedback": None}

def explain_results_node(state: AgentState) -> dict:
    explanation = get_english_explanation(state["user_question"], state["db_results"])
    return {"final_explanation": explanation}

# ==========================================
# 3. GRAPH CONFIGURATION
# ==========================================
builder = StateGraph(AgentState)
builder.add_node("guardrail", guardrail_node)
builder.add_node("generate_query", generate_query_node)
builder.add_node("execute_query", execute_query_node)
builder.add_node("verify_results", verify_results_node)
builder.add_node("explain_results", explain_results_node)

builder.set_entry_point("guardrail")

def route_after_guardrail(state: AgentState) -> str:
    if state.get("error_feedback") == "OUT_OF_SCOPE":
        return "stop_unsupported"
    return "proceed_to_sql"

builder.add_conditional_edges("guardrail", route_after_guardrail, {"stop_unsupported": END, "proceed_to_sql": "generate_query"})
builder.add_edge("generate_query", "execute_query")
builder.add_edge("execute_query", "verify_results")

def route_after_verification(state: AgentState) -> str:
    if state.get("error_feedback"):
        if state.get("attempt_count", 0) < 3:
            return "retry"
        return "fail"
    return "success"

builder.add_conditional_edges("verify_results", route_after_verification, {"retry": "generate_query", "fail": END, "success": "explain_results"})
builder.add_edge("explain_results", END)

sql_agent = builder.compile(checkpointer=MemorySaver())
