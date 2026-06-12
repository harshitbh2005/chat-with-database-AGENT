import sqlite3
import datetime
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
    UPDATED: Generates deterministic SQLite code based on conversation context, enforcing strict
    relational database rules and anchoring dynamic calendar calculations.
    """
    # Dynamically pull the exact active calendar year inside the runtime environment context
    current_year = datetime.datetime.now().year
    last_year = current_year - 1
    
    system_prompt = f"""You are an expert SQLite Data Assistant for a Nike Retail Store.
    Convert the user's natural language question into valid, executable SQLite syntax based on the schema below.
    
    ### CURRENT DATE ENVIRONMENT:
    - The active runtime calendar year right now is: {current_year}.
    - Relative phrases like "last year" mean explicitly year number: {last_year}.
    - "This year" means explicitly year number: {current_year}.
    
    ### DATABASE SCHEMA Matrix:
    {DATABASE_SCHEMA}
    
    ### EXPLICIT BREAKDOWN IMPLEMENTATION PATTERNS:
    User Question: "how much money we made last year?"
    -> CORRECT QUERY: SELECT SUM(total_amount) FROM orders WHERE order_year = {last_year}
    
    User Question: "which gender purchase our products the most?"
    -> WRONG QUERY: SELECT gender, SUM(total_amount) FROM customers JOIN orders ... LIMIT 1
    -> CORRECT QUERY: SELECT gender, SUM(total_amount) FROM customers INNER JOIN orders ON customers.customer_id = orders.customer_id GROUP BY gender ORDER BY SUM(total_amount) DESC
    (Reason: Do NOT use LIMIT 1 when comparing distributions or relative majorities, as it hides categories from the breakdown analyst).

    User Question: "which year we made most sales?"
    -> CORRECT QUERY: SELECT order_year, SUM(total_amount) FROM orders GROUP BY order_year ORDER BY SUM(total_amount) DESC
    (Reason: You must SELECT the calculation metrics and avoid LIMIT cuts so the analyst sees all records).

    ### STRICTOR STRUCTURAL LAWS:
    1. NO COLUMN HALLUCINATIONS: Do NOT invent fields like 'city', 'state', 'location', 'tracking_status', or 'shipping_code'. 
    2. DEMOGRAPHIC TARGETS: If asked "where" or "which place" to advertise, query customer 'profession' or 'gender' spending groups.
    3. RELATIVE TIME FILTERS: Always map relative dates using the specific integers provided in the CURRENT DATE ENVIRONMENT section.
    4. MANDATORY MATRIX DATA: For any comparison, breakdown, majority search ("most", "highest distribution", "percentages"), you MUST return ALL categories via GROUP BY. Do NOT append LIMIT clauses.
    5. NO CODE WRAPPERS: Return ONLY raw SQL text. Never wrap output in markdown syntax (no backticks, no ```sql).
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
            'content': f"CRITICAL REWRITE REQUIRED: Your previous query failed verification or returned a blind spot. Fix instruction: {error_feedback}"
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
    """
    UPDATED: Strictly grounds the analyst to your local dataset and handles empty states gracefully.
    """
    system_prompt = """You are a direct Nike Retail Analytics Consultant.
    
    CRITICAL RESTRICTION LAWS:
    1. You are an analyst for THIS specific store database only. Never mention global corporate revenue, billions, or market reports.
    2. If the Data Matrix is empty, None, or evaluates to no entries (e.g., '[]', '[(None,)]'), you MUST state directly: "No sales records match this criteria in our store database." Do NOT make up numbers.
    3. Give the exact metric answers from the provided data matrix in your very first sentence. No storytelling.
    4. Use short sentences and punchy markdown bullet points.
    """
    
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