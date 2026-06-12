# run_agent.py
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
    
    ### CRITICAL AS ALIAS RULE:
    Whenever you generate an aggregate calculation function (COUNT, SUM, AVG, MIN, MAX), you MUST explicitly name the output column using an descriptive 'AS' alias (e.g., SELECT COUNT(order_id) AS total_orders_found ...). Never return raw un-aliased aggregate functions.

    ### EXPLICIT BREAKDOWN IMPLEMENTATION PATTERNS:
    User Question: "how many in 2025?"
    -> CORRECT QUERY: SELECT COUNT(order_id) AS total_orders FROM orders WHERE order_year = 2025
    
    User Question: "which gender purchase our products the most?"
    -> CORRECT QUERY: SELECT gender, SUM(total_amount) AS total_spending FROM customers INNER JOIN orders ON customers.customer_id = orders.customer_id GROUP BY gender ORDER BY total_spending DESC

    ### STRICTOR STRUCTURAL LAWS:
    1. NO COLUMN HALLUCINATIONS: Do NOT invent fields.
    2. MANDATORY MATRIX DATA: For any comparison breakdown, do NOT append LIMIT clauses unless looking for a single top entry.
    3. NO CODE WRAPPERS: Return ONLY raw SQL text. Never wrap output in markdown syntax (no backticks, no ```sql).
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
            'content': f"CRITICAL REWRITE REQUIRED: Your previous query failed verification. Fix instruction: {error_feedback}"
        })

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant', 
        messages=messages, 
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def verify_sql_logic(user_question: str, generated_sql: str, db_results: list) -> str:
    system_prompt = f"""You are a QA Database Auditor. Check if the generated SQL statement safely answers the question using valid columns and proper descriptive 'AS' aliases.
    Respond with exactly 'PASSED' if the query logic is complete. If it is wrong, describe what to fix."""
    
    user_content = f"Question: {user_question}\nGenerated SQL: {generated_sql}\nReturned Data Matrix: {str(db_results)}"
    response = client.chat.completions.create(
        model='llama-3.1-8b-instant',
        messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


def get_english_explanation(user_question: str, db_results: list) -> str:
    system_prompt = """You are a direct Nike Retail Analytics Consultant.
    Review the structured column-value data records carefully and answer the user's question.
    
    CRITICAL INSTRUCTIONS:
    1. Give the exact number response matching the labeled keys in your very first sentence. 
    2. Read the label names carefully (e.g., if total_orders is 5, it means 5 orders were placed, NOT \$5).
    3. Use short sentences and punchy markdown bullet points. Never display raw brackets or dictionary formats.
    """
    
    user_content = f"User Question: {user_question}\nStructured Data Matrix Input: {str(db_results)}"
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
        # FIX: Force sqlite3 to return rows as mapped column dictionaries instead of blind tuples
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        raw_rows = cursor.execute(cleaned_sql).fetchall()
        
        # Unpack SQL rows into explicit key-value dictionary lists for the explanation node
        structured_results = [dict(row) for row in raw_rows]
        conn.close()
        
        return {"db_results": structured_results, "error_feedback": None, "generated_sql": cleaned_sql}
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
