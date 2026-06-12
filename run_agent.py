import sqlite3
import re
import streamlit as st  
from groq import Groq   
from typing import TypedDict, Optional, List, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver 

# Initialize the Groq Client safely using Streamlit secrets
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ==========================================
# 1. CORE LLM FUNCTIONS (The Agent's Brain)
# ==========================================
def get_sql_from_llm(user_question, error_feedback=None, history=None):
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
    4. MULTI-YEAR / RESPECTIVELY BREAKDOWNS: If the user asks for metrics broken down by year, month, or "respectively", you MUST select the year expression AND use a GROUP BY clause to return a vertical list of rows.

    GOOD EXAMPLE FOR RESPECTIVELY/BREAKDOWNS:
    User Question: "how many orders did we get in 2025 and 2026 respectively?"
    Correct Output: SELECT strftime('%Y', order_date) AS order_year, COUNT(*) FROM orders WHERE strftime('%Y', order_date) IN ('2025', '2026') GROUP BY order_year;
    
    BAD EXAMPLES (NEVER DO THIS):
    - Do NOT combine counts into subqueries horizontally.
    - Do NOT use multiple semicolons.
    """

    messages = [{'role': 'system', 'content': system_prompt}]

    if history:
        for turn in history:
            messages.append({'role': 'user', 'content': turn['question']})
            messages.append({'role': 'assistant', 'content': turn['sql']})

    messages.append({'role': 'user', 'content': user_question})

    if error_feedback:
        healing_context = f"""
        ⚠️ Your previous query failed with this error:
        {error_feedback}
        Fix the syntax error. Output ONLY the corrected raw SQL query.
        """
        messages.append({'role': 'user', 'content': healing_context})

    response = client.chat.completions.create(
        model='llama-3.1-8b-instant', 
        messages=messages
    )
    return response.choices[0].message.content.strip()


def get_english_explanation(user_question, db_results):
    """Translates raw database rows into a clear, natural English sentence."""
    system_prompt = """
    You are a precise Data Analyst Assistant. Read the raw database rows and answer the user's question directly.
    
    CRITICAL INSTRUCTIONS FOR COMPARISONS:
    If the database returns multiple rows (e.g., grouped values or list data), you MUST mention each value separately in your final sentence.
    Never group them into a single total or make generalizations if the user asked for a breakdown. Be exact and literal with the rows returned.
    
    CRITICAL ZERO-DATA RULE:
    If a user asks about a specific year, group, or item (like 2025), and that item is completely missing from the raw database output or returns empty results, it means the count is exactly 0.
    Explain clearly that we have 0 records or 0 orders for that specific item. Do not say the data is unavailable or missing.
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
# 2. STATE & GRAPH NODES (The Engine Layout)
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
    print(f"\n[AI Workstation]: Designing query... (Attempt {attempt}/3)")
    
    sql = get_sql_from_llm(
        user_question=state["user_question"], 
        error_feedback=state.get("error_feedback"), 
        history=state.get("chat_history", [])
    )
    return {"generated_sql": sql, "attempt_count": attempt}


def execute_query_node(state: AgentState) -> dict:
    raw_sql = state.get("generated_sql", "")
    cleaned_sql = re.sub(r"```sql|```", "", raw_sql).strip()
    
    try:
        conn = sqlite3.connect('ecommerce.db')
        cursor = conn.cursor()
        cursor.execute(cleaned_sql)
        results = cursor.fetchall()
        conn.close()
        
        print("✅ [AI Workstation]: SQL executed successfully!")
        return {"db_results": results, "error_feedback": None, "generated_sql": cleaned_sql}
    except Exception as e:
        print(f"⚠️ [AI Workstation]: SQL failed -> {e}")
        # FIXED: Only return clean raw exception string so the next healing prompt doesn't degrade
        return {"error_feedback": str(e), "db_results": None}


def explain_results_node(state: AgentState) -> dict:
    print("[AI Workstation]: Translating raw data to plain English...")
    explanation = get_english_explanation(state["user_question"], state["db_results"])
    return {"final_explanation": explanation}


# ==========================================
# 3. GRAPH COMPOSITION (The Router Map)
# ==========================================
builder = StateGraph(AgentState)
builder.add_node("generate_query", generate_query_node)
builder.add_node("execute_query", execute_query_node)
builder.add_node("explain_results", explain_results_node)

builder.set_entry_point("generate_query")
builder.add_edge("generate_query", "execute_query")

def route_after_execution(state: AgentState) -> str:
    if state.get("error_feedback") is not None:
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


# ==========================================
# 4. RUNTIME TERMINAL LOOP
# ==========================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "session_1"}}
    session_history = []
    
    print("\n=======================================================")
    print("--- Text-to-SQL Agent with Real Context Memory ---")
    print("=======================================================")

    while True:
        question = input("\nAsk your database a question: ")
        if question.lower() in ['exit', 'quit']: break
        if not question.strip(): continue

        initial_state: AgentState = {
            "user_question": question, 
            "attempt_count": 0, 
            "error_feedback": None,
            "generated_sql": None,
            "db_results": None,
            "final_explanation": None,
            "chat_history": session_history
        }
        
        output = sql_agent.invoke(initial_state, config=config)

        if output.get("final_explanation"):
            print(f"\n[Final System Response]: {output.get('final_explanation')}")
            print(f"📊 (Under the hood SQL used: {output.get('generated_sql')})")
            
            session_history.append({
                "question": question,
                "sql": output.get("generated_sql")
            })
