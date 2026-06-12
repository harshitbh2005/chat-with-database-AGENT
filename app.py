import streamlit as st
import os
import sqlite3
from run_agent import sql_agent, AgentState

# ============================================================
# FIXED AUTOMATIC DATABASE INITIALIZER (NO NULL BYTE IMPORT)
# ============================================================
def verify_and_init_db():
    db_file = 'ecommerce.db'
    # Check if the database doesn't exist or is an empty 0-byte file
    if not os.path.exists(db_file) or os.path.getsize(db_file) == 0:
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Create Tables
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                join_date DATE NOT NULL
            );
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL,
                product_name TEXT NOT NULL,
                order_date DATE NOT NULL,
                total_amount REAL NOT NULL,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
            );
            """)
            
            # Seed Starting Data safely
            cursor.execute("SELECT COUNT(*) FROM customers")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("""
                INSERT INTO customers (name, email, join_date) VALUES (?, ?, ?);
                """, [
                    ('Alice Johnson', 'alice@example.com', '2025-01-15'),
                    ('Bob Smith', 'bob@example.com', '2025-02-20'),
                    ('Charlie Brown', 'charlie@example.com', '2025-03-10'),
                    ('Diana Prince', 'diana@example.com', '2025-04-05')
                ])
                
                cursor.executemany("""
                INSERT INTO orders (customer_id, product_name, order_date, total_amount) VALUES (?, ?, ?, ?);
                """, [
                    (1, 'Laptop', '2025-05-10', 1200.00),
                    (2, 'Smartphone', '2025-06-14', 800.00),
                    (1, 'Headphones', '2025-07-22', 150.00),
                    (3, 'Desk Chair', '2025-08-01', 250.00),
                    (4, 'Monitor', '2025-09-18', 350.00),
                    (2, 'Keyboard', '2025-10-05', 100.00),
                    (3, 'Bookshelf', '2026-01-12', 180.00),
                    (4, 'Coffee Maker', '2026-02-28', 90.00),
                    (1, 'Tablet', '2026-03-05', 450.00),
                    (2, 'Smart Watch', '2026-04-20', 299.99)
                ])
            
            conn.commit()
            conn.close()
            st.sidebar.success("🚀 Cloud database seeded successfully!")
        except Exception as e:
            st.sidebar.error(f"Database auto-seeding failed: {e}")

verify_and_init_db()
# ============================================================

# 1. Page Configurations
st.set_page_config(
    page_title="Agentic Text-to-SQL Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Cloud-Agentic AI Chat with Database")
st.markdown("Ask natural language questions against a cloud-hosted SQLite `ecommerce.db` instance backed by Llama 3.1 (via Groq API) & LangGraph.")

# 2. Initialize Persistent Session States for Web Browser
if "web_history" not in st.session_state:
    st.session_state.web_history = []  # Display logs for chat bubbles

if "graph_history" not in st.session_state:
    st.session_state.graph_history = []  # Context payload lists passed to Llama 3

# 3. Sidebar System Logs Layout
with st.sidebar:
    st.header("⚙️ Agentic Engine Monitoring")
    st.markdown("Watch the self-healing and graph execution parameters in real-time.")
    status_box = st.empty()
    status_box.info("System Ready. Awaiting user question...")

# 4. Display Past Chat History Bubbles
for message in st.session_state.web_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message:
            st.code(message["sql"], language="sql")

# 5. Handle New User Input
if user_question := st.chat_input("Ask your database a question (e.g., 'How many orders in 2026 respectively?')"):
    
    # Render user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.web_history.append({"role": "user", "content": user_question})

    # Render assistant bubble with spinner processing node graph
    with st.chat_message("assistant"):
        explanation_placeholder = st.empty()
        sql_placeholder = st.empty()
        
        status_box.warning("🤖 Graph State Processing Active...")
        
        # Build initial dictionary state frame matching our TypedDict schema
        initial_state: AgentState = {
            "user_question": user_question,
            "attempt_count": 0,
            "error_feedback": None,
            "generated_sql": None,
            "db_results": None,
            "final_explanation": None,
            "chat_history": st.session_state.graph_history
        }
        
        config = {"configurable": {"thread_id": "streamlit_session"}}
        output = sql_agent.invoke(initial_state, config=config)
        
        # Render responses based on state exit outcomes
        if output.get("error_feedback"):
            error_msg = f"❌ Sorry, I couldn't resolve the database query syntax safely within 3 automated tries.\n\n*Last Engine Error:* `{output.get('error_feedback')}`"
            explanation_placeholder.markdown(error_msg)
            status_box.error("Graph Ended: Execution Failures encountered.")
            st.session_state.web_history.append({"role": "assistant", "content": error_msg})
        
        elif output.get("final_explanation"):
            final_ans = output.get("final_explanation")
            final_sql = output.get("generated_sql")
            
            # Print values cleanly onto display slots
            explanation_placeholder.markdown(final_ans)
            sql_placeholder.code(final_sql, language="sql")
            
            status_box.success("✅ Execution Completed Successfully!")
            
            # Save elements to persistent visual context loops
            st.session_state.web_history.append({
                "role": "assistant", 
                "content": final_ans,
                "sql": final_sql
            })
            
            # Append query details to the background Llama 3 memory payload
            st.session_state.graph_history.append({
                "question": user_question,
                "sql": final_sql
            })
