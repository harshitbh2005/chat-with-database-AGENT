# app.py
import streamlit as st
import os
import sqlite3
from run_agent import sql_agent, AgentState

# ============================================================
# NIKE DATABASE REBUILDER
# ============================================================
def rebuild_nike_database():
    db_file = 'ecommerce.db'
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Reset tables cleanly
        cursor.execute("DROP TABLE IF EXISTS orders;")
        cursor.execute("DROP TABLE IF EXISTS customers;")
        
        cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            profession TEXT NOT NULL,
            join_date DATE NOT NULL
        );
        """)
        
        cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            shoe_model TEXT NOT NULL,
            order_year INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            quantity INTEGER NOT NULL,
            total_amount REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        );
        """)
        
        cursor.executemany("""
        INSERT INTO customers (name, email, age, gender, profession, join_date) 
        VALUES (?, ?, ?, ?, ?, ?);
        """, [
            ('Rohit Verma', 'rohit.v@gmail.com', 29, 'Male', 'Pro Cricketer', '2024-02-10'),
            ('Sarah Jenkins', 'sarah.j@tech.com', 34, 'Female', 'Tech CEO', '2024-05-15'),
            ('Amit Patel', 'amit.p@fitness.com', 25, 'Male', 'Gym Trainer', '2025-01-11'),
            ('Priya Sharma', 'priya.s@design.com', 28, 'Female', 'Fashion Designer', '2025-03-22'),
            ('Mike Ross', 'mike.r@legal.com', 40, 'Male', 'Corporate Lawyer', '2025-08-30')
        ])
        
        cursor.executemany("""
        INSERT INTO orders (customer_id, shoe_model, order_year, unit_price, quantity, total_amount) 
        VALUES (?, ?, ?, ?, ?, ?);
        """, [
            (1, 'Nike Air Jordan 1 High', 2024, 200.00, 5, 1000.00),
            (1, 'Nike Vaporfly 3', 2025, 250.00, 4, 1000.00),
            (1, 'Nike Air Max 97', 2026, 180.00, 2, 360.00),
            (2, 'Nike Alphafly 3', 2024, 285.00, 1, 285.00),
            (2, 'Nike Metcon 9', 2025, 150.00, 2, 300.00),
            (3, 'Nike Pegasus 40', 2025, 130.00, 1, 130.00),
            (3, 'Nike Dunk Low', 2025, 115.00, 1, 115.00),
            (4, 'Nike Air Force 1', 2026, 110.00, 1, 110.00),
            (5, 'Nike Monarch IV', 2025, 75.00, 1, 75.00)
        ])
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.sidebar.error(f"DB Init Failed: {e}")

# Rebuild database structure
rebuild_nike_database()

# Page Configurations
st.set_page_config(
    page_title="Nike Retail Analytics",
    page_icon="👟",
    layout="wide"
)

st.title("👟 Nike AgenticAI Database Dashboard")
st.markdown("Analyze Nike's customer demographics, sales trends, and golden buyers using **Llama 3.1 & LangGraph** over a cloud-hosted SQLite database.")

# Session States initialization
if "web_history" not in st.session_state:
    st.session_state.web_history = []
if "graph_history" not in st.session_state:
    st.session_state.graph_history = []

# Sidebar Audit inspector panels
with st.sidebar:
    st.header("🗄️ Database Audit")
    if os.path.exists('ecommerce.db'):
        try:
            conn = sqlite3.connect('ecommerce.db')
            cust = conn.execute("SELECT * FROM customers").fetchall()
            st.expander(f"👥 Customers ({len(cust)})").dataframe(cust)
            
            ords = conn.execute("SELECT * FROM orders").fetchall()
            st.expander(f"📦 Orders ({len(ords)})").dataframe(ords)
            conn.close()
        except:
            pass

# Present Chat History UI Blocks
for message in st.session_state.web_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message:
            st.code(message["sql"], language="sql")

# Process Incoming Submissions
if user_question := st.chat_input("Ask: 'Who is our golden customer and what do they do?'"):
    
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.web_history.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        # Setup static dynamic display blocks
        verbose_expander = st.expander("⚙️ Agent Processing Logs (Verbose Trail)", expanded=True)
        explanation_placeholder = st.empty()
        sql_placeholder = st.empty()
        
        initial_state = {
            "user_question": user_question,
            "attempt_count": 0,
            "error_feedback": None,
            "generated_sql": None,
            "db_results": None,
            "final_explanation": None,
            "chat_history": st.session_state.graph_history
        }
        
        unique_id = f"turn_{len(st.session_state.web_history)}"
        config = {"configurable": {"thread_id": unique_id}}
        
        # Track active operational states manually out of the stream iterator
        output = {}
        
        # Open live container block inside expander
        with verbose_expander:
            status_logs = st.container()
            
            # STREAM TURNS LIVE FROM LANGGRAPH AGENT ENGINE
            for event in sql_agent.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    # CRITICAL PARSING PATCH: Only update state tracker dictionary if node_output contains actual map items
                    if node_output:
                        output.update(node_output)
                    
                    # 1. Handle Guardrail Output Event Logging
                    if node_name == "guardrail":
                        if node_output.get("error_feedback") == "OUT_OF_SCOPE":
                            status_logs.error("🚫 **Guardrail Node**: Question evaluated as OUT OF SCOPE.")
                        else:
                            status_logs.success("✅ **Guardrail Node**: Input text structure approved.")
                            
                    # 2. Handle Generation Node Event Logging
                    elif node_name == "generate_query":
                        attempt = node_output.get("attempt_count", 1)
                        status_logs.info(f"🧠 **Generation Node (Attempt {attempt}/3)**: Drafting SQLite query code...")
                        status_logs.code(node_output.get("generated_sql"), language="sql")
                        
                    # 3. Handle Execution Node Event Logging
                    elif node_name == "execute_query":
                        if node_output.get("error_feedback"):
                            status_logs.warning(f"⚠️ **Execution Node**: Database threw runtime error -> `{node_output.get('error_feedback')}`")
                        else:
                            status_logs.success(f"📊 **Execution Node**: Query processed successfully. Found {len(node_output.get('db_results', []))} row entries.")
                            
                    # 4. Handle Logic Auditor Verification Event Logging
                    elif node_name == "verify_results":
                        if node_output.get("error_feedback"):
                            status_logs.warning(f"🔄 **Verification Node**: Logic check failed. Instruction: *{node_output.get('error_feedback')}*. Routing to Self-Healing Loop...")
                        else:
                            status_logs.success("🎯 **Verification Node**: Logic check passed. Data structure is perfect.")

                    # 5. Handle Explanation Node Event Logging
                    elif node_name == "explain_results":
                        status_logs.info("📝 **Explanation Node**: Translating data values to direct markdown response...")

        # FINAL STATE OUTPUT RENDERING OUTSIDE LOG CONTAINER
        if output.get("error_feedback") == "OUT_OF_SCOPE":
            refusal_text = "⚠️ This request references fields or topics that do not exist within our database schema. Please ask a question related to available Customer profiles or Order history metrics."
            explanation_placeholder.markdown(refusal_text)
            
            st.session_state.web_history.append({
                "role": "assistant", 
                "content": refusal_text
            })
            
        elif output.get("error_feedback") and output.get("attempt_count", 0) >= 3:
            st.error(f"🚨 **Self-Healing Exhausted**: Failed to complete after 3 internal cycles. Error context: {output.get('error_feedback')}")
        
        elif output.get("final_explanation"):
            final_ans = output.get("final_explanation")
            final_sql = output.get("generated_sql")
            
            explanation_placeholder.markdown(final_ans)
            sql_placeholder.code(final_sql, language="sql")
            
            st.session_state.web_history.append({
                "role": "assistant", 
                "content": final_ans,
                "sql": final_sql
            })
            st.session_state.graph_history.append({
                "question": user_question,
                "sql": final_sql
            })
