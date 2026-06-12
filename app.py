import streamlit as st
import os
import sqlite3
from run_agent import sql_agent, AgentState

# ============================================================
# NIKE DATABASE REBUILDER (Run once to fix schema)
# ============================================================
def rebuild_nike_database():
    db_file = 'ecommerce.db'
    
    # Force connection and reset
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Drop old tables to ensure clean slate
        cursor.execute("DROP TABLE IF EXISTS orders;")
        cursor.execute("DROP TABLE IF EXISTS customers;")
        
        # 1. Create Detailed Customers Table
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
        
        # 2. Create Detailed Orders Table
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
        
        # 3. Seed High-Detail Customers (Including Rohit!)
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
        
        # 4. Seed Transaction History
        cursor.executemany("""
        INSERT INTO orders (customer_id, shoe_model, order_year, unit_price, quantity, total_amount) 
        VALUES (?, ?, ?, ?, ?, ?);
        """, [
            # Rohit (Golden Customer) - High value, recurring
            (1, 'Nike Air Jordan 1 High', 2024, 200.00, 5, 1000.00),
            (1, 'Nike Vaporfly 3', 2025, 250.00, 4, 1000.00),
            (1, 'Nike Air Max 97', 2026, 180.00, 2, 360.00),
            
            # Sarah - High value single items
            (2, 'Nike Alphafly 3', 2024, 285.00, 1, 285.00),
            (2, 'Nike Metcon 9', 2025, 150.00, 2, 300.00),
            
            # Amit - Frequent budget buyer
            (3, 'Nike Pegasus 40', 2025, 130.00, 1, 130.00),
            (3, 'Nike Dunk Low', 2025, 115.00, 1, 115.00),
            
            # Priya - Trend buyer
            (4, 'Nike Air Force 1', 2026, 110.00, 1, 110.00),
            
            # Mike - Occasional
            (5, 'Nike Monarch IV', 2025, 75.00, 1, 75.00)
        ])
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.sidebar.error(f"DB Init Failed: {e}")

# Run the rebuilder every time the app loads to guarantee data exists
rebuild_nike_database()
# ============================================================


# 1. Page Configurations
st.set_page_config(
    page_title="Nike Retail Analytics",
    page_icon="👟",
    layout="wide"
)

st.title("👟 Nike Retail Analytics Dashboard")
st.markdown("Analyze customer demographics, sales trends, and golden buyers using **Llama 3.1 & LangGraph** over a cloud-hosted SQLite database.")

# 2. Session States
if "web_history" not in st.session_state:
    st.session_state.web_history = []
if "graph_history" not in st.session_state:
    st.session_state.graph_history = []

# 3. Sidebar Audit (To Prove Data Exists)
with st.sidebar:
    st.header("🗄️ Database Audit")
    if os.path.exists('ecommerce.db'):
        try:
            conn = sqlite3.connect('ecommerce.db')
            # Fetch Customers
            cust = conn.execute("SELECT * FROM customers").fetchall()
            st.expander(f"👥 Customers ({len(cust)})").dataframe(cust)
            
            # Fetch Orders
            ords = conn.execute("SELECT * FROM orders").fetchall()
            st.expander(f"📦 Orders ({len(ords)})").dataframe(ords)
            conn.close()
        except:
            pass

# 4. Chat History
for message in st.session_state.web_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message:
            st.code(message["sql"], language="sql")

# 5. User Input
if user_question := st.chat_input("Ask: 'Who is our golden customer and what do they do?'"):
    
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.web_history.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        explanation_placeholder = st.empty()
        sql_placeholder = st.empty()
        
        initial_state: AgentState = {
            "user_question": user_question,
            "attempt_count": 0,
            "error_feedback": None,
            "generated_sql": None,
            "db_results": None,
            "final_explanation": None,
            "chat_history": st.session_state.graph_history
        }
        
        # UNIQUE THREAD ID per question to prevent hallucination loops
        unique_id = f"turn_{len(st.session_state.web_history)}"
        config = {"configurable": {"thread_id": unique_id}}
        
        with st.spinner("🧠 Analyst AI is thinking..."):
            output = sql_agent.invoke(initial_state, config=config)
        
        if output.get("error_feedback"):
            st.error(f"SQL Generation Failed: {output.get('error_feedback')}")
        
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
            # Add to graph history for context
            st.session_state.graph_history.append({
                "question": user_question,
                "sql": final_sql
            })
