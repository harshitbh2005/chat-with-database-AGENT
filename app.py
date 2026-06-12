import streamlit as st
from run_agent import sql_agent, AgentState

# 1. Page Configurations
st.set_page_config(
    page_title="Agentic Text-to-SQL Dashboard",
    page_icon="📊",
    layout="wide"
)

# FIXED HEADER FOR CLOUD-HOSTED RECRUITER METRICS
st.title("📊 Cloud-Agentic AI Chat with Database")
st.markdown("Ask natural language questions against a cloud-hosted SQLite `ecommerce.db` instance backed by **Llama 3.1 (via Groq API)** & **LangGraph**.")

# 2. Initialize Persistent Session States for Web Browser
if "web_history" not in st.session_state:
    st.session_state.web_history = []  # Display logs for chat bubbles

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

    # Render assistant bubble
    with st.chat_message("assistant"):
        explanation_placeholder = st.empty()
        sql_placeholder = st.empty()
        
        status_box.warning("🤖 Graph State Processing Active...")
        
        # Build initial state with an empty history array to keep execution clean
        initial_state: AgentState = {
            "user_question": user_question,
            "attempt_count": 0,
            "error_feedback": None,
            "generated_sql": None,
            "db_results": None,
            "final_explanation": None,
            "chat_history": []
        }
        
        # Create a completely fresh thread ID config for each turn to avoid memory leakage
        config = {"configurable": {"thread_id": f"query_id_{len(st.session_state.web_history)}"}}

        # Use a stable loading spinner instead of unstable streaming loops
        with st.spinner("🧠 LangGraph Self-Healing Agent Execution Tree Active..."):
            output = sql_agent.invoke(initial_state, config=config)
        
        # Extract operational values
        final_explanation = output.get("final_explanation")
        final_sql = output.get("generated_sql")
        error_feedback = output.get("error_feedback")
        
        # Render responses based on state exit outcomes
        if error_feedback and not final_explanation:
            error_msg = f"❌ Sorry, I couldn't resolve the database query safely within 3 automated tries.\n\n*Database Error Catch:* `{error_feedback}`"
            explanation_placeholder.markdown(error_msg)
            status_box.error("Graph Ended: Execution Failures encountered.")
            st.session_state.web_history.append({"role": "assistant", "content": error_msg})
        
        elif final_explanation:
            # Print values cleanly onto display slots
            explanation_placeholder.markdown(final_explanation)
            if final_sql:
                sql_placeholder.code(final_sql, language="sql")
            
            status_box.success("✅ Execution Completed Successfully!")
            
            # Save elements to persistent visual context loops
            st.session_state.web_history.append({
                "role": "assistant", 
                "content": final_explanation,
                "sql": final_sql
            })
