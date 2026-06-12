import streamlit as st
from run_agent import sql_agent, AgentState
import time  # NEW: Added to create unique thread configurations

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

    # Render assistant bubble with verbose processing tracker
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
        
        # FIX: Generate a unique thread ID based on current time so the attempt loop completely resets
        unique_thread_id = f"session_{int(time.time())}"
        config = {"configurable": {"thread_id": unique_thread_id}}

        # ============================================================
        # VERBOSE STATUS CONTAINER FOR LANGGRAPH STEP STREAMING
        # ============================================================
        with st.status("🧠 Agent Execution Path Running...", expanded=True) as status:
            # Use .stream to trigger nodes and animate logs live
            for chunk in sql_agent.stream(initial_state, config=config, stream_mode="updates"):
                for node_name, state_update in chunk.items():
                    
                    if node_name == "generate_query":
                        # Fetch the loop's current iteration dynamically from the active state update
                        attempt = state_update.get("attempt_count", 1)
                        status.write(f"📝 **Node `generate_query` (Attempt {attempt}/3):** Llama 3.1 is evaluating the schema rules and crafting raw SQL syntax...")
                    
                    elif node_name == "execute_query":
                        if state_update.get("error_feedback"):
                            status.write("⚠️ **Node `execute_query`:** SQLite engine returned a syntax error! Activating LangGraph self-healing loop routing...")
                        else:
                            status.write("📊 **Node `execute_query`:** Connection established. Executed successfully and fetched raw rows from `ecommerce.db`.")
                    
                    elif node_name == "explain_results":
                        status.write("🗣️ **Node `explain_results`:** Synthesizing the relational raw table results back into a user-friendly conversational English explanation...")
            
            # Finish up container workflow animation
            status.update(label="✅ LangGraph Execution Completed Successfully!", state="complete", expanded=False)
        
        # ============================================================
        # EXTRACT UNIFIED STATE DIRECTLY FROM THE SPECIFIC THREAD
        # ============================================================
        compiled_state = sql_agent.get_state(config).values
        
        final_explanation = compiled_state.get("final_explanation")
        final_sql = compiled_state.get("generated_sql")
        error_feedback = compiled_state.get("error_feedback")
        
        # Render responses based on state exit outcomes
        if error_feedback and not final_explanation:
            error_msg = "❌ Sorry, I couldn't resolve the database query syntax safely within 3 automated tries."
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
            
            # Append query details to the background Llama 3 memory payload
            st.session_state.graph_history.append({
                "question": user_question,
                "sql": final_sql
            })
