import sqlite3
import ollama

# ==========================================
# 1. AI GENERATION LAYER (Text to SQL)
# ==========================================
def get_sql_from_llm(user_question):
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
    4. For date matching like '2026' or 'May 2026', prefer using simple patterns like order_date LIKE '2026-%' or order_date LIKE '2026-05-%' instead of strftime functions.
    5. If the user's request is talking about things completely missing from the schema (like spaceships, employees, cars, etc.), reply with the exact word: CLARIFY
    """

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_question}
        ]
    )
    return response['message']['content'].strip()


# ==========================================
# AI EXPLANATION LAYER (Data to Strict English)
# ==========================================
def get_english_explanation(user_question, db_results):
    """Takes the raw data and user question, and returns an completely accurate sentence."""
    
    system_prompt = """
    You are a highly precise Data Analyst Assistant. 
    You will be given a user's original question and the raw data fetched from a database.
    Your job is to read the raw database data carefully and answer the user's question directly in a clear, conversational, plain English sentence.
    
    CRITICAL RULE: Treat the raw database data literally. If the value inside the brackets/parentheses is a number (e.g., 3), that is the total count. Say exactly that number in your response. Never say 'one record' or 'one row' if the number itself represents a larger count.
    """
    
    # We clean up the db_results format to make it easier for the LLM to interpret numbers
    user_content = f"""
    User Question: {user_question}
    Raw Database Data Output: {str(db_results)}
    """

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]
    )
    return response['message']['content'].strip()


# ==========================================
# 2. DATABASE EXECUTION LAYER (SQL Runner)
# ==========================================
def execute_query(sql_query):
    try:
        conn = sqlite3.connect('ecommerce.db')
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        conn.close()
        return results
    except Exception as e:
        return f"SQL_ERROR: {e}"


# ==========================================
# 3. THE AGENTIC LOOP (The Coordinator)
# ==========================================
def run_agent():
    print("\n==================================================")
    print("--- Local Agentic AI Text-to-SQL System Active ---")
    print("      (Type 'exit' or 'quit' to stop the agent)   ")
    print("==================================================")
    
    while True:
        user_question = input("\nAsk your database a question: ")
        
        if user_question.lower() in ['exit', 'quit']:
            print("\nShutting down the agent. Goodbye!")
            break
            
        if not user_question.strip():
            continue
            
        # 1. Ask Llama 3 to create the query
        generated_sql = get_sql_from_llm(user_question)
        
        # 2. Guardrail Check
        if "CLARIFY" in generated_sql.upper() or "PLEASE" in generated_sql.upper():
            print("\n[Agent Response]: I'm sorry, I cannot find that information in the current database schema. Could you please rephrase or ask about customers or orders?")
            continue

        print(f"\n[Agent Thought]: Generated SQL -> {generated_sql}")

        # 3. Execute the SQL query
        db_results = execute_query(generated_sql)

        # 4. Handle results/non-existing data cases gracefully
        if isinstance(db_results, str) and db_results.startswith("SQL_ERROR"):
            print(f"\n[Agent Response]: Oops, I generated an invalid query. Error: {db_results}")
            
        elif len(db_results) == 0:
            print("\n[Agent Response]: I ran the query successfully, but there is no data matching your request in the database.")
            
        else:
            # Pass results to our strict AI explanation layer
            english_answer = get_english_explanation(user_question, db_results)
            print(f"\n[Agent Response]: {english_answer}")

if __name__ == "__main__":
    run_agent()