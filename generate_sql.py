import ollama

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
    4. If the user's request is too vague to write a query, reply with the exact word: PLEASE CLARIFY
    """

    response = ollama.chat(
        model='llama3',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_question}
        ]
    )

    sql_query = response['message']['content'].strip()
    return sql_query

# This is the magic part that forces the terminal to pause and wait for YOU!
if __name__ == "__main__":
    user_question = input("Ask your database a question: ")
    
    print(f"\nProcessing your question: {user_question}")
    
    generated_sql = get_sql_from_llm(user_question)
    print("\nGenerated SQL:")
    print(generated_sql)