import sqlite3

def create_database():
    conn = sqlite3.connect('ecommerce.db')
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        join_date DATE
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        product_name TEXT,
        order_date DATE,
        total_amount REAL,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    )
    ''')

    customers_data = [
        ('Alice Smith', 'alice@email.com', '2025-01-15'),
        ('Bob Jones', 'bob@email.com', '2025-02-20'),
        ('Charlie Brown', 'charlie@email.com', '2025-03-05')
    ]
    
    orders_data = [
        (1, 'Laptop', '2026-05-10', 1200.00),
        (1, 'Mouse', '2026-05-12', 25.00),
        (2, 'Smartphone', '2026-05-15', 800.00),
        (3, 'Headphones', '2026-06-01', 150.00),
        (2, 'Keyboard', '2026-06-02', 75.00)
    ]

    cursor.execute("DELETE FROM orders")
    cursor.execute("DELETE FROM customers")
    
    cursor.executemany("INSERT INTO customers (name, email, join_date) VALUES (?, ?, ?)", customers_data)
    cursor.executemany("INSERT INTO orders (customer_id, product_name, order_date, total_amount) VALUES (?, ?, ?, ?)", orders_data)

    conn.commit()
    conn.close()
    print("Database 'ecommerce.db' created successfully with sample data!")

if __name__ == "__main__":
    create_database()