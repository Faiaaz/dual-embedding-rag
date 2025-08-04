#!/usr/bin/env python3
"""
Script to create a sample SQLite database for testing
"""

import sqlite3
import pandas as pd

def create_sample_database():
    """Create a sample SQLite database with multiple tables"""
    
    # Create a new database
    conn = sqlite3.connect('sample_database.db')
    
    # Create users table
    users_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com'],
        'age': [25, 30, 35, 28, 32],
        'department': ['Engineering', 'Marketing', 'Sales', 'Engineering', 'HR']
    }
    
    users_df = pd.DataFrame(users_data)
    users_df.to_sql('users', conn, if_exists='replace', index=False)
    
    # Create products table
    products_data = {
        'product_id': [101, 102, 103, 104, 105],
        'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'price': [999.99, 29.99, 79.99, 299.99, 149.99],
        'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
        'stock_quantity': [50, 200, 150, 30, 75]
    }
    
    products_df = pd.DataFrame(products_data)
    products_df.to_sql('products', conn, if_exists='replace', index=False)
    
    # Create orders table
    orders_data = {
        'order_id': [1001, 1002, 1003, 1004, 1005],
        'user_id': [1, 2, 3, 1, 4],
        'product_id': [101, 102, 103, 104, 105],
        'quantity': [1, 2, 1, 1, 3],
        'order_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        'total_amount': [999.99, 59.98, 79.99, 299.99, 449.97]
    }
    
    orders_df = pd.DataFrame(orders_data)
    orders_df.to_sql('orders', conn, if_exists='replace', index=False)
    
    # Create departments table
    departments_data = {
        'dept_id': [1, 2, 3, 4, 5],
        'dept_name': ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance'],
        'manager': ['John Manager', 'Jane Manager', 'Bob Manager', 'Alice Manager', 'Charlie Manager'],
        'budget': [500000, 200000, 300000, 100000, 400000]
    }
    
    departments_df = pd.DataFrame(departments_data)
    departments_df.to_sql('departments', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("✅ Sample database created successfully!")
    print("📊 Tables created:")
    print("   - users (5 rows)")
    print("   - products (5 rows)")
    print("   - orders (5 rows)")
    print("   - departments (5 rows)")
    print("📁 File: sample_database.db")

if __name__ == "__main__":
    create_sample_database() 