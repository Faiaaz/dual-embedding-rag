#!/usr/bin/env python3
"""
Script to view SQLite database contents in a readable format
"""

import sqlite3
import pandas as pd

def view_database(db_path):
    """View all tables in the database"""
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get list of tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"📊 Database: {db_path}")
    print(f"📋 Tables found: {len(tables)}")
    print("=" * 50)
    
    for table in tables:
        table_name = table[0]
        print(f"\n🔍 Table: {table_name}")
        print("-" * 30)
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        print("📝 Schema:")
        for col in columns:
            col_name, col_type = col[1], col[2]
            print(f"  - {col_name} ({col_type})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"📊 Total rows: {row_count}")
        
        if row_count > 0:
            # Display data using pandas for better formatting
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            print("\n📋 Data:")
            print(df.to_string(index=False))
        
        print()
    
    conn.close()

if __name__ == "__main__":
    view_database('sample_database.db') 