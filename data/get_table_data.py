import sqlite3
import os

# Configuration
DB_PATH = 'kilter.db'  # Make sure this matches your downloaded file name


def connect_db(db_file):
    """Creates a database connection."""
    if not os.path.exists(db_file):
        print(f"Error: Database file '{db_file}' not found.")
        return None

    try:
        conn = sqlite3.connect(db_file)
        # This enables accessing columns by name: row['name']
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    return None


def list_tables(conn):
    """Lists all tables in the database."""
    print("--- Database Tables ---")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table in tables:
        print(f"- {table['name']}")
    print("-" * 20)


def describe_table(conn, table_name):
    """Prints column names and types for a specific table."""
    print(f"\n--- Schema for table: '{table_name}' ---")
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    # Printing formatted columns
    print(f"{'Column ID':<10} {'Name':<20} {'Type':<10}")
    for col in columns:
        print(f"{col['cid']:<10} {col['name']:<20} {col['type']:<10}")
    print("-" * 20)


def get_sample_climbs(conn, limit=5):
    """Fetches a few sample rows from the 'climbs' table."""
    print(f"\n--- First {limit} Climbs ---")
    cursor = conn.cursor()

    # Note: 'climbs' is the standard table name, but check list_tables() if it fails
    try:
        cursor.execute("SELECT uuid, name, frames_count FROM climbs LIMIT ?", (limit,))
        rows = cursor.fetchall()

        for row in rows:
            print(f"Name: {row['name']} | UUID: {row['uuid']} | Frames: {row['frames_count']}")
    except sqlite3.Error as e:
        print(f"Could not query 'climbs' table: {e}")


def main():
    conn = connect_db(DB_PATH)
    if conn:
        # 1. See what tables exist
        list_tables(conn)

        # 2. Inspect the structure of the main 'climbs' table
        # (Change 'climbs' to any table name found in step 1)
        describe_table(conn, 'climbs')

        # 3. Get actual data
        get_sample_climbs(conn)

        conn.close()
        print("\nConnection closed.")


if __name__ == '__main__':
    main()