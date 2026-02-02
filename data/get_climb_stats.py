import sqlite3
import json
import os

# Configuration
DB_PATH = 'kilter.db'  # Ensure this points to your actual database file
OUTPUT_FILE = 'climb_stats.json'


def export_stats_to_json():
    # 1. Connect to database
    if not os.path.exists(DB_PATH):
        print(f"Error: {DB_PATH} not found.")
        return

    conn = sqlite3.connect(DB_PATH)

    # This allows us to access columns by name and easily convert to dict
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 2. The Query
    # We select 'rowid' explicitly to match your "_rowid_" requirement
    query = """
            SELECT rowid as _rowid_, \
                   * \
            FROM climb_stats \
            """

    try:
        print(f"Executing query on '{DB_PATH}'...")
        cursor.execute(query)
        rows = cursor.fetchall()

        # 3. Convert SQLite rows to a list of dictionaries
        data = [dict(row) for row in rows]

        # 4. Save to JSON
        print(f"Exporting {len(data)} records to '{OUTPUT_FILE}'...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        print("Done!")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    export_stats_to_json()