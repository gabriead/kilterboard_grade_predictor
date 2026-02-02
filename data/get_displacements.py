import sqlite3
import json
import os

# ==========================================
# CONFIGURATION
# ==========================================
DB_PATH = "kilter.db"
OUTPUT_FILE = "kilter_layout.json"
LAYOUT_ID = 1

def save_layout_json():
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database file '{DB_PATH}' not found.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        print(f"   Querying database for Layout {LAYOUT_ID}...")

        # Modified SQL: Removes 'set_id' constraint to get ALL placements for the layout
        query = """
        SELECT
            placements.id as placement_id,
            mirrored_placements.id as mirrored_placement_id,
            holes.x,
            holes.y
        FROM holes
        INNER JOIN placements
            ON placements.hole_id = holes.id
            AND placements.layout_id = ?
        LEFT JOIN placements mirrored_placements
            ON mirrored_placements.hole_id = holes.mirrored_hole_id
            AND mirrored_placements.layout_id = ?
        """

        cursor.execute(query, (LAYOUT_ID, LAYOUT_ID))
        rows = cursor.fetchall()

        # Build list of dictionaries
        output_data = []
        for row in rows:
            item = {
                "placement_id": row['placement_id'],
                "mirrored_placement_id": row['mirrored_placement_id'],
                "x": row['x'],
                "y": row['y']
            }
            output_data.append(item)

        conn.close()

        # Save to JSON file
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ Successfully saved {len(output_data)} holds to '{OUTPUT_FILE}'.")

    except sqlite3.Error as e:
        print(f"❌ Database Error: {e}")

if __name__ == "__main__":
    save_layout_json()