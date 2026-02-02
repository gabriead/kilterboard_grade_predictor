import json
import re
import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 1280
MARGIN_X = 80
MARGIN_Y = 80
LAYOUT_PATH = "../data/kilter_layout.json"

# Standard Kilter Board Original Grid Dimensions (Fixed)
# This prevents the grid from shifting if your JSON is incomplete.
BOARD_MIN_X = 0
BOARD_MAX_X = 144
BOARD_MIN_Y = 0
BOARD_MAX_Y = 156

COLORS = {
    12: (0, 255, 0),  # Green (Start)
    13: (255, 255, 0),  # Cyan (Hand/Foot)
    14: (255, 0, 255),  # Magenta (Finish)
    15: (0, 165, 255)  # Orange (Feet)
}

# Your provided climb data
TARGET_CLIMB =   {
    "_rowid_": 26385,
    "uuid": "0E1692C3B26B49C7A5204B774E52431A",
    "layout_id": 1,
    "setter_id": 1410,
    "setter_username": "jwilder",
    "name": "Jivin’ Pete",
    "description": "No matching",
    "hsm": 3,
    "edge_left": 64,
    "edge_right": 136,
    "edge_bottom": 8,
    "edge_top": 152,
    "frames_count": 1,
    "frames_pace": 0,
    "frames": "p1073r15p1104r15p1188r12p1191r12p1255r13p1291r13p1304r13p1320r13p1352r13p1389r14p1509r15p1524r15",
    "is_draft": 0,
    "is_listed": 1,
    "created_at": "2019-04-08 21:11:10.443673"
  }


# ==========================================
# 1. HELPERS
# ==========================================
def load_layout():
    if not os.path.exists(LAYOUT_PATH):
        print(f"❌ Error: {LAYOUT_PATH} not found.")
        exit()
    with open(LAYOUT_PATH, 'r') as f:
        layout = json.load(f)
    return {i['placement_id']: {'x': i['x'], 'y': i['y']} for i in layout}


def parse_frames(frame_str):
    return re.findall(r'p(\d+)r(\d+)', frame_str)


def to_pixel(board_x, board_y, draw_w, draw_h):
    """
    Converts logical board coordinates (0-144, 0-156) to pixel coordinates.
    """
    # Normalize 0.0 -> 1.0
    norm_x = (board_x - BOARD_MIN_X) / (BOARD_MAX_X - BOARD_MIN_X)
    norm_y = (board_y - BOARD_MIN_Y) / (BOARD_MAX_Y - BOARD_MIN_Y)

    # Scale to Pixel
    # Invert Y because Image(0,0) is Top-Left, Board(0,0) is Bottom-Left
    px = int(MARGIN_X + (norm_x * draw_w))
    py = int((CANVAS_HEIGHT - MARGIN_Y) - (norm_y * draw_h))
    return px, py


# ==========================================
# 2. DRAWING
# ==========================================
def draw_climb(img, frames, db, climb_meta):
    h_img, w_img = img.shape[:2]
    draw_w = w_img - (2 * MARGIN_X)
    draw_h = h_img - (2 * MARGIN_Y)

    # --- 1. DRAW BOUNDING BOX ---
    # Convert edge coordinates to pixels
    # Top edge in climb data (Y=120) corresponds to the visual top of the box
    # Bottom edge (Y=8) corresponds to visual bottom

    # Note: Because of Y-inversion in to_pixel, 'top' board coord becomes 'top' pixel coord visually
    bx1, by_top = to_pixel(climb_meta['edge_left'], climb_meta['edge_top'], draw_w, draw_h)
    bx2, by_bot = to_pixel(climb_meta['edge_right'], climb_meta['edge_bottom'], draw_w, draw_h)

    # Draw Rectangle (Red, Thin line)
    # We add padding (+/- 10px) so the box doesn't cut through the center of the edge holds
    cv2.rectangle(img, (bx1 - 15, by_top - 15), (bx2 + 15, by_bot + 15), (0, 0, 255), 2)

    print(
        f"📦 Bounding Box: X[{climb_meta['edge_left']}-{climb_meta['edge_right']}], Y[{climb_meta['edge_bottom']}-{climb_meta['edge_top']}]")

    # --- 2. DRAW HOLDS ---
    missing_ids = []

    for pid_str, role_str in frames:
        pid = int(pid_str)
        role = int(role_str)

        if pid not in db:
            missing_ids.append(pid)
            continue

        raw_x = db[pid]['x']
        raw_y = db[pid]['y']

        px, py = to_pixel(raw_x, raw_y, draw_w, draw_h)

        color = COLORS.get(role, (255, 255, 255))
        base_radius = 14

        # Draw Rings based on Role
        if role == 12:  # Start
            cv2.circle(img, (px, py), base_radius + 4, color, 3)
            cv2.circle(img, (px, py), 6, color, -1)
        elif role == 14:  # Finish
            cv2.circle(img, (px, py), base_radius + 4, color, 3)
            cv2.circle(img, (px, py), 6, color, -1)
            cv2.circle(img, (px, py), base_radius + 8, color, 1)  # Extra ring
        elif role == 15:  # Feet
            cv2.circle(img, (px, py), base_radius - 2, color, 2)
        else:  # Middle
            cv2.circle(img, (px, py), base_radius, color, 3)

    if missing_ids:
        print(f"⚠️ Warning: {len(missing_ids)} holds were in the climb but NOT in your layout JSON.")
        print(f"   Missing IDs: {missing_ids}")
        print("   (This is why they were invisible. Update kilter_layout_orig.json)")

    return img


# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    hold_db = load_layout()

    # Create Black Canvas
    img = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

    # Draw
    holds = parse_frames(TARGET_CLIMB['frames'])
    result = draw_climb(img, holds, hold_db, TARGET_CLIMB)

    # Add Text
    cv2.putText(result, TARGET_CLIMB['name'], (32, 52), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite("kilter_vis_bbox.jpg", result)
    print("✅ Saved to kilter_vis_bbox.jpg")