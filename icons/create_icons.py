"""
create_icons.py — Generate placeholder PNG icons for the ForestDL plugin.

Run this script once (outside QGIS) to create the icon files:
    python create_icons.py

Requires Pillow:
    pip install Pillow

Icons produced (32×32 px each):
    icon.png  — green forest/tree icon (main plugin icon)
    odm.png   — blue drone icon (ODM tab)
    yolo.png  — orange detection/target icon (YOLO tab)
"""

import os
import sys
import math

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow est requis. Installez-le avec: pip install Pillow")
    sys.exit(1)

# Output directory = same directory as this script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SIZE = 32
CORNER_RADIUS = 6


def rounded_rect(draw: ImageDraw.ImageDraw, xy, fill, radius=CORNER_RADIUS):
    """Draw a filled rounded rectangle."""
    x0, y0, x1, y1 = xy
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.ellipse([x0, y0, x0 + 2 * radius, y0 + 2 * radius], fill=fill)
    draw.ellipse([x1 - 2 * radius, y0, x1, y0 + 2 * radius], fill=fill)
    draw.ellipse([x0, y1 - 2 * radius, x0 + 2 * radius, y1], fill=fill)
    draw.ellipse([x1 - 2 * radius, y1 - 2 * radius, x1, y1], fill=fill)


def create_icon_png(filename: str):
    """
    Create the main ForestDL icon: green background with a white tree silhouette.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Green rounded background
    bg_color = (46, 125, 50, 255)   # dark green
    rounded_rect(draw, [0, 0, SIZE - 1, SIZE - 1], bg_color)

    # White tree: trunk
    trunk_color = (255, 255, 255, 220)
    draw.rectangle([14, 22, 18, 30], fill=trunk_color)

    # White tree: three triangular canopy layers
    canopy_color = (255, 255, 255, 240)
    # Bottom layer
    draw.polygon([(7, 23), (16, 12), (25, 23)], fill=canopy_color)
    # Middle layer
    draw.polygon([(9, 19), (16, 9), (23, 19)], fill=canopy_color)
    # Top layer
    draw.polygon([(11, 15), (16, 5), (21, 15)], fill=canopy_color)

    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, "PNG")
    print(f"Créé: {path}")
    return path


def create_odm_png(filename: str):
    """
    Create the ODM icon: blue background with a white drone silhouette.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Blue rounded background
    bg_color = (13, 71, 161, 255)   # dark blue
    rounded_rect(draw, [0, 0, SIZE - 1, SIZE - 1], bg_color)

    white = (255, 255, 255, 240)

    # Drone body (center rectangle)
    draw.rectangle([12, 14, 20, 18], fill=white)

    # Four arms extending diagonally from corners of body
    arm_color = (200, 230, 255, 200)
    # Top-left arm
    draw.line([(12, 14), (6, 8)], fill=arm_color, width=2)
    # Top-right arm
    draw.line([(20, 14), (26, 8)], fill=arm_color, width=2)
    # Bottom-left arm
    draw.line([(12, 18), (6, 24)], fill=arm_color, width=2)
    # Bottom-right arm
    draw.line([(20, 18), (26, 24)], fill=arm_color, width=2)

    # Four rotors (small ellipses at arm ends)
    rotor_color = (255, 255, 255, 200)
    draw.ellipse([3, 5, 9, 11], outline=rotor_color, width=1)
    draw.ellipse([23, 5, 29, 11], outline=rotor_color, width=1)
    draw.ellipse([3, 21, 9, 27], outline=rotor_color, width=1)
    draw.ellipse([23, 21, 29, 27], outline=rotor_color, width=1)

    # Camera (small square below body)
    draw.rectangle([14, 18, 18, 22], fill=white)

    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, "PNG")
    print(f"Créé: {path}")
    return path


def create_yolo_png(filename: str):
    """
    Create the YOLO icon: orange background with a white crosshair/target.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Orange rounded background
    bg_color = (230, 81, 0, 255)   # deep orange
    rounded_rect(draw, [0, 0, SIZE - 1, SIZE - 1], bg_color)

    white = (255, 255, 255, 240)
    cx, cy = SIZE // 2, SIZE // 2

    # Outer circle
    r_outer = 11
    draw.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        outline=white,
        width=2,
    )

    # Inner circle
    r_inner = 5
    draw.ellipse(
        [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
        outline=white,
        width=1,
    )

    # Center dot
    draw.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=white)

    # Crosshair lines (with gap in the middle)
    gap = 6
    # Horizontal
    draw.line([(3, cy), (cx - gap, cy)], fill=white, width=2)
    draw.line([(cx + gap, cy), (SIZE - 4, cy)], fill=white, width=2)
    # Vertical
    draw.line([(cx, 3), (cx, cy - gap)], fill=white, width=2)
    draw.line([(cx, cy + gap), (cx, SIZE - 4)], fill=white, width=2)

    # Corner bracket marks inside outer circle
    bracket_color = (255, 220, 150, 200)
    bracket_len = 4
    bracket_offset = 12
    # Top-left
    draw.line([(cx - bracket_offset, cy - bracket_offset),
               (cx - bracket_offset + bracket_len, cy - bracket_offset)], fill=bracket_color, width=1)
    draw.line([(cx - bracket_offset, cy - bracket_offset),
               (cx - bracket_offset, cy - bracket_offset + bracket_len)], fill=bracket_color, width=1)
    # Top-right
    draw.line([(cx + bracket_offset, cy - bracket_offset),
               (cx + bracket_offset - bracket_len, cy - bracket_offset)], fill=bracket_color, width=1)
    draw.line([(cx + bracket_offset, cy - bracket_offset),
               (cx + bracket_offset, cy - bracket_offset + bracket_len)], fill=bracket_color, width=1)
    # Bottom-left
    draw.line([(cx - bracket_offset, cy + bracket_offset),
               (cx - bracket_offset + bracket_len, cy + bracket_offset)], fill=bracket_color, width=1)
    draw.line([(cx - bracket_offset, cy + bracket_offset),
               (cx - bracket_offset, cy + bracket_offset - bracket_len)], fill=bracket_color, width=1)
    # Bottom-right
    draw.line([(cx + bracket_offset, cy + bracket_offset),
               (cx + bracket_offset - bracket_len, cy + bracket_offset)], fill=bracket_color, width=1)
    draw.line([(cx + bracket_offset, cy + bracket_offset),
               (cx + bracket_offset, cy + bracket_offset - bracket_len)], fill=bracket_color, width=1)

    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path, "PNG")
    print(f"Créé: {path}")
    return path


def main():
    print(f"Génération des icônes dans: {OUTPUT_DIR}")
    create_icon_png("icon.png")
    create_odm_png("odm.png")
    create_yolo_png("yolo.png")
    print("Terminé ! Tous les icônes ont été créés.")


if __name__ == "__main__":
    main()
