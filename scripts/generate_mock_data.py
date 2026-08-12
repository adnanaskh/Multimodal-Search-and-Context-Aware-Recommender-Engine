import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

categories = [
    "Clothing",
    "Electronics",
    "Accessories",
    "Sports",
    "Home",
    "Fitness",
    "Office",
    "Beauty",
    "Outdoors",
    "Toys",
]

adjectives = [
    "Classic",
    "Modern",
    "Premium",
    "Eco",
    "Smart",
    "Compact",
    "Ultra",
    "Performance",
    "Elegant",
    "Travel",
]

nouns = [
    "Jacket",
    "Earbuds",
    "Bottle",
    "Shoes",
    "Backpack",
    "T-Shirt",
    "Watch",
    "Headphones",
    "Mug",
    "Mat",
    "Lamp",
    "Chair",
    "Speaker",
    "Monitor",
    "Keyboard",
    "Camera",
    "Sunglasses",
    "Wallet",
    "Notebook",
    "Travel Bag",
    "Yoga Block",
    "Waterproof Jacket",
    "Coffee Grinder",
    "Fitness Band",
    "Gaming Mouse",
    "Desk Organizer",
    "Air Purifier",
    "Gloves",
    "Heater",
    "Drone",
]

colors = [
    (66, 133, 244),
    (232, 17, 35),
    (52, 168, 83),
    (255, 193, 7),
    (153, 102, 51),
    (102, 45, 145),
    (0, 120, 215),
    (255, 87, 34),
    (126, 87, 194),
    (60, 179, 113),
    (241, 90, 96),
    (255, 140, 0),
    (0, 150, 136),
    (33, 150, 243),
    (156, 39, 176),
]

num_products = 100
rows = []
for idx in range(1, num_products + 1):
    category = random.choice(categories)
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    title = f"{adjective} {noun}"
    description = (
        f"{title} with premium materials and thoughtful features for everyday use. "
        f"Designed to deliver comfort, durability, and style in the {category.lower()} category."
    )
    image_filename = f"product_{idx:03d}.jpg"
    price = round(random.uniform(12.99, 249.99), 2)
    rows.append((str(idx), title, description, image_filename, category, f"{price:.2f}"))

with open(DATA_DIR / "mock_catalog.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["item_id", "title", "description", "image_filename", "category", "price"])
    writer.writerows(rows)

try:
    font = ImageFont.load_default()
except Exception:
    font = None

for item_id, title, description, image_filename, category, price in rows:
    img = Image.new("RGB", (400, 400), color=random.choice(colors))
    draw = ImageDraw.Draw(img)
    text = f"{item_id}: {title}"

    if font is not None:
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        elif hasattr(draw, "textsize"):
            text_width, text_height = draw.textsize(text, font=font)
        elif hasattr(font, "getbbox"):
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width, text_height = 0, 0
    else:
        text_width, text_height = 0, 0

    x = max((400 - text_width) / 2, 10)
    y = max((400 - text_height) / 2, 10)
    draw.text((x, y), text, fill="white", font=font)
    img.save(IMAGE_DIR / image_filename)

num_users = 10
user_ids = [f"u{uid:03d}" for uid in range(1, num_users + 1)]
interaction_types = ["click", "purchase", "rating"]
interaction_weights = [0.6, 0.25, 0.15]
interactions = []
start_time = datetime(2026, 8, 1, 8, 0, 0)

def next_timestamp(base, minutes):
    return (base + timedelta(minutes=minutes)).isoformat()

for user in user_ids:
    if user == "u001":
        count = 40
    elif user in {"u002", "u003", "u004", "u005"}:
        count = random.randint(25, 35)
    else:
        count = random.randint(10, 20)

    for i in range(count):
        item_id = str(random.randint(1, num_products))
        interaction_type = random.choices(interaction_types, weights=interaction_weights, k=1)[0]
        rating = ""
        if interaction_type == "purchase":
            rating = str(random.randint(3, 5))
        elif interaction_type == "rating":
            rating = str(random.randint(1, 5))
            if random.random() < 0.4:
                interaction_type = "click"
        timestamp = next_timestamp(start_time, random.randint(0, 60 * 24 * 30))
        interactions.append((user, item_id, interaction_type, rating, timestamp))

with open(DATA_DIR / "user_interactions.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["user_id", "item_id", "interaction_type", "rating", "timestamp"])
    writer.writerows(interactions)

print(f"Generated {num_products} mock products and {len(interactions)} interaction records in {DATA_DIR}")
