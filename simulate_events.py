import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ----------------------
# 1️⃣ Generate Users (500 per category)
# ----------------------
num_per_category = 500
categories = ['electronics', 'sports', 'fashion']

user_records = []
user_counter = 1
for cat in categories:
    for _ in range(num_per_category):
        user_records.append({
            'user_id': f"U{user_counter:04d}",
            'name': f"User{user_counter}",
            'age': np.random.randint(18, 60),
            'location': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai']),
            'preferred_genres': cat
        })
        user_counter += 1

users_df = pd.DataFrame(user_records)

# ----------------------
# 2️⃣ Define Brands, Types & Variants
# ----------------------
electronics_items = {
    "Laptop": ['Apple', 'Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'Microsoft'],
    "Smartphone": ['Samsung', 'Apple', 'OnePlus', 'Xiaomi', 'Google', 'Oppo', 'Huawei'],
    "Headphones": ['Bose', 'Sony', 'JBL', 'Sennheiser', 'Beats', 'Bang & Olufsen'],
    "TV": ['LG', 'Samsung', 'Panasonic', 'Sony', 'Philips'],
    "Console": ['Nintendo', 'Sony PlayStation', 'Microsoft Xbox', 'Razer', 'Corsair']
}

electronics_variants = {
    "Laptop": ['Gaming', 'Ultrabook', '2-in-1', 'Business'],
    "Smartphone": ['Flagship', 'Mid-range', 'Budget'],
    "Headphones": ['Over-ear', 'In-ear', 'Wireless', 'Noise-cancelling'],
    "TV": ['OLED', 'LED', 'QLED', 'Smart'],
    "Console": ['Home', 'Handheld', 'Bundle Pack']
}

sports_items = {
    "Running Shoes": ['Nike', 'Adidas', 'Puma', 'Reebok', 'Under Armour'],
    "Football": ['Nike', 'Adidas', 'Puma', 'Wilson'],
    "Tennis Racket": ['Wilson', 'Yonex', 'Head'],
    "Fitness Tracker": ['Fitbit', 'Garmin', 'Samsung'],
    "Jersey": ['Nike', 'Adidas', 'Puma']
}

sports_variants = {
    "Running Shoes": ['Men', 'Women', 'Kids'],
    "Football": ['Match', 'Training'],
    "Tennis Racket": ['Beginner', 'Pro', 'Junior'],
    "Fitness Tracker": ['Basic', 'Advanced', 'Pro'],
    "Jersey": ['Home', 'Away', 'Training']
}

fashion_items = {
    "T-shirt": ['Nike', 'Adidas', 'Puma', 'Zara', 'H&M'],
    "Jeans": ['Levi\'s', 'Zara', 'H&M'],
    "Jacket": ['Nike', 'Adidas', 'Gucci', 'Louis Vuitton'],
    "Sneakers": ['Nike', 'Adidas', 'Puma'],
    "Dress": ['Zara', 'H&M', 'Gucci'],
    "Handbag": ['Gucci', 'Louis Vuitton', 'H&M']
}

fashion_variants = {
    "T-shirt": ['V-neck', 'Round-neck', 'Long-sleeve', 'Sleeveless'],
    "Jeans": ['Skinny', 'Slim', 'Straight', 'Bootcut'],
    "Jacket": ['Bomber', 'Hooded', 'Leather', 'Denim'],
    "Sneakers": ['Running', 'Casual', 'Training'],
    "Dress": ['Casual', 'Party', 'Evening'],
    "Handbag": ['Clutch', 'Tote', 'Crossbody']
}

adjectives = ["premium", "sleek", "durable", "lightweight", "high-performance", "comfortable", "stylish"]

# ----------------------
# 3️⃣ Description Generator
# ----------------------
def generate_description(category, item_type, variant, brand):
    adj = random.choice(adjectives)
    
    if category == 'electronics':
        if item_type == 'Laptop':
            cpu = random.choice(['Intel i5', 'Intel i7', 'AMD Ryzen 5', 'AMD Ryzen 7'])
            ram = random.choice(['8GB', '16GB', '32GB'])
            storage = random.choice(['256GB SSD', '512GB SSD', '1TB HDD'])
            display = random.choice(['13-inch', '15-inch', '17-inch'])
            return f"{brand} {variant} Laptop with {cpu}, {ram} RAM, {storage} storage, {display} display. {adj.capitalize()} for work, gaming, and productivity."
        elif item_type == 'Smartphone':
            battery = random.choice(['3000mAh', '4000mAh', '5000mAh'])
            camera = random.choice(['12MP', '48MP', '108MP'])
            display = random.choice(['6.1-inch', '6.5-inch', '6.7-inch'])
            storage = random.choice(['64GB', '128GB', '256GB'])
            return f"{brand} {variant} Smartphone with {display} display, {camera} camera, {battery} battery, and {storage} storage. {adj.capitalize()} for daily use and photography."
        elif item_type == 'Headphones':
            connectivity = random.choice(['Bluetooth', 'Wired', 'Wireless'])
            battery = random.choice(['20h', '30h', '40h'])
            return f"{brand} {variant} Headphones with {connectivity}, {battery} battery life, and superior sound quality. {adj.capitalize()} listening experience."
        elif item_type == 'TV':
            size = random.choice(['42-inch', '50-inch', '65-inch'])
            resolution = random.choice(['Full HD', '4K UHD', '8K UHD'])
            smart = random.choice(['Smart TV features', 'Alexa integration', 'Google Assistant'])
            return f"{brand} {variant} TV {size} {resolution} with {smart}. {adj.capitalize()} for immersive home entertainment."
        elif item_type == 'Console':
            storage = random.choice(['512GB', '1TB', '2TB'])
            return f"{brand} {variant} Console with {storage} storage and exclusive games. {adj.capitalize()} for ultimate gaming experience."
    
    elif category == 'sports':
        feature = random.choice(['lightweight design', 'durable build', 'ergonomic grip', 'moisture-wicking material'])
        return f"{brand} {variant} {item_type} with {feature}. {adj.capitalize()} for athletes and sports enthusiasts."
    
    elif category == 'fashion':
        material = random.choice(['cotton', 'denim', 'leather', 'synthetic blend'])
        feature = random.choice(['comfortable fit', 'breathable fabric', 'stylish design'])
        return f"{brand} {variant} {item_type} made from {material}, {feature}. {adj.capitalize()} for everyday wear."

# ----------------------
# 4️⃣ Generate Items (500+ per category)
# ----------------------
items = []
item_counter = 1

def generate_items(category_dict, variant_dict, category_name, num_per_type=100, price_range=(500,1500)):
    global item_counter
    local_items = []
    for item_type, brands in category_dict.items():
        for _ in range(num_per_type):
            brand = random.choice(brands)
            variant = random.choice(variant_dict[item_type])
            price = round(random.uniform(price_range[0], price_range[1]),2)
            description = generate_description(category_name, item_type, variant, brand)
            local_items.append({
                'item_id': f"P{item_counter:05d}",
                'name': f"{brand} {variant} {item_type} {item_counter}",
                'category': category_name,
                'brand': brand,
                'price': price,
                'description': description
            })
            item_counter +=1
    return local_items

items += generate_items(electronics_items, electronics_variants, 'electronics', num_per_type=100, price_range=(500,2000))
items += generate_items(sports_items, sports_variants, 'sports', num_per_type=100, price_range=(500,1500))
items += generate_items(fashion_items, fashion_variants, 'fashion', num_per_type=100, price_range=(300,1500))

items_df = pd.DataFrame(items)

# ----------------------
# 5️⃣ Generate Interactions
# ----------------------
interactions_records = []
session_counter = 1

for user in users_df['user_id']:
    user_pref = users_df.loc[users_df['user_id']==user, 'preferred_genres'].values[0]
    candidate_items = items_df[items_df['category']==user_pref]
    interacted_items = candidate_items.sample(n=np.random.randint(5,15), replace=False)['item_id'].tolist()
    
    for item in interacted_items:
        clicked = np.random.choice([0,1], p=[0.3,0.7])
        purchased = np.random.choice([0,1], p=[0.8,0.2])
        watch_time = round(np.random.uniform(10, 300),2)
        skip_after_seconds = round(np.random.uniform(0, watch_time),2)
        rewatched = np.random.choice([0,1], p=[0.9,0.1])
        session_id = f"S{session_counter:05d}"
        timestamp = datetime.now() - timedelta(days=np.random.randint(0,30))
        interactions_records.append({
            'user_id': user,
            'item_id': item,
            'clicked': clicked,
            'purchased': purchased,
            'watch_time': watch_time,
            'skip_after_seconds': skip_after_seconds,
            'rewatched': rewatched,
            'session_id': session_id,
            'timestamp': timestamp
        })
        session_counter += 1

interactions_df = pd.DataFrame(interactions_records)

# ----------------------
# 6️⃣ Save CSVs
# ----------------------
users_df.to_csv("users.csv", index=False)
items_df.to_csv("items.csv", index=False)
interactions_df.to_csv("interactions.csv", index=False)

print("✅ Realistic dataset generated!")
print(users_df.head())
print(items_df.head())
print(interactions_df.head())
