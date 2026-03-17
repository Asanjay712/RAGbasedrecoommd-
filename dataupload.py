import pandas as pd
import re
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# -------------------------------
# 1️⃣ Setup database
# -------------------------------
Base = declarative_base()
engine = create_engine("sqlite:///mydatabase.db", echo=True)
Session = sessionmaker(bind=engine)
session = Session()

# -------------------------------
# 2️⃣ Define tables
# -------------------------------
class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True)
    name = Column(String)
    age = Column(Integer)
    location = Column(String)
    preferred_genres = Column(String)

class Item(Base):
    __tablename__ = "items"
    item_id = Column(Integer, primary_key=True)
    name = Column(String)
    category = Column(String)
    brand = Column(String)
    price = Column(Float)
    description = Column(Text)

class Interaction(Base):  # historical user-item interactions
    __tablename__ = "interactions"
    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    item_id = Column(Integer, ForeignKey("items.item_id"))
    clicked = Column(Integer)
    purchased = Column(Integer)
    watch_time = Column(Float)
    skip_after_seconds = Column(Float)
    rewatched = Column(Integer)
    session_id = Column(String)
    timestamp = Column(DateTime)

class Impression(Base):
    __tablename__ = "impressions"
    impression_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    item_id = Column(Integer, ForeignKey("items.item_id"))
    prompt = Column(Text)
    explanation = Column(Text)
    rank_position = Column(Integer)  # New column added
    timestamp = Column(DateTime, default=datetime.utcnow)

class Feedback(Base):  # extended feedback with interaction metrics
    __tablename__ = "feedback"
    feedback_id = Column(Integer, primary_key=True, autoincrement=True)
    impression_id = Column(Integer, ForeignKey("impressions.impression_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    rating = Column(Float)           # explicit rating 1–5
    comment = Column(Text)           # optional textual comment
    clicked = Column(Integer)        # 0/1
    purchased = Column(Integer)      # 0/1
    watch_time = Column(Float)       # seconds
    skip_after_seconds = Column(Float)
    rewatched = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

# -------------------------------
# 3️⃣ Create tables
# -------------------------------
Base.metadata.create_all(engine)
print("✅ Tables created successfully!")

# -------------------------------
# 4️⃣ Delete previous data
# -------------------------------
print("Deleting previous data from database...")
session.query(Interaction).delete()
session.query(Item).delete()
session.query(User).delete()
session.commit()
print("✅ Previous data cleared!")

# -------------------------------
# 5️⃣ Load CSVs
# -------------------------------
users_df = pd.read_csv("users.csv")
items_df = pd.read_csv("items.csv")
interactions_df = pd.read_csv("interactions.csv")

# Clean & convert IDs
users_df['user_id'] = users_df['user_id'].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
items_df['item_id'] = items_df['item_id'].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
interactions_df['user_id'] = interactions_df['user_id'].apply(lambda x: int(re.search(r'\d+', str(x)).group()))
interactions_df['item_id'] = interactions_df['item_id'].apply(lambda x: int(re.search(r'\d+', str(x)).group()))

# Numeric conversions
users_df['age'] = pd.to_numeric(users_df['age'], errors='coerce').fillna(0).astype(int)
items_df['price'] = pd.to_numeric(items_df['price'], errors='coerce').fillna(0.0)
for col in ['clicked', 'purchased', 'watch_time', 'skip_after_seconds', 'rewatched']:
    interactions_df[col] = pd.to_numeric(interactions_df[col], errors='coerce').fillna(0)

# Parse timestamp
interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
interactions_df['timestamp'] = interactions_df['timestamp'].fillna(datetime.utcnow())

# -------------------------------
# 6️⃣ Insert data using ORM
# -------------------------------
# Insert users
for _, row in users_df.iterrows():
    session.add(User(
        user_id=row['user_id'],
        name=row['name'],
        age=row['age'],
        location=row['location'],
        preferred_genres=row['preferred_genres']
    ))

# Insert items
for _, row in items_df.iterrows():
    session.add(Item(
        item_id=row['item_id'],
        name=row['name'],
        category=row['category'],
        brand=row['brand'],
        price=row['price'],
        description=row['description']
    ))

# Insert historical interactions
for _, row in interactions_df.iterrows():
    session.add(Interaction(
        user_id=row['user_id'],
        item_id=row['item_id'],
        clicked=row['clicked'],
        purchased=row['purchased'],
        watch_time=row['watch_time'],
        skip_after_seconds=row['skip_after_seconds'],
        rewatched=row['rewatched'],
        session_id=row['session_id'],
        timestamp=row['timestamp']
    ))

# Commit all changes
session.commit()
session.close()

print("✅ CSVs inserted successfully! Impressions & feedback are empty for runtime collection.")
