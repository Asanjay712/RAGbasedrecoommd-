# app/main.py - Enhanced FastAPI Backend with Vector Database Search and Retrieval Sessions

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine, Column, Integer, Float, Text, DateTime, ForeignKey, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import text
from sqlalchemy import and_
import comet_ml
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq, APIError
from datetime import datetime
import json
import os
from typing import Optional, List, Dict, Any
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import logging
from functools import lru_cache
import re
from comet_ml import Experiment
from evidently.presets import RegressionPreset, DataDriftPreset
from evidently import Report
from scipy.stats import pearsonr, spearmanr

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import TextEvals
from evidently.tests import lte, gte, eq
from evidently.descriptors import LLMEval, TestSummary, DeclineLLMEval, Sentiment, TextLength, IncludesWords
from evidently.llm.templates import BinaryClassificationPromptTemplate
from evidently.ui.workspace import CloudWorkspace
from sklearn import datasets
from evidently import Dataset
from evidently import DataDefinition, Regression
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset 
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import DashboardPanelPlot
import genai

from evidently import MulticlassClassification
from evidently import BinaryClassification
from evidently import Report
from evidently.metrics import *
from evidently.presets import *
from evidently import compare
from evidently.metrics.group_by import GroupBy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, confusion_matrix
from sklearn.metrics import ndcg_score, average_precision_score, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

import time
import uuid
from tracely import init_tracing, create_trace_event
from evidently.ui.workspace import CloudWorkspace
import logging
from evidently.metrics import RecallTopK, PrecisionTopK, FBetaTopK, MAP, NDCG, HitRate, MRR, ScoreDistribution

# top-level objects
from evidently import Report, Dataset, DataDefinition

# ranking / recsys metrics (current names)
from evidently.metrics import (
    RecallTopK,
    PrecisionTopK,
    FBetaTopK,
    MAP,
    
    NDCG,
    HitRate,
    MRR,
    ScoreDistribution,   # score-entropy metric for score-based recs
)


# presets (if you want presets instead of hand-picking metrics)
from evidently.presets import *
# imports (as above)
from evidently import Report, Dataset, DataDefinition
from evidently.metrics import RecallTopK, PrecisionTopK, NDCG, MRR
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Database setup
Base = declarative_base()
engine = create_engine("sqlite:///./mydatabase.db", echo=False)

# Comet.ml Configuration
COMET_API_KEY = "I1kQa57lnn3J3JGWpt56S5E9L"

# Evidently Configuration
EVIDENTLY_API_TOKEN = "dG9rbgH4nmfHlFJFD6jpXzD6aJEPglYafbDTljkJGlGbMt7gEABQrLfd+5Jk4zSp0U3FXhpcEug2syl35hVgD1o7/roMtghiPtVWXjMz9Gi0kFbB0wGKT4rOG6bj17YrSmJVHDNGBPZhjL9/YfdwL4qZZ7BzsYxVqYKw"
EVIDENTLY_ORG_ID = "019931b3-8662-7b61-816a-d577d2c12edf"
PROJECT_NAME = "llmeval"

# Database Models
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

class Interaction(Base):
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
    time_of_day = Column(String)

class Impression(Base):
    __tablename__ = "impressions"
    impression_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    item_id = Column(Integer, ForeignKey("items.item_id"))
    prompt = Column(Text)
    explanation = Column(Text)
    rank_position = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    relevance_score = Column(Float)

class Feedback(Base):
    __tablename__ = "feedback"
    feedback_id = Column(Integer, primary_key=True, autoincrement=True)
    impression_id = Column(Integer, ForeignKey("impressions.impression_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    rating = Column(Float)
    comment = Column(Text)
    clicked = Column(Integer)
    purchased = Column(Integer)
    watch_time = Column(Float)
    skip_after_seconds = Column(Float)
    rewatched = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)

class RetrievalSession(Base):
    __tablename__ = "retrieval_sessions"
    session_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    prompt = Column(Text)
    candidates = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Drop and recreate retrieval_sessions to fix user_id type
#Base.metadata.drop_all(engine, tables=[RetrievalSession.__table__])
Base.metadata.create_all(engine)

# Enhanced Retrieval System Classes
class FixedVectorDatabase:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.index = None
        self.item_ids = []
        self.item_texts = []
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None

    def build_index(self, items_df: pd.DataFrame) -> None:
        try:
            logger.info(f"Building index for {len(items_df)} items...")
            self.item_ids = []
            self.item_texts = []
            
            for _, row in items_df.iterrows():
                text_parts = []
                name = str(row.get('name', '')).strip()
                category = str(row.get('category', '')).strip()
                brand = str(row.get('brand', '')).strip()
                description = str(row.get('description', '')).strip()
                
                for part in [name, category, brand, description]:
                    if part and part.lower() not in ['nan', 'none', '']:
                        text_parts.append(part)
                
                if not text_parts:
                    logger.warning(f"No text content for item {row.get('item_id', 'unknown')}")
                    continue
                
                combined_text = ' '.join(text_parts)
                self.item_texts.append(combined_text)
                self.item_ids.append(int(row['item_id']))  # Ensure int
            
            if not self.item_texts:
                raise Exception("No valid item texts to index")
            
            logger.info(f"Processing {len(self.item_texts)} items with text content")
            embeddings = self.model.encode(self.item_texts, convert_to_numpy=True, show_progress_bar=False)
            
            self.index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=5000,
                ngram_range=(1, 2),
                min_df=1,
                lowercase=True
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.item_texts)
            
            logger.info(f"Index built successfully: {len(self.item_ids)} items indexed")
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise

    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        try:
            if self.index is None:
                logger.error("Index not built")
                return []
                
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            dense_scores, dense_indices = self.index.search(query_embedding.astype('float32'), min(top_k * 2, len(self.item_ids)))
            
            query_tfidf = self.tfidf_vectorizer.transform([query])
            sparse_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
            
            results = {}
            
            for idx, (score, item_idx) in enumerate(zip(dense_scores[0], dense_indices[0])):
                if item_idx < len(self.item_ids):
                    item_id = self.item_ids[item_idx]
                    results[item_id] = {
                        'item_id': item_id,
                        'dense_similarity': float(max(0, score)),
                        'sparse_similarity': 0.0,
                        'dense_rank': idx + 1
                    }
            
            sparse_indices = np.argsort(sparse_scores)[::-1][:top_k * 2]
            
            for idx, item_idx in enumerate(sparse_indices):
                if item_idx < len(self.item_ids):
                    item_id = self.item_ids[item_idx]
                    sparse_score = sparse_scores[item_idx]
                    
                    if item_id in results:
                        results[item_id]['sparse_similarity'] = float(sparse_score)
                        results[item_id]['sparse_rank'] = idx + 1
                    else:
                        results[item_id] = {
                            'item_id': item_id,
                            'dense_similarity': 0.0,
                            'sparse_similarity': float(sparse_score),
                            'sparse_rank': idx + 1
                        }
            
            for item_id, scores in results.items():
                hybrid_score = 0.7 * scores['dense_similarity'] + 0.3 * scores['sparse_similarity']
                scores['hybrid_score'] = hybrid_score
                scores['final_score'] = hybrid_score
            
            sorted_results = sorted(results.values(), key=lambda x: x['hybrid_score'], reverse=True)[:top_k]
            return sorted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

# Initialize global vector database
vector_db = FixedVectorDatabase()

# FastAPI app
app = FastAPI(title="Enhanced Recommendation System with Vector Database", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Groq Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", "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"))

# Pydantic Models
class RecommendationRequest(BaseModel):
    user_id: int
    prompt: str
    top_n: int = 20

class RecommendationResponse(BaseModel):
    user_id: int
    prompt: str
    impressions: List[Dict[str, Any]]
    total_candidates: int
    session_id: int
    status: str = "success"

class FeedbackRequest(BaseModel):
    impression_id: int
    rating: Optional[float] = None
    comment: Optional[str] = None
    clicked: Optional[bool] = None
    purchased: Optional[bool] = None
    watch_time: Optional[float] = None
    skip_after_seconds: Optional[float] = None
    rewatched: Optional[int] = None

class FeedbackResponse(BaseModel):
    status: str = "success"
    feedback_id: int
    message: str

class UserInfo(BaseModel):
    user_id: int
    name: str
    age: int
    location: str
    preferred_genres: str

class RerankRequest(BaseModel):
    user_id: int
    prompt: str
    candidates: List[Dict[str, Any]]
    top_k: int = 5

class RetrievalSessionRequest(BaseModel):
    user_id: int
    prompt: str
    candidates: List[Dict[str, Any]]

class RetrievalSessionResponse(BaseModel):
    session_id: int
    user_id: int
    prompt: str
    candidates: List[Dict[str, Any]]
    timestamp: datetime
    status: str = "success"

class CometMetricsRequest(BaseModel):
    user_id: Optional[int] = None
    experiment_name: str = "recommendation_metrics"
    project_name: str = "recsys_backend"

class TraceSessionRequest(BaseModel):
    user_id: int
# Pydantic Model for the POST request
# Request model

# Helper Functions
def get_user_profile(db: Session, user_id: int) -> Optional[Dict]:
    user = db.query(User).filter(User.user_id == user_id).first()
    if user:
        return {
            "user_id": user.user_id,
            "name": user.name,
            "age": user.age,
            "location": user.location,
            "preferred_genres": user.preferred_genres
        }
    return None

def get_all_users_profiles() -> pd.DataFrame:
    try:
        users_df = pd.read_sql("SELECT user_id, age, location, preferred_genres FROM users", engine)
        users_df['user_id'] = users_df['user_id'].astype(int)
        return users_df
    except Exception as e:
        logger.error(f"Error getting all users profiles: {e}")
        return pd.DataFrame()

def get_all_interactions() -> pd.DataFrame:
    try:
        query = """
        SELECT user_id, item_id, clicked, purchased, watch_time, skip_after_seconds, rewatched
        FROM interactions
        """
        interactions_df = pd.read_sql(query, engine)
        interactions_df['user_id'] = interactions_df['user_id'].astype(int)
        interactions_df['item_id'] = interactions_df['item_id'].astype(int)
        interactions_df['engagement_score'] = (
            interactions_df['purchased'] * 1.0 +
            interactions_df['clicked'] * 0.5 +
            interactions_df['rewatched'] * 0.5 +
            (interactions_df['watch_time'] - interactions_df['skip_after_seconds']).clip(lower=0) / 300
        )
        return interactions_df
    except Exception as e:
        logger.error(f"Error getting all interactions: {e}")
        return pd.DataFrame()

def create_user_embeddings() -> Dict[int, np.ndarray]:
    users_df = get_all_users_profiles()
    interactions_df = get_all_interactions()
    
    if users_df.empty or interactions_df.empty:
        return {}
    
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        location_encoded = encoder.fit_transform(users_df[['location']])
        genres_encoded = encoder.fit_transform(users_df[['preferred_genres']])
        
        scaler = StandardScaler()
        age_normalized = scaler.fit_transform(users_df[['age']].fillna(0))
        
        profile_features = np.hstack([location_encoded, genres_encoded, age_normalized])
        
        item_ids = pd.read_sql("SELECT item_id FROM items", engine)['item_id'].astype(int).tolist()
        num_items = len(item_ids)
        item_id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        engagement_matrix = np.zeros((len(users_df), num_items))
        user_id_to_index = {user_id: idx for idx, user_id in enumerate(users_df['user_id'])}
        
        for _, row in interactions_df.iterrows():
            user_idx = user_id_to_index.get(row['user_id'])
            item_idx = item_id_to_index.get(row['item_id'])
            if user_idx is not None and item_idx is not None:
                engagement_matrix[user_idx, item_idx] = row['engagement_score']
        
        user_embeddings = np.hstack([profile_features, engagement_matrix])
        embeddings_dict = {int(users_df['user_id'].iloc[i]): user_embeddings[i] for i in range(len(users_df))}
        
        return embeddings_dict
        
    except Exception as e:
        logger.error(f"Error creating user embeddings: {e}")
        return {}

def create_item_embeddings() -> tuple[List[int], np.ndarray]:
    try:
        items_df = pd.read_sql("SELECT item_id, category, brand, price FROM items", engine)
        items_df['item_id'] = items_df['item_id'].astype(int)
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        categorical_features = encoder.fit_transform(items_df[['category', 'brand']])
        
        scaler = StandardScaler()
        price_features = scaler.fit_transform(items_df[['price']].fillna(0))
        
        item_embeddings = np.hstack([categorical_features, price_features])
        return items_df['item_id'].tolist(), item_embeddings
    except Exception as e:
        logger.error(f"Error creating item embeddings: {e}")
        return [], np.array([])

def get_similar_users(target_user_id: int, top_k: int = 5) -> List[int]:
    user_embeddings = create_user_embeddings()
    if not user_embeddings:
        return []
    
    target_embedding = user_embeddings.get(target_user_id)
    if target_embedding is None:
        return []
    
    other_users = {uid: emb for uid, emb in user_embeddings.items() if uid != target_user_id}
    other_uids = list(other_users.keys())
    other_embeddings = np.array(list(other_users.values()))
    
    similarities = cosine_similarity([target_embedding], other_embeddings)[0]
    similar_indices = np.argsort(similarities)[::-1][:top_k]
    similar_users = [other_uids[i] for i in similar_indices]
    
    return similar_users

def vector_search_no_cache(query: str, top_k: int = 50) -> List[Dict]:
    try:
        if vector_db.index is None:
            logger.warning("Vector index is None, rebuilding...")
            items_df = pd.read_sql("SELECT item_id, name, category, brand, description FROM items", engine)
            items_df['item_id'] = items_df['item_id'].astype(int)
            if items_df.empty:
                logger.error("No items in database to build index")
                return []
            vector_db.build_index(items_df)
        
        results = vector_db.search(query, top_k=top_k)
        logger.info(f"Vector search for '{query}' returned {len(results)} results")
        
        for i, result in enumerate(results[:3]):
            logger.info(f"  Result {i+1}: item_id={result['item_id']}, hybrid_score={result.get('hybrid_score', 0):.3f}")
        
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

def retrieve_candidates_fixed(user_id: int, prompt: str, top_n: int = 20, db: Session = None) -> tuple[List[Dict], Optional[int]]:
    db = db or SessionLocal()
    try:
        logger.info(f"=== RETRIEVAL START === User: {user_id}, Query: '{prompt}', Top: {top_n}")
        
        user_profile = get_user_profile(db, user_id)
        if not user_profile:
            logger.error(f"User {user_id} not found")
            return [], None
        
        items_df = pd.read_sql("SELECT item_id, name, category, brand, price, description FROM items", engine)
        items_df['item_id'] = items_df['item_id'].astype(int)
        logger.info(f"Total items in database: {len(items_df)}")
        
        if items_df.empty:
            logger.error("No items in database")
            return [], None
        
        interactions = get_all_interactions()
        user_interactions = interactions[interactions['user_id'] == user_id]
        interacted_items = set(user_interactions['item_id']) if not user_interactions.empty else set()
        logger.info(f"User has {len(interacted_items)} previous interactions")
        
        global vector_db
        if not hasattr(vector_db, 'index') or vector_db.index is None:
            logger.info("Initializing vector database...")
            vector_db = FixedVectorDatabase()
            vector_db.build_index(items_df)
        
        logger.info(f"Performing vector search for: '{prompt}'")
        search_results = vector_db.search(prompt, top_k=top_n * 3)
        logger.info(f"Vector search returned {len(search_results)} results")
        
        if not search_results:
            logger.warning("Vector search returned no results, using fallback")
            return retrieve_candidates_simple_fallback(user_id, prompt, top_n, items_df, interacted_items, db)
        
        candidates = []
        for result in search_results:
            item_id = int(result['item_id'])
            if item_id in interacted_items:
                continue
            
            item_row = items_df[items_df['item_id'] == item_id]
            if item_row.empty:
                logger.warning(f"Item {item_id} not found in items_df")
                continue
            
            item_info = item_row.iloc[0]
            
            candidate = {
                'item_id': item_id,
                'name': str(item_info['name']),
                'category': str(item_info['category']),
                'brand': str(item_info['brand']),
                'price': float(item_info['price']) if pd.notna(item_info['price']) else 0.0,
                'description': str(item_info['description']) if pd.notna(item_info['description']) else '',
                'similarity_score': result['hybrid_score'],
                'final_score': result['hybrid_score'],
                'dense_similarity': result['dense_similarity'],
                'sparse_similarity': result['sparse_similarity']
            }
            
            candidates.append(candidate)
            if len(candidates) >= top_n:
                break
        
        preferred_genres = user_profile.get('preferred_genres', '').split(',') if user_profile else []
        preferred_genres = [g.strip() for g in preferred_genres if g.strip()]
        
        if preferred_genres:
            for candidate in candidates:
                if candidate['category'] in preferred_genres:
                    candidate['final_score'] += 0.2
                    logger.debug(f"Boosted {candidate['name']} for preferred genre {candidate['category']}")
        
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        final_candidates = candidates[:top_n]
        
        try:
            candidates_json = json.dumps(final_candidates)
            retrieval_session = RetrievalSession(
                user_id=user_id,
                prompt=prompt,
                candidates=candidates_json,
                timestamp=datetime.utcnow()
            )
            
            db.add(retrieval_session)
            db.commit()
            db.refresh(retrieval_session)
            session_id = retrieval_session.session_id
        except Exception as e:
            logger.error(f"Failed to log retrieval session: {e}")
            db.rollback()
            session_id = None
        
        logger.info(f"=== RETRIEVAL COMPLETE === Returning {len(final_candidates)} candidates, Session ID: {session_id}")
        for i, cand in enumerate(final_candidates[:3]):
            logger.info(f"  Top {i+1}: {cand['name']} (score: {cand['final_score']:.3f})")
        
        return final_candidates, session_id
        
    except Exception as e:
        logger.error(f"retrieve_candidates_fixed failed: {e}")
        return retrieve_candidates_simple_fallback(user_id, prompt, top_n, items_df, interacted_items, db), None
    finally:
        db.close()

def retrieve_candidates_simple_fallback(user_id: int, prompt: str, top_n: int, items_df: pd.DataFrame, interacted_items: set, db: Session) -> tuple[List[Dict], Optional[int]]:
    try:
        logger.info("Using simple text matching fallback")
        available_items = items_df[~items_df['item_id'].isin(interacted_items)]
        
        if available_items.empty:
            return [], None
        
        query_words = prompt.lower().split()
        scores = []
        
        for _, row in available_items.iterrows():
            searchable = f"{row['name']} {row['category']} {row['brand']} {row.get('description', '')}".lower()
            matches = sum(1 for word in query_words if word in searchable)
            score = matches / len(query_words) if query_words else 0
            scores.append(score)
        
        available_items = available_items.copy()
        available_items['score'] = scores
        top_matches = available_items.nlargest(top_n, 'score')
        
        candidates = []
        for _, row in top_matches.iterrows():
            candidates.append({
                'item_id': int(row['item_id']),
                'name': str(row['name']),
                'category': str(row['category']),
                'brand': str(row['brand']),
                'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                'description': str(row['description']) if pd.notna(row['description']) else '',
                'similarity_score': row['score'],
                'final_score': row['score']
            })
        
        try:
            candidates_json = json.dumps(candidates)
            retrieval_session = RetrievalSession(
                user_id=user_id,
                prompt=prompt,
                candidates=candidates_json,
                timestamp=datetime.utcnow()
            )
            
            db.add(retrieval_session)
            db.commit()
            db.refresh(retrieval_session)
            session_id = retrieval_session.session_id
        except Exception as e:
            logger.error(f"Failed to log retrieval session in fallback: {e}")
            db.rollback()
            session_id = None
        
        logger.info(f"Fallback returned {len(candidates)} candidates, Session ID: {session_id}")
        return candidates, session_id
        
    except Exception as e:
        logger.error(f"Even fallback failed: {e}")
        return [], None
    finally:
        db.close()

def safe_get_text(obj: Dict, key: str, default: str = '') -> str:
    return str(obj.get(key, default)).strip() if obj.get(key) is not None else default

def llm_rerank_and_explain(user_id: int, user_profile: Dict, prompt: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
    try:
        if not candidates:
            return []

        candidates_str = ""
        for i, cand in enumerate(candidates[:15]):
            item_id = f"Item {cand['item_id']}"
            candidates_str += f"{item_id}: {cand['name']}\n"
            candidates_str += f"  Category: {cand['category']}, Brand: {cand['brand']}, Price: ${cand['price']:.2f}\n"
            candidates_str += f"  Similarity Score: {cand.get('similarity_score', 0):.3f}\n\n"

        current_time = datetime.now()
        hour = current_time.hour
        time_of_day = (
            "morning" if 6 <= hour < 12 else
            "afternoon" if 12 <= hour < 18 else
            "evening" if 18 <= hour < 24 else "night"
        )

        llm_prompt = f"""
You are an expert recommendation system.

User Profile:
- User ID: {user_id}
- Name: {user_profile.get('name', 'Unknown')}
- Age: {user_profile.get('age', 'Unknown')}
- Location: {user_profile.get('location', 'Unknown')}
- Preferred Genres: {user_profile.get('preferred_genres', 'Unknown')}

Current Context:
- Time of Day: {time_of_day}

User Query: "{prompt}"

Top Candidate Items:
{candidates_str}

Task: Rerank and select the top {top_k} most relevant items for this user's query. Use the exact 'Item X' format (e.g., 'Item 30') from the candidate list as the 'item_id' in your response. Respond ONLY with valid JSON, no extra text or code blocks:

{{
  "ranked_items": [
    {{
      "item_id": "Item 30",
      "rank": 1,
      "explanation": "Why this item is recommended",
      "relevance_score": 0.95
    }}
  ]
}}
"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.3,
            max_tokens=1500
        )

        response_content = ""
        try:
            if hasattr(response, "choices") and response.choices:
                response_content = response.choices[0].message.content.strip()
                logger.info(f"LLM raw response: {response_content}")
        except Exception as e:
            logger.error(f"Failed to extract LLM response content: {e}")
            response_content = ""

        result = {"ranked_items": []}
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.error("Manual JSON parsing failed; using empty result")
            else:
                logger.error("No valid JSON found in response")

        ranked_items = result.get("ranked_items", [])[:top_k]

        if not ranked_items:
            logger.warning("LLM returned no valid ranked items; using fallback")

        valid_ranked_items = []
        for item in ranked_items:
            item_id = safe_get_text(item, 'item_id')
            if not item_id.startswith("Item "):
                item_id = f"Item {item_id}"
            
            try:
                numeric_item_id = int(item_id.replace("Item ", ""))
            except ValueError:
                logger.warning(f"Invalid item_id format: {item_id}")
                continue

            original_candidate = next((c for c in candidates if c['item_id'] == numeric_item_id), None)
            
            if original_candidate:
                enriched_item = {
                    'item_id': numeric_item_id,
                    'rank': int(safe_get_text(item, 'rank', '1')),
                    'explanation': safe_get_text(item, 'explanation', f"Recommended based on user preferences for '{prompt}'."),
                    'relevance_score': float(safe_get_text(item, 'relevance_score', str(original_candidate.get('final_score', 0.8)))),
                    'name': original_candidate.get('name', ''),
                    'category': original_candidate.get('category', ''),
                    'brand': original_candidate.get('brand', ''),
                    'price': original_candidate.get('price', 0.0),
                    'description': original_candidate.get('description', ''),
                    'original_score': original_candidate.get('final_score', 0.0)
                }
                
                valid_ranked_items.append(enriched_item)
            else:
                logger.warning(f"No matching candidate for item_id: {item_id}")

        if not valid_ranked_items:
            logger.warning("Falling back to original candidates due to LLM failure")
            for i, cand in enumerate(candidates[:top_k]):
                valid_ranked_items.append({
                    'item_id': cand['item_id'],
                    'rank': i + 1,
                    'explanation': f"Fallback recommendation based on vector search for '{prompt}'.",
                    'relevance_score': cand.get('final_score', 0.8),
                    'name': cand.get('name', ''),
                    'category': cand.get('category', ''),
                    'brand': cand.get('brand', ''),
                    'price': cand.get('price', 0.0),
                    'description': cand.get('description', ''),
                    'original_score': cand.get('final_score', 0.0)
                })

        return valid_ranked_items

    except Exception as e:
        logger.error(f"LLM reranking failed completely: {e}")
        return [{
            'item_id': c['item_id'],
            'rank': i + 1,
            'explanation': f"Final fallback based on vector search for '{prompt}'.",
            'relevance_score': c.get('final_score', 0.8),
            'name': c.get('name', ''),
            'category': c.get('category', ''),
            'brand': c.get('brand', ''),
            'price': c.get('price', 0.0),
            'description': c.get('description', ''),
            'original_score': c.get('final_score', 0.0)
        } for i, c in enumerate(candidates[:top_k])]

# API Endpoints
@app.get("/")
def read_root():
    return {
        "message": "Enhanced Recommendation System API with Vector Database",
        "version": "2.0",
        "features": ["Dense Vector Search", "Hybrid Retrieval", "FAISS Indexing", "Semantic Embeddings", "Retrieval Sessions", "Comet Metrics", "Evidently Reporting"],
        "endpoints": [
            "/users", "/recommend", "/feedback", "/recommendations/{user_id}",
            "/feedback/{impression_id}", "/analytics/user/{user_id}", "/evaluate/{user_id}",
            "/retrieve", "/rerank", "/rebuild-index", "/vector-stats", "/retrieval-sessions",
            "/retrieval-sessions/{user_id}", "/comet/log-metrics", "/evidently-report/{user_id}",
            "/run_classification_report"
        ]
    }

@app.get("/users", response_model=List[UserInfo])
def get_all_users(db: Session = Depends(get_db)):
    try:
        users = db.query(User).all()
        return [UserInfo(
            user_id=user.user_id,
            name=user.name,
            age=user.age,
            location=user.location,
            preferred_genres=user.preferred_genres
        ) for user in users]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")

@app.get("/users/{user_id}", response_model=UserInfo)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfo(
        user_id=user.user_id,
        name=user.name,
        age=user.age,
        location=user.location,
        preferred_genres=user.preferred_genres
    )

@app.post("/recommend", response_model=RecommendationResponse)
def generate_recommendations_fixed(request: RecommendationRequest, db: Session = Depends(get_db)):
    try:
        user_profile = get_user_profile(db, request.user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(
            f"=== RECOMMENDATION REQUEST === User: {request.user_id}, "
            f"Prompt: '{request.prompt}', Top: {request.top_n}"
        )

        candidates, session_id = retrieve_candidates_fixed(
            request.user_id, request.prompt, request.top_n, db
        )

        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant items found for your query"
            )

        logger.info(f"Retrieved {len(candidates)} candidates, proceeding to LLM reranking...")

        top_k = min(5, len(candidates))
        ranked_items = llm_rerank_and_explain(
            request.user_id, user_profile, request.prompt, candidates, top_k
        )

        impressions_data = []
        for item in ranked_items:
            impression = Impression(
                user_id=request.user_id,
                item_id=item['item_id'],
                prompt=request.prompt,
                explanation=item['explanation'],
                rank_position=item['rank'],
                relevance_score=item['relevance_score'],
                timestamp=datetime.utcnow()
            )

            db.add(impression)
            db.flush()

            impressions_data.append({
                "impression_id": impression.impression_id,
                "item_id": item['item_id'],
                "rank": item['rank'],
                "name": item['name'],
                "category": item['category'],
                "brand": item['brand'],
                "price": item['price'],
                "description": (
                    item['description'][:300] + "..." if len(item['description']) > 300 else item['description']
                ),
                "explanation": item['explanation'],
                "relevance_score": impression.relevance_score
            })

        db.commit()

        logger.info(
            f"=== RECOMMENDATION COMPLETE === Returning {len(impressions_data)} "
            f"recommendations, Session ID: {session_id}"
        )

        return RecommendationResponse(
            user_id=request.user_id,
            prompt=request.prompt,
            impressions=impressions_data,
            total_candidates=len(candidates),
            session_id=session_id,
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Recommendation generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/rerank")
def rerank(request: RerankRequest, db: Session = Depends(get_db)):
    try:
        user_profile = get_user_profile(db, request.user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")

        ranked_items = llm_rerank_and_explain(
            request.user_id,
            user_profile,
            request.prompt,
            request.candidates,
            request.top_k
        )

        impressions_data = []
        for item in ranked_items:
            if not item.get('item_id') or not item.get('explanation'):
                logger.warning(f"Skipping invalid item: {item}")
                continue

            try:
                impression = Impression(
                    user_id=request.user_id,
                    item_id=item['item_id'],
                    prompt=request.prompt,
                    explanation=item['explanation'],
                    rank_position=item['rank'],
                    relevance_score=item.get('relevance_score', 0.8),
                    timestamp=datetime.utcnow()
                )

                db.add(impression)
                db.flush()

                impressions_data.append({
                    "impression_id": impression.impression_id,
                    "item_id": item['item_id'],
                    "rank": item['rank'],
                    "name": item.get('name', ''),
                    "category": item.get('category', ''),
                    "brand": item.get('brand', ''),
                    "price": item.get('price', 0.0),
                    "description": (item.get('description','')[:300] + "...") if len(item.get('description','')) > 300 else item.get('description',''),
                    "explanation": item['explanation'],
                    "relevance_score": impression.relevance_score
                })

            except Exception as e:
                logger.error(f"Failed to save impression for item {item.get('item_id')}: {e}")
                db.rollback()
                continue

        db.commit()
        return impressions_data

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error in rerank endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/recommendations/{user_id}")
def get_user_recommendations(user_id: int, limit: int = 50, db: Session = Depends(get_db)):
    try:
        query = """
        SELECT
            i.impression_id,
            i.item_id,
            i.prompt,
            i.explanation,
            i.rank_position,
            i.timestamp,
            i.relevance_score,
            it.name,
            it.category,
            it.brand,
            it.price
        FROM impressions i
        JOIN items it ON i.item_id = it.item_id
        WHERE i.user_id = ?
        ORDER BY i.timestamp DESC
        LIMIT ?
        """
        
        recommendations_df = pd.read_sql(query, engine, params=(user_id, limit))
        
        if recommendations_df.empty:
            return {"user_id": user_id, "recommendations": [], "count": 0}
        
        recommendations = recommendations_df.to_dict(orient='records')
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recommendations: {str(e)}")

@app.post("/feedback", response_model=FeedbackResponse)
def log_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    try:
        impression = db.query(Impression).filter(Impression.impression_id == request.impression_id).first()
        if not impression:
            raise HTTPException(status_code=404, detail="Impression not found")

        watch_time = request.watch_time if request.watch_time is not None else 0.0
        skip_after_seconds = request.skip_after_seconds if request.skip_after_seconds is not None else 0.0
        rewatched = request.rewatched if request.rewatched is not None else 0

        feedback = db.query(Feedback).filter(
            Feedback.impression_id == request.impression_id
        ).first()

        if feedback:
            if request.clicked:
                feedback.clicked = (feedback.clicked or 0) + 1
            if request.purchased:
                feedback.purchased = (feedback.purchased or 0) + 1
            feedback.watch_time = max(feedback.watch_time or 0, watch_time)
            feedback.skip_after_seconds = skip_after_seconds
            feedback.rewatched = (feedback.rewatched or 0) + rewatched
            if request.rating is not None:
                feedback.rating = request.rating
            if request.comment:
                feedback.comment = request.comment
            feedback.timestamp = datetime.utcnow()
            db.commit()
            feedback_id = feedback.feedback_id
        else:
            feedback = Feedback(
                impression_id=request.impression_id,
                user_id=impression.user_id,
                rating=request.rating,
                comment=request.comment,
                clicked=1 if request.clicked else 0,
                purchased=1 if request.purchased else 0,
                watch_time=watch_time,
                skip_after_seconds=skip_after_seconds,
                rewatched=rewatched,
                timestamp=datetime.utcnow()
            )

            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            feedback_id = feedback.feedback_id

        if request.clicked or request.purchased:
            try:
                current_time = datetime.now()
                hour = current_time.hour
                time_of_day = (
                    "morning" if 6 <= hour < 12 else
                    "afternoon" if 12 <= hour < 18 else
                    "evening" if 18 <= hour < 24 else "night"
                )

                interaction = Interaction(
                    user_id=impression.user_id,
                    item_id=impression.item_id,
                    clicked=1 if request.clicked else 0,
                    purchased=1 if request.purchased else 0,
                    watch_time=watch_time,
                    skip_after_seconds=skip_after_seconds,
                    rewatched=rewatched,
                    session_id=f"feedback_{feedback_id}",
                    timestamp=datetime.utcnow(),
                    time_of_day=time_of_day
                )

                db.add(interaction)
                db.commit()
            except Exception as interaction_error:
                logger.error(f"Failed to create interaction: {interaction_error}")

        return FeedbackResponse(
            status="success",
            feedback_id=feedback_id,
            message="Feedback logged successfully"
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error logging feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging feedback: {str(e)}")

@app.get("/feedback/{impression_id}")
def get_feedback(impression_id: int, db: Session = Depends(get_db)):
    try:
        query = """
        SELECT
            f.feedback_id,
            f.impression_id,
            f.rating,
            f.comment,
            f.clicked,
            f.purchased,
            f.watch_time,
            f.skip_after_seconds,
            f.rewatched,
            f.timestamp,
            i.user_id,
            i.item_id,
            i.prompt,
            i.relevance_score,
            it.name as item_name
        FROM feedback f
        JOIN impressions i ON f.impression_id = i.impression_id
        JOIN items it ON i.item_id = it.item_id
        WHERE f.impression_id = ?
        """
        
        feedback_df = pd.read_sql(query, engine, params=(impression_id,))
        
        if feedback_df.empty:
            return {"impression_id": impression_id, "feedback": [], "count": 0}
        
        feedback_records = feedback_df.to_dict(orient='records')
        
        return {
            "impression_id": impression_id,
            "feedback": feedback_records,
            "count": len(feedback_records)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching feedback: {str(e)}")

@app.get("/analytics/user/{user_id}")
def get_user_analytics(user_id: int, db: Session = Depends(get_db)):
    try:
        impressions_query = """
        SELECT COUNT(*) as total_impressions,
               COUNT(DISTINCT prompt) as unique_prompts,
               COUNT(DISTINCT item_id) as unique_items_recommended,
               AVG(relevance_score) as avg_relevance_score
        FROM impressions
        WHERE user_id = ?
        """
        
        impressions_stats = pd.read_sql(impressions_query, engine, params=(user_id,)).iloc[0].to_dict()

        feedback_query = """
        SELECT
            COUNT(*) as total_feedback,
            SUM(CASE WHEN clicked = 1 THEN 1 ELSE 0 END) as total_clicks,
            SUM(CASE WHEN purchased = 1 THEN 1 ELSE 0 END) as total_purchases,
            AVG(CASE WHEN watch_time IS NOT NULL THEN watch_time ELSE NULL END) as avg_watch_time
        FROM feedback f
        JOIN impressions i ON f.impression_id = i.impression_id
        WHERE i.user_id = ?
        """
        
        feedback_stats = pd.read_sql(feedback_query, engine, params=(user_id,)).iloc[0].to_dict()

        return {
            "user_id": user_id,
            "impressions_stats": impressions_stats,
            "feedback_stats": feedback_stats
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user analytics: {str(e)}")

@app.get("/evaluate/{user_id}")
def evaluate_recommendations(user_id: int, db: Session = Depends(get_db)):
    try:
        query = """
        SELECT i.item_id
        FROM impressions i
        WHERE i.user_id = ?
        ORDER BY i.timestamp DESC
        LIMIT 10
        """
        
        recommended_items = pd.read_sql(query, engine, params=(user_id,))['item_id'].tolist()

        query = """
        SELECT item_id
        FROM interactions
        WHERE user_id = ? AND (purchased = 1 OR (watch_time > 100 AND rewatched = 1))
        """
        
        relevant_items = set(pd.read_sql(query, engine, params=(user_id,))['item_id'].tolist())

        y_true = [1 if item in relevant_items else 0 for item in recommended_items]

        precision = sum(y_true) / len(recommended_items) if recommended_items else 0.0
        recall = sum(y_true) / len(relevant_items) if relevant_items else 0.0

        query = """
        SELECT DISTINCT it.category
        FROM impressions i
        JOIN items it ON i.item_id = it.item_id
        WHERE i.user_id = ?
        ORDER BY i.timestamp DESC
        LIMIT 10
        """
        
        categories = pd.read_sql(query, engine, params=(user_id,))['category'].tolist()
        diversity = len(set(categories)) / max(len(categories), 1)

        return {
            "user_id": user_id,
            "precision": precision,
            "recall": recall,
            "diversity": diversity,
            "recommended_items_count": len(recommended_items),
            "relevant_items_count": len(relevant_items)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating recommendations: {str(e)}")

@app.post("/retrieve")
def retrieve(request: RecommendationRequest, db: Session = Depends(get_db)):
    try:
        user_profile = get_user_profile(db, request.user_id)
        if not user_profile:
            raise HTTPException(status_code=404, detail="User not found")

        candidates, session_id = retrieve_candidates_fixed(request.user_id, request.prompt, request.top_n, db)

        if not candidates:
            raise HTTPException(status_code=404, detail="No relevant items found")

        return {
            "user_id": request.user_id,
            "prompt": request.prompt,
            "candidates": candidates,
            "total_candidates": len(candidates),
            "session_id": session_id,
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in retrieve: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/retrieval-sessions", response_model=RetrievalSessionResponse)
def log_retrieval_session(request: RetrievalSessionRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.user_id == request.user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        try:
            candidates_json = json.dumps(request.candidates)
        except Exception as e:
            logger.error(f"Failed to serialize candidates: {e}")
            raise HTTPException(status_code=400, detail="Invalid candidates format")

        retrieval_session = RetrievalSession(
            user_id=request.user_id,
            prompt=request.prompt,
            candidates=candidates_json,
            timestamp=datetime.utcnow()
        )

        db.add(retrieval_session)
        db.commit()
        db.refresh(retrieval_session)

        return RetrievalSessionResponse(
            session_id=retrieval_session.session_id,
            user_id=retrieval_session.user_id,
            prompt=retrieval_session.prompt,
            candidates=json.loads(retrieval_session.candidates),
            timestamp=retrieval_session.timestamp,
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error logging retrieval session: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/retrieval-sessions/{user_id}")
def get_retrieval_sessions(user_id: int, limit: int = 10, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        sessions = db.query(RetrievalSession).filter(RetrievalSession.user_id == user_id).order_by(RetrievalSession.timestamp.desc()).limit(limit).all()

        sessions_data = []
        for session in sessions:
            try:
                candidates = json.loads(session.candidates) if session.candidates else []
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse candidates for session {session.session_id}: {e}")
                candidates = []

            sessions_data.append({
                "session_id": session.session_id,
                "user_id": session.user_id,
                "prompt": session.prompt,
                "candidates": candidates,
                "timestamp": session.timestamp
            })

        return {
            "user_id": user_id,
            "sessions": sessions_data,
            "count": len(sessions_data)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching retrieval sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching retrieval sessions: {str(e)}")

@app.post("/rebuild-index")
def rebuild_vector_index():
    try:
        logger.info("Starting vector index rebuild...")
        items_df = pd.read_sql("SELECT item_id, name, category, brand, description FROM items", engine)
        items_df['item_id'] = items_df['item_id'].astype(int)
        
        if items_df.empty:
            return {"status": "error", "message": "No items found in database"}
        
        vector_db.build_index(items_df)
        
        return {
            "status": "success",
            "message": f"Vector index rebuilt successfully with {len(items_df)} items",
            "item_count": len(items_df)
        }
        
    except Exception as e:
        logger.error(f"Error rebuilding index: {e}")
        return {"status": "error", "message": f"Failed to rebuild index: {str(e)}"}

@app.get("/vector-stats")
def get_vector_stats():
    try:
        stats = {
            "index_exists": vector_db.index is not None,
            "item_count": len(vector_db.item_ids) if vector_db.item_ids else 0,
            "dimension": vector_db.dimension,
            "cache_info": bool(hasattr(vector_db, 'embeddings_cache')),
            "last_updated": getattr(vector_db, 'last_updated', 'Never')
        }
        
        return {"status": "success", "stats": stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/comet/log-metrics")
def log_comet_metrics(request: CometMetricsRequest, db: Session = Depends(get_db)):
    try:
        experiment = Experiment(
            api_key=COMET_API_KEY,
            project_name=request.project_name,
            experiment_name=request.experiment_name
        )

        try:
            db.execute(text("SELECT 1"))
            logger.info("SQLite3 database connection verified")
        except Exception as e:
            logger.error(f"Failed to connect to SQLite3 database: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

        impressions_query = """
        SELECT COUNT(*) as total_impressions,
               COUNT(DISTINCT user_id) as unique_users,
               COUNT(DISTINCT item_id) as unique_items,
               COUNT(DISTINCT prompt) as unique_prompts,
               AVG(relevance_score) as avg_relevance_score
        FROM impressions
        """
        
        if request.user_id:
            impressions_query += " WHERE user_id = ?"
            impressions_params = (request.user_id,)
            logger.info(f"Querying impressions for user_id: {request.user_id}")
        else:
            impressions_params = ()
            logger.info("Querying impressions for all users")

        impressions_stats = pd.read_sql(impressions_query, engine, params=impressions_params)

        if impressions_stats.empty:
            logger.warning("No impressions data found")
            impressions_stats = pd.DataFrame({
                'total_impressions': [0],
                'unique_users': [0],
                'unique_items': [0],
                'unique_prompts': [0],
                'avg_relevance_score': [0.0]
            })

        impressions_stats = impressions_stats.iloc[0].to_dict()

        feedback_query = """
        SELECT
            SUM(clicked) as total_clicks,
            SUM(purchased) as total_purchases,
            AVG(watch_time) as avg_watch_time,
            AVG(skip_after_seconds) as avg_skip_time,
            SUM(rewatched) as total_rewatched
        FROM feedback
        """
        
        if request.user_id:
            feedback_query += " WHERE user_id = ?"
            feedback_params = (request.user_id,)
            logger.info(f"Querying feedback for user_id: {request.user_id}")
        else:
            feedback_params = ()
            logger.info("Querying feedback for all users")

        feedback_stats = pd.read_sql(feedback_query, engine, params=feedback_params)

        if feedback_stats.empty:
            logger.warning("No feedback data found")
            feedback_stats = pd.DataFrame({
                'total_clicks': [0],
                'total_purchases': [0],
                'avg_watch_time': [0.0],
                'avg_skip_time': [0.0],
                'total_rewatched': [0]
            })

        feedback_stats = feedback_stats.iloc[0].to_dict()

        total_impressions = impressions_stats.get('total_impressions', 0)
        total_clicks = feedback_stats.get('total_clicks', 0) or 0
        total_purchases = feedback_stats.get('total_purchases', 0) or 0

        click_through_rate = total_clicks / total_impressions if total_impressions > 0 else 0.0
        purchase_rate = total_purchases / total_impressions if total_impressions > 0 else 0.0

        avg_watch_time = feedback_stats.get('avg_watch_time', 0.0) or 0.0
        avg_skip_time = feedback_stats.get('avg_skip_time', 0.0) or 0.0
        total_rewatched = feedback_stats.get('total_rewatched', 0) or 0
        avg_relevance_score = impressions_stats.get('avg_relevance_score', 0.0) or 0.0

        experiment.log_metric("total_impressions", total_impressions)
        experiment.log_metric("unique_users", impressions_stats.get('unique_users', 0))
        experiment.log_metric("unique_items", impressions_stats.get('unique_items', 0))
        experiment.log_metric("unique_prompts", impressions_stats.get('unique_prompts', 0))
        experiment.log_metric("total_clicks", total_clicks)
        experiment.log_metric("total_purchases", total_purchases)
        experiment.log_metric("click_through_rate", click_through_rate)
        experiment.log_metric("purchase_rate", purchase_rate)
        experiment.log_metric("avg_watch_time", avg_watch_time)
        experiment.log_metric("avg_skip_time", avg_skip_time)
        experiment.log_metric("total_rewatched", total_rewatched)
        experiment.log_metric("avg_relevance_score", avg_relevance_score)

        experiment.log_parameter("user_id", request.user_id if request.user_id else "all_users")
        experiment.log_parameter("timestamp", datetime.utcnow().isoformat())

        experiment.end()

        return {
            "status": "success",
            "message": f"Metrics logged to Comet.ml under experiment '{request.experiment_name}'",
            "metrics": {
                "total_impressions": total_impressions,
                "unique_users": impressions_stats.get('unique_users', 0),
                "unique_items": impressions_stats.get('unique_items', 0),
                "unique_prompts": impressions_stats.get('unique_prompts', 0),
                "total_clicks": total_clicks,
                "total_purchases": total_purchases,
                "click_through_rate": click_through_rate,
                "purchase_rate": purchase_rate,
                "avg_watch_time": avg_watch_time,
                "avg_skip_time": avg_skip_time,
                "total_rewatched": total_rewatched,
                "avg_relevance_score": avg_relevance_score
            }
        }

    except Exception as e:
        logger.error(f"Error logging metrics to Comet: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error logging metrics to Comet: {str(e)}")

@app.get("/api/evaluate-impressions/{userid}", response_class=JSONResponse)
def evaluate_impressions(userid: int, db: Session = Depends(get_db)):
    try:
        logger.info(f"Starting evaluation for user_id: {userid}")
        
        # Verify database connection
        try:
            db.execute(text("SELECT 1"))
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

        # Initialize Evidently Cloud workspace and project
        try:
            ws = CloudWorkspace(token=EVIDENTLY_API_TOKEN, url="https://app.evidently.cloud")
            try:
                project = ws.create_project(PROJECT_NAME, org_id=EVIDENTLY_ORG_ID)
                project.description = "Evaluation of impression prompts and explanations"
                project.save()
                logger.info(f"Created new Evidently project: {PROJECT_NAME}")
            except Exception:
                project = ws.get_project(PROJECT_NAME)
                logger.info(f"Retrieved existing Evidently project: {PROJECT_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Evidently Cloud workspace: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently Cloud initialization failed: {str(e)}")

        # Query impressions from database table
        query = """
            SELECT prompt, explanation
            FROM impressions
            WHERE user_id = :userid
            ORDER BY timestamp DESC
            LIMIT 100
        """
        try:
            impressions_df = pd.read_sql(query, db.bind, params={"userid": userid})
            logger.info(f"Retrieved {len(impressions_df)} impressions for user_id: {userid}")
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

        if impressions_df.empty:
            logger.info(f"No impressions found for user_id: {userid}")
            return {"message": "No impressions data found for the given user.", "data": [], "userid": userid}

        # Prepare Evidently Dataset for evaluation
        try:
            eval_dataset = Dataset.from_pandas(
                impressions_df,
                data_definition=DataDefinition(),
                descriptors=[
                    Sentiment("explanation", alias="Sentiment", tests=[gte(0, alias="Is_non_negative")]),
                    TextLength("explanation", alias="Length", tests=[lte(150, alias="Has_expected_length")]),
                    DeclineLLMEval("explanation", alias="Denials", tests=[eq("OK", column="Denials", alias="Is_not_a_refusal")]),
                    TestSummary(success_all=True, alias="All_tests_passed"),
                ],
            )
            logger.info("Evidently dataset prepared successfully")
        except Exception as e:
            logger.error(f"Failed to prepare Evidently dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently dataset preparation failed: {str(e)}")

        # Create and run an Evidently report
        try:
            report = Report([TextEvals()])
            evaluation_result = report.run(eval_dataset, None)
            logger.info("Evidently report generated successfully")
        except Exception as e:
            logger.error(f"Failed to run Evidently report: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently report generation failed: {str(e)}")

        # Upload evaluation results to Evidently Cloud
        try:
            ws.add_run(project.id, evaluation_result, include_data=True)
            logger.info(f"Uploaded evaluation results to Evidently Cloud for project: {PROJECT_NAME}")
        except Exception as e:
            logger.error(f"Failed to upload results to Evidently Cloud: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently Cloud upload failed: {str(e)}")

        # Return JSON evaluation report
        logger.info(f"Returning evaluation report for user_id: {userid}")
        import json
        return {"userid": userid, "evaluation_report": json.loads(evaluation_result.json())}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluate_impressions for user_id {userid}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/trace-session")
def trace_session(request: TraceSessionRequest, db: Session = Depends(get_db)):
    try:
        # Initialize Evidently Cloud workspace
        ws = CloudWorkspace(token=EVIDENTLY_API_TOKEN, url="https://app.evidently.cloud")

        # Check for existing llmeval project or create it
        project = None
        try:
            projects = ws.list_projects(org_id=EVIDENTLY_ORG_ID)
            project = next((p for p in projects if p.name == PROJECT_NAME), None)
            if not project:
                logger.info(f"Project '{PROJECT_NAME}' not found, creating new project")
                project = ws.create_project(PROJECT_NAME, org_id=EVIDENTLY_ORG_ID)
                project.description = "Tracing for LLM evaluation and recommendations"
                project.save()
                logger.info(f"Created project '{PROJECT_NAME}' with ID: {project.id}")
            else:
                logger.info(f"Using existing project '{PROJECT_NAME}' with ID: {project.id}")
        except Exception as e:
            logger.error(f"Failed to check or create project: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize Evidently project: {str(e)}")

        # Initialize tracing with the project ID
        init_tracing(
            address="https://app.evidently.cloud/",
            api_key=EVIDENTLY_API_TOKEN,
            project_id=str(project.id),
            export_name="RECOMMENDER_TRACES"
        )

        # Verify user exists
        user = db.query(User).filter(User.user_id == request.user_id).first()
        if not user:
            logger.warning(f"User {request.user_id} not found for tracing")
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch the most recent prompt from impressions table
        impression = db.query(Impression).filter(
            Impression.user_id == request.user_id
        ).order_by(Impression.timestamp.desc()).first()
        
        if not impression:
            logger.warning(f"No impressions found for user {request.user_id}")
            raise HTTPException(status_code=404, detail="No impressions found for user")

        prompt = impression.prompt

        # Fetch recommendations from impressions table for the prompt
        impressions = db.query(Impression).filter(
            Impression.user_id == request.user_id,
            Impression.prompt == prompt
        ).order_by(Impression.rank_position).all()
        
        recommendations = [
            {
                "item_id": imp.item_id,
                "rank_position": imp.rank_position,
                "explanation": imp.explanation
            } for imp in impressions
        ]

        # Fetch latest feedback for this user and prompt from feedback table
        feedback = db.query(Feedback).join(Impression).filter(
            Feedback.user_id == request.user_id,
            Impression.prompt == prompt
        ).order_by(Feedback.timestamp.desc()).first()
        
        feedback_data = {}
        if feedback:
            feedback_data = {
                "rating": feedback.rating,
                "comment": feedback.comment,
                "clicked": feedback.clicked,
                "purchased": feedback.purchased,
                "watch_time": feedback.watch_time,
                "skip_after_seconds": feedback.skip_after_seconds,
                "rewatched": feedback.rewatched
            }
        else:
            logger.info(f"No feedback found for user {request.user_id} and prompt '{prompt}'")

        session_id = str(uuid.uuid4())
        
        # Create trace event
        with create_trace_event("recommendation", session_id=session_id) as event:
            # Input attributes
            event.set_attribute("user_id", user.user_id)
            event.set_attribute("preferred_genres", user.preferred_genres)
            event.set_attribute("prompt", prompt)
            
            # Output attributes (serialize to JSON string)
            event.set_attribute("recommendations", json.dumps(recommendations))
            
            # Feedback attributes (serialize to JSON string)
            event.set_attribute("feedback", json.dumps(feedback_data))
            
            time.sleep(0.5)  # Simulate processing delay if needed
        
        logger.info(f"Traced session {session_id} for user {request.user_id} with prompt '{prompt}'")
        return {"status": "success", "session_id": session_id}
    
    except HTTPException as e:
        raise
    except Exception as e:
        logger.error(f"Tracing failed for user {request.user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tracing failed: {str(e)}")

# [Previous imports and code remain unchanged]

@app.get("/run_classification_report", response_class=JSONResponse)
def run_classification_report(db: Session = Depends(get_db)):
    try:
        logger.info("Starting classification report generation")

        # Verify database connection
        try:
            db.execute(text("SELECT 1"))
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

        # Initialize Evidently Cloud workspace and project
        try:
            ws = CloudWorkspace(token=EVIDENTLY_API_TOKEN, url="https://app.evidently.cloud")
            try:
                project = ws.create_project(PROJECT_NAME, org_id=EVIDENTLY_ORG_ID)
                project.description = "Classification report for recommendation system"
                project.save()
                logger.info(f"Created new Evidently project: {PROJECT_NAME}")
            except Exception:
                project = ws.get_project(PROJECT_NAME)
                logger.info(f"Retrieved existing Evidently project: {PROJECT_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Evidently Cloud workspace: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently Cloud initialization failed: {str(e)}")

        # Query impressions and feedback from database
        query = """
            SELECT
                i.impression_id,
                i.user_id,
                i.item_id,
                i.relevance_score,
                i.timestamp,
                f.clicked,
                f.purchased
            FROM impressions i
            LEFT JOIN feedback f ON i.impression_id = f.impression_id
            WHERE f.clicked IS NOT NULL
            LIMIT 1000
        """
        try:
            eval_df = pd.read_sql(query, db.bind)
            logger.info(f"Retrieved {len(eval_df)} records for classification report")
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

        if eval_df.empty:
            logger.info("No data found for classification report")
            return {"message": "No relevant data found for classification report", "data": []}

        # Prepare evaluation DataFrame
        try:
            # Ensure clicked is binary (0 or 1)
            eval_df["clicked"] = eval_df["clicked"].fillna(0).astype(int).clip(0, 1)
            eval_df["target"] = eval_df["clicked"]
            eval_df["prediction_proba"] = eval_df["relevance_score"].astype(float).clip(0, 1)
            eval_df["prediction"] = (eval_df["prediction_proba"] >= 0.5).astype(int)

            # Log unique target values for debugging
            unique_targets = eval_df["target"].unique().tolist()
            logger.info(f"Unique target values: {unique_targets}")

            # Check if target is binary or multiclass
            if len(unique_targets) > 2:
                logger.warning("Target column contains more than two classes; treating as multiclass")
            elif len(unique_targets) <= 1:
                logger.warning("Target column has only one class or is empty")
                return {"message": "Insufficient target variability for classification report", "data": []}
            else:
                logger.info("Target column is binary, proceeding with binary classification")
        except Exception as e:
            logger.error(f"Failed to prepare evaluation DataFrame: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data preparation error: {str(e)}")

        # Build Evidently DataDefinition
        try:
            if len(unique_targets) <= 2:
                data_definition = DataDefinition(
                    classification=[
                        BinaryClassification(
                            target="target",
                            prediction_labels="prediction",
                            prediction_probas="prediction_proba",
                            pos_label=1
                        )
                    ],
                    categorical_columns=["user_id", "item_id"],
                    datetime_columns=["timestamp"]
                )
            else:
                data_definition = DataDefinition(
                    classification=[
                        MulticlassClassification(
                            target="target",
                            prediction_labels="prediction",
                            prediction_probas="prediction_proba",
                            labels=unique_targets
                        )
                    ],
                    categorical_columns=["user_id", "item_id"],
                    datetime_columns=["timestamp"]
                )
            logger.info("Evidently DataDefinition created successfully")
        except Exception as e:
            logger.error(f"Failed to create DataDefinition: {str(e)}")
            raise HTTPException(status_code=500, detail=f"DataDefinition creation failed: {str(e)}")

        # Create Evidently Dataset
        try:
            evidently_dataset = Dataset.from_pandas(
                eval_df[[
                    "target",
                    "prediction",
                    "prediction_proba",
                    "user_id",
                    "item_id",
                    "timestamp"
                ]],
                data_definition=data_definition
            )
            logger.info("Evidently dataset created successfully")
        except Exception as e:
            logger.error(f"Failed to create Evidently dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently dataset creation failed: {str(e)}")

        # Run the ClassificationPreset with appropriate average setting
        try:
            report = Report([ClassificationPreset()])
            evaluation_result = report.run(evidently_dataset, None)
            logger.info("Evidently classification report generated successfully")
        except Exception as e:
            logger.error(f"Failed to run Evidently report: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently report generation failed: {str(e)}")

        # Upload evaluation results to Evidently Cloud
        try:
            ws.add_run(project.id, evaluation_result, include_data=True)
            logger.info(f"Uploaded classification report to Evidently Cloud for project: {PROJECT_NAME}")
        except Exception as e:
            logger.error(f"Failed to upload results to Evidently Cloud: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently Cloud upload failed: {str(e)}")

        # Return JSON evaluation report
        logger.info("Returning classification report")
        return {"evaluation_report": json.loads(evaluation_result.json())}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run_classification_report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")





@app.get("/run_regression_report", response_class=JSONResponse)
def run_regression_report(db: Session = Depends(get_db)):
    """
    Evaluates recommendation system as regression task:
    - Target: User watch time (seconds)
    - Prediction: Relevance scores (0.0-1.0)
    """
    try:
        logger.info("Starting regression report generation")

        # Verify database connection
        try:
            db.execute(text("SELECT 1"))
            logger.info("Database connection verified")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")

        # Initialize Evidently Cloud workspace and project
        try:
            ws = CloudWorkspace(token=EVIDENTLY_API_TOKEN, url="https://app.evidently.cloud")
            try:
                project = ws.create_project(PROJECT_NAME, org_id=EVIDENTLY_ORG_ID)
                project.description = "Regression report for recommendation system"
                project.save()
                logger.info(f"Created new Evidently project: {PROJECT_NAME}")
            except Exception:
                project = ws.get_project(PROJECT_NAME)
                logger.info(f"Retrieved existing Evidently project: {PROJECT_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialize Evidently Cloud workspace: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently Cloud initialization failed: {str(e)}")

        # Query impressions and feedback from database
        query = """
            SELECT
                i.impression_id,
                i.user_id,
                i.item_id,
                i.relevance_score,
                i.timestamp,
                f.watch_time
            FROM impressions i
            LEFT JOIN feedback f ON i.impression_id = f.impression_id
            WHERE f.watch_time IS NOT NULL
            LIMIT 1000
        """
        try:
            eval_df = pd.read_sql(query, db.bind)
            logger.info(f"Retrieved {len(eval_df)} records for regression report")
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Database query error: {str(e)}")

        if eval_df.empty:
            logger.info("No data found for regression report")
            return {"message": "No relevant data found for regression report", "data": []}

        # Prepare evaluation DataFrame
        try:
            eval_df["target"] = eval_df["watch_time"].astype(float)  # Actual watch time
            eval_df["prediction"] = eval_df["relevance_score"].astype(float).clip(0, 1)  # Model scores

            # Optional: Normalize watch_time if range varies widely (e.g., to 0-1)
            # max_watch_time = eval_df["watch_time"].max()
            # if max_watch_time > 0:
            #     eval_df["target"] = eval_df["watch_time"] / max_watch_time
            # else:
            #     eval_df["target"] = 0.0

            logger.info(f"Prepared DataFrame with {len(eval_df)} records")
        except Exception as e:
            logger.error(f"Failed to prepare evaluation DataFrame: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Data preparation error: {str(e)}")

        # Build Evidently DataDefinition
        try:
            data_definition = DataDefinition(
                regression=[
                    Regression(
                        target="target",
                        prediction="prediction"
                    )
                ],
                numerical_columns=["target", "prediction"],
                categorical_columns=["user_id", "item_id"],
                datetime_columns=["timestamp"]
            )
            logger.info("Evidently DataDefinition created successfully")
        except Exception as e:
            logger.error(f"Failed to create DataDefinition: {str(e)}")
            raise HTTPException(status_code=500, detail=f"DataDefinition creation failed: {str(e)}")

        # Create Evidently Dataset
        try:
            evidently_dataset = Dataset.from_pandas(
                eval_df[["target", "prediction", "user_id", "item_id", "timestamp"]],
                data_definition=data_definition
            )
            logger.info("Evidently dataset created successfully")
        except Exception as e:
            logger.error(f"Failed to create Evidently dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently dataset creation failed: {str(e)}")

        # Run the RegressionPreset
        try:
            report = Report([RegressionPreset()])
            evaluation_result = report.run(evidently_dataset, None)
            logger.info("Evidently regression report generated successfully")
        except Exception as e:
            logger.error(f"Failed to run Evidently report: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently report generation failed: {str(e)}")

        # Upload evaluation results to Evidently Cloud
        try:
            ws.add_run(project.id, evaluation_result, include_data=True)
            logger.info(f"Uploaded regression report to Evidently Cloud for project: {PROJECT_NAME}")
        except Exception as e:
            logger.error(f"Failed to upload results to Evidently Cloud: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evidently Cloud upload failed: {str(e)}")

        # Return JSON evaluation report
        logger.info("Returning regression report")
        return {
            "status": "success",
            "evaluation_type": "regression",
            "data_points": len(eval_df),
            "evaluation_report": json.loads(evaluation_result.json())
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run_regression_report: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
# [Rest of the main.py file remains unchanged]



# ==============================================================================
# CLEANED IMPORTS (REMOVED RecSys)

# ==============================================================================
# UPDATED HELPER FUNCTION: extract_item_ids (No Change to Logic)
# ==============================================================================


# NOTE: Ensure other necessary imports (app, get_db, CloudWorkspace, etc.) 
# are correctly defined elsewhere in your main.py file.
# We include minimal imports here for clarity.

# Assuming logger is defined globally
#logger = logging.getLogger(__name__)

# ==============================================================================
# HELPER FUNCTION: extract_item_ids
# ==============================================================================

import math # Used for NDCG log2 calculation



# ==============================================================================
# PYDANTIC MODEL (For frontend input)
# ==============================================================================

class ListRankingRequest(BaseModel):
    user_id: Optional[int] = None 

# ==============================================================================
# HELPER FUNCTION: extract_item_ids
# ==============================================================================

def extract_item_ids(candidates) -> List[str]:
    """
    Extracts item IDs from the candidates JSON string/list and ensures they are strings.
    """
    try:
        if candidates is None:
            return []
        
        if isinstance(candidates, list):
            return [str(item['item_id']) for item in candidates 
                    if isinstance(item, dict) and 'item_id' in item and item['item_id'] is not None]
        
        elif isinstance(candidates, str):
            try:
                parsed = json.loads(candidates)
                if isinstance(parsed, list):
                    return [str(item['item_id']) for item in parsed
                            if isinstance(item, dict) and 'item_id' in item and item['item_id'] is not None]
                return []
            except json.JSONDecodeError:
                # Handle non-JSON string fallback
                return [str(item.strip()) for item in candidates.split(",") if item.strip()]
        else:
            return []
    except Exception as e:
        logger.error(f"Error extracting item IDs: {str(e)}, candidates type: {type(candidates)}")
        return []

# ==============================================================================
# HELPER FUNCTION: get_recsys_data_list_format
# ==============================================================================

def get_recsys_data_list_format(db: Session, user_id: Optional[int] = None, limit: int = 1000) -> pd.DataFrame:
    """
    Extracts, deduplicates, and aligns TARGET (RetrievalSession.candidates) 
    and PREDICTION (Impressions.item_id list) data, optionally filtered by user_id.
    """
    
    user_filter = f"WHERE user_id = {user_id}" if user_id is not None and user_id > 0 else ""
    
    # --- 1. Get TARGET (Candidates) from retrieval_sessions ---
    target_query = f"""
        SELECT session_id, user_id, prompt, candidates, timestamp
        FROM retrieval_sessions
        {user_filter}
        ORDER BY timestamp DESC
        LIMIT {limit * 2} 
    """
    targets_df = pd.read_sql(target_query, db.bind)
    
    # 🚨 DEDUPLICATION STEP
    targets_df.sort_values(by='timestamp', ascending=False, inplace=True)
    targets_df.drop_duplicates(subset=['session_id'], keep='first', inplace=True)
    targets_df.drop(columns=['session_id', 'timestamp'], inplace=True) 
    logger.info(f"Targets: {len(targets_df)} unique sessions retained.")
    
    targets_df['target'] = targets_df['candidates'].apply(extract_item_ids)
    targets_df.drop(columns=['candidates'], inplace=True)
    
    # --- 2. Get PREDICTION (Impressions) from impressions ---
    prediction_query = f"""
        SELECT user_id, prompt, item_id
        FROM impressions
        {user_filter}
        ORDER BY user_id, prompt, rank_position
        LIMIT {limit * 10} 
    """
    impressions_df = pd.read_sql(prediction_query, db.bind)
    
    impressions_df['item_id'] = impressions_df['item_id'].astype(str)
    
    predictions_df = impressions_df.groupby(['user_id', 'prompt'])['item_id'].apply(
        lambda x: x.tolist()
    ).reset_index()
    predictions_df.rename(columns={'item_id': 'prediction'}, inplace=True)
    logger.info(f"Predictions: {len(predictions_df)} unique user/prompt combinations processed.")
    
    # --- 3. Merge and Finalize DataFrame ---
    final_df = pd.merge(
        targets_df, 
        predictions_df, 
        on=['user_id', 'prompt'], 
        how='inner' 
    )

    # Final data quality checks: drop empty lists
    final_df = final_df[final_df['target'].apply(len) > 0]
    final_df = final_df[final_df['prediction'].apply(len) > 0]
    
    final_df['user_id'] = final_df['user_id'].astype(str)
    
    logger.info(f"Final list-based DF created with {len(final_df)} aligned records.")
    return final_df

# ==============================================================================
# MANUAL RANKING CALCULATION LOGIC
# ==============================================================================

def calculate_ranking_metrics_manually(df: pd.DataFrame, k: int = 5, beta: float = 1.0) -> Dict[str, float]:
    """
    Manually calculates all required ranking metrics (Precision, Recall, FBeta, MAP, MAR, NDCG, HitRate, MRR).
    """
    
    all_precision_at_k = []
    all_recall_at_k = []
    all_mrr = []
    all_ap_at_k = []
    all_ndcg_at_k = []
    all_ar_at_k = []
    all_hit_rate = []

    for _, row in df.iterrows():
        y_true_set = set(row['target'])
        y_pred_list = row['prediction'][:k]

        num_relevant_in_session = len(y_true_set)
        
        # Binary relevance array: 1 if item is relevant, 0 otherwise
        relevance = np.array([1 if item in y_true_set else 0 for item in y_pred_list])
        hits = np.sum(relevance)
        
        # --- Metrics requiring Ground Truth items to be non-zero ---
        if num_relevant_in_session == 0:
            continue
            
        first_relevant_pos = np.argmax(relevance) if hits > 0 else 0 

        # 1 & 2. Precision@K and Recall@K
        precision_k = hits / k
        recall_k = hits / num_relevant_in_session
        all_precision_at_k.append(precision_k)
        all_recall_at_k.append(recall_k)

        # 3. MRR@K
        mrr = 1.0 / (first_relevant_pos + 1) if hits > 0 else 0.0
        all_mrr.append(mrr)
        
        # 4. Hit Rate@K
        hit_rate = 1.0 if hits > 0 else 0.0
        all_hit_rate.append(hit_rate)

        # 5. MAP@K
        ap_at_k = 0.0
        if hits > 0:
            num_hits_so_far = 0
            for i in range(k):
                if relevance[i] == 1:
                    num_hits_so_far += 1
                    ap_at_k += num_hits_so_far / (i + 1.0)
            ap_at_k /= hits
            all_ap_at_k.append(ap_at_k)

        # 6. MAR@K
        ar_at_k = 0.0
        if hits > 0:
            num_hits_so_far = 0
            for i in range(k):
                if relevance[i] == 1:
                    num_hits_so_far += 1
                    ar_at_k += num_hits_so_far / num_relevant_in_session
            ar_at_k /= hits
            all_ar_at_k.append(ar_at_k)

        # 7. NDCG@K
        discounts = np.log2(np.arange(k) + 2)
        dcg = np.sum(relevance / discounts)
        ideal_relevance = np.zeros(k)
        ideal_relevance[:min(k, num_relevant_in_session)] = 1
        idcg = np.sum(ideal_relevance / discounts)
        ndcg_k = dcg / idcg if idcg > 0 else 0.0
        all_ndcg_at_k.append(ndcg_k)
        
    # --- Final Averaging and F-Beta Calculation ---
    
    if not all_precision_at_k:
         return {
            'Precision@5': 0.0, 'Recall@5': 0.0, 'FBeta@5': 0.0, 'MAP@5': 0.0, 'MAR@5': 0.0, 
            'NDCG@5': 0.0, 'HitRate@5': 0.0, 'MRR@5': 0.0, 'ScoreDistribution': float('nan')
        }

    mean_precision = np.mean(all_precision_at_k)
    mean_recall = np.mean(all_recall_at_k)

    # 8. FBetaTopK (F1-score if beta=1)
    fbeta_denominator = ((beta**2 * mean_precision) + mean_recall)
    fbeta = (1 + beta**2) * mean_precision * mean_recall / fbeta_denominator if fbeta_denominator > 0 else 0.0

    return {
        'Precision@5': mean_precision,
        'Recall@5': mean_recall,
        'FBeta@5': fbeta,
        'MAP@5': np.mean(all_ap_at_k) if all_ap_at_k else 0.0,
        'MAR@5': np.mean(all_ar_at_k) if all_ar_at_k else 0.0,
        'NDCG@5': np.mean(all_ndcg_at_k),
        'HitRate@5': np.mean(all_hit_rate),
        'MRR@5': np.mean(all_mrr),
        'ScoreDistribution': float('nan') # Conceptual placeholder
    }


# ==============================================================================
# FASTAPI ENDPOINT (For React Integration)
# ==============================================================================

@app.post("/run_list_ranking_report", response_class=JSONResponse)
async def run_list_ranking_report(request: ListRankingRequest, db: Session = Depends(get_db)):
    """
    Calculates ranking metrics manually for the global or a specific user.
    """
    try:
        logger.info(f"Starting manual list-based ranking calculation for user_id: {request.user_id}")

        # --- Data Preparation (Passing user_id to helper) ---
        eval_df = get_recsys_data_list_format(db, user_id=request.user_id)
        
        if eval_df.empty:
            return {"message": "No aligned data found for list-based ranking report", "data_points": 0}

        # --- CALCULATION STEP ---
        calculated_metrics = calculate_ranking_metrics_manually(eval_df, k=5)
        
        logger.info("Manual ranking calculation successful.")
        
        # --- Format Output for Frontend ---
        metric_results = []
        for name, value in calculated_metrics.items():
            metric_results.append({
                "metric": name,
                "result": {
                    "value": round(float(value), 4) if not math.isnan(value) else None # Handle NaN for ScoreDistribution
                }
            })
        
        return {
            "status": "success",
            "evaluation_type": "manual_list_ranking",
            "data_points": len(eval_df),
            "user_id": request.user_id, # Echo back the user_id evaluated
            "evaluation_report": {
                "metrics": metric_results
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run_list_ranking_report (Manual): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Assuming imports (Report, Dataset, DataDefinition, RecallTopK, etc.) are defined.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)




    
