import os
import sys
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

load_dotenv()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

missing_env = [
    env_name for env_name, env_value in {
        "DB_USER": DB_USER,
        "DB_PASSWORD": DB_PASSWORD,
        "DB_HOST": DB_HOST,
        "DB_NAME": DB_NAME,
    }.items()
    if env_value is None
]

if missing_env:
    print(f"Configuration error: missing environment variables: {', '.join(missing_env)}")
    sys.exit(1)

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


def get_table_columns(table_name):
    try:
        return {column["name"] for column in inspect(engine).get_columns(table_name)}
    except Exception:
        return set()


def table_exists(table_name):
    try:
        return inspect(engine).has_table(table_name)
    except Exception:
        return False


def build_books_query():
    books_columns = get_table_columns("books")
    categories_columns = get_table_columns("categories")

    book_id_expr = "b.book_id" if "book_id" in books_columns else "b.id"
    title_expr = "b.title" if "title" in books_columns else "''"

    if "category" in books_columns:
        category_expr = "COALESCE(b.category, '')"
        category_join = ""
    elif "category_id" in books_columns and {"id", "name"}.issubset(categories_columns):
        category_expr = "COALESCE(c.name, '')"
        category_join = "LEFT JOIN categories c ON b.category_id = c.id"
    else:
        category_expr = "''"
        category_join = ""

    keyword_parts = ["b.title"]
    if "description" in books_columns:
        keyword_parts.append("b.description")
    if "author" in books_columns:
        keyword_parts.append("b.author")
    fallback_keywords = f"CONCAT_WS(' ', {', '.join(keyword_parts)})"
    keywords_expr = f"COALESCE(b.keywords, {fallback_keywords})" if "keywords" in books_columns else fallback_keywords

    query = f"""
        SELECT
            {book_id_expr} AS book_id,
            {title_expr} AS title,
            {category_expr} AS category,
            {keywords_expr} AS keywords
        FROM books b
    """
    if category_join:
        query += f"\n        {category_join}"
    return query


def build_users_query():
    user_columns = get_table_columns("users")
    user_id_expr = "user_id" if "user_id" in user_columns else "id"
    preferred_categories_expr = "preferred_categories" if "preferred_categories" in user_columns else "''"
    return f"SELECT {user_id_expr} AS user_id, name, {preferred_categories_expr} AS preferred_categories FROM users"

try:
    # Read books and alias columns to match recommender expectations
    df_books = pd.read_sql(build_books_query(), con=engine)

    # Read users using the available schema
    df_users = pd.read_sql(build_users_query(), con=engine)

    # Read interactions when the recommender table exists
    if table_exists("interactions"):
        df_interactions = pd.read_sql(
            "SELECT interaction_id, user_id, book_id, action, rating, created_at FROM interactions",
            con=engine
        )
    else:
        df_interactions = pd.DataFrame(
            columns=["interaction_id", "user_id", "book_id", "action", "rating", "created_at"]
        )

    # Remove duplicates in interactions to ensure data integrity
    if not df_interactions.empty:
        df_interactions.drop_duplicates(subset=['user_id', 'book_id'], keep='last', inplace=True)
    print("Data successfully loaded from MySQL!")
except Exception as e:
    print(f"Database error: {e}")
    sys.exit(1) 

# 2.(TF-IDF & User Similarity)

# tfidf = TfidfVectorizer()

# (Content-Based)
df_books['Features'] = (
    df_books['title'].fillna('') + " " +
    df_books['keywords'].fillna('')
)
df_books = df_books[df_books['Features'].str.strip() != '']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_books['Features'])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# (Collaborative)
if not df_interactions.empty and len(df_interactions[df_interactions['rating'].notnull()]) > 0:
    user_item_matrix = df_interactions.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
else:
    user_item_matrix = pd.DataFrame()
    user_similarity_df = pd.DataFrame()


def refresh_recommender_data():
    global df_books, df_users, df_interactions, tfidf_matrix, content_similarity, user_item_matrix, user_similarity_df

    df_books = pd.read_sql(build_books_query(), con=engine)
    df_users = pd.read_sql(build_users_query(), con=engine)

    if table_exists("interactions"):
        df_interactions = pd.read_sql(
            "SELECT interaction_id, user_id, book_id, action, rating, created_at FROM interactions",
            con=engine
        )
    else:
        df_interactions = pd.DataFrame(
            columns=["interaction_id", "user_id", "book_id", "action", "rating", "created_at"]
        )

    if not df_interactions.empty:
        df_interactions.drop_duplicates(subset=['user_id', 'book_id'], keep='last', inplace=True)

    df_books['Features'] = (
        df_books['title'].fillna('') + " " +
        df_books['keywords'].fillna('')
    )
    df_books = df_books[df_books['Features'].str.strip() != '']

    tfidf_matrix = tfidf.fit_transform(df_books['Features'])
    content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

    if not df_interactions.empty and len(df_interactions[df_interactions['rating'].notnull()]) > 0:
        user_item_matrix = df_interactions.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
        user_similarity = cosine_similarity(user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    else:
        user_item_matrix = pd.DataFrame()
        user_similarity_df = pd.DataFrame()

# 3.Filtering Algorithm Functions

def get_content_based_recommendations(book_title, top_n=3):
    try:
        idx = df_books.index[df_books['title'] == book_title].tolist()[0]
        sim_scores = list(enumerate(content_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        book_indices = [i[0] for i in sim_scores]
        recommended_books = df_books.iloc[book_indices][['book_id', 'title', 'category']].to_dict(orient='records')
        return recommended_books
    except:
        return []


def get_collaborative_recommendations(target_user_id, top_n=3):
    if target_user_id not in user_similarity_df.columns or user_similarity_df.empty:
        return []
    
    similar_users = user_similarity_df[target_user_id].sort_values(ascending=False)
    if len(similar_users) < 2: return [] # لا يوجد مستخدمين آخرين للتشابه
    
    most_similar_user_id = similar_users.index[1]
    similar_user_books = user_item_matrix.loc[most_similar_user_id]
    target_user_books = user_item_matrix.loc[target_user_id]
    
    recommendations = []
    for book_id, rating in similar_user_books.items():
        if rating >= 3.0 and target_user_books.get(book_id, 0) == 0:
            #إرجاع قاموس يحتوي على تفاصيل الكتاب بدلاً من العنوان فقط
            book_data = df_books[df_books['book_id'] == book_id].iloc[0]
            recommendations.append({
                "book_id": int(book_data['book_id']), 
                "title": book_data['title'], 
                "category": book_data['category']
            })
            if len(recommendations) == top_n: break
    return recommendations


def get_hybrid_recommendations(target_user_id, top_n=3):
    # Plan A: Collaborative
    collab_result = get_collaborative_recommendations(target_user_id, top_n)
    if collab_result:
        return {"strategy": "Collaborative", "books": collab_result}
        
    # Plan B: Content-Based
    user_history = df_interactions[df_interactions['user_id'] == target_user_id]
    if not user_history.empty and not user_history[user_history['rating'].notnull()].empty:
        best_book_id = user_history.sort_values(by='rating', ascending=False).iloc[0]['book_id']
        best_book_title = df_books[df_books['book_id'] == best_book_id]['title'].values[0]
        content_result = get_content_based_recommendations(best_book_title, top_n)
        if content_result:
            return {"strategy": "Content-Based", "books": content_result}
            
    # Plan C: Cold Start
    user_data = df_users[df_users['user_id'] == target_user_id]
    if not user_data.empty:
        user_prefs = user_data['preferred_categories'].values[0]
        prefs_list = [p.strip() for p in user_prefs.split(',')]
        # اإرجاع قائمة قواميس بدلاً من DataFrame
        cold_start_recs = df_books[df_books['category'].isin(prefs_list)].head(top_n)[['book_id', 'title', 'category']].to_dict(orient='records')
        return {"strategy": "Cold Start (Profile)", "books": cold_start_recs}
        
    # إرجاع قائمة قواميس
    return {"strategy": "General Popular", "books": df_books.head(top_n)[['book_id', 'title', 'category']].to_dict(orient='records')}


# API (FastAPI) و CORS
app = FastAPI(title="Scientific Books Recommendation API")

# السماح للواجهة الأمامية (Front-end) بالاتصال
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # في مرحلة الإطلاق الفعلي، ضع رابط المنصة هنا بدلاً من النجمة
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InteractionData(BaseModel):
    user_id: int
    book_id: int
    action: str
    rating: int = None

@app.get("/")
def read_root():
    return {"status": "Active", "message": "API is securely connected to MySQL."}

@app.get("/api/recommendations/{user_id}")
def fetch_recommendations(user_id: int, top_n: int = 3):
    try:
        refresh_recommender_data()
        recs = get_hybrid_recommendations(user_id, top_n)
        return {"success": True, "user_id": user_id, "data": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interactions")
def add_interaction(data: InteractionData):
    try:
        if not table_exists("interactions"):
            raise HTTPException(
                status_code=503,
                detail="The interactions table is not available yet. Run backend/migrations/create_database_tables.php first."
            )

        query = text("""
            INSERT INTO interactions (user_id, book_id, action, rating) 
            VALUES (:u_id, :b_id, :act, :rtg)
        """)
        with engine.begin() as conn:
            conn.execute(query, {
                "u_id": data.user_id,
                "b_id": data.book_id,
                "act": data.action,
                "rtg": data.rating
            })
        return {"success": True, "message": "Interaction saved successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
