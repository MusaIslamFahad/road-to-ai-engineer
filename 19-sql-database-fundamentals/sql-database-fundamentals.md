# SQL & Database Fundamentals — Main Guide

Complete learning material for SQL, Python+SQL integration, NoSQL, and Vector Databases.

---

## Part 1: SQL Foundations

### DDL — Data Definition Language

```sql
-- Create a table
CREATE TABLE customers (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    email         VARCHAR(150) UNIQUE NOT NULL,
    age           INTEGER CHECK (age >= 0 AND age <= 150),
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active     BOOLEAN DEFAULT TRUE
);

-- Add a column
ALTER TABLE customers ADD COLUMN country VARCHAR(50);

-- Modify a column type
ALTER TABLE customers ALTER COLUMN age TYPE SMALLINT;

-- Create an index (speeds up lookups)
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_country ON customers(country, is_active);  -- composite

-- Drop a table
DROP TABLE IF EXISTS customers;
```

### DML — Data Manipulation Language

```sql
-- Insert
INSERT INTO customers (name, email, age, country)
VALUES ('Alice Smith', 'alice@example.com', 32, 'USA');

-- Insert multiple rows
INSERT INTO customers (name, email, age, country) VALUES
    ('Bob Jones',    'bob@example.com',    28, 'UK'),
    ('Carol White',  'carol@example.com',  45, 'Canada');

-- Update
UPDATE customers
SET    is_active = FALSE, country = 'Germany'
WHERE  email = 'bob@example.com';

-- Delete
DELETE FROM customers WHERE is_active = FALSE;

-- Upsert (insert or update on conflict)
INSERT INTO customers (email, name, age)
VALUES ('alice@example.com', 'Alice Updated', 33)
ON CONFLICT (email)
DO UPDATE SET name = EXCLUDED.name, age = EXCLUDED.age;
```

### SELECT — Querying Data

```sql
-- Basic SELECT
SELECT name, email, age
FROM   customers
WHERE  country = 'USA'
  AND  age BETWEEN 25 AND 40
  AND  is_active = TRUE
ORDER BY age DESC
LIMIT 10 OFFSET 0;

-- Aggregations
SELECT
    country,
    COUNT(*)                       AS customer_count,
    AVG(age)                       AS avg_age,
    MIN(age)                       AS min_age,
    MAX(age)                       AS max_age,
    SUM(CASE WHEN is_active THEN 1 ELSE 0 END) AS active_count
FROM   customers
GROUP BY country
HAVING COUNT(*) > 5
ORDER BY customer_count DESC;

-- LIKE and pattern matching
SELECT * FROM customers
WHERE  name LIKE 'A%'          -- starts with A
  OR   email LIKE '%@gmail.%'  -- gmail addresses
  OR   name ILIKE '%smith%';   -- case-insensitive

-- NULL handling
SELECT name, COALESCE(country, 'Unknown') AS country
FROM   customers
WHERE  country IS NULL;
```

---

## Part 2: JOINs

```sql
-- Setup: orders table
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    product     VARCHAR(100),
    amount      DECIMAL(10, 2),
    order_date  DATE DEFAULT CURRENT_DATE
);

-- INNER JOIN: only rows that match in both tables
SELECT c.name, o.product, o.amount, o.order_date
FROM   customers c
INNER JOIN orders o ON c.id = o.customer_id
WHERE  o.amount > 100;

-- LEFT JOIN: all customers, even those with no orders
SELECT c.name, COUNT(o.id) AS order_count, COALESCE(SUM(o.amount), 0) AS total_spend
FROM   customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
ORDER BY total_spend DESC;

-- RIGHT JOIN: all orders even if customer was deleted
SELECT c.name, o.product
FROM   customers c
RIGHT JOIN orders o ON c.id = o.customer_id;

-- FULL OUTER JOIN: all rows from both tables
SELECT c.name, o.product
FROM   customers c
FULL OUTER JOIN orders o ON c.id = o.customer_id;

-- Self-join: find customers in the same country
SELECT a.name AS customer1, b.name AS customer2, a.country
FROM   customers a
JOIN   customers b ON a.country = b.country AND a.id < b.id;

-- Cross join: every combination
SELECT c.name, p.category
FROM   customers c
CROSS JOIN (VALUES ('Electronics'), ('Books'), ('Clothing')) AS p(category);
```

---

## Part 3: Subqueries

```sql
-- Scalar subquery: returns a single value
SELECT name, age,
       (SELECT AVG(age) FROM customers) AS avg_age,
       age - (SELECT AVG(age) FROM customers) AS diff_from_avg
FROM   customers;

-- IN subquery: customers who placed at least one order
SELECT name
FROM   customers
WHERE  id IN (
    SELECT DISTINCT customer_id
    FROM   orders
    WHERE  amount > 500
);

-- NOT IN: customers who have never ordered
SELECT name
FROM   customers
WHERE  id NOT IN (
    SELECT customer_id FROM orders WHERE customer_id IS NOT NULL
);

-- EXISTS (more efficient than IN for large tables)
SELECT name
FROM   customers c
WHERE  EXISTS (
    SELECT 1 FROM orders o
    WHERE  o.customer_id = c.id AND o.amount > 1000
);

-- Correlated subquery: count orders per customer
SELECT
    name,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count
FROM customers c;
```

---

## Part 4: CTEs (Common Table Expressions)

```sql
-- Simple CTE
WITH high_spenders AS (
    SELECT customer_id, SUM(amount) AS total_spend
    FROM   orders
    GROUP BY customer_id
    HAVING SUM(amount) > 1000
)
SELECT c.name, hs.total_spend
FROM   customers c
JOIN   high_spenders hs ON c.id = hs.customer_id
ORDER BY hs.total_spend DESC;

-- Multiple CTEs chained
WITH
monthly_revenue AS (
    SELECT
        DATE_TRUNC('month', order_date) AS month,
        SUM(amount) AS revenue
    FROM orders
    GROUP BY 1
),
monthly_growth AS (
    SELECT
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) AS prev_revenue
    FROM monthly_revenue
)
SELECT
    month,
    revenue,
    ROUND((revenue - prev_revenue) / prev_revenue * 100, 1) AS growth_pct
FROM monthly_growth
WHERE prev_revenue IS NOT NULL;

-- Recursive CTE: build a date series
WITH RECURSIVE date_series AS (
    SELECT DATE '2025-01-01' AS dt
    UNION ALL
    SELECT dt + INTERVAL '1 day'
    FROM   date_series
    WHERE  dt < '2025-12-31'
)
SELECT dt FROM date_series;

-- Recursive CTE: org chart traversal
WITH RECURSIVE org_chart AS (
    -- Base: top-level managers (no manager)
    SELECT id, name, manager_id, 1 AS level
    FROM   employees
    WHERE  manager_id IS NULL
    UNION ALL
    -- Recursive: find direct reports
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM   employees e
    JOIN   org_chart oc ON e.manager_id = oc.id
)
SELECT name, level FROM org_chart ORDER BY level, name;
```

---

## Part 5: Window Functions

```sql
-- ROW_NUMBER, RANK, DENSE_RANK
SELECT
    name,
    country,
    age,
    ROW_NUMBER() OVER (PARTITION BY country ORDER BY age DESC) AS rn,
    RANK()       OVER (PARTITION BY country ORDER BY age DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY country ORDER BY age DESC) AS dense_rnk
FROM customers;

-- Get top 3 customers per country by spend
WITH ranked AS (
    SELECT
        c.name, c.country, SUM(o.amount) AS total_spend,
        RANK() OVER (PARTITION BY c.country ORDER BY SUM(o.amount) DESC) AS rnk
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    GROUP BY c.id, c.name, c.country
)
SELECT name, country, total_spend FROM ranked WHERE rnk <= 3;

-- LAG and LEAD: compare to previous/next row
SELECT
    order_date,
    amount,
    LAG(amount, 1, 0) OVER (ORDER BY order_date) AS prev_amount,
    LEAD(amount)      OVER (ORDER BY order_date) AS next_amount,
    amount - LAG(amount) OVER (ORDER BY order_date) AS change
FROM orders;

-- Running total and moving average
SELECT
    order_date,
    amount,
    SUM(amount)  OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)    AS running_total,
    AVG(amount)  OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)            AS moving_avg_7d,
    COUNT(*)     OVER (PARTITION BY DATE_TRUNC('month', order_date))                            AS orders_this_month
FROM orders;

-- NTILE: bucket customers into quartiles by spend
SELECT name, total_spend,
    NTILE(4) OVER (ORDER BY total_spend) AS spend_quartile
FROM (
    SELECT c.name, COALESCE(SUM(o.amount), 0) AS total_spend
    FROM customers c LEFT JOIN orders o ON c.id = o.customer_id
    GROUP BY c.id, c.name
) t;

-- FIRST_VALUE, LAST_VALUE, NTH_VALUE
SELECT
    name, order_date, amount,
    FIRST_VALUE(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS first_order_amount,
    LAST_VALUE(amount)  OVER (PARTITION BY customer_id ORDER BY order_date
                               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_order_amount
FROM orders o
JOIN customers c ON o.customer_id = c.id;
```

---

## Part 6: Views, Stored Procedures & Indexes

```sql
-- Create a view (reusable saved query)
CREATE OR REPLACE VIEW customer_summary AS
SELECT
    c.id, c.name, c.email, c.country,
    COUNT(o.id)        AS order_count,
    COALESCE(SUM(o.amount), 0)  AS total_spend,
    MAX(o.order_date)  AS last_order_date
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name, c.email, c.country;

SELECT * FROM customer_summary WHERE total_spend > 500;

-- Materialized view (stores results physically — faster reads)
CREATE MATERIALIZED VIEW monthly_stats AS
SELECT DATE_TRUNC('month', order_date) AS month,
       COUNT(*) AS orders, SUM(amount) AS revenue
FROM orders GROUP BY 1;

REFRESH MATERIALIZED VIEW monthly_stats;

-- Stored procedure (PostgreSQL)
CREATE OR REPLACE FUNCTION get_top_customers(min_spend DECIMAL)
RETURNS TABLE(customer_name VARCHAR, total DECIMAL) AS $$
BEGIN
    RETURN QUERY
    SELECT c.name::VARCHAR, SUM(o.amount)::DECIMAL
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    GROUP BY c.id, c.name
    HAVING SUM(o.amount) >= min_spend
    ORDER BY SUM(o.amount) DESC;
END;
$$ LANGUAGE plpgsql;

SELECT * FROM get_top_customers(500);

-- Index types
CREATE INDEX idx_orders_date      ON orders(order_date);              -- B-tree (default)
CREATE INDEX idx_orders_amount    ON orders(amount DESC);             -- sorted
CREATE INDEX idx_customers_email  ON customers(LOWER(email));         -- functional
CREATE INDEX idx_orders_covering  ON orders(customer_id, amount, order_date); -- covering index
-- Partial index (only index rows matching condition)
CREATE INDEX idx_active_customers ON customers(email) WHERE is_active = TRUE;

-- EXPLAIN ANALYZE to find slow queries
EXPLAIN ANALYZE
SELECT c.name, SUM(o.amount)
FROM customers c JOIN orders o ON c.id = o.customer_id
GROUP BY c.id, c.name
HAVING SUM(o.amount) > 1000;
```

---

## Part 7: Data Cleaning with SQL

```sql
-- Remove duplicates (keep latest)
DELETE FROM customers
WHERE id NOT IN (
    SELECT MAX(id) FROM customers GROUP BY email
);

-- Standardise text
UPDATE customers
SET    name    = INITCAP(TRIM(name)),
       email   = LOWER(TRIM(email)),
       country = UPPER(TRIM(country));

-- Fix bad dates
UPDATE orders
SET order_date = CURRENT_DATE
WHERE order_date > CURRENT_DATE OR order_date < '2000-01-01';

-- Handle nulls: fill missing country from most common per age group
UPDATE customers c
SET    country = (
    SELECT country
    FROM   customers
    WHERE  ABS(age - c.age) <= 5 AND country IS NOT NULL
    GROUP BY country ORDER BY COUNT(*) DESC LIMIT 1
)
WHERE country IS NULL;

-- Detect outliers (orders more than 3 std devs above mean)
SELECT *
FROM   orders
WHERE  amount > (SELECT AVG(amount) + 3 * STDDEV(amount) FROM orders);
```

---

## Part 8: Python + SQL Integration

```python
import pandas as pd
from sqlalchemy import create_engine, text
import psycopg2
import sqlite3

# ─── SQLAlchemy (recommended — works with pandas) ─────────────────────────────
engine = create_engine(
    "postgresql://user:password@localhost:5432/mydb",
    pool_size=5, max_overflow=10, pool_pre_ping=True
)

# Read query into DataFrame
df = pd.read_sql(
    "SELECT * FROM customer_summary WHERE total_spend > %(min)s",
    engine,
    params={"min": 500}
)

# Write DataFrame to database
df_new.to_sql("temp_customers", engine, if_exists="replace", index=False, chunksize=1000)

# Execute raw SQL
with engine.connect() as conn:
    result = conn.execute(text("UPDATE customers SET is_active = FALSE WHERE age > 90"))
    conn.commit()

# ─── sqlite3 (no install, great for prototyping) ─────────────────────────────
conn = sqlite3.connect(":memory:")   # or "database.db" for persistent
cursor = conn.cursor()
cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
cursor.executemany("INSERT INTO test VALUES (?, ?)", [(1, "Alice"), (2, "Bob")])
conn.commit()
rows = cursor.fetchall()
conn.close()

# ─── psycopg2 (PostgreSQL native) ────────────────────────────────────────────
conn = psycopg2.connect(host="localhost", dbname="mydb", user="user", password="pass")
with conn.cursor() as cur:
    cur.execute("SELECT id, name FROM customers WHERE age > %s", (30,))
    customers = cur.fetchall()
    # Bulk insert
    psycopg2.extras.execute_values(cur,
        "INSERT INTO orders (customer_id, amount) VALUES %s",
        [(1, 150.0), (2, 299.99)], page_size=100
    )
conn.commit()
conn.close()

# ─── ML workflow: load data from DB ──────────────────────────────────────────
query = """
WITH features AS (
    SELECT
        c.id,
        c.age,
        EXTRACT(YEAR FROM AGE(NOW(), c.created_at)) AS account_years,
        COUNT(o.id)             AS order_count,
        COALESCE(SUM(o.amount), 0) AS total_spend,
        COALESCE(AVG(o.amount), 0) AS avg_order_value,
        COALESCE(MAX(o.order_date) - MIN(o.order_date), 0) AS purchase_span_days,
        c.is_active             AS label
    FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
    GROUP BY c.id, c.age, c.created_at, c.is_active
)
SELECT * FROM features
"""
df = pd.read_sql(query, engine)
X = df.drop(columns=["id", "label"])
y = df["label"]
```

---

## Part 9: NoSQL Databases

### MongoDB

```python
from pymongo import MongoClient
import pymongo

client = MongoClient("mongodb://localhost:27017/")
db     = client["ml_database"]
users  = db["users"]

# Insert
users.insert_one({"name": "Alice", "age": 32, "tags": ["ml", "python"]})
users.insert_many([
    {"name": "Bob",   "age": 25, "tags": ["data", "sql"]},
    {"name": "Carol", "age": 40, "tags": ["ml", "nlp"]},
])

# Query
user = users.find_one({"name": "Alice"})
results = users.find({"age": {"$gte": 30}, "tags": "ml"})
for r in results: print(r["name"], r["age"])

# Update
users.update_one({"name": "Alice"}, {"$set": {"age": 33}, "$push": {"tags": "ai"}})
users.update_many({"age": {"$lt": 25}}, {"$set": {"is_junior": True}})

# Aggregation pipeline
pipeline = [
    {"$match":  {"tags": "ml"}},
    {"$group":  {"_id": None, "avg_age": {"$avg": "$age"}, "count": {"$sum": 1}}},
    {"$project":{"_id": 0, "avg_age": {"$round": ["$avg_age", 1]}, "count": 1}}
]
result = list(users.aggregate(pipeline))
print(result)

# Index
users.create_index([("name", pymongo.ASCENDING), ("age", pymongo.DESCENDING)])
```

### Redis

```python
import redis, json

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

# String (great for caching ML predictions)
r.set("prediction:user:123", json.dumps({"score": 0.87, "label": "churn"}), ex=3600)  # TTL 1hr
cached = json.loads(r.get("prediction:user:123"))

# Hash (store structured user features)
r.hset("user:456", mapping={"age": 35, "tenure": 12, "country": "US"})
features = r.hgetall("user:456")

# List (FIFO queue for ML job queue)
r.rpush("ml_queue", json.dumps({"user_id": 789, "model": "churn_v2"}))
job = json.loads(r.blpop("ml_queue", timeout=5)[1])

# Sorted set (leaderboard / top-k predictions)
r.zadd("top_scores", {"user:101": 0.92, "user:202": 0.85, "user:303": 0.78})
top_3 = r.zrevrange("top_scores", 0, 2, withscores=True)

# Pub/Sub (real-time event streaming)
pubsub = r.pubsub()
pubsub.subscribe("predictions")
r.publish("predictions", json.dumps({"user_id": 123, "score": 0.91}))
```

---

## Part 10: Vector Databases for AI

### pgvector (PostgreSQL extension)

```sql
-- Enable extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table with embedding column
CREATE TABLE documents (
    id        SERIAL PRIMARY KEY,
    content   TEXT,
    source    VARCHAR(200),
    embedding vector(1536)   -- dimension matches your embedding model
);

-- Insert with embedding (normally done via Python)
-- INSERT INTO documents (content, embedding) VALUES ('text', '[0.1, 0.2, ...]'::vector)

-- L2 distance search (closest vectors)
SELECT id, content, embedding <-> '[0.1, 0.2, ...]'::vector AS distance
FROM   documents
ORDER BY distance
LIMIT  5;

-- Cosine similarity search (most semantically similar)
SELECT id, content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM   documents
ORDER BY similarity DESC
LIMIT  5;

-- Inner product (fastest, requires normalised vectors)
SELECT id, content, embedding <#> '[0.1, 0.2, ...]'::vector AS neg_inner_product
FROM   documents
ORDER BY neg_inner_product
LIMIT  5;

-- Create HNSW index for fast ANN search
CREATE INDEX idx_docs_embedding_hnsw ON documents
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### pgvector with Python (LangChain)

```python
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

CONNECTION_STRING = "postgresql://user:pass@localhost:5432/vectordb"
COLLECTION_NAME   = "documents"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store and add documents
vectorstore = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING
)

# Similarity search
results = vectorstore.similarity_search("What is gradient descent?", k=5)
results_with_scores = vectorstore.similarity_search_with_score("neural networks", k=3)

# As retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### ChromaDB

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")

oai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-...", model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="ml_papers",
    embedding_function=oai_ef,
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["Paper 1 text...", "Paper 2 text..."],
    metadatas=[{"source": "arxiv", "year": 2024}, {"source": "neurips", "year": 2024}],
    ids=["doc1", "doc2"]
)

# Query
results = collection.query(
    query_texts=["explain transformer architecture"],
    n_results=5,
    where={"year": {"$gte": 2023}},    # metadata filter
    include=["documents", "distances", "metadatas"]
)
```
