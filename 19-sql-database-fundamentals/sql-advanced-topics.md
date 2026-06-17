# SQL & Databases — Advanced Topics

---

## 1. Query Optimisation

```sql
-- EXPLAIN: shows query plan without running
EXPLAIN SELECT * FROM orders WHERE customer_id = 42;

-- EXPLAIN ANALYZE: runs query, shows actual vs estimated rows
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT c.name, SUM(o.amount)
FROM customers c JOIN orders o ON c.id = o.customer_id
GROUP BY c.id;

-- Key things to look for:
-- "Seq Scan"   → full table scan — consider adding an index
-- "Index Scan" → good — using an index
-- "Hash Join" vs "Nested Loop" → hash join better for large tables
-- "rows=1000 actual rows=89000" → bad estimate → run ANALYZE to refresh stats

-- Force PostgreSQL to refresh table statistics
ANALYZE customers;
VACUUM ANALYZE orders;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read
FROM   pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

---

## 2. Advanced Query Patterns

### Pivot (CROSSTAB)

```sql
-- Manual pivot with conditional aggregation
SELECT
    country,
    SUM(CASE WHEN EXTRACT(YEAR FROM order_date) = 2023 THEN amount ELSE 0 END) AS revenue_2023,
    SUM(CASE WHEN EXTRACT(YEAR FROM order_date) = 2024 THEN amount ELSE 0 END) AS revenue_2024,
    SUM(CASE WHEN EXTRACT(YEAR FROM order_date) = 2025 THEN amount ELSE 0 END) AS revenue_2025
FROM   orders o
JOIN   customers c ON o.customer_id = c.id
GROUP BY country;
```

### Gap and Island Detection (Time Series)

```sql
-- Find gaps in daily data (missing dates)
WITH date_series AS (
    SELECT GENERATE_SERIES(
        (SELECT MIN(order_date) FROM orders),
        (SELECT MAX(order_date) FROM orders),
        INTERVAL '1 day'
    )::DATE AS dt
),
actual_dates AS (SELECT DISTINCT order_date FROM orders)
SELECT dt AS missing_date
FROM   date_series
LEFT JOIN actual_dates ON dt = order_date
WHERE  order_date IS NULL;

-- Find consecutive "islands" of activity
WITH gaps AS (
    SELECT
        customer_id, order_date,
        order_date - LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS gap_days
    FROM orders
),
islands AS (
    SELECT *, SUM(CASE WHEN gap_days > 30 OR gap_days IS NULL THEN 1 ELSE 0 END)
              OVER (PARTITION BY customer_id ORDER BY order_date) AS island_id
    FROM gaps
)
SELECT customer_id, island_id, MIN(order_date) AS start_date, MAX(order_date) AS end_date
FROM   islands
GROUP BY customer_id, island_id;
```

### Funnel Analysis (ML use case)

```sql
-- Count users at each stage of a conversion funnel
WITH events AS (
    SELECT user_id, event_type, MIN(created_at) AS first_occurrence
    FROM   user_events
    GROUP BY user_id, event_type
),
funnel AS (
    SELECT
        COUNT(DISTINCT CASE WHEN event_type = 'page_view'  THEN user_id END) AS stage1_views,
        COUNT(DISTINCT CASE WHEN event_type = 'add_to_cart' THEN user_id END) AS stage2_cart,
        COUNT(DISTINCT CASE WHEN event_type = 'checkout'   THEN user_id END) AS stage3_checkout,
        COUNT(DISTINCT CASE WHEN event_type = 'purchase'   THEN user_id END) AS stage4_purchase
    FROM events
)
SELECT
    stage1_views,
    stage2_cart,
    stage3_checkout,
    stage4_purchase,
    ROUND(stage2_cart   * 100.0 / stage1_views,   1) AS view_to_cart_pct,
    ROUND(stage3_checkout * 100.0 / stage2_cart,  1) AS cart_to_checkout_pct,
    ROUND(stage4_purchase * 100.0 / stage3_checkout, 1) AS checkout_to_purchase_pct
FROM funnel;
```

---

## 3. Full-Text Search

```sql
-- Add full-text search column
ALTER TABLE documents ADD COLUMN search_vector TSVECTOR;

-- Populate it
UPDATE documents
SET    search_vector = TO_TSVECTOR('english', content);

-- Create GIN index for fast full-text search
CREATE INDEX idx_documents_fts ON documents USING GIN(search_vector);

-- Search
SELECT id, content,
       TS_RANK(search_vector, query) AS rank
FROM   documents,
       TO_TSQUERY('english', 'machine & learning & transformer') AS query
WHERE  search_vector @@ query
ORDER BY rank DESC
LIMIT 10;

-- Highlight matching words
SELECT TS_HEADLINE('english', content,
    TO_TSQUERY('machine & learning'),
    'StartSel=<b>, StopSel=</b>, MaxWords=35, MinWords=15'
) FROM documents LIMIT 3;
```

---

## 4. JSON / JSONB (Semi-Structured Data)

```sql
-- JSONB column
CREATE TABLE model_metadata (
    id        SERIAL PRIMARY KEY,
    run_id    VARCHAR(50) UNIQUE,
    params    JSONB,
    metrics   JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO model_metadata (run_id, params, metrics) VALUES
('run_001', '{"n_estimators": 500, "max_depth": 6, "lr": 0.05}',
            '{"val_auc": 0.924, "test_auc": 0.918, "f1": 0.712}');

-- Query JSONB fields
SELECT run_id,
    params->>'n_estimators'  AS n_estimators,
    (metrics->>'val_auc')::FLOAT AS val_auc
FROM model_metadata
WHERE (metrics->>'val_auc')::FLOAT > 0.90
ORDER BY (metrics->>'val_auc')::FLOAT DESC;

-- JSONB operators
-- -> returns JSON     -->  params->'n_estimators'  →  500 (as JSON)
-- ->> returns TEXT    -->  params->>'n_estimators' →  "500"
-- @> contains         -->  params @> '{"lr": 0.05}'
-- ? key exists        -->  params ? 'max_depth'
-- jsonb_object_keys   -->  get all keys

-- Update nested JSONB
UPDATE model_metadata
SET    metrics = metrics || '{"pr_auc": 0.841}'::jsonb
WHERE  run_id = 'run_001';

-- Index on JSONB field
CREATE INDEX idx_models_val_auc ON model_metadata
USING BTREE ((metrics->>'val_auc'));
```

---

## 5. Transactions & Concurrency

```sql
-- ACID transaction
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    UPDATE accounts SET balance = balance + 100 WHERE id = 2;
    -- Both or neither — atomicity
COMMIT;

-- Rollback on error
BEGIN;
    UPDATE orders SET amount = -999 WHERE id = 1;  -- mistake
ROLLBACK;

-- Savepoints (partial rollback)
BEGIN;
    INSERT INTO orders (customer_id, amount) VALUES (1, 500);
    SAVEPOINT sp1;
    INSERT INTO orders (customer_id, amount) VALUES (2, -100);  -- error
    ROLLBACK TO sp1;  -- only rolls back to savepoint
    INSERT INTO orders (customer_id, amount) VALUES (2, 100);
COMMIT;

-- Isolation levels
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;   -- default; no dirty reads
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;  -- no phantom reads within transaction
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;     -- strongest; slowest
```

---

## 6. Data Partitioning (Large Tables)

```sql
-- Range partitioning by date (great for time-series ML data)
CREATE TABLE events (
    id         BIGSERIAL,
    event_date DATE NOT NULL,
    user_id    INTEGER,
    event_type VARCHAR(50),
    payload    JSONB
) PARTITION BY RANGE (event_date);

-- Create partitions
CREATE TABLE events_2024 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE events_2025 PARTITION OF events
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Queries automatically route to correct partition
SELECT * FROM events WHERE event_date BETWEEN '2024-06-01' AND '2024-06-30';
-- → only scans events_2024, not events_2025
```

---

## 7. SQLAlchemy ORM (Advanced)

```python
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, Session
from sqlalchemy import select, func
from datetime import datetime

Base = declarative_base()
engine = create_engine("postgresql://user:pass@localhost:5432/db")

class Customer(Base):
    __tablename__ = "customers"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(100), nullable=False)
    email      = Column(String(150), unique=True, nullable=False)
    age        = Column(Integer)
    orders     = relationship("Order", back_populates="customer", lazy="select")

    def __repr__(self): return f"Customer(name={self.name!r})"

class Order(Base):
    __tablename__ = "orders"
    id          = Column(Integer, primary_key=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False)
    amount      = Column(Float)
    order_date  = Column(DateTime, default=datetime.utcnow)
    customer    = relationship("Customer", back_populates="orders")

Base.metadata.create_all(engine)

# ORM queries
with Session(engine) as session:
    # Add
    alice = Customer(name="Alice", email="alice@example.com", age=32)
    session.add(alice)
    session.commit()

    # Query
    customers = session.execute(
        select(Customer).where(Customer.age > 30)
    ).scalars().all()

    # Join + aggregate (2.0 style)
    result = session.execute(
        select(Customer.name, func.sum(Order.amount).label("total"))
        .join(Customer.orders)
        .group_by(Customer.id)
        .having(func.sum(Order.amount) > 1000)
        .order_by(func.sum(Order.amount).desc())
    ).all()
    for name, total in result:
        print(f"{name}: ${total:.2f}")
```
