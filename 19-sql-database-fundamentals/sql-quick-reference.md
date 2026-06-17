# SQL Quick Reference

All the SQL syntax you'll use daily — organised for instant lookup.

---

## SELECT Anatomy

```sql
SELECT   column1, column2, AGG_FUNC(column3) AS alias   -- 5. What to return
FROM     table1                                          -- 1. Source table
JOIN     table2 ON table1.id = table2.fk               -- 2. Combine tables
WHERE    condition1 AND condition2                       -- 3. Filter rows
GROUP BY column1, column2                               -- 4. Group for aggregation
HAVING   AGG_FUNC(column3) > value                     -- 4b. Filter groups
ORDER BY alias DESC                                     -- 6. Sort results
LIMIT    10 OFFSET 20;                                 -- 7. Paginate
```

---

## JOINs At a Glance

```sql
INNER JOIN   -- only matching rows in BOTH tables
LEFT JOIN    -- ALL rows from left, matched rows from right (NULL if no match)
RIGHT JOIN   -- ALL rows from right, matched rows from left (NULL if no match)
FULL JOIN    -- ALL rows from BOTH tables
CROSS JOIN   -- every combination (m × n rows)
SELF JOIN    -- join table to itself (use aliases)
```

---

## Aggregate Functions

```sql
COUNT(*)            -- count all rows
COUNT(col)          -- count non-null values
COUNT(DISTINCT col) -- count unique non-null values
SUM(col)
AVG(col)
MIN(col) / MAX(col)
STRING_AGG(col, ', ' ORDER BY col)   -- concatenate strings (PostgreSQL)
ARRAY_AGG(col)                        -- collect into array
```

---

## Window Functions

```sql
-- Syntax
function() OVER (
    PARTITION BY col1        -- group within
    ORDER BY col2 DESC       -- order within group
    ROWS BETWEEN 2 PRECEDING AND CURRENT ROW  -- frame
)

-- Ranking
ROW_NUMBER()    -- unique: 1, 2, 3, 4
RANK()          -- gaps on ties: 1, 2, 2, 4
DENSE_RANK()    -- no gaps: 1, 2, 2, 3
NTILE(4)        -- assign to N buckets (quartiles)

-- Offset
LAG(col, n, default)    -- value n rows before
LEAD(col, n, default)   -- value n rows after
FIRST_VALUE(col)        -- first value in window
LAST_VALUE(col)         -- last value in window

-- Aggregate as window
SUM(amount) OVER (ORDER BY date)  -- running total
AVG(amount) OVER (PARTITION BY customer_id ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)  -- 7-day moving avg
```

---

## CTEs

```sql
WITH cte_name AS (
    SELECT ...
),
cte_name2 AS (
    SELECT ... FROM cte_name
)
SELECT * FROM cte_name2;

-- Recursive
WITH RECURSIVE r AS (
    SELECT 1 AS n          -- base case
    UNION ALL
    SELECT n + 1 FROM r WHERE n < 10   -- recursive case
)
SELECT n FROM r;
```

---

## String Functions

```sql
LENGTH(str)                   -- character count
UPPER(str) / LOWER(str)
TRIM(str) / LTRIM / RTRIM
SUBSTRING(str FROM 1 FOR 5)  -- extract 5 chars from position 1
POSITION('x' IN str)         -- find position
REPLACE(str, 'old', 'new')
SPLIT_PART(str, '.', 2)      -- split on delimiter, get 2nd part
CONCAT(str1, ' ', str2)      -- or use ||
LPAD(str, 10, '0')           -- left-pad to width 10 with zeros
REGEXP_REPLACE(str, '[0-9]+', '', 'g')   -- regex replace
str LIKE 'A%'                -- starts with A
str ILIKE '%smith%'          -- case-insensitive match
```

---

## Date & Time Functions

```sql
NOW() / CURRENT_TIMESTAMP    -- current datetime with timezone
CURRENT_DATE                 -- today's date
CURRENT_TIME

-- Arithmetic
order_date + INTERVAL '7 days'
order_date - INTERVAL '1 month'
DATEDIFF('2025-12-31', '2025-01-01')   -- MySQL
'2025-12-31'::DATE - '2025-01-01'::DATE  -- PostgreSQL (returns integer days)

-- Extraction
EXTRACT(YEAR   FROM order_date)
EXTRACT(MONTH  FROM order_date)
EXTRACT(DOW    FROM order_date)   -- day of week 0=Sunday
DATE_TRUNC('month', order_date)  -- truncate to start of month
TO_CHAR(order_date, 'YYYY-MM-DD')  -- format as string
TO_DATE('2025-01-15', 'YYYY-MM-DD')  -- parse string to date
```

---

## NULL Handling

```sql
IS NULL / IS NOT NULL
COALESCE(val1, val2, 'default')   -- return first non-null
NULLIF(val, 0)                    -- return NULL if val = 0 (avoid divide-by-zero)
NVL(val, 'default')               -- Oracle equivalent of COALESCE

-- NULL in aggregates: NULL values are IGNORED by SUM, AVG, COUNT(col)
-- NULL in comparisons: NULL = NULL → FALSE; use IS NULL instead
-- NULL in ORDER BY: NULLS FIRST or NULLS LAST
ORDER BY col DESC NULLS LAST
```

---

## CASE Expressions

```sql
-- Simple CASE
SELECT CASE country
    WHEN 'USA'    THEN 'North America'
    WHEN 'Canada' THEN 'North America'
    WHEN 'UK'     THEN 'Europe'
    ELSE 'Other'
END AS region
FROM customers;

-- Searched CASE
SELECT name,
    CASE
        WHEN age < 25 THEN 'Young'
        WHEN age < 40 THEN 'Adult'
        WHEN age < 60 THEN 'Mid-age'
        ELSE               'Senior'
    END AS age_group,
    CASE WHEN is_active THEN 'Active' ELSE 'Inactive' END AS status
FROM customers;
```

---

## Data Type Casting

```sql
CAST(col AS INTEGER)
col::INTEGER         -- PostgreSQL shorthand
col::FLOAT
col::TEXT
col::DATE
col::TIMESTAMP
col::BOOLEAN
col::JSONB
col::vector(1536)    -- pgvector
```

---

## Indexes — When to Use

| Index Type | Use Case | Syntax |
|---|---|---|
| B-tree (default) | =, <, >, BETWEEN, IN, LIKE 'x%' | `CREATE INDEX ON t(col)` |
| Hash | = only, faster than B-tree | `CREATE INDEX ON t USING HASH(col)` |
| GIN | Arrays, JSONB, full-text | `CREATE INDEX ON t USING GIN(col)` |
| GiST | Geometry, ranges, full-text | `CREATE INDEX ON t USING GIST(col)` |
| HNSW | Vector similarity (pgvector) | `CREATE INDEX ON t USING hnsw(emb vector_cosine_ops)` |
| Partial | Filter condition on index | `CREATE INDEX ON t(col) WHERE is_active = TRUE` |
| Covering | Include extra cols in index | `CREATE INDEX ON t(a) INCLUDE (b, c)` |

---

## Common Patterns for ML

```sql
-- 1. Feature table for ML (one row per entity)
SELECT
    u.id                                           AS user_id,
    u.age, u.country,
    COUNT(o.id)                                    AS order_count,
    COALESCE(SUM(o.amount), 0)                     AS total_spend,
    COALESCE(AVG(o.amount), 0)                     AS avg_order,
    COALESCE(STDDEV(o.amount), 0)                  AS std_order,
    MAX(o.order_date) - MIN(o.order_date)          AS purchase_span,
    EXTRACT(DAY FROM NOW() - MAX(o.order_date))    AS days_since_last,
    u.is_active                                    AS label
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.age, u.country, u.is_active;

-- 2. Train/test split in SQL
SELECT *, CASE WHEN RANDOM() < 0.8 THEN 'train' ELSE 'test' END AS split
FROM features;

-- 3. Find class distribution
SELECT label, COUNT(*) AS n, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct
FROM features GROUP BY label;

-- 4. Check for data leakage (label column not null before cutoff)
SELECT COUNT(*) FROM events
WHERE event_date < '2024-01-01' AND label IS NOT NULL;

-- 5. Time-based train/test split
SELECT *, CASE WHEN order_date < '2024-10-01' THEN 'train' ELSE 'test' END AS split
FROM orders;
```

---

## pgvector Quick Reference

```sql
-- Install
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table
CREATE TABLE embeddings (id SERIAL PRIMARY KEY, content TEXT, emb vector(1536));

-- Insert
INSERT INTO embeddings (content, emb) VALUES ('text', '[0.1,0.2,...]'::vector);

-- Search operators
emb <->  query_vec   -- L2 distance (Euclidean)
emb <#>  query_vec   -- negative inner product
emb <=>  query_vec   -- cosine distance (1 - similarity)

-- Find similar
SELECT id, content, emb <=> '[...]'::vector AS cos_dist
FROM embeddings ORDER BY cos_dist LIMIT 5;

-- Indexes
CREATE INDEX ON embeddings USING hnsw (emb vector_cosine_ops) WITH (m=16, ef_construction=64);
CREATE INDEX ON embeddings USING ivfflat (emb vector_l2_ops) WITH (lists=100);
```

---

## Python + SQL Cheatsheet

```python
import pandas as pd
from sqlalchemy import create_engine, text

# Connect
engine = create_engine("postgresql://user:pass@host:5432/db")

# Read
df = pd.read_sql("SELECT * FROM customers WHERE age > 30", engine)
df = pd.read_sql(text("SELECT * FROM customers WHERE age > :age"), engine, params={"age": 30})

# Write
df.to_sql("table_name", engine, if_exists="replace", index=False, chunksize=1000)
df.to_sql("table_name", engine, if_exists="append",  index=False)

# Execute
with engine.begin() as conn:    # auto-commit on exit
    conn.execute(text("UPDATE customers SET is_active = FALSE WHERE age > 90"))

# Transaction
with engine.connect() as conn:
    with conn.begin():
        conn.execute(text("INSERT INTO ..."))
        conn.execute(text("UPDATE ..."))
        # auto-rollback on exception
```
