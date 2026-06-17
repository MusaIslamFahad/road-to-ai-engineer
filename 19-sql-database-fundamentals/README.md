# Module 19: SQL & Database Fundamentals

**Phase 7.5 — Essential Skills** | Est. time: 1 month (full-time) · 2–3 months (part-time)

*Can be taken any time after Module 01. Recommended alongside or after Phase 2.*

---

## Learning Objectives

By the end of this module you will:
- Write complex SQL: JOINs, subqueries, CTEs, window functions
- Clean and transform data entirely in SQL for ML feature tables
- Integrate SQL with Python via SQLAlchemy and Pandas
- Work with MongoDB, Redis, and Neo4j (NoSQL)
- Use pgvector and ChromaDB for vector similarity search

---

## Prerequisites

- Module 01: Python for Data Science (Pandas basics)

---

## Files in This Module

```
19-sql-database-fundamentals/
├── README.md                           ← You are here — overview and quick-start
├── sql-database-fundamentals.md        ← Full guide: DDL/DML, JOINs, CTEs, Python+SQL, NoSQL, vector DBs
├── sql-advanced-topics.md              ← Query optimisation, JSON/JSONB, partitioning, full-text search
├── sql-quick-reference.md              ← Cheatsheet: all syntax in one place for fast lookup
└── sql-project-tutorial.md             ← End-to-end project: SQL feature engineering → ML pipeline
```

---

## Quick Start

```sql
-- Test your database connection
SELECT version();

-- The 5 most important SQL patterns for ML engineers:

-- 1. Left join + aggregation (build feature tables)
SELECT c.id, COUNT(o.id) AS order_count, COALESCE(SUM(o.amount), 0) AS total_spend
FROM customers c LEFT JOIN orders o ON c.id = o.customer_id
GROUP BY c.id;

-- 2. Window function (running total, ranking)
SELECT customer_id, order_date, amount,
       SUM(amount) OVER (PARTITION BY customer_id ORDER BY order_date) AS running_total
FROM orders;

-- 3. CTE (readable multi-step queries)
WITH top_customers AS (
    SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY 1 HAVING SUM(amount) > 1000
)
SELECT c.name, tc.total FROM customers c JOIN top_customers tc ON c.id = tc.customer_id;

-- 4. Time-based train/test split
SELECT *, CASE WHEN order_date < '2024-10-01' THEN 'train' ELSE 'test' END AS split
FROM orders;

-- 5. Load into Pandas for ML
-- df = pd.read_sql("SELECT * FROM ml_features", engine)
```

---

## Topics Covered

| Topic | File | Key Concepts |
|---|---|---|
| DDL & DML | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | CREATE, ALTER, INSERT, UPDATE, DELETE, UPSERT |
| SELECT & Filtering | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | WHERE, LIKE, NULL, CASE, GROUP BY, HAVING |
| JOINs | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | INNER, LEFT, RIGHT, FULL OUTER, SELF, CROSS |
| Subqueries | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | Scalar, IN, NOT IN, EXISTS, correlated |
| CTEs | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | Simple, chained, recursive |
| Window Functions | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | ROW_NUMBER, RANK, LAG, LEAD, running totals, moving averages |
| Views & Indexes | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | Materialized views, B-tree, GIN, partial, covering |
| Data Cleaning | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | Deduplication, normalisation, outlier detection |
| Python + SQL | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | SQLAlchemy, psycopg2, sqlite3, pandas read_sql |
| NoSQL | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | MongoDB, Redis, Neo4j |
| Vector DBs | [sql-database-fundamentals.md](./sql-database-fundamentals.md) | pgvector, ChromaDB for AI search |
| Query Optimisation | [sql-advanced-topics.md](./sql-advanced-topics.md) | EXPLAIN ANALYZE, index types, partitioning |
| JSONB | [sql-advanced-topics.md](./sql-advanced-topics.md) | Semi-structured data, operators, indexing |
| Full-Text Search | [sql-advanced-topics.md](./sql-advanced-topics.md) | tsvector, tsquery, GIN index, ranking |
| Transactions | [sql-advanced-topics.md](./sql-advanced-topics.md) | ACID, isolation levels, savepoints |
| SQLAlchemy ORM | [sql-advanced-topics.md](./sql-advanced-topics.md) | Declarative models, relationships, queries |
| Project | [sql-project-tutorial.md](./sql-project-tutorial.md) | Full churn prediction feature pipeline |
| Quick Reference | [sql-quick-reference.md](./sql-quick-reference.md) | All syntax at a glance |

---

## Related Resources

- [Data Science Cheatsheet](../resources/data_science_cheatsheet.md) — Python+SQL integration patterns
- [ML Glossary](../resources/ml_glossary.md) — Database terminology

**[← Module 25: Generative AI & Modern LLMs](../25-generative-ai-llms/README.md)** | **[→ Module 20: Imbalanced Data](../20-handling-imbalanced-data/README.md)**

