# Module 19: SQL & Database Fundamentals

**Phase 7.5 — Essential Skills** | Est. time: 1 month (full-time) · 2–3 months (part-time)

*Can be taken any time after Module 01*

## Topics Covered

### SQL
- DDL: `CREATE`, `ALTER`, `DROP`; DML: `SELECT`, `INSERT`, `UPDATE`, `DELETE`
- Joins: `INNER`, `LEFT`, `RIGHT`, `FULL OUTER`, self-join, cross-join
- Subqueries, CTEs (`WITH`), recursive CTEs
- Window functions: `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LEAD()`, `LAG()`, `SUM() OVER()`
- Stored procedures, views, indexes, `EXPLAIN` / `EXPLAIN ANALYZE`
- Data cleaning with SQL: dedup, type casting, string operations

### Python + SQL
- `sqlalchemy` — ORM and Core; connection pooling
- `psycopg2` — PostgreSQL native driver
- `sqlite3` — stdlib, no install needed (great for prototyping)
- Pandas `.read_sql()` and `.to_sql()`

### NoSQL Databases
| Database | Type | Best For |
|---|---|---|
| **MongoDB** | Document | Semi-structured data, flexible schema |
| **Redis** | Key-Value | Caching, sessions, pub/sub |
| **Cassandra** | Wide-Column | High-write, time series, IoT |
| **Neo4j** | Graph | Relationship-heavy data, knowledge graphs |

### Vector Databases (AI-specific)
- **pgvector**: Postgres extension for vector similarity; `<->` (L2), `<#>` (inner product), `<=>` (cosine)
- **ChromaDB**: local-first, great for prototyping RAG
- **Pinecone**: serverless, production-ready
- **Weaviate**: hybrid search (dense + BM25), GraphQL API

## Files

```
19-sql-database-fundamentals/
├── README.md
├── sql-database-fundamentals.md
├── sql-advanced-topics.md
├── sql-project-tutorial.md
└── sql-quick-reference.md
```

**[← Main README](../README.md)**
