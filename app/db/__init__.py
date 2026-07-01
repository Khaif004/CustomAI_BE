# app.db — async SQLAlchemy layer for the Tool Registry.
#
# This package is INTENTIONALLY separate from the rest of the backend, which
# uses sync psycopg2 (app/api/apps.py:_neon_conn, app/knowledge/vector_store.py).
# Nothing here touches those code paths; the two drivers/pools stay independent.
