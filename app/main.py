import sys
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
import logging

# psycopg v3 async (used by the Tool Registry) cannot run on Windows'
# default ProactorEventLoop — it requires a SelectorEventLoop. uvicorn picks up
# the policy set here at import time, before it creates the loop. This is a
# no-op on Linux/macOS (incl. Cloud Foundry, which uses uvloop), so production
# is unaffected; it only fixes local Windows development.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Agent System for SAP BTP Development",
    debug=settings.debug
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    # Service tool registry is now lazy-loaded per app_id on first access.
    # No bulk DB query on startup — server is ready in < 1 s.

    # Tool Registry: bring up the async engine and ensure its tables exist.
    # Best-effort — a failure here must never block startup of the rest of the app.
    # NOTE: we intentionally keep using @app.on_event (not lifespan); switching to
    # a lifespan handler would silently disable these on_event hooks and break the
    # service-registry pre-load above.
    try:
        from app.db.session import init_engine
        from app.db.ddl_tool_catalog import ensure_tool_tables
        engine = init_engine()
        if engine is not None:
            await ensure_tool_tables(engine)
    except Exception as e:
        logger.warning(f"Tool Registry init skipped: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")
    try:
        from app.db.session import dispose_engine
        await dispose_engine()
    except Exception as e:
        logger.warning(f"Tool Registry engine dispose skipped: {e}")


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.app_name}", "version": settings.app_version, "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.app_name, "version": settings.app_version}


try:
    from app.api import chat
    app.include_router(chat.router)
    logger.info("Chat API routes registered")
except Exception as e:
    logger.warning(f"Chat routes not available: {e}")

try:
    from app.api import auth
    app.include_router(auth.router)
    logger.info("Auth API routes registered")
except Exception as e:
    logger.error(f"Failed to register auth routes: {e}")

try:
    from app.api import apps as apps_api
    app.include_router(apps_api.router)
    logger.info("Apps context API routes registered")
except Exception as e:
    logger.warning(f"Apps context routes not available: {e}")

try:
    from app.api import tool_catalog_routes
    app.include_router(tool_catalog_routes.router)
    logger.info("Tool Registry API routes registered")
except Exception as e:
    logger.warning(f"Tool Registry routes not available: {e}")

try:
    from app.api import planner_routes
    app.include_router(planner_routes.router)
    logger.info("Planner API routes registered")
except Exception as e:
    logger.warning(f"Planner routes not available: {e}")

try:
    from app.api import retrieval_routes
    app.include_router(retrieval_routes.router)
    logger.info("Retrieval Orchestrator API routes registered")
except Exception as e:
    logger.warning(f"Retrieval routes not available: {e}")

try:
    from app.api import context_routes
    app.include_router(context_routes.router)
    logger.info("Context Builder API routes registered")
except Exception as e:
    logger.warning(f"Context Builder routes not available: {e}")

try:
    from app.api import documents
    app.include_router(documents.router)
    logger.info("Documents API routes registered")
except Exception as e:
    logger.warning(f"Documents routes not available: {e}")

try:
    from app.api import export as export_api
    app.include_router(export_api.router)
    logger.info("Export API routes registered")
except Exception as e:
    logger.warning(f"Export routes not available: {e}")

try:
    from app.api import action_execution_routes
    app.include_router(action_execution_routes.router)
    logger.info("Action Execution API routes registered")
except Exception as e:
    logger.warning(f"Action Execution routes not available: {e}")

try:
    from app.api import navigation as navigation_api
    app.include_router(navigation_api.router)
    logger.info("Navigation relay API routes registered")
except Exception as e:
    logger.warning(f"Navigation relay routes not available: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)