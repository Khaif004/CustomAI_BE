from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
import logging

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


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application")


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=settings.debug)