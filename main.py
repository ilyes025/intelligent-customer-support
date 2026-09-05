"""
FastAPI application main entry point.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.auth import router as auth_router
from app.api.health import router as health_router
from app.api.support import router as support_router
from app.core.config import settings
from typing import AsyncGenerator
from app.db.seed import seed_knowledge_base
from app.db.session import sessionmanager
from contextlib import asynccontextmanager

# Note: password hashing no longer goes through passlib (see
# app/core/security.py for why), so the passlib/bcrypt `__about__`
# compatibility shim that used to live here is no longer needed.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Function that handles startup and shutdown events.
    To understand more, read https://fastapi.tiangolo.com/advanced/events/
    """
    async with sessionmanager.session() as session:
        await seed_knowledge_base(session)
    yield
    if sessionmanager._engine is not None:
        # Close the DB connection
        await sessionmanager.close()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    lifespan=lifespan,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["system"])
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(support_router, prefix="/support", tags=["support-agent"])

# Static dashboard (app/static/dashboard.html). Not auth-gated itself since
# it's just HTML/JS/CSS - it calls the real, staff-only /support/metrics/summary
# API using a token pasted in by the person viewing it, so data access is
# still protected even though the page shell is public.
app.mount("/static", StaticFiles(directory="app/static"), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
