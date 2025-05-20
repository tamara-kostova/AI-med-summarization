import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from groq import Groq

from api.routes import router
from config.settings import Settings
from services.error_handler import setup_error_handlers
from summarization.evaluator import Evaluator
from summarization.summarizer import Summarizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.groq_client = Groq(api_key="")
    app.state.summarizer = Summarizer(groq_client=app.state.groq_client)
    app.state.evaluator = Evaluator(summarizer=app.state.summarizer)
    yield


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Medical Summarization API",
        version="1.0.0",
        routes=app.routes,
    )

    openapi_schema["info"]["x-logo"] = {"url": "https://placeholder.com/logo.png"}

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="Medical Summarization API",
        description="AI-powered medical text summarization",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.mount("/static", StaticFiles(directory="static"), name="static")

    app.openapi = custom_openapi

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="Medical Summarization API - Docs",
            swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        )

    @app.get("/redoc", include_in_schema=False)
    async def redoc_html():
        return get_redoc_html(
            openapi_url="/openapi.json", title="Medical Summarization API - ReDoc"
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    setup_error_handlers(app)

    return app


app = create_app()
