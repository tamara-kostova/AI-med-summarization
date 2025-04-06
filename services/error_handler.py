import logging
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def setup_error_handlers(app: FastAPI):
    """
    Setup centralized error handling for the FastAPI application
    """

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """
        Handle HTTP exceptions with custom logging
        """
        logger.error(f"HTTP Exception: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "message": exc.detail,
                "status_code": exc.status_code,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request, exc):
        """
        Handle unexpected exceptions with detailed logging
        """
        logger.error(f"Unexpected Error: {str(exc)}")
        logger.error(traceback.format_exc())

        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "message": "An unexpected error occurred",
                "details": str(exc),
            },
        )
