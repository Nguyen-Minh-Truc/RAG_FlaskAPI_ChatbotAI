"""Global error handlers for the Flask app."""

from __future__ import annotations

from flask import request
from flask import Flask
from werkzeug.exceptions import HTTPException

from app.api.response import error_response


def register_error_handlers(app: Flask) -> None:
    """Register app-wide error handlers that always return JSON."""

    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        app.logger.warning(
            "HTTP error %s on %s %s: %s",
            exc.code,
            request.method,
            request.path,
            exc.description,
        )
        return error_response(
            message=exc.description or exc.name,
            status_code=exc.code or 500,
            details=None,
        )

    @app.errorhandler(Exception)
    def handle_unhandled_exception(exc: Exception):
        app.logger.exception("Unhandled exception on %s %s", request.method, request.path)
        return error_response(
            message="Internal server error",
            status_code=500,
            details=None,
        )