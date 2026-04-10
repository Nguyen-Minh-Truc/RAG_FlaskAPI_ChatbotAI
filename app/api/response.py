"""Standardized API response helpers."""

from __future__ import annotations

from flask import jsonify


def success_response(data=None, message: str = "Success", status_code: int = 200):
    """Return a consistent success payload."""
    payload = {
        "success": True,
        "message": message,
        "data": data,
        "error": None,
    }
    return jsonify(payload), status_code


def error_response(message: str, status_code: int = 400, details=None):
    """Return a consistent error payload."""
    payload = {
        "success": False,
        "message": message,
        "data": None,
        "error": details,
    }
    return jsonify(payload), status_code