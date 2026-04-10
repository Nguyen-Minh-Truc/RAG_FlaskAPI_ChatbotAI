"""Flask application entry point for the RAG learning skeleton."""

from flask import Flask

from app import config
from app.api.routes import api_bp
from app.errors import register_error_handlers


def create_app() -> Flask:
    """Create and configure the Flask app."""
    app = Flask(__name__)
    app.register_blueprint(api_bp)
    register_error_handlers(app)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=config.FLASK_DEBUG)
