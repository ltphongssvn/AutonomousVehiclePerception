# AutonomousVehiclePerception/src/django_backend/config/__init__.py
"""Django config package — loads Celery on startup."""

from config.celery import app as celery_app

__all__ = ["celery_app"]
