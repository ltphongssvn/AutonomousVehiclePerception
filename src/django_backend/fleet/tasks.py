# AutonomousVehiclePerception/src/django_backend/fleet/tasks.py
"""Celery tasks for async fleet data processing."""

import requests
from celery import shared_task
from django.conf import settings


@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def process_driving_session(self, session_id):
    """Process all frames in a driving session through the perception pipeline.

    1. Load session and its frames from DB
    2. Send each frame to FastAPI model service
    3. Store detection results back in DB
    """
    from fleet.models import DrivingSession
    from perception.models import DetectedObject, DetectionResult, MLModel

    try:
        session = DrivingSession.objects.get(session_id=session_id)
    except DrivingSession.DoesNotExist:
        return {"error": f"Session {session_id} not found"}

    # Get deployed model
    model = MLModel.objects.filter(status="deployed", model_type="cnn_2d").first()
    if not model:
        return {"error": "No deployed 2D CNN model found"}

    frames = session.frames.all()
    results_count = 0

    for frame in frames:
        try:
            # In production, fetch image from S3 and send to FastAPI
            # Here we record a placeholder detection result
            detection = DetectionResult.objects.create(
                frame=frame,
                model=model,
                inference_time_ms=0.0,
                num_objects_detected=0,
            )
            results_count += 1
        except Exception as e:
            print(f"Error processing frame {frame.frame_number}: {e}")
            continue

    # Mark session as processed
    session.processed = True
    session.save(update_fields=["processed"])

    return {
        "session_id": session_id,
        "frames_processed": results_count,
        "total_frames": frames.count(),
    }


@shared_task
def health_check_model_service():
    """Periodic health check on FastAPI model service."""
    try:
        resp = requests.get(f"{settings.MODEL_SERVICE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
