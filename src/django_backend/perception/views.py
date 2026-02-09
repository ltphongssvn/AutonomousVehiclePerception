# AutonomousVehiclePerception/src/django_backend/perception/views.py
"""REST API views for perception results and model management."""

import requests
from django.conf import settings
from rest_framework import filters, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from perception.models import DetectedObject, DetectionResult, MLModel
from perception.serializers import (
    DetectedObjectSerializer,
    DetectionResultDetailSerializer,
    DetectionResultSerializer,
    MLModelSerializer,
)


class MLModelViewSet(viewsets.ModelViewSet):
    """CRUD operations for ML models.

    list:   GET /api/perception/models/
    create: POST /api/perception/models/
    read:   GET /api/perception/models/{id}/
    """

    queryset = MLModel.objects.all()
    serializer_class = MLModelSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "model_type", "training_dataset"]
    ordering_fields = ["accuracy", "inference_time_ms", "created_at"]

    @action(detail=False, methods=["get"])
    def deployed(self, request):
        """GET /api/perception/models/deployed/ — List currently deployed models."""
        deployed = MLModel.objects.filter(status="deployed")
        serializer = self.get_serializer(deployed, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def run_inference(self, request, pk=None):
        """POST /api/perception/models/{id}/run_inference/ — Trigger inference via FastAPI.

        Proxies the request to the FastAPI model service.
        """
        ml_model = self.get_object()
        endpoint_map = {
            "cnn_2d": "/predict/2d",
            "cnn_3d": "/predict/3d",
            "fpn_resnet": "/predict/fpn",
        }
        endpoint = endpoint_map.get(ml_model.model_type)
        if not endpoint:
            return Response({"error": f"Unknown model type: {ml_model.model_type}"}, status=status.HTTP_400_BAD_REQUEST)

        # Forward uploaded file to FastAPI
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            api_url = f"{settings.MODEL_SERVICE_URL}{endpoint}"
            resp = requests.post(
                api_url,
                files={"file": (file.name, file.read(), file.content_type)},
                timeout=60,
            )
            resp.raise_for_status()
            return Response(resp.json())
        except requests.ConnectionError:
            return Response(
                {"error": "Model service unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except requests.Timeout:
            return Response(
                {"error": "Model service timeout"},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )


class DetectionResultViewSet(viewsets.ModelViewSet):
    """CRUD operations for detection results.

    list: GET /api/perception/detections/
    read: GET /api/perception/detections/{id}/
    """

    queryset = DetectionResult.objects.select_related("frame", "model").all()
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ["processed_at", "inference_time_ms", "num_objects_detected"]

    def get_serializer_class(self):
        if self.action == "retrieve":
            return DetectionResultDetailSerializer
        return DetectionResultSerializer

    @action(detail=False, methods=["get"])
    def stats(self, request):
        """GET /api/perception/detections/stats/ — Detection statistics."""
        from django.db.models import Avg, Count, Sum

        stats = DetectionResult.objects.aggregate(
            total_detections=Count("id"),
            total_objects=Sum("num_objects_detected"),
            avg_inference_ms=Avg("inference_time_ms"),
        )
        # Object class distribution
        class_dist = DetectedObject.objects.values("object_class").annotate(count=Count("id")).order_by("-count")
        stats["class_distribution"] = list(class_dist)
        return Response(stats)


class DetectedObjectViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only access to detected objects.

    list: GET /api/perception/objects/
    read: GET /api/perception/objects/{id}/
    """

    queryset = DetectedObject.objects.select_related("detection", "detection__frame", "detection__model").all()
    serializer_class = DetectedObjectSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["object_class"]
    ordering_fields = ["confidence", "object_class"]
