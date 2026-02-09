# AutonomousVehiclePerception/src/django_backend/fleet/views.py
"""REST API views for fleet management."""

from rest_framework import filters, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from fleet.models import DrivingSession, SensorFrame, Vehicle
from fleet.serializers import (
    DrivingSessionDetailSerializer,
    DrivingSessionSerializer,
    SensorFrameSerializer,
    VehicleSerializer,
)


class VehicleViewSet(viewsets.ModelViewSet):
    """CRUD operations for vehicles in the fleet.

    list:   GET /api/fleet/vehicles/
    create: POST /api/fleet/vehicles/
    read:   GET /api/fleet/vehicles/{id}/
    update: PUT /api/fleet/vehicles/{id}/
    delete: DELETE /api/fleet/vehicles/{id}/
    """

    queryset = Vehicle.objects.all()
    serializer_class = VehicleSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["name", "vin", "model_type"]
    ordering_fields = ["name", "status", "registered_at", "updated_at"]

    @action(detail=True, methods=["get"])
    def sessions(self, request, pk=None):
        """GET /api/fleet/vehicles/{id}/sessions/ — List sessions for a vehicle."""
        vehicle = self.get_object()
        sessions = vehicle.sessions.all()
        serializer = DrivingSessionSerializer(sessions, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=["get"])
    def summary(self, request):
        """GET /api/fleet/vehicles/summary/ — Fleet summary statistics."""
        total = Vehicle.objects.count()
        active = Vehicle.objects.filter(status="active").count()
        total_sessions = DrivingSession.objects.count()
        total_frames = SensorFrame.objects.count()
        return Response(
            {
                "total_vehicles": total,
                "active_vehicles": active,
                "total_sessions": total_sessions,
                "total_frames": total_frames,
            }
        )


class DrivingSessionViewSet(viewsets.ModelViewSet):
    """CRUD operations for driving sessions.

    list:   GET /api/fleet/sessions/
    create: POST /api/fleet/sessions/
    read:   GET /api/fleet/sessions/{id}/
    """

    queryset = DrivingSession.objects.select_related("vehicle").all()
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ["session_id", "route_description", "weather"]
    ordering_fields = ["start_time", "distance_km", "num_frames"]

    def get_serializer_class(self):
        if self.action == "retrieve":
            return DrivingSessionDetailSerializer
        return DrivingSessionSerializer

    @action(detail=True, methods=["get"])
    def frames(self, request, pk=None):
        """GET /api/fleet/sessions/{id}/frames/ — List frames in a session."""
        session = self.get_object()
        frames = session.frames.all()
        page = self.paginate_queryset(frames)
        if page is not None:
            serializer = SensorFrameSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = SensorFrameSerializer(frames, many=True)
        return Response(serializer.data)


class SensorFrameViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only access to sensor frames.

    list: GET /api/fleet/frames/
    read: GET /api/fleet/frames/{id}/
    """

    queryset = SensorFrame.objects.select_related("session", "session__vehicle").all()
    serializer_class = SensorFrameSerializer
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ["timestamp", "frame_number"]
