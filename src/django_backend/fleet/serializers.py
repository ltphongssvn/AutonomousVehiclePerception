# AutonomousVehiclePerception/src/django_backend/fleet/serializers.py
"""REST API serializers for fleet management."""

from rest_framework import serializers

from fleet.models import DrivingSession, SensorFrame, Vehicle


class VehicleSerializer(serializers.ModelSerializer):
    session_count = serializers.SerializerMethodField()

    class Meta:
        model = Vehicle
        fields = "__all__"

    def get_session_count(self, obj):
        return obj.sessions.count()


class SensorFrameSerializer(serializers.ModelSerializer):
    class Meta:
        model = SensorFrame
        fields = "__all__"


class DrivingSessionSerializer(serializers.ModelSerializer):
    vehicle_name = serializers.CharField(source="vehicle.name", read_only=True)
    duration_minutes = serializers.FloatField(read_only=True)
    frame_count = serializers.SerializerMethodField()

    class Meta:
        model = DrivingSession
        fields = "__all__"

    def get_frame_count(self, obj):
        return obj.frames.count()


class DrivingSessionDetailSerializer(DrivingSessionSerializer):
    frames = SensorFrameSerializer(many=True, read_only=True)

    class Meta(DrivingSessionSerializer.Meta):
        pass
