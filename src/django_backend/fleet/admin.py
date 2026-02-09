# AutonomousVehiclePerception/src/django_backend/fleet/admin.py
"""Fleet management admin configuration."""

from django.contrib import admin

from fleet.models import DrivingSession, SensorFrame, Vehicle


class SensorFrameInline(admin.TabularInline):
    model = SensorFrame
    extra = 0
    fields = ["frame_number", "timestamp", "camera_image_path", "lidar_path", "speed_kmh"]
    readonly_fields = ["frame_number", "timestamp"]


@admin.register(Vehicle)
class VehicleAdmin(admin.ModelAdmin):
    list_display = ["name", "vin", "status", "model_type", "num_cameras", "has_lidar", "updated_at"]
    list_filter = ["status", "has_lidar"]
    search_fields = ["name", "vin", "model_type"]


@admin.register(DrivingSession)
class DrivingSessionAdmin(admin.ModelAdmin):
    list_display = ["session_id", "vehicle", "start_time", "distance_km", "num_frames", "weather", "processed"]
    list_filter = ["processed", "weather"]
    search_fields = ["session_id", "route_description"]
    inlines = [SensorFrameInline]


@admin.register(SensorFrame)
class SensorFrameAdmin(admin.ModelAdmin):
    list_display = ["frame_number", "session", "timestamp", "speed_kmh"]
    list_filter = ["session__vehicle"]
