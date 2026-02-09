# AutonomousVehiclePerception/src/django_backend/fleet/models.py
"""Fleet management models for autonomous vehicle tracking."""

from django.db import models


class Vehicle(models.Model):
    """Autonomous vehicle in the fleet."""

    STATUS_CHOICES = [
        ("active", "Active"),
        ("maintenance", "Maintenance"),
        ("offline", "Offline"),
        ("testing", "Testing"),
    ]

    vin = models.CharField(max_length=17, unique=True, help_text="Vehicle Identification Number")
    name = models.CharField(max_length=100)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="offline")
    model_type = models.CharField(max_length=100, help_text="Vehicle model (e.g., Model X Sensor Suite v3)")
    firmware_version = models.CharField(max_length=50, blank=True)
    num_cameras = models.PositiveIntegerField(default=6)
    has_lidar = models.BooleanField(default=True)
    registered_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.name} ({self.vin})"


class DrivingSession(models.Model):
    """A recorded driving session from a vehicle."""

    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, related_name="sessions")
    session_id = models.CharField(max_length=64, unique=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    distance_km = models.FloatField(default=0.0)
    num_frames = models.PositiveIntegerField(default=0)
    route_description = models.TextField(blank=True)
    weather = models.CharField(max_length=50, blank=True, help_text="e.g., clear, rain, fog, night")
    s3_path = models.CharField(max_length=500, blank=True, help_text="S3 path to raw sensor data")
    processed = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-start_time"]

    def __str__(self):
        return f"Session {self.session_id} - {self.vehicle.name}"

    @property
    def duration_minutes(self):
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds() / 60
        return 0


class SensorFrame(models.Model):
    """Individual sensor frame (camera + LiDAR) from a driving session."""

    session = models.ForeignKey(DrivingSession, on_delete=models.CASCADE, related_name="frames")
    frame_number = models.PositiveIntegerField()
    timestamp = models.DateTimeField()
    camera_image_path = models.CharField(max_length=500, help_text="S3 path to camera image")
    lidar_path = models.CharField(max_length=500, blank=True, help_text="S3 path to LiDAR .bin")
    gps_lat = models.FloatField(null=True, blank=True)
    gps_lon = models.FloatField(null=True, blank=True)
    speed_kmh = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["session", "frame_number"]
        unique_together = ["session", "frame_number"]

    def __str__(self):
        return f"Frame {self.frame_number} - Session {self.session.session_id}"
