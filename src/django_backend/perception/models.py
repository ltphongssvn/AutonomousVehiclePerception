# AutonomousVehiclePerception/src/django_backend/perception/models.py
"""Perception models for object detection results and model management."""

from django.db import models

from fleet.models import SensorFrame


class MLModel(models.Model):
    """Registered ML model for inference."""

    MODEL_TYPE_CHOICES = [
        ("cnn_2d", "2D CNN"),
        ("cnn_3d", "3D Voxel CNN"),
        ("fpn_resnet", "FPN-ResNet50"),
    ]
    STATUS_CHOICES = [
        ("training", "Training"),
        ("validating", "Validating"),
        ("deployed", "Deployed"),
        ("archived", "Archived"),
    ]

    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPE_CHOICES)
    version = models.CharField(max_length=50)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="training")
    accuracy = models.FloatField(null=True, blank=True, help_text="Validation mAP")
    inference_time_ms = models.FloatField(null=True, blank=True, help_text="Average inference latency")
    num_parameters = models.PositiveIntegerField(null=True, blank=True)
    weights_path = models.CharField(max_length=500, blank=True, help_text="S3 path to model weights")
    onnx_path = models.CharField(max_length=500, blank=True, help_text="S3 path to ONNX export")
    training_dataset = models.CharField(max_length=100, blank=True, help_text="e.g., KITTI, nuScenes")
    training_epochs = models.PositiveIntegerField(null=True, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        unique_together = ["name", "version"]

    def __str__(self):
        return f"{self.name} v{self.version} ({self.get_status_display()})"


class DetectionResult(models.Model):
    """Object detection result for a single sensor frame."""

    frame = models.ForeignKey(SensorFrame, on_delete=models.CASCADE, related_name="detections")
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE, related_name="detections")
    processed_at = models.DateTimeField(auto_now_add=True)
    inference_time_ms = models.FloatField()
    num_objects_detected = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["-processed_at"]

    def __str__(self):
        return f"Detection on Frame {self.frame.frame_number} by {self.model.name}"


class DetectedObject(models.Model):
    """Individual detected object within a frame."""

    OBJECT_CLASSES = [
        ("car", "Car"),
        ("truck", "Truck"),
        ("bus", "Bus"),
        ("pedestrian", "Pedestrian"),
        ("cyclist", "Cyclist"),
        ("motorcycle", "Motorcycle"),
        ("traffic_sign", "Traffic Sign"),
        ("traffic_light", "Traffic Light"),
        ("barrier", "Barrier"),
        ("other", "Other"),
    ]

    detection = models.ForeignKey(DetectionResult, on_delete=models.CASCADE, related_name="detected_objects")
    object_class = models.CharField(max_length=20, choices=OBJECT_CLASSES)
    confidence = models.FloatField(help_text="Detection confidence 0-1")
    # 2D bounding box (pixel coordinates)
    bbox_x_min = models.FloatField()
    bbox_y_min = models.FloatField()
    bbox_x_max = models.FloatField()
    bbox_y_max = models.FloatField()
    # 3D bounding box (world coordinates, optional)
    location_x = models.FloatField(null=True, blank=True)
    location_y = models.FloatField(null=True, blank=True)
    location_z = models.FloatField(null=True, blank=True)
    dimension_h = models.FloatField(null=True, blank=True)
    dimension_w = models.FloatField(null=True, blank=True)
    dimension_l = models.FloatField(null=True, blank=True)
    rotation_y = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-confidence"]

    def __str__(self):
        return f"{self.get_object_class_display()} ({self.confidence:.2f})"
