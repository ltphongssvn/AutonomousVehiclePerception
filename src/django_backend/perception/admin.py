# AutonomousVehiclePerception/src/django_backend/perception/admin.py
"""Perception admin configuration."""

from django.contrib import admin

from perception.models import DetectedObject, DetectionResult, MLModel


class DetectedObjectInline(admin.TabularInline):
    model = DetectedObject
    extra = 0
    fields = ["object_class", "confidence", "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]
    readonly_fields = ["object_class", "confidence"]


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = [
        "name",
        "model_type",
        "version",
        "status",
        "accuracy",
        "inference_time_ms",
        "training_dataset",
        "updated_at",
    ]
    list_filter = ["status", "model_type", "training_dataset"]
    search_fields = ["name", "version"]


@admin.register(DetectionResult)
class DetectionResultAdmin(admin.ModelAdmin):
    list_display = ["id", "frame", "model", "num_objects_detected", "inference_time_ms", "processed_at"]
    list_filter = ["model__model_type"]
    inlines = [DetectedObjectInline]


@admin.register(DetectedObject)
class DetectedObjectAdmin(admin.ModelAdmin):
    list_display = ["id", "detection", "object_class", "confidence"]
    list_filter = ["object_class"]
    search_fields = ["object_class"]
