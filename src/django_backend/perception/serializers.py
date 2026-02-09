# AutonomousVehiclePerception/src/django_backend/perception/serializers.py
"""REST API serializers for perception results."""

from rest_framework import serializers

from perception.models import DetectedObject, DetectionResult, MLModel


class MLModelSerializer(serializers.ModelSerializer):
    detection_count = serializers.SerializerMethodField()

    class Meta:
        model = MLModel
        fields = "__all__"

    def get_detection_count(self, obj):
        return obj.detections.count()


class DetectedObjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = DetectedObject
        fields = "__all__"


class DetectionResultSerializer(serializers.ModelSerializer):
    model_name = serializers.CharField(source="model.name", read_only=True)
    frame_number = serializers.IntegerField(source="frame.frame_number", read_only=True)
    object_count = serializers.SerializerMethodField()

    class Meta:
        model = DetectionResult
        fields = "__all__"

    def get_object_count(self, obj):
        return obj.objects.count()


class DetectionResultDetailSerializer(DetectionResultSerializer):
    objects = DetectedObjectSerializer(many=True, read_only=True)

    class Meta(DetectionResultSerializer.Meta):
        pass
