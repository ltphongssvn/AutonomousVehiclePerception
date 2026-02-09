# AutonomousVehiclePerception/deploy/terraform/outputs.tf

output "eks_cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = module.eks.cluster_endpoint
}

output "ecr_django_url" {
  description = "ECR repository URL for Django"
  value       = aws_ecr_repository.django.repository_url
}

output "ecr_fastapi_url" {
  description = "ECR repository URL for FastAPI"
  value       = aws_ecr_repository.fastapi.repository_url
}

output "rds_endpoint" {
  description = "RDS PostgreSQL endpoint"
  value       = aws_db_instance.postgres.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "s3_data_bucket" {
  description = "S3 bucket for sensor data"
  value       = aws_s3_bucket.data.id
}
