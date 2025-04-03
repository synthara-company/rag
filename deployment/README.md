# RAG Transformer Deployment Guide

This guide provides instructions for deploying the RAG Transformer system using Docker and various cloud providers.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ installed
- For cloud deployments:
  - AWS CLI (for AWS deployment)
  - gcloud CLI (for GCP deployment)
  - Appropriate cloud credentials configured

## Local Deployment with Docker

The simplest way to deploy RAG Transformer is using Docker Compose:

1. Navigate to the deployment directory:
   ```bash
   cd rag-transformer/deployment
   ```

2. Create a `.env` file with your API keys:
   ```bash
   # API Keys for RAG Transformer
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   TMDB_API_KEY=your_tmdb_api_key
   NASA_API_KEY=your_nasa_api_key
   ```

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access the web interface at http://localhost:8501

5. To stop the containers:
   ```bash
   docker-compose down
   ```

## Cloud Deployment

The `cloud_deployment.py` script provides utilities for deploying to various cloud providers.

### AWS Deployment

1. Create a configuration file `aws_config.json`:
   ```json
   {
     "region": "us-west-2",
     "ecr_repository": "rag-transformer",
     "ecs_cluster": "default",
     "execution_role_arn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
     "task_role_arn": "arn:aws:iam::123456789012:role/ecsTaskRole",
     "subnet_id": "subnet-0123456789abcdef0",
     "security_group_id": "sg-0123456789abcdef0",
     "openai_api_key": "your_openai_api_key",
     "anthropic_api_key": "your_anthropic_api_key",
     "tmdb_api_key": "your_tmdb_api_key",
     "nasa_api_key": "your_nasa_api_key"
   }
   ```

2. Run the deployment script:
   ```bash
   python cloud_deployment.py --provider aws --config aws_config.json
   ```

### GCP Deployment

1. Create a configuration file `gcp_config.json`:
   ```json
   {
     "project_id": "your-gcp-project-id",
     "region": "us-central1",
     "openai_api_key": "your_openai_api_key",
     "anthropic_api_key": "your_anthropic_api_key",
     "tmdb_api_key": "your_tmdb_api_key",
     "nasa_api_key": "your_nasa_api_key"
   }
   ```

2. Run the deployment script:
   ```bash
   python cloud_deployment.py --provider gcp --config gcp_config.json
   ```

## Deployment Options

The deployment script supports several actions:

- `validate`: Validate the environment
- `build`: Build the Docker image
- `deploy`: Build and deploy the application
- `stop`: Stop the application (Docker only)

Example:
```bash
python cloud_deployment.py --provider docker --action validate
python cloud_deployment.py --provider aws --action build --config aws_config.json
python cloud_deployment.py --provider gcp --action deploy --config gcp_config.json
python cloud_deployment.py --provider docker --action stop
```

## Customization

You can customize the deployment by modifying the following files:

- `Dockerfile`: Docker image configuration
- `docker-compose.yml`: Docker Compose configuration
- `requirements.txt`: Python dependencies
- `cloud_deployment.py`: Cloud deployment utilities

## Troubleshooting

### Docker Issues

- **Error: Port already in use**
  - Change the port mapping in `docker-compose.yml`
  - Check for other services using port 8501

- **Error: Memory limit exceeded**
  - Increase memory limit in Docker settings
  - Use a smaller model by modifying `advanced_rag_pipeline.py`

### Cloud Deployment Issues

- **AWS: Task definition not found**
  - Check that the task definition was created successfully
  - Verify IAM roles and permissions

- **GCP: Deployment failed**
  - Check that the Docker image was pushed to GCR
  - Verify project ID and region
  - Check Cloud Run service account permissions

## Security Considerations

- API keys are stored in environment variables
- For production deployments, consider using a secrets manager
- Restrict access to the web interface using authentication
- Use HTTPS for all communications
