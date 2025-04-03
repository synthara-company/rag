"""
Cloud deployment utilities for the RAG Transformer system.
This module provides functionality to deploy the system to various cloud providers.
"""

import os
import subprocess
import json
import argparse
from typing import Dict, Any, List, Optional

class CloudDeployer:
    """Base class for cloud deployment"""
    
    def __init__(self, project_dir: str, config_file: Optional[str] = None):
        """
        Initialize cloud deployer
        
        Args:
            project_dir (str): Path to the project directory
            config_file (Optional[str]): Path to the configuration file
        """
        self.project_dir = os.path.abspath(project_dir)
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
    
    def validate_environment(self) -> bool:
        """
        Validate that the environment is properly set up
        
        Returns:
            bool: True if environment is valid
        """
        return True
    
    def build(self) -> bool:
        """
        Build the application
        
        Returns:
            bool: True if build was successful
        """
        return True
    
    def deploy(self) -> bool:
        """
        Deploy the application
        
        Returns:
            bool: True if deployment was successful
        """
        return True
    
    def run_command(self, command: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        """
        Run a shell command
        
        Args:
            command (List[str]): Command to run
            cwd (Optional[str]): Working directory
            
        Returns:
            subprocess.CompletedProcess: Command result
        """
        print(f"Running command: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            cwd=cwd or self.project_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}")
            print(f"Error: {result.stderr}")
        else:
            print(f"Command succeeded")
            if result.stdout:
                print(f"Output: {result.stdout[:500]}...")
        
        return result


class DockerDeployer(CloudDeployer):
    """Docker deployment"""
    
    def validate_environment(self) -> bool:
        """
        Validate that Docker is installed and running
        
        Returns:
            bool: True if Docker is available
        """
        try:
            result = self.run_command(["docker", "--version"])
            if result.returncode != 0:
                print("Docker is not installed or not in PATH")
                return False
            
            result = self.run_command(["docker-compose", "--version"])
            if result.returncode != 0:
                print("Docker Compose is not installed or not in PATH")
                return False
            
            return True
        except Exception as e:
            print(f"Error validating Docker environment: {e}")
            return False
    
    def build(self) -> bool:
        """
        Build Docker image
        
        Returns:
            bool: True if build was successful
        """
        try:
            deployment_dir = os.path.join(self.project_dir, "deployment")
            
            result = self.run_command(
                ["docker-compose", "build"],
                cwd=deployment_dir
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Error building Docker image: {e}")
            return False
    
    def deploy(self) -> bool:
        """
        Deploy using Docker Compose
        
        Returns:
            bool: True if deployment was successful
        """
        try:
            deployment_dir = os.path.join(self.project_dir, "deployment")
            
            result = self.run_command(
                ["docker-compose", "up", "-d"],
                cwd=deployment_dir
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Error deploying with Docker Compose: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop Docker containers
        
        Returns:
            bool: True if stop was successful
        """
        try:
            deployment_dir = os.path.join(self.project_dir, "deployment")
            
            result = self.run_command(
                ["docker-compose", "down"],
                cwd=deployment_dir
            )
            
            return result.returncode == 0
        except Exception as e:
            print(f"Error stopping Docker containers: {e}")
            return False


class AWSDeployer(CloudDeployer):
    """AWS deployment"""
    
    def validate_environment(self) -> bool:
        """
        Validate that AWS CLI is installed and configured
        
        Returns:
            bool: True if AWS CLI is available
        """
        try:
            result = self.run_command(["aws", "--version"])
            if result.returncode != 0:
                print("AWS CLI is not installed or not in PATH")
                return False
            
            result = self.run_command(["aws", "sts", "get-caller-identity"])
            if result.returncode != 0:
                print("AWS CLI is not configured properly")
                return False
            
            return True
        except Exception as e:
            print(f"Error validating AWS environment: {e}")
            return False
    
    def build(self) -> bool:
        """
        Build for AWS deployment
        
        Returns:
            bool: True if build was successful
        """
        try:
            # Create ECR repository if it doesn't exist
            repository_name = self.config.get("ecr_repository", "rag-transformer")
            
            result = self.run_command([
                "aws", "ecr", "describe-repositories",
                "--repository-names", repository_name
            ])
            
            if result.returncode != 0:
                print(f"Creating ECR repository: {repository_name}")
                result = self.run_command([
                    "aws", "ecr", "create-repository",
                    "--repository-name", repository_name
                ])
                
                if result.returncode != 0:
                    print(f"Failed to create ECR repository: {repository_name}")
                    return False
            
            # Get ECR login token
            result = self.run_command([
                "aws", "ecr", "get-login-password", "--region", self.config.get("region", "us-west-2")
            ])
            
            if result.returncode != 0:
                print("Failed to get ECR login token")
                return False
            
            ecr_token = result.stdout.strip()
            
            # Get ECR repository URI
            result = self.run_command([
                "aws", "ecr", "describe-repositories",
                "--repository-names", repository_name,
                "--query", "repositories[0].repositoryUri",
                "--output", "text"
            ])
            
            if result.returncode != 0:
                print(f"Failed to get ECR repository URI for {repository_name}")
                return False
            
            repository_uri = result.stdout.strip()
            
            # Login to ECR
            result = self.run_command([
                "docker", "login",
                "--username", "AWS",
                "--password-stdin",
                repository_uri
            ], input=ecr_token)
            
            if result.returncode != 0:
                print("Failed to login to ECR")
                return False
            
            # Build and tag Docker image
            deployment_dir = os.path.join(self.project_dir, "deployment")
            
            result = self.run_command([
                "docker", "build",
                "-t", f"{repository_uri}:latest",
                "-f", "Dockerfile",
                ".."
            ], cwd=deployment_dir)
            
            if result.returncode != 0:
                print("Failed to build Docker image")
                return False
            
            # Push Docker image to ECR
            result = self.run_command([
                "docker", "push",
                f"{repository_uri}:latest"
            ])
            
            if result.returncode != 0:
                print("Failed to push Docker image to ECR")
                return False
            
            return True
        except Exception as e:
            print(f"Error building for AWS deployment: {e}")
            return False
    
    def deploy(self) -> bool:
        """
        Deploy to AWS ECS
        
        Returns:
            bool: True if deployment was successful
        """
        try:
            # Create ECS task definition
            task_definition = {
                "family": "rag-transformer",
                "networkMode": "awsvpc",
                "executionRoleArn": self.config.get("execution_role_arn"),
                "taskRoleArn": self.config.get("task_role_arn"),
                "containerDefinitions": [
                    {
                        "name": "rag-transformer",
                        "image": f"{self.config.get('ecr_repository_uri')}:latest",
                        "essential": True,
                        "portMappings": [
                            {
                                "containerPort": 8501,
                                "hostPort": 8501,
                                "protocol": "tcp"
                            }
                        ],
                        "environment": [
                            {
                                "name": "OPENAI_API_KEY",
                                "value": self.config.get("openai_api_key", "")
                            },
                            {
                                "name": "ANTHROPIC_API_KEY",
                                "value": self.config.get("anthropic_api_key", "")
                            },
                            {
                                "name": "TMDB_API_KEY",
                                "value": self.config.get("tmdb_api_key", "")
                            },
                            {
                                "name": "NASA_API_KEY",
                                "value": self.config.get("nasa_api_key", "")
                            }
                        ],
                        "logConfiguration": {
                            "logDriver": "awslogs",
                            "options": {
                                "awslogs-group": "/ecs/rag-transformer",
                                "awslogs-region": self.config.get("region", "us-west-2"),
                                "awslogs-stream-prefix": "ecs"
                            }
                        }
                    }
                ],
                "requiresCompatibilities": ["FARGATE"],
                "cpu": "1024",
                "memory": "2048"
            }
            
            # Write task definition to file
            task_def_file = os.path.join(self.project_dir, "deployment", "task-definition.json")
            with open(task_def_file, 'w') as f:
                json.dump(task_definition, f, indent=2)
            
            # Register task definition
            result = self.run_command([
                "aws", "ecs", "register-task-definition",
                "--cli-input-json", f"file://{task_def_file}"
            ])
            
            if result.returncode != 0:
                print("Failed to register ECS task definition")
                return False
            
            # Create or update ECS service
            cluster = self.config.get("ecs_cluster", "default")
            service_name = "rag-transformer"
            
            # Check if service exists
            result = self.run_command([
                "aws", "ecs", "describe-services",
                "--cluster", cluster,
                "--services", service_name
            ])
            
            if "MISSING" in result.stdout:
                # Create new service
                result = self.run_command([
                    "aws", "ecs", "create-service",
                    "--cluster", cluster,
                    "--service-name", service_name,
                    "--task-definition", "rag-transformer",
                    "--desired-count", "1",
                    "--launch-type", "FARGATE",
                    "--network-configuration", f"awsvpcConfiguration={{subnets=[{self.config.get('subnet_id')}],securityGroups=[{self.config.get('security_group_id')}],assignPublicIp=ENABLED}}"
                ])
                
                if result.returncode != 0:
                    print("Failed to create ECS service")
                    return False
            else:
                # Update existing service
                result = self.run_command([
                    "aws", "ecs", "update-service",
                    "--cluster", cluster,
                    "--service", service_name,
                    "--task-definition", "rag-transformer",
                    "--force-new-deployment"
                ])
                
                if result.returncode != 0:
                    print("Failed to update ECS service")
                    return False
            
            print("Deployment to AWS ECS completed successfully")
            return True
        except Exception as e:
            print(f"Error deploying to AWS ECS: {e}")
            return False


class GCPDeployer(CloudDeployer):
    """Google Cloud Platform deployment"""
    
    def validate_environment(self) -> bool:
        """
        Validate that gcloud CLI is installed and configured
        
        Returns:
            bool: True if gcloud CLI is available
        """
        try:
            result = self.run_command(["gcloud", "--version"])
            if result.returncode != 0:
                print("gcloud CLI is not installed or not in PATH")
                return False
            
            result = self.run_command(["gcloud", "auth", "list"])
            if result.returncode != 0:
                print("gcloud CLI is not configured properly")
                return False
            
            return True
        except Exception as e:
            print(f"Error validating GCP environment: {e}")
            return False
    
    def build(self) -> bool:
        """
        Build for GCP deployment
        
        Returns:
            bool: True if build was successful
        """
        try:
            # Set GCP project
            project_id = self.config.get("project_id")
            if not project_id:
                print("GCP project ID not specified in config")
                return False
            
            result = self.run_command([
                "gcloud", "config", "set", "project", project_id
            ])
            
            if result.returncode != 0:
                print(f"Failed to set GCP project to {project_id}")
                return False
            
            # Build and push Docker image to GCR
            image_name = f"gcr.io/{project_id}/rag-transformer:latest"
            
            deployment_dir = os.path.join(self.project_dir, "deployment")
            
            result = self.run_command([
                "gcloud", "builds", "submit",
                "--tag", image_name,
                ".."
            ], cwd=deployment_dir)
            
            if result.returncode != 0:
                print("Failed to build and push Docker image to GCR")
                return False
            
            return True
        except Exception as e:
            print(f"Error building for GCP deployment: {e}")
            return False
    
    def deploy(self) -> bool:
        """
        Deploy to Google Cloud Run
        
        Returns:
            bool: True if deployment was successful
        """
        try:
            # Set GCP project
            project_id = self.config.get("project_id")
            if not project_id:
                print("GCP project ID not specified in config")
                return False
            
            # Deploy to Cloud Run
            image_name = f"gcr.io/{project_id}/rag-transformer:latest"
            service_name = "rag-transformer"
            region = self.config.get("region", "us-central1")
            
            result = self.run_command([
                "gcloud", "run", "deploy", service_name,
                "--image", image_name,
                "--platform", "managed",
                "--region", region,
                "--allow-unauthenticated",
                "--memory", "2Gi",
                "--cpu", "1",
                "--set-env-vars", f"OPENAI_API_KEY={self.config.get('openai_api_key', '')}",
                "--set-env-vars", f"ANTHROPIC_API_KEY={self.config.get('anthropic_api_key', '')}",
                "--set-env-vars", f"TMDB_API_KEY={self.config.get('tmdb_api_key', '')}",
                "--set-env-vars", f"NASA_API_KEY={self.config.get('nasa_api_key', '')}"
            ])
            
            if result.returncode != 0:
                print("Failed to deploy to Google Cloud Run")
                return False
            
            # Get service URL
            result = self.run_command([
                "gcloud", "run", "services", "describe", service_name,
                "--platform", "managed",
                "--region", region,
                "--format", "value(status.url)"
            ])
            
            if result.returncode == 0:
                service_url = result.stdout.strip()
                print(f"Deployed to Google Cloud Run: {service_url}")
            
            return True
        except Exception as e:
            print(f"Error deploying to Google Cloud Run: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Deploy RAG Transformer to cloud")
    parser.add_argument("--provider", choices=["docker", "aws", "gcp"], default="docker", help="Cloud provider")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--project-dir", default="..", help="Path to project directory")
    parser.add_argument("--action", choices=["validate", "build", "deploy", "stop"], default="deploy", help="Action to perform")
    
    args = parser.parse_args()
    
    # Create deployer
    if args.provider == "docker":
        deployer = DockerDeployer(args.project_dir, args.config)
    elif args.provider == "aws":
        deployer = AWSDeployer(args.project_dir, args.config)
    elif args.provider == "gcp":
        deployer = GCPDeployer(args.project_dir, args.config)
    else:
        print(f"Unknown provider: {args.provider}")
        return
    
    # Perform action
    if args.action == "validate":
        if deployer.validate_environment():
            print("Environment validation successful")
        else:
            print("Environment validation failed")
    elif args.action == "build":
        if deployer.validate_environment() and deployer.build():
            print("Build successful")
        else:
            print("Build failed")
    elif args.action == "deploy":
        if deployer.validate_environment() and deployer.build() and deployer.deploy():
            print("Deployment successful")
        else:
            print("Deployment failed")
    elif args.action == "stop" and args.provider == "docker":
        if deployer.stop():
            print("Stop successful")
        else:
            print("Stop failed")
    else:
        print(f"Action {args.action} not supported for provider {args.provider}")


if __name__ == "__main__":
    main()
