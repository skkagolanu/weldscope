# Docker Deployment

## Quick Start

### Using Docker Compose

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Using Docker

```bash
# Build image
docker build -t weldscope .

# Run container
docker run -d \
  -p 8501:8501 \
  -e HF_TOKEN=your_token \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/vectorstore:/app/vectorstore:ro \
  --name weldscope \
  weldscope

# View logs
docker logs -f weldscope

# Stop container
docker stop weldscope
docker rm weldscope
```

## Building Vector Store

```bash
# Using Docker Compose
docker-compose --profile build run vectorstore-builder

# Using Docker
docker run --rm \
  -e HF_TOKEN=your_token \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/vectorstore:/app/vectorstore \
  weldscope \
  python build_vectorstore.py --pdf-dir /app/data --output-dir /app/vectorstore
```

## Environment Variables

Create `.env` file:

```env
HF_TOKEN=your_huggingface_token
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

## Production Deployment

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker build -t weldscope .
docker tag weldscope:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/weldscope:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/weldscope:latest

# Create ECS task definition and service
aws ecs create-service --cluster rag-cluster --service-name weldscope --task-definition weldscope:1 --desired-count 2
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weldscope
spec:
  replicas: 2
  selector:
    matchLabels:
      app: weldscope
  template:
    metadata:
      labels:
        app: weldscope
    spec:
      containers:
      - name: weldscope
        image: weldscope:latest
        ports:
        - containerPort: 8501
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: hf-token
        volumeMounts:
        - name: vectorstore
          mountPath: /app/vectorstore
          readOnly: true
      volumes:
      - name: vectorstore
        persistentVolumeClaim:
          claimName: vectorstore-pvc
```

## Health Checks

The container includes a health check endpoint:

```bash
curl http://localhost:8501/_stcore/health
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker logs weldscope

# Check if port is available
lsof -i :8501

# Verify environment variables
docker exec weldscope env | grep HF_TOKEN
```

### Out of memory

```bash
# Increase memory limit
docker run -d -p 8501:8501 --memory=4g weldscope
```

### Slow performance

- Use GPU-enabled base image for faster inference
- Pre-build vector store before deployment
- Enable caching
- Use multiple replicas with load balancer
