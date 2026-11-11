# Deployment Guide

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run

```bash
docker build -t weldscope .
docker run -p 8501:8501 --env-file .env weldscope
```

## Streamlit Cloud

1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Add `HF_TOKEN` to secrets
4. Deploy

## AWS Deployment

### EC2

```bash
# Launch EC2 instance (t3.medium or larger)
ssh -i key.pem ubuntu@<instance-ip>

# Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv

# Clone and setup
git clone https://github.com/skkagolanu/weldscope.git
cd weldscope
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with credentials

# Run with nohup
nohup streamlit run app.py --server.port=8501 &
```

### ECS (Fargate)

1. Build and push Docker image to ECR
2. Create ECS task definition
3. Create ECS service with Fargate
4. Configure ALB for HTTPS

## Environment Variables

Required:
- `HF_TOKEN`: HuggingFace API token

Optional:
- `LANGFUSE_ENABLED`: Enable tracing (true/false)
- `LANGFUSE_PUBLIC_KEY`: Langfuse public key
- `LANGFUSE_SECRET_KEY`: Langfuse secret key

## Performance Tuning

### Vector Store
- Pre-build vector store before deployment
- Use persistent storage for vectorstore/

### Caching
- Enable query caching for repeated queries
- Adjust similarity threshold based on use case

### Scaling
- Use load balancer for multiple instances
- Consider GPU instances for faster inference
- Implement request queuing for high load

## Monitoring

### Metrics to Track
- Query latency (P50, P95, P99)
- Cache hit rate
- Error rate
- Token usage

### Logging
- Application logs to CloudWatch/ELK
- Langfuse for query tracing
- Performance metrics to Prometheus

## Security

### Best Practices
- Store secrets in AWS Secrets Manager
- Use IAM roles for AWS access
- Enable HTTPS with valid certificates
- Implement rate limiting
- Regular security updates

### Network Security
- Use VPC with private subnets
- Security groups for access control
- WAF for application protection
