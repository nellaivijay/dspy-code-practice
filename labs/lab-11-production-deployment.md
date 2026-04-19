# Lab 11: Production Deployment

## Overview

This lab covers deploying DSPy applications to production environments. You'll learn about infrastructure, monitoring, scaling, and maintaining production-ready AI systems.

## Learning Objectives

- Understand production deployment requirements
- Set up production infrastructure
- Implement monitoring and logging
- Handle scaling and load balancing
- Maintain and update production systems

## Prerequisites

- Completed Lab 10: Real-World Applications
- Understanding of cloud infrastructure
- Experience with deployment concepts

## Production Architecture

### Components of Production DSPy Systems

1. **Application Layer**: DSPy programs and business logic
2. **API Layer**: REST/GraphQL APIs for client access
3. **Infrastructure Layer**: Servers, databases, caching
4. **Monitoring Layer**: Logging, metrics, alerting
5. **Security Layer**: Authentication, authorization, encryption

### Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                  Load Balancer                   │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼──────────┐
│  Application   │  │  Application   │
│  Server 1      │  │  Server 2      │
│  (DSPy Apps)   │  │  (DSPy Apps)   │
└───────┬────────┘  └──────┬──────────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │   Shared Services  │
        │  - Database       │
        │  - Cache          │
        │  - Message Queue  │
        └───────────────────┘
```

## Containerization with Docker

### Dockerfile for DSPy Application

```dockerfile
# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DSPY_MODEL=gpt-3.5-turbo

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  dspy-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:password@db:5432/dspy
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app/app
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=dspy
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

## FastAPI Application Wrapper

### Basic FastAPI App

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import dspy
import os

app = FastAPI(title="DSPy Production API")

# Configure DSPy
lm = dspy.OpenAI(
    model=os.getenv("DSPY_MODEL", "gpt-3.5-turbo"),
    api_key=os.getenv("OPENAI_API_KEY")
)
dspy.settings.configure(lm=lm)

# Define DSPy program
class TextSummarizer(dspy.Signature):
    """Summarize text."""
    text = dspy.InputField(desc="text to summarize")
    summary = dspy.OutputField(desc="summary")

summarizer = dspy.Predict(TextSummarizer)

# API Models
class SummarizeRequest(BaseModel):
    text: str
    max_length: int = 100

class SummarizeResponse(BaseModel):
    summary: str
    processing_time: float

import time

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """Summarize text using DSPy."""
    try:
        start_time = time.time()
        
        result = summarizer(text=request.text)
        processing_time = time.time() - start_time
        
        return SummarizeResponse(
            summary=result.summary,
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Monitoring and Logging

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# Usage
logger.info("Processing request", extra_data={
    "request_id": "12345",
    "user_id": "user123",
    "endpoint": "/summarize"
})
```

### Metrics Collection

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Define metrics
request_counter = Counter(
    'dspy_requests_total',
    'Total number of requests',
    ['endpoint', 'status']
)

request_duration = Histogram(
    'dspy_request_duration_seconds',
    'Request duration in seconds',
    ['endpoint']
)

active_requests = Gauge(
    'dspy_active_requests',
    'Number of active requests'
)

# Middleware to track metrics
from fastapi import Request

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    active_requests.inc()
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    request_duration.labels(
        endpoint=request.url.path
    ).observe(duration)
    
    request_counter.labels(
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    active_requests.dec()
    
    return response
```

## Caching Strategy

### Redis Caching

```python
import redis
import json
import hashlib

class CacheManager:
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour
    
    def generate_key(self, prefix, **kwargs):
        """Generate cache key from parameters."""
        key_data = f"{prefix}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prefix, **kwargs):
        """Get cached result."""
        key = self.generate_key(prefix, **kwargs)
        cached = self.redis_client.get(key)
        
        if cached:
            return json.loads(cached)
        return None
    
    def set(self, prefix, value, ttl=None, **kwargs):
        """Set cached result."""
        key = self.generate_key(prefix, **kwargs)
        ttl = ttl or self.default_ttl
        
        self.redis_client.setex(
            key,
            ttl,
            json.dumps(value)
        )
    
    def invalidate(self, prefix):
        """Invalidate all keys with prefix."""
        pattern = f"{prefix}:*"
        keys = self.redis_client.keys(pattern)
        
        if keys:
            self.redis_client.delete(*keys)

# Usage in DSPy application
cache = CacheManager()

def cached_summarize(text):
    cache_key = cache.get("summarize", text=text)
    
    if cache_key:
        return cache_key
    
    result = summarizer(text=text)
    cache.set("summarize", result.summary, text=text)
    
    return result.summary
```

## Error Handling and Resilience

### Circuit Breaker Pattern

```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def resilient_dspy_call(func, *args, **kwargs):
    """Call DSPy function with circuit breaker."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"DSPy call failed: {str(e)}")
        raise

# Usage
try:
    result = resilient_dspy_call(summarizer, text=sample_text)
except Exception as e:
    # Fallback logic
    result = "Service temporarily unavailable"
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from fastapi import Request

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(_rate_limit_exceeded_handler)
async def rate_limit_exceeded_handler(request: Request, exc):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )

@app.post("/summarize")
@limiter.limit("10/minute")
async def summarize_text(request: SummarizeRequest):
    """Summarize with rate limiting."""
    # ... existing logic
```

## Database Integration

### Storing Results and Analytics

```python
import asyncpg
from datetime import datetime

class DatabaseManager:
    def __init__(self, database_url):
        self.pool = None
        self.database_url = database_url
    
    async def initialize(self):
        """Initialize database connection pool."""
        self.pool = await asyncpg.create_pool(self.database_url)
    
    async def store_result(self, request_data, result, metrics):
        """Store processing result."""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO processing_results 
                (request_id, input_text, output_text, 
                 processing_time, timestamp, metrics)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
                request_data["request_id"],
                request_data["input_text"],
                result["output"],
                metrics["processing_time"],
                datetime.utcnow(),
                json.dumps(metrics)
            )
    
    async def get_analytics(self, start_date, end_date):
        """Get analytics for date range."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM processing_results
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp DESC
            """, start_date, end_date)
            return rows
```

## Deployment Strategies

### Blue-Green Deployment

```bash
# Deploy to green environment
kubectl apply -f deployment-green.yaml

# Wait for green to be healthy
kubectl wait --for=condition=available --timeout=600s deployment/dspy-app-green

# Switch traffic to green
kubectl patch service dspy-service -p '{"spec":{"selector":{"version":"green"}}}'

# Monitor green deployment
# If issues, rollback to blue
kubectl patch service dspy-service -p '{"spec":{"selector":{"version":"blue"}}}'
```

### Canary Deployment

```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: dspy-app
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dspy-app
  service:
    port: 80
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
```

## Security Considerations

### API Key Management

```python
from cryptography.fernet import Fernet
import os

class SecretManager:
    def __init__(self):
        self.cipher = Fernet(os.getenv("ENCRYPTION_KEY"))
    
    def encrypt_secret(self, secret):
        """Encrypt sensitive data."""
        return self.cipher.encrypt(secret.encode()).decode()
    
    def decrypt_secret(self, encrypted_secret):
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_secret.encode()).decode()
    
    def get_api_key(self):
        """Get and decrypt API key."""
        encrypted_key = os.getenv("ENCRYPTED_OPENAI_API_KEY")
        return self.decrypt_secret(encrypted_key)
```

### Input Validation

```python
from pydantic import validator, constr

class SummarizeRequest(BaseModel):
    text: constr(max_length=10000)  # Limit text length
    max_length: int = 100
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Check for malicious content
        dangerous_patterns = ['<script', 'javascript:', 'data:']
        if any(pattern in v.lower() for pattern in dangerous_patterns):
            raise ValueError("Invalid content detected")
        
        return v
```

## Performance Optimization

### Connection Pooling

```python
from httpx import AsyncClient, Limits

class DSPyClient:
    def __init__(self):
        self.client = AsyncClient(
            limits=Limits(
                max_connections=100,
                max_keepalive_connections=20
            ),
            timeout=30.0
        )
    
    async def close(self):
        await self.client.aclose()
```

### Batch Processing

```python
async def batch_process(items, batch_size=10):
    """Process items in batches."""
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        batch_results = await asyncio.gather(*[
            process_item(item) for item in batch
        ])
        results.extend(batch_results)
    
    return results
```

## Monitoring Dashboard

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "DSPy Application Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(dspy_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Request Duration",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, dspy_request_duration_seconds)"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(dspy_requests_total{status=\"5xx\"}[5m]) / rate(dspy_requests_total[5m])"
          }
        ]
      }
    ]
  }
}
```

## Best Practices

1. **Infrastructure as Code**: Use Terraform or CloudFormation for infrastructure
2. **Continuous Integration**: Automate testing and deployment
3. **Monitoring**: Implement comprehensive monitoring and alerting
4. **Security**: Encrypt secrets, validate inputs, implement rate limiting
5. **Scalability**: Design for horizontal scaling from the start
6. **Cost Optimization**: Monitor resource usage and optimize costs
7. **Documentation**: Maintain clear documentation for operations

## Summary

In this lab, you learned:

- **Production Architecture**: Understanding production system design
- **Containerization**: Docker and Docker Compose for deployment
- **API Development**: FastAPI wrapper for DSPy applications
- **Monitoring**: Logging, metrics, and performance tracking
- **Caching**: Redis caching for performance optimization
- **Resilience**: Error handling, circuit breakers, rate limiting
- **Security**: Secret management, input validation
- **Deployment**: Blue-green and canary deployment strategies

## Final Challenge

Deploy a complete DSPy application that:
1. Includes proper monitoring and logging
2. Implements caching and error handling
3. Uses containerization for deployment
4. Provides comprehensive metrics
5. Handles scaling gracefully

This completes your journey through DSPy mastery. You're now ready to build production-ready AI systems!

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Redis Caching](https://redis.io/docs/manual/patterns/)