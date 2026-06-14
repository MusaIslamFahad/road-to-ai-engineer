# Docker Tutorial for ML Engineers

From zero to containerized ML APIs — everything you need to deploy ML models with Docker.

---

## Why Docker for ML?

| Problem | Without Docker | With Docker |
|---|---|---|
| "Works on my machine" | Common | Eliminated |
| Dependency conflicts | Frequent | Each container is isolated |
| Reproducible environments | Hard | Built in |
| Deployment | Complex | `docker run` |
| Scaling | Manual | Orchestrated (Compose, Kubernetes) |

---

## Core Concepts

| Term | What It Is |
|---|---|
| **Image** | Read-only template: OS + runtime + code + dependencies |
| **Container** | Running instance of an image (like a lightweight VM) |
| **Dockerfile** | Script that defines how to build an image |
| **Registry** | Repository for images (Docker Hub, AWS ECR, GCR) |
| **Volume** | Persistent storage that survives container restarts |
| **Network** | Virtual network connecting containers |
| **docker-compose** | Tool for multi-container applications |

---

## Part 1: Your First ML Dockerfile

### Project Structure

```
ml-api/
├── app.py              ← FastAPI application
├── model.pkl           ← Trained model
├── requirements.txt    ← Python dependencies
├── Dockerfile          ← Container definition
├── .dockerignore       ← Files to exclude
└── docker-compose.yml  ← Multi-service orchestration
```

### Dockerfile — Standard ML API

```dockerfile
# ─── Base image ───────────────────────────────────────────────────────────────
# Use slim variant: smaller image size, no development tools
FROM python:3.10-slim

# ─── System dependencies ───────────────────────────────────────────────────────
# Install OS packages needed by Python libraries (e.g., libgomp for LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ─── Dependencies (layer cache optimization) ───────────────────────────────────
# Copy requirements FIRST — only re-runs pip install when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application code ──────────────────────────────────────────────────────────
COPY . .

# ─── Security: non-root user ───────────────────────────────────────────────────
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# ─── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### .dockerignore

```
__pycache__/
*.pyc
*.pyo
.env
.git/
.gitignore
.pytest_cache/
venv/
.venv/
*.ipynb
*.ipynb_checkpoints/
notebooks/
data/raw/
mlruns/
wandb/
*.log
README.md
```

---

## Part 2: Build & Run

```bash
# Build image (tag it with a name and version)
docker build -t ml-api:1.0 .
docker build -t ml-api:1.0 --no-cache .        # Ignore cache

# Run container
docker run -p 8000:8000 ml-api:1.0             # Map host:container port
docker run -d -p 8000:8000 ml-api:1.0           # Detached (background)
docker run -d -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \          # Pass env vars
  -v $(pwd)/models:/app/models \               # Mount volume
  --name my-api \                              # Name the container
  ml-api:1.0

# Container management
docker ps                                      # List running containers
docker ps -a                                   # All containers (incl. stopped)
docker logs my-api                             # View stdout/stderr
docker logs -f my-api                          # Follow logs (live)
docker exec -it my-api bash                    # Open shell in container
docker stop my-api                             # Graceful shutdown
docker rm my-api                               # Remove container
docker rmi ml-api:1.0                          # Remove image

# Inspect
docker inspect my-api                          # Full container info
docker stats my-api                            # Resource usage (live)
```

---

## Part 3: Multi-Stage Builds (Smaller Images)

Multi-stage builds let you use heavy build tools only in intermediate stages and produce a lean final image.

```dockerfile
# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.10 AS builder

WORKDIR /app
COPY requirements.txt .

# Install all deps including build tools
RUN pip install --user --no-cache-dir -r requirements.txt

# ─── Stage 2: Production ──────────────────────────────────────────────────────
FROM python:3.10-slim AS production

WORKDIR /app

# Copy only the installed packages from builder (no pip, no build tools)
COPY --from=builder /root/.local /root/.local

# Copy application
COPY app.py .
COPY model.pkl .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Compare image sizes
docker images | grep ml-api
# ml-api:slim    ~400MB  (multi-stage)
# ml-api:full    ~1.2GB  (single-stage)
```

---

## Part 4: docker-compose for ML Stacks

```yaml
# docker-compose.yml
version: '3.8'

services:

  # ─── ML API ───────────────────────────────────────────────────────────────
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://user:pass@db:5432/mldb
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped
    volumes:
      - ./models:/app/models:ro           # Read-only model mount
    networks:
      - ml-network

  # ─── MLflow Tracking Server ───────────────────────────────────────────────
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://user:pass@db:5432/mlflowdb
      --default-artifact-root s3://my-bucket/mlflow-artifacts
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - db
    networks:
      - ml-network

  # ─── PostgreSQL ───────────────────────────────────────────────────────────
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mldb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mldb"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ml-network

  # ─── Redis (prediction cache) ─────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - ml-network

  # ─── NGINX reverse proxy ──────────────────────────────────────────────────
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - api
    networks:
      - ml-network

volumes:
  postgres_data:
  redis_data:

networks:
  ml-network:
    driver: bridge
```

```bash
# docker-compose commands
docker-compose up -d              # Start all services (detached)
docker-compose up --build         # Rebuild images then start
docker-compose down               # Stop and remove containers
docker-compose down -v            # Also remove volumes (wipes data!)
docker-compose logs -f api        # Follow logs for a specific service
docker-compose exec api bash      # Shell into running service
docker-compose ps                 # Status of all services
docker-compose pull               # Pull latest images
```

---

## Part 5: GPU Support (for DL Models)

```dockerfile
# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Run with GPU access
docker run --gpus all -p 8000:8000 ml-gpu-api:1.0
docker run --gpus '"device=0"' -p 8000:8000 ml-gpu-api:1.0  # Specific GPU

# docker-compose with GPU
services:
  api:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

---

## Part 6: Pushing to a Registry

### Docker Hub

```bash
docker login
docker tag ml-api:1.0 yourusername/ml-api:1.0
docker push yourusername/ml-api:1.0

# Pull anywhere
docker pull yourusername/ml-api:1.0
```

### AWS ECR (Elastic Container Registry)

```bash
# Authenticate
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag ml-api:1.0 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-api:1.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-api:1.0
```

---

## Part 7: CI/CD with Docker + GitHub Actions

```yaml
# .github/workflows/docker-build-push.yml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            ${{ secrets.DOCKERHUB_USERNAME }}/ml-api:latest
            ${{ secrets.DOCKERHUB_USERNAME }}/ml-api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## Part 8: Common Issues & Fixes

| Problem | Cause | Fix |
|---|---|---|
| `No space left on device` | Too many unused images/containers | `docker system prune -af` |
| `Port already in use` | Another process using port | `lsof -i :8000` then kill, or use different port |
| `Permission denied` | Container runs as root, volume as user | Add `USER appuser` or fix volume permissions |
| `Module not found` | requirements.txt not up to date | Add to requirements.txt + rebuild |
| `File not found` | Wrong WORKDIR or COPY path | Check Dockerfile paths and .dockerignore |
| Slow builds | No layer caching | Copy requirements.txt before `COPY . .` |
| Large image size | Using full base image | Switch to `-slim` or use multi-stage build |
| Can't connect between containers | Using `localhost` | Use the service name as hostname in compose |

---

## Quick Reference

```bash
# Lifecycle
docker build -t name:tag .      # Build
docker run -p host:container name:tag  # Run
docker stop <id>                # Stop
docker rm <id>                  # Remove container
docker rmi name:tag             # Remove image

# Debugging
docker logs <id>
docker exec -it <id> bash
docker inspect <id>
docker stats

# Cleanup
docker system prune             # Remove stopped containers + dangling images
docker system prune -af --volumes  # Nuclear cleanup (removes everything unused)

# Images
docker images
docker pull name:tag
docker push name:tag

# Compose
docker-compose up -d
docker-compose down
docker-compose logs -f service
docker-compose exec service bash
```
