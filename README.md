# Consensus Monorepo

A monorepo for YouTube live chat analysis, featuring microservices in different languages (NestJS and Python).

## Structure

- `chat-api/` - NestJS microservice (TypeScript)
- `python-microservice/` - Python FastAPI microservice
- `services/chat-api/` – NestJS microservice (TypeScript)
- `services/python-microservice/` – Python FastAPI microservice

## Running Microservices with Docker Compose

To build and run both microservices together:

```bash
docker-compose up --build
```

- The NestJS service will be available at http://localhost:3000
- The Python FastAPI service will be available at http://localhost:8000

## Building Services with Docker

To build the Docker images for each service individually:

### NestJS (services/chat-api)
```bash
docker build -t chat-api ./services/chat-api
```

### Python FastAPI (services/python-microservice)
```bash
docker build -t python-microservice ./services/python-microservice
```

You can then run each service with:

```bash
docker run -p 3000:3000 chat-api
```

```bash
docker run -p 8000:8000 python-microservice
```

## Individual Service Usage

### NestJS (services/chat-api)

```bash
cd services/chat-api
npm install
npm run start:dev
```

### Python FastAPI (services/python-microservice)

```bash
cd services/python-microservice
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Adding More Microservices

You can add more microservices in any language by creating a new directory and adding a Dockerfile for each.

---

This monorepo is managed with Nx for TypeScript/Node.js projects. Python and other language services are managed manually within the same repo for easy orchestration with Docker Compose. 