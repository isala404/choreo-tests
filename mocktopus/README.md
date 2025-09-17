# Mocktopus API

A simple FastAPI mock API with various endpoints for testing purposes.

## Features

- **GET /** - Hello from HOSTNAME
- **GET /healthz** - Health check endpoint
- **GET /echo** - Echo query parameters
- **POST /proxy** - Proxy requests to other URLs with support for different HTTP methods
- **GET /debug** - Returns extensive system and request information
- **GET /failure** - Returns a 500 error
- **GET /success** - Returns a 200 success
- **GET /random** - Returns random status codes (200, 400, or 500)

## Running the Application

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py
```

### Using Docker

```bash
# Build the Docker image
docker build -t mocktopus .

# Run the container
docker run -p 8000:8000 mocktopus
```

### Using Docker Compose

```bash
# Start the application
docker-compose up --build
```

## API Documentation

Once the application is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Endpoints

### GET /
Returns a hello message with the hostname.

**Response:**
```json
{
  "message": "Hello from HOSTNAME"
}
```

### GET /healthz
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /echo
Echoes back all query parameters.

**Example:** `GET /echo?any=arg&should=return`

**Response:**
```json
{
  "any": "arg",
  "should": "return"
}
```

### POST /proxy
Proxies requests to other URLs.

**Request Body:**
```json
{
  "method": "GET",
  "url": "https://httpbin.org/get",
  "body": {},
  "repeat": 1
}
```

**Response:**
```json
{
  "results": [
    {
      "attempt": 1,
      "status_code": 200,
      "headers": {...},
      "content": "..."
    }
  ]
}
```

### GET /debug
Returns extensive system and request information including:
- Platform information
- Python environment details
- Network information
- Process information
- Request headers and client details
- System resource information (CPU, memory, disk)

### GET /failure
Returns a 500 error for testing error handling.

### GET /success
Returns a 200 success response.

### GET /random
Returns random status codes (200, 400, or 500) for testing different scenarios.

## Development

The application is built with:
- **FastAPI** - Modern, fast web framework for building APIs
- **Uvicorn** - ASGI server for running FastAPI applications
- **httpx** - HTTP client for the proxy functionality
- **uv** - Fast Python package manager

## Docker

The application includes Docker support with:
- `Dockerfile` - Multi-stage build for production
- `docker-compose.yml` - For easy development setup
- `requirements.txt` - Generated from uv dependencies
