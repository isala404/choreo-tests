# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY main.go ./

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o ffmpeg-server .

# Runtime stage
FROM alpine:3.19

# Install ffmpeg and required dependencies
RUN apk add --no-cache \
    ffmpeg \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Create non-root user with UID 10001
RUN addgroup -g 10001 appgroup && \
    adduser -u 10001 -G appgroup -D -H appuser

# Create temp directories for file processing
RUN mkdir -p /tmp/ffmpeg/input /tmp/ffmpeg/output && \
    chown -R appuser:appgroup /tmp/ffmpeg

# Copy binary from builder
COPY --from=builder /app/ffmpeg-server .

# Change ownership of app directory
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER 10001

# Expose port
EXPOSE 8000

# Set environment variable
ENV PORT=8000

# Run the server
CMD ["./ffmpeg-server"]
