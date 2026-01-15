# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY main.go ./

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o retirement-api .

# Runtime stage
FROM alpine:3.19

WORKDIR /app

RUN addgroup -g 10001 appgroup && \
    adduser -u 10001 -G appgroup -D -H appuser

COPY --from=builder /app/retirement-api .

RUN chown -R appuser:appgroup /app

USER 10001

EXPOSE 8000

ENV PORT=8000

CMD ["./retirement-api"]
