# SRE Assistant MCP Server

A Model Context Protocol (MCP) server that provides intelligent SRE operations and monitoring capabilities by integrating with Prometheus and OpenTelemetry traces. Ask questions about metrics, alerts, service health, traces, and troubleshoot issues directly from Claude Desktop.

## Features

- **Metrics Querying**: Execute PromQL queries for instant and time-series data
- **Service Health Analysis**: Analyze service availability, error rates, and latency
- **Infrastructure Monitoring**: Check node health, CPU, memory, and pod status
- **Alert Management**: View active alerts and Prometheus targets
- **Service Troubleshooting**: Comprehensive analysis for specific services
- **Trace Analysis**: Search and analyze OpenTelemetry traces via Jaeger
- **Error Detection**: Find traces with errors and performance issues
- **Service Dependencies**: Visualize service call patterns from traces
- **Metrics-Traces Correlation**: Cross-reference metrics and trace data

## Quick Start - Local Testing

### Prerequisites

- Python 3.8+
- Prometheus running locally
- OpenTelemetry Collector with Jaeger endpoint
- Claude Desktop app

### Step 1: Setup Environment

```bash
# Clone and setup
cd sre-assistant-mcp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Port Forward Services

```bash
# Port forward Prometheus
kubectl port-forward svc/prometheus 9090:9090

# Port forward OTEL Collector (Jaeger endpoint)
kubectl port-forward svc/opentelemetry-collector 16686:16686

# Alternative: If you have separate Jaeger UI
kubectl port-forward svc/jaeger-query 16686:16686
```

### Step 3: Test Connectivity

```bash
# Test Prometheus
curl http://localhost:9090/api/v1/query?query=up

# Test Jaeger
curl http://localhost:16686/api/services
```

### Step 4: Configure Claude Desktop

Add to your Claude Desktop MCP settings (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "sre-assistant": {
      "command": "python",
      "args": ["/path/to/your/sre-assistant-mcp/sre_assistant.py"],
      "env": {
        "PROMETHEUS_URL": "http://localhost:9090",
        "JAEGER_URL": "http://localhost:16686"
      }
    }
  }
}
```

### Step 5: Test in Claude Desktop

Restart Claude Desktop and try these example queries:

**📊 Metrics & Monitoring:**
- "What's the current status of my Prometheus targets?"
- "Show me any active alerts"
- "Analyze the health of my services"
- "Check error rates across all services"

**🔍 Trace Analysis:**
- "Show me all services that are reporting traces"
- "Search for traces from the 'frontend' service"
- "Find traces with errors in the last hour"
- "Analyze service dependencies for 'api-gateway'"

**🔧 Advanced Analysis:**
- "Correlate metrics and traces for the 'checkout' service"
- "Find slow traces over 2 seconds"
- "Troubleshoot service 'payment-service'"

## Docker Deployment

### Step 1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY sre_assistant.py .

CMD ["python", "sre_assistant.py"]
```

### Step 2: Build and Run Container

```bash
# Build the image
docker build -t sre-assistant-mcp .

# Run with custom URLs
docker run -d \
  --name sre-assistant \
  -e PROMETHEUS_URL=http://host.docker.internal:9090 \
  -e JAEGER_URL=http://host.docker.internal:16686 \
  sre-assistant-mcp
```

### Step 3: Update Claude Desktop Config

```json
{
  "mcpServers": {
    "sre-assistant": {
      "command": "docker",
      "args": ["exec", "sre-assistant", "python", "sre_assistant.py"]
    }
  }
}
```

## Environment Variables

- `PROMETHEUS_URL`: Prometheus server URL (default: `http://localhost:9090`)
- `JAEGER_URL`: Jaeger UI URL (default: `http://localhost:16686`)

## Available Tools

### Metrics Tools
1. **query_metrics** - Execute PromQL queries
2. **query_metrics_range** - Time-series data queries
3. **analyze_service_health** - Service availability analysis
4. **check_error_rates** - Error rate monitoring
5. **analyze_latency** - Latency pattern analysis
6. **get_infrastructure_health** - Infrastructure overview
7. **get_prometheus_targets** - Scrape target status
8. **get_active_alerts** - Current firing alerts
9. **troubleshoot_service** - Comprehensive service analysis

### Trace Tools
10. **search_traces** - Search OpenTelemetry traces
11. **analyze_trace_errors** - Find error traces and slow requests
12. **get_trace_details** - Detailed trace analysis
13. **analyze_service_dependencies** - Service call patterns
14. **get_trace_services** - List services with traces
15. **correlate_metrics_traces** - Cross-reference metrics and traces

## Example Queries

### 🔍 Basic Queries
- "Show me all services that are currently up"
- "What's the current CPU usage across my nodes?"
- "Get the memory utilization for my infrastructure"
- "Show me the status of all Prometheus scrape targets"

### 🚨 Health & Monitoring
- "Analyze the overall health of my services"
- "Check if any services have high error rates"
- "Show me services with latency issues"
- "Are there any active alerts firing right now?"
- "Get the infrastructure health overview"

### 🔧 Troubleshooting
- "Troubleshoot the [your-service-name] service"
- "Show me error rates for services in the last hour"
- "Analyze P95 latency across all my applications"
- "Which services are experiencing issues?"

### 📊 Custom PromQL Queries
- "Run this PromQL query: up{job='kubernetes-pods'}"
- "Query the request rate: rate(http_requests_total[5m])"
- "Get time series data for CPU usage over the last 2 hours"

### 🔍 Trace Analysis
- "Show me all services reporting traces"
- "Search for traces from the 'checkout' service"
- "Find traces with errors in the last 30 minutes"
- "Show me the slowest traces over 5 seconds"
- "Get details for trace ID abc123..."
- "What the duration for the trace 99bbdc0b0e6d6285f799717516e590f1"
- "what the longest span ? and what the tags and any errors in the process section ?"

### 🌐 Service Dependencies
- "Analyze service dependencies for 'user-service'"
- "Show me the call patterns between services"
- "Which services does 'order-api' depend on?"

### 🔗 Correlation Analysis
- "Correlate metrics and traces for 'payment-gateway'"
- "Compare error rates from metrics vs traces"
- "Show me discrepancies between monitoring data"

## OTEL Collector Setup

Your OpenTelemetry Collector should have Jaeger receiver/exporter configured:

```yaml
receivers:
  jaeger:
    protocols:
      grpc:
        endpoint: 0.0.0.0:14250
      thrift_http:
        endpoint: 0.0.0.0:14268

exporters:
  jaeger:
    endpoint: jaeger-collector:14250
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [jaeger]
      exporters: [jaeger]
```

## Troubleshooting

### Common Issues

1. **Connection refused**: Ensure services are accessible at configured URLs
2. **No trace data**: Check if your applications are sending traces to OTEL collector
3. **No metrics data**: Verify Prometheus has the expected metrics
4. **Claude Desktop not detecting**: Restart Claude Desktop after config changes

### Verify Connectivity

```bash
# Test Prometheus API
curl http://localhost:9090/api/v1/query?query=up

# Test Jaeger API
curl http://localhost:16686/api/services

# Test OTEL Collector metrics endpoint
curl http://localhost:8888/metrics
```

### Debug Mode

Set logging level for debugging:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Port Forward Commands

```bash
# Prometheus
kubectl port-forward svc/prometheus 9090:9090

# OTEL Collector endpoints
kubectl port-forward svc/opentelemetry-collector 16686:16686  # Jaeger UI access
kubectl port-forward svc/opentelemetry-collector 14250:14250  # Jaeger gRPC
kubectl port-forward svc/opentelemetry-collector 14268:14268  # Jaeger Thrift
kubectl port-forward svc/opentelemetry-collector 8888:8888   # Metrics
kubectl port-forward svc/opentelemetry-collector 4317:4317   # OTLP gRPC  
kubectl port-forward svc/opentelemetry-collector 4318:4318   # OTLP HTTP
kubectl port-forward svc/opentelemetry-collector 9411:9411   # Zipkin
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with your Prometheus and OTEL setup
5. Submit a pull request