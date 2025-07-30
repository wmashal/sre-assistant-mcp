# SRE Assistant - Cloud Foundry & Dynatrace Integration

This document describes how to set up and use the SRE Assistant MCP Server with Cloud Foundry applications monitored by Dynatrace.

## Overview

The `sre_assistant_cf.py` server provides specialized tools for monitoring Cloud Foundry applications using Dynatrace APM. It offers comprehensive observability for CF apps with intelligent analysis capabilities.

## Features

- **Service Discovery**: Automatically discover CF services from Dynatrace
- **Performance Analysis**: Monitor response times, throughput, and error rates
- **Problem Detection**: Get active alerts and problems from Dynatrace
- **Infrastructure Health**: Check CF hosts and process groups
- **Service Dependencies**: Analyze service relationships and call patterns
- **Custom Metrics**: Query any Dynatrace metric with flexible selectors

## Prerequisites

- Cloud Foundry application deployed and running
- Dynatrace OneAgent installed on CF (should be automatic if bound)
- Dynatrace API token with required permissions
- Python 3.8+
- Claude Desktop app

## Setup Instructions

### Step 1: Environment Configuration

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Update `.env` with your Dynatrace credentials:
   ```bash
   # Your Dynatrace environment API URL
   DYNATRACE_API_URL=https://staging.us10.apm.services.cloud.sap/e/YOUR-ENVIRONMENT-ID/api
   
   # Your Dynatrace API token
   DYNATRACE_API_TOKEN=dt0c01.YOUR-API-TOKEN-HERE
   ```

### Step 2: Install Dependencies

```bash
cd sre-assistant-mcp
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Test the Server

```bash
# Test the CF server directly
python sre_assistant_cf.py
```

### Step 4: Configure Claude Desktop

Add to your Claude Desktop MCP settings:

```json
{
  "mcpServers": {
    "sre-assistant-cf": {
      "command": "python",
      "args": ["/path/to/your/sre-assistant-mcp/sre_assistant_cf.py"]
    }
  }
}
```

## Available Tools

### Service Management
1. **get_cf_services** - List all CF services discovered by Dynatrace
2. **analyze_cf_service_health** - Comprehensive service health analysis
3. **troubleshoot_cf_service** - Deep-dive troubleshooting for specific services

### Performance Monitoring  
4. **analyze_cf_performance** - Application performance analysis
5. **get_cf_metrics** - Query custom Dynatrace metrics
6. **get_cf_problems** - Active problems and alerts

### Infrastructure
7. **get_cf_infrastructure** - CF infrastructure health (hosts, processes)
8. **get_cf_dependencies** - Service dependency mapping

## Example Queries

### 🔍 Service Discovery
- "Show me all Cloud Foundry services"
- "List services in Dynatrace"
- "What CF applications are being monitored?"

### 📊 Health Monitoring
- "Analyze the health of my 'checkout-service'"
- "Check for any active problems in CF"
- "Show me services with performance issues"

### 🎯 Performance Analysis
- "What's the response time for 'payment-api'?"
- "Analyze performance of 'user-service' over the last 2 hours"
- "Show me error rates across all CF services"

### 🔧 Troubleshooting
- "Troubleshoot the 'order-processor' service"
- "Why is 'notification-service' slow?"
- "Check infrastructure health for CF"

### 📈 Custom Metrics
- "Get metric 'builtin:service.response.time' for all services"
- "Query CPU usage for CF hosts"
- "Show me custom business metrics"

## Dynatrace API Token Permissions

Your API token needs these permissions:

- **Read entities** (`entities.read`)
- **Read metrics** (`metrics.read`)  
- **Read problems** (`problems.read`)
- **Access problem and event feed** (`events.read`)
- **Read configuration** (`settings.read`)

## Metric Selectors

Common Dynatrace metric selectors you can use:

```bash
# Service metrics
"builtin:service.response.time"
"builtin:service.errors.total.rate"
"builtin:service.requestCount.total"

# Host metrics  
"builtin:host.cpu.usage"
"builtin:host.mem.usage"
"builtin:host.disk.usedPct"

# Custom metrics
"calc:service.custom_metric"
```

## Entity Selectors

Filter entities with these selectors:

```bash
# Services
'type("SERVICE"),entityName.equals("my-service")'
'type("SERVICE"),managementZone("Production")'

# Hosts
'type("HOST"),tag("environment:prod")'

# Process Groups
'type("PROCESS_GROUP"),fromRelationships.runsOnHost(type("HOST"))'
```

## Troubleshooting

### Common Issues

1. **Authentication Error**: Verify your API token and URL
2. **No Services Found**: Check if Dynatrace OneAgent is properly installed
3. **Missing Metrics**: Some metrics may not be available for CF applications

### Debug Mode

Enable detailed logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Verify Dynatrace Connection

```bash
# Test API access
curl -H "Authorization: Api-Token YOUR-TOKEN" \
  "https://staging.us10.apm.services.cloud.sap/e/YOUR-ENV-ID/api/v2/entities?entitySelector=type(SERVICE)"
```

## Cloud Foundry Specific Notes

- **OneAgent**: Should be automatically bound to your CF apps
- **Management Zones**: Use to organize your CF environments (dev/staging/prod)
- **Tags**: Automatically applied based on CF org/space/app metadata
- **Service Detection**: Dynatrace automatically detects CF services and dependencies

## Advanced Usage

### Custom Analysis Scripts

You can extend the server with custom analysis:

```python
async def custom_cf_analysis(self, dt: DynatraceClient, app_name: str) -> str:
    # Your custom logic here
    pass
```

### Integration with CF CLI

Combine with CF CLI for enhanced workflows:

```bash
# Get app GUID and use in Dynatrace queries
cf app my-app --guid
```

## Best Practices

1. **Use Management Zones** to separate environments
2. **Tag Resources** consistently for better filtering
3. **Set up Alerts** in Dynatrace for proactive monitoring
4. **Regular Health Checks** using the troubleshooting tools
5. **Monitor Dependencies** to understand service relationships

## Support

For issues specific to:
- **Dynatrace**: Check Dynatrace documentation and support
- **Cloud Foundry**: Refer to CF documentation
- **MCP Server**: Check logs and verify configuration