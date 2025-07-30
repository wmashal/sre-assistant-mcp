#!/usr/bin/env python3
"""
SRE Assistant MCP Server - Prometheus & OpenTelemetry Integration
A Model Context Protocol server for intelligent SRE operations with Prometheus and OTEL traces
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrometheusClient:
    """Async Prometheus client for querying metrics"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query(self, query: str, time: Optional[str] = None) -> Dict[str, Any]:
        """Execute instant query"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = urljoin(self.base_url, "/api/v1/query")
        params = {"query": query}
        if time:
            params["time"] = time
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    async def query_range(self, query: str, start: str, end: str, step: str = "30s") -> Dict[str, Any]:
        """Execute range query"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = urljoin(self.base_url, "/api/v1/query_range")
        params = {
            "query": query,
            "start": start,
            "end": end,
            "step": step
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Range query failed: {e}")
            raise
    
    async def get_targets(self) -> Dict[str, Any]:
        """Get scrape targets"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = urljoin(self.base_url, "/api/v1/targets")
        
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to get targets: {e}")
            raise
    
    async def get_rules(self) -> Dict[str, Any]:
        """Get Prometheus rules"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = urljoin(self.base_url, "/api/v1/rules")
        
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to get rules: {e}")
            raise

class JaegerClient:
    """Async Jaeger client for querying traces via OTEL collector"""
    
    def __init__(self, base_url: str = "http://localhost:16686"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_traces(self, service: str = None, operation: str = None, 
                           start_time: Optional[datetime] = None, 
                           end_time: Optional[datetime] = None,
                           limit: int = 20) -> Dict[str, Any]:
        """Search for traces"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Default to last hour if no time specified
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        url = urljoin(self.base_url, "/api/traces")
        params = {
            "start": int(start_time.timestamp() * 1000000),  # microseconds
            "end": int(end_time.timestamp() * 1000000),
            "limit": limit
        }
        
        if service:
            params["service"] = service
        if operation:
            params["operation"] = operation
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Trace search failed: {e}")
            raise
    
    async def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get specific trace by ID"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = urljoin(self.base_url, f"/api/traces/{trace_id}")
        
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get trace failed: {e}")
            raise
    
    async def get_services(self) -> Dict[str, Any]:
        """Get list of services"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = urljoin(self.base_url, "/api/services")
        
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get services failed: {e}")
            raise
    
    async def get_operations(self, service: str) -> Dict[str, Any]:
        """Get operations for a service"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = urljoin(self.base_url, "/api/services/{}/operations".format(service))
        
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get operations failed: {e}")
            raise

class SREAssistantMCPServer:
    """SRE Assistant MCP Server"""
    
    def __init__(self, prometheus_url: str = "http://localhost:9090", 
                 jaeger_url: str = "http://localhost:16686"):
        self.prometheus_url = prometheus_url
        self.jaeger_url = jaeger_url
        self.server = Server("sre-assistant")
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="query_metrics",
                    description="Execute PromQL query for instant metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "PromQL query"},
                            "time": {"type": "string", "description": "Optional evaluation time (RFC3339 or Unix timestamp)"}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="query_metrics_range",
                    description="Execute PromQL range query for time series data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "PromQL query"},
                            "duration": {"type": "string", "description": "Time range (e.g., '1h', '24h', '7d')", "default": "1h"},
                            "step": {"type": "string", "description": "Query resolution step", "default": "30s"}
                        },
                        "required": ["query"]
                    }
                ),
                types.Tool(
                    name="analyze_service_health",
                    description="Analyze overall health of services based on common SRE metrics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service_filter": {"type": "string", "description": "Service name filter (regex)", "default": ".*"},
                            "duration": {"type": "string", "description": "Analysis time range", "default": "1h"}
                        }
                    }
                ),
                types.Tool(
                    name="check_error_rates",
                    description="Check error rates across services and identify anomalies",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "threshold": {"type": "number", "description": "Error rate threshold (0-1)", "default": 0.05},
                            "duration": {"type": "string", "description": "Analysis time range", "default": "1h"}
                        }
                    }
                ),
                types.Tool(
                    name="analyze_latency",
                    description="Analyze service latency patterns and identify outliers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "percentile": {"type": "string", "description": "Latency percentile to analyze", "default": "95"},
                            "duration": {"type": "string", "description": "Analysis time range", "default": "1h"}
                        }
                    }
                ),
                types.Tool(
                    name="get_infrastructure_health",
                    description="Check infrastructure health (nodes, pods, resources)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_details": {"type": "boolean", "description": "Include detailed metrics", "default": True}
                        }
                    }
                ),
                types.Tool(
                    name="get_prometheus_targets",
                    description="Get status of Prometheus scrape targets",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="get_active_alerts",
                    description="Get currently firing alerts from Prometheus rules",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="troubleshoot_service",
                    description="Comprehensive troubleshooting analysis for a specific service",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service_name": {"type": "string", "description": "Service name to troubleshoot"},
                            "duration": {"type": "string", "description": "Analysis time range", "default": "1h"}
                        },
                        "required": ["service_name"]
                    }
                ),
                types.Tool(
                    name="search_traces",
                    description="Search for traces from OpenTelemetry/Jaeger",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "description": "Service name to filter traces"},
                            "operation": {"type": "string", "description": "Operation name to filter traces"},
                            "duration": {"type": "string", "description": "Time range for search", "default": "1h"},
                            "limit": {"type": "number", "description": "Maximum number of traces to return", "default": 20}
                        }
                    }
                ),
                types.Tool(
                    name="analyze_trace_errors",
                    description="Find and analyze traces with errors or high latency",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "description": "Service name to analyze"},
                            "duration": {"type": "string", "description": "Time range for analysis", "default": "1h"},
                            "min_duration_ms": {"type": "number", "description": "Minimum trace duration in ms to consider slow", "default": 1000}
                        }
                    }
                ),
                types.Tool(
                    name="get_trace_details",
                    description="Get detailed information about a specific trace",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "trace_id": {"type": "string", "description": "Trace ID to analyze"}
                        },
                        "required": ["trace_id"]
                    }
                ),
                types.Tool(
                    name="analyze_service_dependencies",
                    description="Analyze service dependencies and call patterns from traces",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "description": "Service name to analyze dependencies for"},
                            "duration": {"type": "string", "description": "Time range for analysis", "default": "1h"}
                        },
                        "required": ["service"]
                    }
                ),
                types.Tool(
                    name="get_trace_services",
                    description="Get list of all services reporting traces",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                types.Tool(
                    name="correlate_metrics_traces",
                    description="Correlate Prometheus metrics with trace data for comprehensive analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service": {"type": "string", "description": "Service name to correlate"},
                            "duration": {"type": "string", "description": "Time range for correlation", "default": "1h"}
                        },
                        "required": ["service"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
            """Handle tool calls"""
            
            try:
                async with PrometheusClient(self.prometheus_url) as prom:
                    
                    if name == "query_metrics":
                        result = await prom.query(
                            arguments["query"],
                            arguments.get("time")
                        )
                        formatted_result = self._format_query_result(result)
                        return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "query_metrics_range":
                        duration = arguments.get("duration", "1h")
                        step = arguments.get("step", "30s")
                        end_time = datetime.now()
                        start_time = end_time - self._parse_duration(duration)
                        
                        result = await prom.query_range(
                            arguments["query"],
                            start_time.isoformat(),
                            end_time.isoformat(),
                            step
                        )
                        formatted_result = self._format_range_result(result)
                        return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "analyze_service_health":
                        service_filter = arguments.get("service_filter", ".*")
                        duration = arguments.get("duration", "1h")
                        analysis = await self._analyze_service_health(prom, service_filter, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "check_error_rates":
                        threshold = arguments.get("threshold", 0.05)
                        duration = arguments.get("duration", "1h")
                        analysis = await self._check_error_rates(prom, threshold, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "analyze_latency":
                        percentile = arguments.get("percentile", "95")
                        duration = arguments.get("duration", "1h")
                        analysis = await self._analyze_latency(prom, percentile, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "get_infrastructure_health":
                        include_details = arguments.get("include_details", True)
                        analysis = await self._get_infrastructure_health(prom, include_details)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "get_prometheus_targets":
                        result = await prom.get_targets()
                        formatted_result = self._format_targets(result)
                        return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "get_active_alerts":
                        result = await prom.get_rules()
                        formatted_result = self._format_alerts(result)
                        return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "troubleshoot_service":
                        service_name = arguments["service_name"]
                        duration = arguments.get("duration", "1h")
                        analysis = await self._troubleshoot_service(prom, service_name, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    # Trace analysis tools
                    elif name == "search_traces":
                        async with JaegerClient(self.jaeger_url) as jaeger:
                            service = arguments.get("service")
                            operation = arguments.get("operation")
                            duration = arguments.get("duration", "1h")
                            limit = arguments.get("limit", 20)
                            
                            end_time = datetime.now()
                            start_time = end_time - self._parse_duration(duration)
                            
                            result = await jaeger.search_traces(
                                service=service, 
                                operation=operation,
                                start_time=start_time,
                                end_time=end_time,
                                limit=limit
                            )
                            formatted_result = self._format_trace_search_results(result)
                            return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "analyze_trace_errors":
                        async with JaegerClient(self.jaeger_url) as jaeger:
                            service = arguments.get("service")
                            duration = arguments.get("duration", "1h")
                            min_duration_ms = arguments.get("min_duration_ms", 1000)
                            analysis = await self._analyze_trace_errors(jaeger, service, duration, min_duration_ms)
                            return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "get_trace_details":
                        async with JaegerClient(self.jaeger_url) as jaeger:
                            trace_id = arguments["trace_id"]
                            result = await jaeger.get_trace(trace_id)
                            formatted_result = self._format_trace_details(result)
                            return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "analyze_service_dependencies":
                        async with JaegerClient(self.jaeger_url) as jaeger:
                            service = arguments["service"]
                            duration = arguments.get("duration", "1h")
                            analysis = await self._analyze_service_dependencies(jaeger, service, duration)
                            return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "get_trace_services":
                        async with JaegerClient(self.jaeger_url) as jaeger:
                            result = await jaeger.get_services()
                            formatted_result = self._format_services_list(result)
                            return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "correlate_metrics_traces":
                        service = arguments["service"]
                        duration = arguments.get("duration", "1h")
                        analysis = await self._correlate_metrics_traces(prom, service, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    else:
                        error_msg = f"Unknown tool: {name}"
                        return [types.TextContent(type="text", text=error_msg)]
            
            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                error_msg = f"Error: {str(e)}"
                return [types.TextContent(type="text", text=error_msg)]
    
    def _parse_duration(self, duration: str) -> timedelta:
        """Parse duration string like '1h', '30m', '7d'"""
        unit = duration[-1]
        value = int(duration[:-1])
        
        if unit == 's':
            return timedelta(seconds=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
        else:
            raise ValueError(f"Unknown duration unit: {unit}")
    
    def _format_query_result(self, result: Dict[str, Any]) -> str:
        """Format Prometheus query result"""
        if result["status"] != "success":
            return f"Query failed: {result.get('error', 'Unknown error')}"
        
        data = result["data"]
        result_type = data["resultType"]
        
        if result_type == "vector":
            if not data["result"]:
                return "No data found"
            
            output = f"Query Results ({len(data['result'])} series):\n\n"
            for series in data["result"]:
                metric = series["metric"]
                value = series["value"][1]
                labels = ", ".join(f"{k}={v}" for k, v in metric.items())
                output += f"{labels}: {value}\n"
            return output
        
        elif result_type == "scalar":
            return f"Scalar result: {data['result'][1]}"
        
        else:
            return json.dumps(result, indent=2)
    
    def _format_range_result(self, result: Dict[str, Any]) -> str:
        """Format Prometheus range query result"""
        if result["status"] != "success":
            return f"Query failed: {result.get('error', 'Unknown error')}"
        
        data = result["data"]
        if not data["result"]:
            return "No data found"
        
        output = f"Range Query Results ({len(data['result'])} series):\n\n"
        for series in data["result"]:
            metric = series["metric"]
            values = series["values"]
            labels = ", ".join(f"{k}={v}" for k, v in metric.items())
            
            output += f"{labels}:\n"
            output += f"  Points: {len(values)}\n"
            if values:
                first_val = values[0][1]
                last_val = values[-1][1]
                output += f"  First: {first_val}, Last: {last_val}\n"
            output += "\n"
        
        return output
    
    def _format_targets(self, result: Dict[str, Any]) -> str:
        """Format Prometheus targets result"""
        if result["status"] != "success":
            return f"Failed to get targets: {result.get('error', 'Unknown error')}"
        
        active_targets = result["data"]["activeTargets"]
        dropped_targets = result["data"]["droppedTargets"]
        
        output = f"Prometheus Targets Status:\n\n"
        output += f"Active Targets: {len(active_targets)}\n"
        output += f"Dropped Targets: {len(dropped_targets)}\n\n"
        
        # Group by job
        jobs = {}
        for target in active_targets:
            job = target["labels"]["job"]
            if job not in jobs:
                jobs[job] = {"up": 0, "down": 0, "total": 0}
            
            jobs[job]["total"] += 1
            if target["health"] == "up":
                jobs[job]["up"] += 1
            else:
                jobs[job]["down"] += 1
        
        output += "Jobs Summary:\n"
        for job, stats in jobs.items():
            health_pct = (stats["up"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            output += f"  {job}: {stats['up']}/{stats['total']} up ({health_pct:.1f}%)\n"
        
        return output
    
    def _format_alerts(self, result: Dict[str, Any]) -> str:
        """Format active alerts from rules"""
        if result["status"] != "success":
            return f"Failed to get rules: {result.get('error', 'Unknown error')}"
        
        firing_alerts = []
        for group in result["data"]["groups"]:
            for rule in group["rules"]:
                if rule["type"] == "alerting" and rule["state"] == "firing":
                    for alert in rule.get("alerts", []):
                        firing_alerts.append({
                            "name": rule["name"],
                            "labels": alert["labels"],
                            "annotations": alert.get("annotations", {}),
                            "state": alert["state"],
                            "activeAt": alert["activeAt"]
                        })
        
        if not firing_alerts:
            return "No active alerts"
        
        output = f"Active Alerts ({len(firing_alerts)}):\n\n"
        for alert in firing_alerts:
            output += f"🚨 {alert['name']}\n"
            output += f"   State: {alert['state']}\n"
            output += f"   Active since: {alert['activeAt']}\n"
            
            if alert['labels']:
                labels = ", ".join(f"{k}={v}" for k, v in alert['labels'].items())
                output += f"   Labels: {labels}\n"
            
            if alert['annotations']:
                for k, v in alert['annotations'].items():
                    output += f"   {k}: {v}\n"
            output += "\n"
        
        return output
    
    async def _analyze_service_health(self, prom: PrometheusClient, service_filter: str, duration: str) -> str:
        """Analyze overall service health"""
        queries = [
            f'up{{job=~"{service_filter}"}}',
            f'rate(http_requests_total{{job=~"{service_filter}"}}[5m])',
            f'rate(http_requests_total{{status=~"5..", job=~"{service_filter}"}}[5m])',
        ]
        
        results = []
        for query in queries:
            try:
                result = await prom.query(query)
                results.append(result)
            except Exception as e:
                results.append({"status": "error", "error": str(e)})
        
        output = f"Service Health Analysis (filter: {service_filter}):\n\n"
        
        # Analyze up/down status
        if results[0]["status"] == "success":
            up_data = results[0]["data"]["result"]
            services = {}
            for series in up_data:
                job = series["metric"].get("job", "unknown")
                instance = series["metric"].get("instance", "unknown")
                is_up = float(series["value"][1]) == 1.0
                
                if job not in services:
                    services[job] = {"up": 0, "down": 0, "total": 0}
                
                services[job]["total"] += 1
                if is_up:
                    services[job]["up"] += 1
                else:
                    services[job]["down"] += 1
            
            output += "Service Availability:\n"
            for job, stats in services.items():
                health_pct = (stats["up"] / stats["total"]) * 100 if stats["total"] > 0 else 0
                status = "✅" if health_pct == 100 else "⚠️" if health_pct > 50 else "❌"
                output += f"  {status} {job}: {stats['up']}/{stats['total']} instances up ({health_pct:.1f}%)\n"
            output += "\n"
        
        return output
    
    async def _check_error_rates(self, prom: PrometheusClient, threshold: float, duration: str) -> str:
        """Check error rates across services"""
        query = f'''
        (
          rate(http_requests_total{{status=~"5.."}}[5m]) /
          rate(http_requests_total[5m])
        ) > {threshold}
        '''
        
        try:
            result = await prom.query(query)
            
            if result["status"] != "success":
                return f"Error rate check failed: {result.get('error', 'Unknown error')}"
            
            high_errors = result["data"]["result"]
            
            if not high_errors:
                return f"✅ No services found with error rate above {threshold*100:.1f}%"
            
            output = f"🚨 Services with high error rates (>{threshold*100:.1f}%):\n\n"
            
            for series in high_errors:
                labels = series["metric"]
                error_rate = float(series["value"][1])
                service = labels.get("job", "unknown")
                method = labels.get("method", "")
                path = labels.get("path", "")
                
                output += f"⚠️ {service}"
                if method or path:
                    output += f" ({method} {path})"
                output += f": {error_rate*100:.2f}% error rate\n"
            
            return output
            
        except Exception as e:
            return f"Error checking error rates: {str(e)}"
    
    async def _analyze_latency(self, prom: PrometheusClient, percentile: str, duration: str) -> str:
        """Analyze service latency patterns"""
        query = f'histogram_quantile(0.{percentile}, rate(http_request_duration_seconds_bucket[5m]))'
        
        try:
            result = await prom.query(query)
            
            if result["status"] != "success":
                return f"Latency analysis failed: {result.get('error', 'Unknown error')}"
            
            latencies = result["data"]["result"]
            
            if not latencies:
                return "No latency data found"
            
            output = f"Latency Analysis (P{percentile}):\n\n"
            
            # Sort by latency value
            sorted_latencies = sorted(latencies, key=lambda x: float(x["value"][1]), reverse=True)
            
            for series in sorted_latencies:
                labels = series["metric"]
                latency_seconds = float(series["value"][1])
                latency_ms = latency_seconds * 1000
                
                service = labels.get("job", "unknown")
                method = labels.get("method", "")
                path = labels.get("path", "")
                
                status = "🐌" if latency_ms > 1000 else "⚠️" if latency_ms > 500 else "✅"
                
                output += f"{status} {service}"
                if method or path:
                    output += f" ({method} {path})"
                output += f": {latency_ms:.1f}ms\n"
            
            return output
            
        except Exception as e:
            return f"Error analyzing latency: {str(e)}"
    
    async def _get_infrastructure_health(self, prom: PrometheusClient, include_details: bool) -> str:
        """Get infrastructure health overview"""
        queries = {
            "node_up": "up{job='node-exporter'}",
            "node_cpu": "100 - (avg(rate(node_cpu_seconds_total{mode='idle'}[5m])) * 100)",
            "node_memory": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "node_disk": "(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100",
            "pod_count": "kube_pod_status_phase",
        }
        
        output = "Infrastructure Health Overview:\n\n"
        
        for metric_name, query in queries.items():
            try:
                result = await prom.query(query)
                if result["status"] == "success" and result["data"]["result"]:
                    output += f"{metric_name.replace('_', ' ').title()}:\n"
                    
                    if metric_name == "node_up":
                        up_count = sum(1 for s in result["data"]["result"] if float(s["value"][1]) == 1.0)
                        total_count = len(result["data"]["result"])
                        output += f"  Nodes up: {up_count}/{total_count}\n"
                    
                    elif metric_name in ["node_cpu", "node_memory"]:
                        values = [float(s["value"][1]) for s in result["data"]["result"]]
                        if values:
                            avg_val = sum(values) / len(values)
                            max_val = max(values)
                            status = "🔴" if max_val > 90 else "🟡" if max_val > 75 else "🟢"
                            output += f"  {status} Average: {avg_val:.1f}%, Max: {max_val:.1f}%\n"
                    
                    elif metric_name == "pod_count":
                        phases = {}
                        for series in result["data"]["result"]:
                            phase = series["metric"].get("phase", "unknown")
                            count = int(float(series["value"][1]))
                            phases[phase] = phases.get(phase, 0) + count
                        
                        for phase, count in phases.items():
                            status = "🟢" if phase == "Running" else "🟡" if phase == "Pending" else "🔴"
                            output += f"  {status} {phase}: {count}\n"
                    
                    output += "\n"
                    
            except Exception as e:
                output += f"  Error querying {metric_name}: {str(e)}\n\n"
        
        return output
    
    async def _troubleshoot_service(self, prom: PrometheusClient, service_name: str, duration: str) -> str:
        """Comprehensive service troubleshooting"""
        output = f"🔍 Troubleshooting Service: {service_name}\n\n"
        
        # Service availability
        try:
            up_query = f'up{{job="{service_name}"}}'
            result = await prom.query(up_query)
            
            if result["status"] == "success":
                instances = result["data"]["result"]
                up_count = sum(1 for s in instances if float(s["value"][1]) == 1.0)
                total_count = len(instances)
                
                status = "✅" if up_count == total_count else "❌"
                output += f"{status} Availability: {up_count}/{total_count} instances up\n"
                
                if up_count < total_count:
                    output += "   Down instances:\n"
                    for instance in instances:
                        if float(instance["value"][1]) == 0.0:
                            inst_name = instance["metric"].get("instance", "unknown")
                            output += f"   - {inst_name}\n"
            output += "\n"
        except Exception as e:
            output += f"❌ Failed to check availability: {str(e)}\n\n"
        
        # Error rates
        try:
            error_query = f'''
            rate(http_requests_total{{status=~"5..", job="{service_name}"}}[5m]) /
            rate(http_requests_total{{job="{service_name}"}}[5m])
            '''
            result = await prom.query(error_query)
            
            if result["status"] == "success" and result["data"]["result"]:
                for series in result["data"]["result"]:
                    error_rate = float(series["value"][1])
                    status = "❌" if error_rate > 0.05 else "⚠️" if error_rate > 0.01 else "✅"
                    output += f"{status} Error Rate: {error_rate*100:.2f}%\n"
            else:
                output += "ℹ️ No error rate data available\n"
        except Exception as e:
            output += f"⚠️ Failed to check error rates: {str(e)}\n"
        
        # Request rate
        try:
            rate_query = f'rate(http_requests_total{{job="{service_name}"}}[5m])'
            result = await prom.query(rate_query)
            
            if result["status"] == "success" and result["data"]["result"]:
                total_rate = sum(float(s["value"][1]) for s in result["data"]["result"])
                output += f"📊 Request Rate: {total_rate:.2f} req/sec\n"
        except Exception as e:
            output += f"⚠️ Failed to check request rate: {str(e)}\n"
        
        # Latency
        try:
            latency_query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job="{service_name}"}}[5m]))'
            result = await prom.query(latency_query)
            
            if result["status"] == "success" and result["data"]["result"]:
                for series in result["data"]["result"]:
                    latency_ms = float(series["value"][1]) * 1000
                    status = "❌" if latency_ms > 1000 else "⚠️" if latency_ms > 500 else "✅"
                    output += f"{status} P95 Latency: {latency_ms:.1f}ms\n"
        except Exception as e:
            output += f"⚠️ Failed to check latency: {str(e)}\n"
        
        return output
    
    # Trace analysis methods
    def _format_trace_search_results(self, result: Dict[str, Any]) -> str:
        """Format trace search results"""
        if "data" not in result or not result["data"]:
            return "No traces found"
        
        traces = result["data"]
        output = f"Found {len(traces)} traces:\n\n"
        
        for trace in traces:
            trace_id = trace.get("traceID", "unknown")
            spans = trace.get("spans", [])
            if not spans:
                continue
                
            root_span = spans[0]  # First span is typically the root
            service_name = root_span.get("process", {}).get("serviceName", "unknown")
            operation_name = root_span.get("operationName", "unknown")
            duration_us = trace.get("duration", 0)
            duration_ms = duration_us / 1000 if duration_us else 0
            
            # Check for errors
            has_error = any(
                span.get("tags", []) and 
                any(tag.get("key") == "error" and tag.get("value") for tag in span.get("tags", []))
                for span in spans
            )
            
            status = "❌" if has_error else "🐌" if duration_ms > 1000 else "✅"
            
            output += f"{status} {trace_id[:16]}... - {service_name}.{operation_name}\n"
            output += f"   Duration: {duration_ms:.1f}ms, Spans: {len(spans)}\n"
            if has_error:
                output += "   ⚠️ Contains errors\n"
            output += "\n"
        
        return output
    
    def _format_trace_details(self, result: Dict[str, Any]) -> str:
        """Format detailed trace information"""
        if "data" not in result or not result["data"]:
            return "Trace not found"
        
        traces = result["data"]
        if not traces:
            return "Trace not found"
        
        trace = traces[0]  # Should be single trace
        spans = trace.get("spans", [])
        
        if not spans:
            return "No spans found in trace"
        
        output = f"Trace Details: {trace.get('traceID', 'unknown')}\n\n"
        
        # Sort spans by start time
        sorted_spans = sorted(spans, key=lambda s: s.get("startTime", 0))
        
        for span in sorted_spans:
            service = span.get("process", {}).get("serviceName", "unknown")
            operation = span.get("operationName", "unknown")
            duration_us = span.get("duration", 0)
            duration_ms = duration_us / 1000 if duration_us else 0
            
            # Check for errors
            tags = span.get("tags", [])
            error_tags = [tag for tag in tags if tag.get("key") == "error" and tag.get("value")]
            has_error = bool(error_tags)
            
            status = "❌" if has_error else "🐌" if duration_ms > 500 else "✅"
            
            output += f"{status} {service}.{operation}\n"
            output += f"   Duration: {duration_ms:.1f}ms\n"
            
            if has_error:
                output += "   Errors: " + ", ".join(str(tag.get("value")) for tag in error_tags) + "\n"
            
            # Show important tags
            important_tags = ["http.method", "http.url", "http.status_code", "db.statement"]
            for tag in tags:
                if tag.get("key") in important_tags:
                    output += f"   {tag['key']}: {tag.get('value')}\n"
            
            output += "\n"
        
        return output
    
    def _format_services_list(self, result: Dict[str, Any]) -> str:
        """Format services list"""
        if "data" not in result:
            return "No services found"
        
        services = result["data"]
        if not services:
            return "No services found"
        
        output = f"Services reporting traces ({len(services)}):\n\n"
        for service in sorted(services):
            output += f"• {service}\n"
        
        return output
    
    async def _analyze_trace_errors(self, jaeger: JaegerClient, service: str, duration: str, min_duration_ms: int) -> str:
        """Analyze traces for errors and performance issues"""
        end_time = datetime.now()
        start_time = end_time - self._parse_duration(duration)
        
        # Search for traces
        result = await jaeger.search_traces(
            service=service,
            start_time=start_time,
            end_time=end_time,
            limit=100
        )
        
        if "data" not in result or not result["data"]:
            return f"No traces found for service '{service or 'all'}'"
        
        traces = result["data"]
        output = f"Trace Analysis for {service or 'all services'} (last {duration}):\n\n"
        
        error_traces = []
        slow_traces = []
        total_traces = len(traces)
        
        for trace in traces:
            trace_id = trace.get("traceID", "unknown")
            spans = trace.get("spans", [])
            duration_us = trace.get("duration", 0)
            duration_ms = duration_us / 1000 if duration_us else 0
            
            # Check for errors
            has_error = any(
                span.get("tags", []) and 
                any(tag.get("key") == "error" and tag.get("value") for tag in span.get("tags", []))
                for span in spans
            )
            
            if has_error:
                error_traces.append((trace_id, duration_ms, spans))
            elif duration_ms > min_duration_ms:
                slow_traces.append((trace_id, duration_ms, spans))
        
        # Summary
        error_rate = len(error_traces) / total_traces * 100 if total_traces > 0 else 0
        slow_rate = len(slow_traces) / total_traces * 100 if total_traces > 0 else 0
        
        output += f"📊 Summary:\n"
        output += f"  Total traces: {total_traces}\n"
        output += f"  Error traces: {len(error_traces)} ({error_rate:.1f}%)\n"
        output += f"  Slow traces (>{min_duration_ms}ms): {len(slow_traces)} ({slow_rate:.1f}%)\n\n"
        
        # Error details
        if error_traces:
            output += f"❌ Error Traces:\n"
            for trace_id, duration_ms, spans in error_traces[:5]:  # Show top 5
                root_span = spans[0] if spans else {}
                service_name = root_span.get("process", {}).get("serviceName", "unknown")
                operation = root_span.get("operationName", "unknown")
                output += f"  {trace_id[:16]}... - {service_name}.{operation} ({duration_ms:.1f}ms)\n"
            output += "\n"
        
        # Slow traces
        if slow_traces:
            output += f"🐌 Slowest Traces:\n"
            sorted_slow = sorted(slow_traces, key=lambda x: x[1], reverse=True)
            for trace_id, duration_ms, spans in sorted_slow[:5]:  # Show top 5
                root_span = spans[0] if spans else {}
                service_name = root_span.get("process", {}).get("serviceName", "unknown")
                operation = root_span.get("operationName", "unknown")
                output += f"  {trace_id[:16]}... - {service_name}.{operation} ({duration_ms:.1f}ms)\n"
        
        return output
    
    async def _analyze_service_dependencies(self, jaeger: JaegerClient, service: str, duration: str) -> str:
        """Analyze service dependencies from traces"""
        end_time = datetime.now()
        start_time = end_time - self._parse_duration(duration)
        
        result = await jaeger.search_traces(
            service=service,
            start_time=start_time,
            end_time=end_time,
            limit=50
        )
        
        if "data" not in result or not result["data"]:
            return f"No traces found for service '{service}'"
        
        traces = result["data"]
        dependencies = {}
        operations = {}
        
        for trace in traces:
            spans = trace.get("spans", [])
            
            for span in spans:
                span_service = span.get("process", {}).get("serviceName", "unknown")
                operation = span.get("operationName", "unknown")
                
                # Count operations per service
                if span_service not in operations:
                    operations[span_service] = set()
                operations[span_service].add(operation)
                
                # Find parent-child relationships
                parent_id = span.get("references", [])
                if parent_id:
                    # This span has a parent, look for it
                    for other_span in spans:
                        if other_span.get("spanID") in [ref.get("spanID") for ref in parent_id]:
                            parent_service = other_span.get("process", {}).get("serviceName", "unknown")
                            if parent_service != span_service:
                                dep_key = f"{parent_service} → {span_service}"
                                dependencies[dep_key] = dependencies.get(dep_key, 0) + 1
        
        output = f"Service Dependencies for '{service}' (last {duration}):\n\n"
        
        # Service operations
        output += f"📋 Service Operations:\n"
        for svc, ops in operations.items():
            output += f"  {svc}: {len(ops)} operations\n"
            for op in sorted(ops):
                output += f"    • {op}\n"
            output += "\n"
        
        # Dependencies
        if dependencies:
            output += f"🔗 Call Patterns:\n"
            sorted_deps = sorted(dependencies.items(), key=lambda x: x[1], reverse=True)
            for dep, count in sorted_deps:
                output += f"  {dep} ({count} calls)\n"
        else:
            output += "🔗 No cross-service dependencies detected\n"
        
        return output
    
    async def _correlate_metrics_traces(self, prom: PrometheusClient, service: str, duration: str) -> str:
        """Correlate Prometheus metrics with trace data"""
        output = f"📊 Metrics & Traces Correlation for '{service}' (last {duration}):\n\n"
        
        # Get metrics
        try:
            # Error rate from metrics
            error_query = f'rate(http_requests_total{{status=~"5..", job="{service}"}}[5m]) / rate(http_requests_total{{job="{service}"}}[5m])'
            error_result = await prom.query(error_query)
            
            # Latency from metrics
            latency_query = f'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{{job="{service}"}}[5m]))'
            latency_result = await prom.query(latency_query)
            
            # Request rate
            rate_query = f'rate(http_requests_total{{job="{service}"}}[5m])'
            rate_result = await prom.query(rate_query)
            
            output += "📈 Prometheus Metrics:\n"
            
            if error_result["status"] == "success" and error_result["data"]["result"]:
                error_rate = float(error_result["data"]["result"][0]["value"][1]) * 100
                output += f"  Error Rate: {error_rate:.2f}%\n"
            
            if latency_result["status"] == "success" and latency_result["data"]["result"]:
                latency_ms = float(latency_result["data"]["result"][0]["value"][1]) * 1000
                output += f"  P95 Latency: {latency_ms:.1f}ms\n"
            
            if rate_result["status"] == "success" and rate_result["data"]["result"]:
                req_rate = sum(float(s["value"][1]) for s in rate_result["data"]["result"])
                output += f"  Request Rate: {req_rate:.2f} req/sec\n"
            
        except Exception as e:
            output += f"  ⚠️ Failed to get metrics: {str(e)}\n"
        
        output += "\n"
        
        # Get trace data
        try:
            async with JaegerClient(self.jaeger_url) as jaeger:
                end_time = datetime.now()
                start_time = end_time - self._parse_duration(duration)
                
                trace_result = await jaeger.search_traces(
                    service=service,
                    start_time=start_time,
                    end_time=end_time,
                    limit=50
                )
                
                if "data" in trace_result and trace_result["data"]:
                    traces = trace_result["data"]
                    
                    # Calculate trace statistics
                    durations = []
                    error_count = 0
                    
                    for trace in traces:
                        duration_us = trace.get("duration", 0)
                        duration_ms = duration_us / 1000 if duration_us else 0
                        durations.append(duration_ms)
                        
                        # Check for errors
                        spans = trace.get("spans", [])
                        has_error = any(
                            span.get("tags", []) and 
                            any(tag.get("key") == "error" and tag.get("value") for tag in span.get("tags", []))
                            for span in spans
                        )
                        if has_error:
                            error_count += 1
                    
                    output += "🔍 Trace Analysis:\n"
                    output += f"  Total traces analyzed: {len(traces)}\n"
                    if durations:
                        avg_duration = sum(durations) / len(durations)
                        max_duration = max(durations)
                        output += f"  Average duration: {avg_duration:.1f}ms\n"
                        output += f"  Max duration: {max_duration:.1f}ms\n"
                    
                    trace_error_rate = error_count / len(traces) * 100 if traces else 0
                    output += f"  Error rate from traces: {trace_error_rate:.1f}%\n"
                    
                    output += "\n💡 Correlation Insights:\n"
                    # Add correlation insights here based on comparing metrics vs traces
                    output += "  • Compare metrics vs trace data for discrepancies\n"
                    output += "  • Check if high latency correlates with error spikes\n"
                    output += "  • Validate monitoring coverage\n"
                
        except Exception as e:
            output += f"⚠️ Failed to get trace data: {str(e)}\n"
        
        return output

    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="sre-assistant",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

def main():
    """Main entry point"""
    prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    jaeger_url = os.getenv("JAEGER_URL", "http://localhost:16686")
    
    server = SREAssistantMCPServer(prometheus_url=prometheus_url, jaeger_url=jaeger_url)
    asyncio.run(server.run())

if __name__ == "__main__":
    main()