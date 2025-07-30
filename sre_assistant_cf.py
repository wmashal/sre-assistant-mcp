#!/usr/bin/env python3
"""
SRE Assistant MCP Server - Cloud Foundry & Dynatrace Integration
A Model Context Protocol server for intelligent SRE operations with Dynatrace APM
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, quote

import aiohttp
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
import mcp.server.stdio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynatraceClient:
    """Async Dynatrace client for querying metrics and traces"""
    
    def __init__(self, api_url: str, api_token: str):
        self.api_url = api_url.rstrip('/')
        self.api_token = api_token
        self.session: Optional[aiohttp.ClientSession] = None
        self.headers = {
            'Authorization': f'Api-Token {self.api_token}',
            'Content-Type': 'application/json'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_entities(self, entity_type: str, entity_selector: str = None) -> Dict[str, Any]:
        """Get entities (services, hosts, applications, etc.)"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.api_url}/v2/entities"
        params = {"entitySelector": entity_selector or f"type({entity_type})"}
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get entities failed: {e}")
            raise
    
    async def get_metrics(self, metric_selector: str, entity_selector: str = None, 
                         from_time: str = None, to_time: str = None) -> Dict[str, Any]:
        """Get metrics data"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.api_url}/v2/metrics/query"
        params = {
            "metricSelector": metric_selector,
            "resolution": "1m"
        }
        
        if entity_selector:
            params["entitySelector"] = entity_selector
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get metrics failed: {e}")
            raise
    
    async def get_problems(self, from_time: str = None, to_time: str = None) -> Dict[str, Any]:
        """Get problems/alerts"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.api_url}/v2/problems"
        params = {}
        
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get problems failed: {e}")
            raise
    
    async def get_service_metrics(self, service_id: str, metric_keys: List[str], 
                                 from_time: str = None, to_time: str = None) -> Dict[str, Any]:
        """Get specific service metrics"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.api_url}/v1/timeseries"
        params = {
            "timeseriesId": ",".join(metric_keys),
            "entityId": service_id,
            "aggregationType": "avg"
        }
        
        if from_time:
            params["startTimestamp"] = from_time
        if to_time:
            params["endTimestamp"] = to_time
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get service metrics failed: {e}")
            raise
    
    async def get_traces(self, service_id: str = None, from_time: str = None, 
                        to_time: str = None, limit: int = 20) -> Dict[str, Any]:
        """Get distributed traces"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.api_url}/v1/userSessionQueryLanguage/table"
        
        # USQL query for traces
        query = "SELECT traceId, duration, startTime, serviceName, operation FROM usersession"
        if service_id:
            query += f" WHERE serviceName = '{service_id}'"
        if from_time and to_time:
            query += f" AND startTime >= {from_time} AND startTime <= {to_time}"
        query += f" LIMIT {limit}"
        
        params = {"query": query}
        
        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Get traces failed: {e}")
            raise

class SREAssistantCFServer:
    """SRE Assistant MCP Server for Cloud Foundry with Dynatrace"""
    
    def __init__(self, dynatrace_url: str, dynatrace_token: str):
        self.dynatrace_url = dynatrace_url
        self.dynatrace_token = dynatrace_token
        self.server = Server("sre-assistant-cf")
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            """List available tools"""
            return [
                types.Tool(
                    name="get_cf_services",
                    description="Get Cloud Foundry services from Dynatrace",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service_type": {"type": "string", "description": "Service type filter", "default": "SERVICE"}
                        }
                    }
                ),
                types.Tool(
                    name="analyze_cf_service_health",
                    description="Analyze Cloud Foundry service health and performance",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service_name": {"type": "string", "description": "Service name to analyze"},
                            "duration": {"type": "string", "description": "Time range (e.g., '1h', '24h')", "default": "1h"}
                        }
                    }
                ),
                types.Tool(
                    name="get_cf_problems",
                    description="Get active problems and alerts from Dynatrace",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "duration": {"type": "string", "description": "Time range to check", "default": "1h"}
                        }
                    }
                ),
                types.Tool(
                    name="get_cf_metrics",
                    description="Get specific metrics from Dynatrace",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "metric_selector": {"type": "string", "description": "Dynatrace metric selector"},
                            "entity_selector": {"type": "string", "description": "Entity selector to filter"},
                            "duration": {"type": "string", "description": "Time range", "default": "1h"}
                        },
                        "required": ["metric_selector"]
                    }
                ),
                types.Tool(
                    name="analyze_cf_performance",
                    description="Analyze CF application performance (response time, throughput, errors)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "app_name": {"type": "string", "description": "CF application name"},
                            "duration": {"type": "string", "description": "Analysis time range", "default": "1h"}
                        }
                    }
                ),
                types.Tool(
                    name="get_cf_infrastructure",
                    description="Get Cloud Foundry infrastructure health (hosts, processes)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_details": {"type": "boolean", "description": "Include detailed metrics", "default": True}
                        }
                    }
                ),
                types.Tool(
                    name="troubleshoot_cf_service",
                    description="Comprehensive troubleshooting for CF service using Dynatrace data",
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
                    name="get_cf_dependencies",
                    description="Analyze service dependencies in Cloud Foundry from Dynatrace",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "service_name": {"type": "string", "description": "Service to analyze dependencies for"},
                            "depth": {"type": "number", "description": "Dependency depth", "default": 2}
                        },
                        "required": ["service_name"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[types.TextContent]:
            """Handle tool calls"""
            
            try:
                async with DynatraceClient(self.dynatrace_url, self.dynatrace_token) as dt:
                    
                    if name == "get_cf_services":
                        service_type = arguments.get("service_type", "SERVICE")
                        result = await dt.get_entities(service_type)
                        formatted_result = self._format_services(result)
                        return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "analyze_cf_service_health":
                        service_name = arguments.get("service_name")
                        duration = arguments.get("duration", "1h")
                        analysis = await self._analyze_service_health(dt, service_name, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "get_cf_problems":
                        duration = arguments.get("duration", "1h")
                        end_time = datetime.now()
                        start_time = end_time - self._parse_duration(duration)
                        
                        result = await dt.get_problems(
                            from_time=self._to_dynatrace_time(start_time),
                            to_time=self._to_dynatrace_time(end_time)
                        )
                        formatted_result = self._format_problems(result)
                        return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "get_cf_metrics":
                        metric_selector = arguments["metric_selector"]
                        entity_selector = arguments.get("entity_selector")
                        duration = arguments.get("duration", "1h")
                        
                        end_time = datetime.now()
                        start_time = end_time - self._parse_duration(duration)
                        
                        result = await dt.get_metrics(
                            metric_selector=metric_selector,
                            entity_selector=entity_selector,
                            from_time=self._to_dynatrace_time(start_time),
                            to_time=self._to_dynatrace_time(end_time)
                        )
                        formatted_result = self._format_metrics(result)
                        return [types.TextContent(type="text", text=formatted_result)]
                    
                    elif name == "analyze_cf_performance":
                        app_name = arguments.get("app_name")
                        duration = arguments.get("duration", "1h")
                        analysis = await self._analyze_performance(dt, app_name, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "get_cf_infrastructure":
                        include_details = arguments.get("include_details", True)
                        analysis = await self._get_infrastructure_health(dt, include_details)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "troubleshoot_cf_service":
                        service_name = arguments["service_name"]
                        duration = arguments.get("duration", "1h")
                        analysis = await self._troubleshoot_service(dt, service_name, duration)
                        return [types.TextContent(type="text", text=analysis)]
                    
                    elif name == "get_cf_dependencies":
                        service_name = arguments["service_name"]
                        depth = arguments.get("depth", 2)
                        analysis = await self._analyze_dependencies(dt, service_name, depth)
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
    
    def _to_dynatrace_time(self, dt: datetime) -> str:
        """Convert datetime to Dynatrace timestamp format"""
        return str(int(dt.timestamp() * 1000))
    
    def _format_services(self, result: Dict[str, Any]) -> str:
        """Format services list"""
        if "entities" not in result:
            return "No services found"
        
        entities = result["entities"]
        if not entities:
            return "No services found"
        
        output = f"Cloud Foundry Services ({len(entities)}):\n\n"
        
        for entity in entities:
            display_name = entity.get("displayName", "Unknown")
            entity_id = entity.get("entityId", "")
            entity_type = entity.get("type", "")
            
            # Get management zones if available
            management_zones = entity.get("managementZones", [])
            mz_names = [mz.get("name", "") for mz in management_zones]
            
            output += f"🔧 {display_name}\n"
            output += f"   Type: {entity_type}\n"
            output += f"   ID: {entity_id[:20]}...\n"
            if mz_names:
                output += f"   Management Zones: {', '.join(mz_names)}\n"
            output += "\n"
        
        return output
    
    def _format_problems(self, result: Dict[str, Any]) -> str:
        """Format problems/alerts"""
        if "problems" not in result:
            return "No problems found"
        
        problems = result["problems"]
        if not problems:
            return "✅ No active problems found"
        
        output = f"🚨 Active Problems ({len(problems)}):\n\n"
        
        for problem in problems:
            title = problem.get("title", "Unknown Problem")
            status = problem.get("status", "UNKNOWN")
            severity = problem.get("severityLevel", "UNKNOWN")
            start_time = problem.get("startTime", 0)
            
            # Convert timestamp
            if start_time:
                start_dt = datetime.fromtimestamp(start_time / 1000)
                start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                start_str = "Unknown"
            
            # Status emoji
            status_emoji = "🔴" if status == "OPEN" else "🟡" if status == "RESOLVED" else "⚪"
            
            output += f"{status_emoji} {title}\n"
            output += f"   Severity: {severity}\n"
            output += f"   Status: {status}\n"
            output += f"   Started: {start_str}\n"
            
            # Affected entities
            affected_entities = problem.get("affectedEntities", [])
            if affected_entities:
                output += f"   Affected: {len(affected_entities)} entities\n"
            
            output += "\n"
        
        return output
    
    def _format_metrics(self, result: Dict[str, Any]) -> str:
        """Format metrics data"""
        if "result" not in result:
            return "No metrics data found"
        
        metrics_result = result["result"]
        if not metrics_result:
            return "No metrics data found"
        
        output = "📊 Metrics Data:\n\n"
        
        for metric in metrics_result:
            metric_id = metric.get("metricId", "unknown")
            data_points = metric.get("data", [])
            
            output += f"Metric: {metric_id}\n"
            
            if data_points:
                output += f"Data points: {len(data_points)}\n"
                
                # Show latest values
                for dp in data_points[-5:]:  # Last 5 data points
                    timestamp = dp.get("timestamp", 0)
                    if timestamp:
                        dt = datetime.fromtimestamp(timestamp / 1000)
                        time_str = dt.strftime("%H:%M:%S")
                    else:
                        time_str = "Unknown"
                    
                    dimensions = dp.get("dimensions", [])
                    values = dp.get("values", [])
                    
                    if values:
                        value = values[0] if isinstance(values, list) else values
                        output += f"  {time_str}: {value}\n"
            
            output += "\n"
        
        return output
    
    async def _analyze_service_health(self, dt: DynatraceClient, service_name: str, duration: str) -> str:
        """Analyze service health"""
        output = f"🔍 Service Health Analysis: {service_name or 'All Services'}\n\n"
        
        try:
            # Get services
            entity_selector = f'type("SERVICE"),entityName.equals("{service_name}")' if service_name else 'type("SERVICE")'
            services_result = await dt.get_entities("SERVICE", entity_selector)
            
            if "entities" not in services_result or not services_result["entities"]:
                return f"Service '{service_name}' not found"
            
            services = services_result["entities"]
            
            for service in services:
                service_display = service.get("displayName", "Unknown")
                service_id = service.get("entityId", "")
                
                output += f"Service: {service_display}\n"
                
                # Get key metrics
                end_time = datetime.now()
                start_time = end_time - self._parse_duration(duration)
                
                # Response time
                try:
                    rt_metrics = await dt.get_metrics(
                        f'builtin:service.response.time:filter(eq("service","{service_id}"))',
                        from_time=self._to_dynatrace_time(start_time),
                        to_time=self._to_dynatrace_time(end_time)
                    )
                    if rt_metrics.get("result"):
                        data = rt_metrics["result"][0].get("data", [])
                        if data:
                            latest_rt = data[-1].get("values", [0])[0]
                            status_rt = "✅" if latest_rt < 500 else "⚠️" if latest_rt < 1000 else "❌"
                            output += f"  {status_rt} Response Time: {latest_rt:.0f}ms\n"
                except Exception as e:
                    output += f"  ⚠️ Response Time: Unable to fetch ({str(e)})\n"
                
                # Error rate
                try:
                    error_metrics = await dt.get_metrics(
                        f'builtin:service.errors.total.rate:filter(eq("service","{service_id}"))',
                        from_time=self._to_dynatrace_time(start_time),
                        to_time=self._to_dynatrace_time(end_time)
                    )
                    if error_metrics.get("result"):
                        data = error_metrics["result"][0].get("data", [])
                        if data:
                            latest_errors = data[-1].get("values", [0])[0]
                            status_err = "✅" if latest_errors < 0.01 else "⚠️" if latest_errors < 0.05 else "❌"
                            output += f"  {status_err} Error Rate: {latest_errors*100:.2f}%\n"
                except Exception as e:
                    output += f"  ⚠️ Error Rate: Unable to fetch ({str(e)})\n"
                
                output += "\n"
        
        except Exception as e:
            output += f"❌ Analysis failed: {str(e)}\n"
        
        return output
    
    async def _analyze_performance(self, dt: DynatraceClient, app_name: str, duration: str) -> str:
        """Analyze application performance"""
        return f"🔍 Performance Analysis for '{app_name}' - Feature coming soon!\n"
    
    async def _get_infrastructure_health(self, dt: DynatraceClient, include_details: bool) -> str:
        """Get CF infrastructure health"""
        output = "🏗️ Cloud Foundry Infrastructure Health:\n\n"
        
        try:
            # Get hosts
            hosts_result = await dt.get_entities("HOST")
            if "entities" in hosts_result:
                hosts = hosts_result["entities"]
                output += f"Hosts: {len(hosts)} detected\n"
                
                if include_details:
                    for host in hosts[:5]:  # Show first 5
                        host_name = host.get("displayName", "Unknown")
                        output += f"  • {host_name}\n"
            
            # Get processes
            processes_result = await dt.get_entities("PROCESS_GROUP")
            if "entities" in processes_result:
                processes = processes_result["entities"]
                output += f"Process Groups: {len(processes)} detected\n"
                
                if include_details:
                    for process in processes[:5]:  # Show first 5
                        process_name = process.get("displayName", "Unknown")
                        output += f"  • {process_name}\n"
        
        except Exception as e:
            output += f"❌ Infrastructure check failed: {str(e)}\n"
        
        return output
    
    async def _troubleshoot_service(self, dt: DynatraceClient, service_name: str, duration: str) -> str:
        """Comprehensive service troubleshooting"""
        output = f"🔍 Troubleshooting: {service_name}\n\n"
        
        # This would include comprehensive analysis
        # For now, delegate to service health analysis
        return await self._analyze_service_health(dt, service_name, duration)
    
    async def _analyze_dependencies(self, dt: DynatraceClient, service_name: str, depth: int) -> str:
        """Analyze service dependencies"""
        return f"🔗 Dependency Analysis for '{service_name}' - Feature coming soon!\n"
    
    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="sre-assistant-cf",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )

def main():
    """Main entry point"""
    dynatrace_url = os.getenv("DYNATRACE_API_URL")
    dynatrace_token = os.getenv("DYNATRACE_API_TOKEN")
    
    if not dynatrace_url or not dynatrace_token:
        logger.error("DYNATRACE_API_URL and DYNATRACE_API_TOKEN environment variables are required")
        return
    
    server = SREAssistantCFServer(dynatrace_url=dynatrace_url, dynatrace_token=dynatrace_token)
    asyncio.run(server.run())

if __name__ == "__main__":
    main()