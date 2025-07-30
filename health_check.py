# health_check.py
#!/usr/bin/env python3
"""Simple health check for the containerized MCP server"""

import sys
import os

# Check if the main server file exists and is readable
if os.path.exists('/app/sre_assistant.py') and os.access('/app/sre_assistant.py', os.R_OK):
    print("MCP server container is healthy")
    sys.exit(0)
else:
    print("MCP server container is unhealthy")
    sys.exit(1)