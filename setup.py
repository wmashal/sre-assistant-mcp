from setuptools import setup, find_packages

setup(
    name="sre-assistant-mcp",
    version="1.0.0",
    description="SRE Assistant MCP Server for Prometheus Integration",
    packages=find_packages(),
    install_requires=[
        "mcp>=1.0.0",
        "aiohttp>=3.8.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "sre-assistant=sre_assistant:main",
        ],
    },
)
