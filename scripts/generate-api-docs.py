#!/usr/bin/env python3
"""
Q2 Platform API Documentation Generator

This script automatically generates API documentation for all services
by discovering FastAPI applications and extracting their OpenAPI specs.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import structlog

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Service configurations
SERVICES = {
    "agentQ": {"port": 8000, "path": "agentQ/app"},
    "managerQ": {"port": 8001, "path": "managerQ/app"},
    "VectorStoreQ": {"port": 8002, "path": "VectorStoreQ/app"},
    "KnowledgeGraphQ": {"port": 8003, "path": "KnowledgeGraphQ/app"},
    "AuthQ": {"port": 8004, "path": "AuthQ/app"},
    "H2M": {"port": 8005, "path": "H2M/app"},
    "AgentSandbox": {"port": 8006, "path": "AgentSandbox/app"},
    "UserProfileQ": {"port": 8007, "path": "UserProfileQ/app"},
}

OUTPUT_DIR = Path("docs/api")


class APIDocGenerator:
    """Generates API documentation for Q2 Platform services."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    def discover_fastapi_services(self) -> Dict[str, Dict]:
        """Discover FastAPI services in the project."""
        discovered_services = {}
        
        for service_name, config in SERVICES.items():
            service_path = Path(config["path"])
            
            if service_path.exists():
                # Look for main.py or app.py
                for main_file in ["main.py", "app.py", "__init__.py"]:
                    main_path = service_path / main_file
                    if main_path.exists():
                        # Check if it contains FastAPI
                        try:
                            content = main_path.read_text()
                            if "FastAPI" in content or "from fastapi" in content:
                                discovered_services[service_name] = config
                                logger.info(f"Discovered FastAPI service: {service_name}")
                                break
                        except Exception as e:
                            logger.warning(f"Error reading {main_path}: {e}")
        
        return discovered_services
    
    async def fetch_openapi_spec(self, service_name: str, port: int) -> Optional[Dict]:
        """Fetch OpenAPI specification from a running service."""
        url = f"http://localhost:{port}/openapi.json"
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            spec = response.json()
            logger.info(f"Retrieved OpenAPI spec for {service_name}")
            return spec
        except httpx.ConnectError:
            logger.warning(f"Service {service_name} not running on port {port}")
            return None
        except Exception as e:
            logger.error(f"Error fetching OpenAPI spec for {service_name}: {e}")
            return None
    
    def generate_service_doc(self, service_name: str, spec: Dict) -> str:
        """Generate markdown documentation from OpenAPI spec."""
        doc = []
        
        # Header
        doc.append(f"# {service_name} API Documentation\n")
        
        # Service info
        info = spec.get("info", {})
        if info.get("description"):
            doc.append(f"{info['description']}\n")
        
        doc.append(f"**Version:** {info.get('version', 'Unknown')}\n")
        
        if info.get("contact"):
            contact = info["contact"]
            if contact.get("name"):
                doc.append(f"**Contact:** {contact['name']}")
                if contact.get("email"):
                    doc.append(f" ({contact['email']})")
                doc.append("\n")
        
        # Base URL
        servers = spec.get("servers", [])
        if servers:
            doc.append(f"**Base URL:** {servers[0]['url']}\n")
        
        doc.append("---\n")
        
        # Endpoints
        paths = spec.get("paths", {})
        if paths:
            doc.append("## Endpoints\n")
            
            for path, methods in paths.items():
                doc.append(f"### `{path}`\n")
                
                for method, details in methods.items():
                    if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                        doc.append(f"#### {method.upper()} {path}\n")
                        
                        # Summary and description
                        if details.get("summary"):
                            doc.append(f"**Summary:** {details['summary']}\n")
                        
                        if details.get("description"):
                            doc.append(f"{details['description']}\n")
                        
                        # Parameters
                        parameters = details.get("parameters", [])
                        if parameters:
                            doc.append("**Parameters:**\n")
                            for param in parameters:
                                param_name = param.get("name", "")
                                param_type = param.get("schema", {}).get("type", "unknown")
                                param_desc = param.get("description", "")
                                param_required = "Required" if param.get("required") else "Optional"
                                doc.append(f"- `{param_name}` ({param_type}) - {param_required}: {param_desc}\n")
                        
                        # Request body
                        request_body = details.get("requestBody")
                        if request_body:
                            doc.append("**Request Body:**\n")
                            content = request_body.get("content", {})
                            for content_type, schema in content.items():
                                doc.append(f"- Content-Type: `{content_type}`\n")
                                if schema.get("schema"):
                                    doc.append(f"- Schema: `{json.dumps(schema['schema'], indent=2)}`\n")
                        
                        # Responses
                        responses = details.get("responses", {})
                        if responses:
                            doc.append("**Responses:**\n")
                            for code, response in responses.items():
                                desc = response.get("description", "")
                                doc.append(f"- `{code}`: {desc}\n")
                        
                        doc.append("\n")
        
        # Models/Schemas
        components = spec.get("components", {})
        schemas = components.get("schemas", {})
        if schemas:
            doc.append("## Data Models\n")
            for schema_name, schema_def in schemas.items():
                doc.append(f"### {schema_name}\n")
                doc.append(f"```json\n{json.dumps(schema_def, indent=2)}\n```\n")
        
        return "\n".join(doc)
    
    def generate_index_doc(self, services: List[str]) -> str:
        """Generate index documentation listing all services."""
        doc = []
        
        doc.append("# Q2 Platform API Documentation\n")
        doc.append("This documentation provides comprehensive API reference for all Q2 Platform services.\n")
        doc.append("## Services\n")
        
        for service in sorted(services):
            doc.append(f"- [{service}](./{service.lower()}.md) - {service} service API")
            port = SERVICES.get(service, {}).get("port", "N/A")
            doc.append(f" (Port: {port})\n")
        
        doc.append("\n## Authentication\n")
        doc.append("Most services require authentication via Keycloak OIDC tokens.\n")
        doc.append("Include the token in the Authorization header:\n")
        doc.append("```\nAuthorization: Bearer <your-token>\n```\n")
        
        doc.append("## Error Handling\n")
        doc.append("All services follow consistent error response format:\n")
        doc.append("```json\n")
        doc.append("{\n")
        doc.append('  "error": {\n')
        doc.append('    "code": "ERROR_CODE",\n')
        doc.append('    "message": "Human readable error message",\n')
        doc.append('    "details": {}\n')
        doc.append("  }\n")
        doc.append("}\n")
        doc.append("```\n")
        
        return "\n".join(doc)
    
    async def generate_docs(self, fetch_from_running: bool = False) -> None:
        """Generate documentation for all services."""
        logger.info("Starting API documentation generation")
        
        # Discover services
        services = self.discover_fastapi_services()
        
        if not services:
            logger.warning("No FastAPI services discovered")
            return
        
        successful_services = []
        
        if fetch_from_running:
            # Try to fetch from running services
            logger.info("Attempting to fetch specs from running services")
            
            for service_name, config in services.items():
                spec = await self.fetch_openapi_spec(service_name, config["port"])
                
                if spec:
                    # Generate documentation
                    doc_content = self.generate_service_doc(service_name, spec)
                    
                    # Save to file
                    doc_file = self.output_dir / f"{service_name.lower()}.md"
                    doc_file.write_text(doc_content)
                    
                    logger.info(f"Generated documentation for {service_name}: {doc_file}")
                    successful_services.append(service_name)
        else:
            # Generate placeholder documentation
            logger.info("Generating placeholder documentation")
            for service_name in services.keys():
                doc_content = self.generate_placeholder_doc(service_name)
                doc_file = self.output_dir / f"{service_name.lower()}.md"
                doc_file.write_text(doc_content)
                successful_services.append(service_name)
        
        # Generate index
        if successful_services:
            index_content = self.generate_index_doc(successful_services)
            index_file = self.output_dir / "index.md"
            index_file.write_text(index_content)
            logger.info(f"Generated API documentation index: {index_file}")
        
        logger.info(f"Documentation generation complete. Generated docs for {len(successful_services)} services.")
    
    def generate_placeholder_doc(self, service_name: str) -> str:
        """Generate placeholder documentation for a service."""
        config = SERVICES.get(service_name, {})
        port = config.get("port", "N/A")
        
        return f"""# {service_name} API Documentation

## Overview

{service_name} is a core service in the Q2 Platform architecture.

**Service Port:** {port}  
**Base URL:** http://localhost:{port}  
**API Documentation:** http://localhost:{port}/docs (when running)  
**OpenAPI Spec:** http://localhost:{port}/openapi.json (when running)

## Getting Started

1. Start the service:
   ```bash
   cd {config.get('path', service_name.lower())}
   python -m {service_name.lower()}.app.main
   ```

2. Visit the interactive API documentation at http://localhost:{port}/docs

3. Test endpoints using the Swagger UI or your preferred HTTP client

## Authentication

This service requires authentication via Keycloak OIDC tokens.

## Error Responses

All endpoints return consistent error responses following the Q2 Platform error format.

---

**Note:** This is placeholder documentation. Start the service and run the documentation generator with `--fetch-running` to get detailed API documentation.
"""


async def main():
    """Main function to run the documentation generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Q2 Platform API documentation")
    parser.add_argument(
        "--fetch-running",
        action="store_true",
        help="Fetch OpenAPI specs from running services (requires services to be running)"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/api",
        help="Output directory for generated documentation"
    )
    
    args = parser.parse_args()
    
    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)
    
    async with APIDocGenerator() as generator:
        await generator.generate_docs(fetch_from_running=args.fetch_running)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Documentation generation cancelled")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        sys.exit(1)