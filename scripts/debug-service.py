#!/usr/bin/env python3
"""
Q2 Platform Service Debugging Utility

This script helps debug Q2 Platform services by checking their health,
connectivity, and providing useful debugging information.
"""

import asyncio
import json
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
        structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "event"])
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Service configurations
SERVICES = {
    "agentQ": {"port": 8000, "path": "agentQ"},
    "managerQ": {"port": 8001, "path": "managerQ"},
    "VectorStoreQ": {"port": 8002, "path": "VectorStoreQ"},
    "KnowledgeGraphQ": {"port": 8003, "path": "KnowledgeGraphQ"},
    "AuthQ": {"port": 8004, "path": "AuthQ"},
    "H2M": {"port": 8005, "path": "H2M"},
    "AgentSandbox": {"port": 8006, "path": "AgentSandbox"},
    "UserProfileQ": {"port": 8007, "path": "UserProfileQ"},
}

# Infrastructure services
INFRA_SERVICES = {
    "Pulsar": {"port": 8080, "health_path": "/admin/v2/brokers/health"},
    "Milvus": {"port": 19530, "health_path": "/health"},
    "Keycloak": {"port": 8080, "health_path": "/health"},
    "Grafana": {"port": 3000, "health_path": "/api/health"},
    "Prometheus": {"port": 9090, "health_path": "/-/healthy"},
}

class ServiceDebugger:
    """Debugging utility for Q2 Platform services."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
        self.results = {}
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def check_service_health(self, service_name: str, port: int, health_path: str = "/health") -> Dict:
        """Check health of a specific service."""
        url = f"http://localhost:{port}{health_path}"
        
        try:
            response = await self.client.get(url)
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "data": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text[:200]
            }
        except httpx.ConnectError:
            return {
                "status": "unreachable",
                "error": "Connection refused - service may not be running"
            }
        except httpx.TimeoutException:
            return {
                "status": "timeout",
                "error": "Service did not respond within timeout"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_service_endpoints(self, service_name: str, port: int) -> Dict:
        """Check common endpoints for a service."""
        endpoints = ["/", "/health", "/ready", "/docs", "/openapi.json"]
        results = {}
        
        for endpoint in endpoints:
            url = f"http://localhost:{port}{endpoint}"
            try:
                response = await self.client.get(url)
                results[endpoint] = {
                    "status_code": response.status_code,
                    "available": response.status_code < 400,
                    "content_type": response.headers.get("content-type", "unknown")
                }
            except Exception:
                results[endpoint] = {
                    "status_code": None,
                    "available": False,
                    "error": "unreachable"
                }
        
        return results
    
    def check_service_files(self, service_name: str, service_path: str) -> Dict:
        """Check if service files exist and are valid."""
        service_dir = Path(service_path)
        
        files_to_check = [
            "requirements.txt",
            "Dockerfile",
            "app/main.py",
            "app/__init__.py",
            "README.md"
        ]
        
        results = {}
        for file_path in files_to_check:
            full_path = service_dir / file_path
            results[file_path] = {
                "exists": full_path.exists(),
                "is_file": full_path.is_file() if full_path.exists() else False,
                "size": full_path.stat().st_size if full_path.exists() else 0
            }
        
        return results
    
    async def check_service_dependencies(self, service_path: str) -> Dict:
        """Check if service dependencies are properly installed."""
        service_dir = Path(service_path)
        requirements_file = service_dir / "requirements.txt"
        
        if not requirements_file.exists():
            return {"status": "no_requirements", "message": "No requirements.txt found"}
        
        try:
            # Read requirements
            requirements = requirements_file.read_text().strip().split('\n')
            requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
            
            # Try to import each package (simplified check)
            import subprocess
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'check'],
                capture_output=True,
                text=True,
                cwd=service_dir
            )
            
            return {
                "status": "ok" if result.returncode == 0 else "issues",
                "requirements_count": len(requirements),
                "pip_check_output": result.stdout + result.stderr if result.returncode != 0 else "All dependencies satisfied"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def debug_service(self, service_name: str) -> Dict:
        """Run comprehensive debugging for a service."""
        if service_name not in SERVICES:
            return {"error": f"Unknown service: {service_name}"}
        
        config = SERVICES[service_name]
        port = config["port"]
        service_path = config["path"]
        
        logger.info(f"Debugging service: {service_name}")
        
        # Run all checks
        health_check = await self.check_service_health(service_name, port)
        endpoints_check = await self.check_service_endpoints(service_name, port)
        files_check = self.check_service_files(service_name, service_path)
        deps_check = await self.check_service_dependencies(service_path)
        
        return {
            "service": service_name,
            "port": port,
            "path": service_path,
            "health": health_check,
            "endpoints": endpoints_check,
            "files": files_check,
            "dependencies": deps_check,
            "overall_status": self._determine_overall_status(health_check, endpoints_check, files_check, deps_check)
        }
    
    def _determine_overall_status(self, health: Dict, endpoints: Dict, files: Dict, deps: Dict) -> str:
        """Determine overall service status based on checks."""
        if health.get("status") == "healthy":
            return "healthy"
        elif health.get("status") in ["unreachable", "timeout"]:
            if all(f["exists"] for f in files.values() if isinstance(f, dict)):
                return "stopped"
            else:
                return "missing_files"
        else:
            return "unhealthy"
    
    async def debug_all_services(self) -> Dict[str, Dict]:
        """Debug all Q2 Platform services."""
        results = {}
        
        for service_name in SERVICES:
            results[service_name] = await self.debug_service(service_name)
        
        return results
    
    async def check_infrastructure(self) -> Dict[str, Dict]:
        """Check infrastructure services."""
        results = {}
        
        for service_name, config in INFRA_SERVICES.items():
            port = config["port"]
            health_path = config.get("health_path", "/health")
            
            logger.info(f"Checking infrastructure service: {service_name}")
            results[service_name] = await self.check_service_health(service_name, port, health_path)
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable debugging report."""
        report = []
        report.append("Q2 Platform Service Debugging Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        healthy_services = []
        unhealthy_services = []
        stopped_services = []
        
        for service_name, result in results.items():
            if isinstance(result, dict) and "overall_status" in result:
                status = result["overall_status"]
                if status == "healthy":
                    healthy_services.append(service_name)
                elif status == "stopped":
                    stopped_services.append(service_name)
                else:
                    unhealthy_services.append(service_name)
        
        report.append(f"Summary:")
        report.append(f"  Healthy services: {len(healthy_services)} - {', '.join(healthy_services) if healthy_services else 'None'}")
        report.append(f"  Stopped services: {len(stopped_services)} - {', '.join(stopped_services) if stopped_services else 'None'}")
        report.append(f"  Unhealthy services: {len(unhealthy_services)} - {', '.join(unhealthy_services) if unhealthy_services else 'None'}")
        report.append("")
        
        # Detailed results
        for service_name, result in results.items():
            if isinstance(result, dict) and "overall_status" in result:
                report.append(f"Service: {service_name}")
                report.append(f"  Port: {result['port']}")
                report.append(f"  Status: {result['overall_status']}")
                
                # Health details
                health = result.get("health", {})
                if health.get("status") == "healthy":
                    report.append(f"  Health: ✅ {health['status']} ({health.get('response_time_ms', 0):.1f}ms)")
                else:
                    report.append(f"  Health: ❌ {health.get('status', 'unknown')} - {health.get('error', 'Unknown error')}")
                
                # File check
                files = result.get("files", {})
                missing_files = [f for f, info in files.items() if isinstance(info, dict) and not info.get("exists", False)]
                if missing_files:
                    report.append(f"  Missing files: {', '.join(missing_files)}")
                
                # Dependencies
                deps = result.get("dependencies", {})
                if deps.get("status") == "issues":
                    report.append(f"  Dependencies: ⚠️  Issues detected")
                elif deps.get("status") == "ok":
                    report.append(f"  Dependencies: ✅ OK ({deps.get('requirements_count', 0)} packages)")
                
                report.append("")
        
        return "\n".join(report)


async def main():
    """Main function to run the debugging utility."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Q2 Platform services")
    parser.add_argument(
        "--service",
        help="Specific service to debug (default: all services)"
    )
    parser.add_argument(
        "--infrastructure",
        action="store_true",
        help="Check infrastructure services instead of application services"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--output",
        help="Save report to file"
    )
    
    args = parser.parse_args()
    
    async with ServiceDebugger() as debugger:
        try:
            if args.infrastructure:
                logger.info("Checking infrastructure services...")
                results = await debugger.check_infrastructure()
            elif args.service:
                logger.info(f"Debugging service: {args.service}")
                results = {args.service: await debugger.debug_service(args.service)}
            else:
                logger.info("Debugging all Q2 Platform services...")
                results = await debugger.debug_all_services()
            
            # Output results
            if args.json:
                output = json.dumps(results, indent=2)
            else:
                output = debugger.generate_report(results)
            
            if args.output:
                Path(args.output).write_text(output)
                logger.info(f"Report saved to {args.output}")
            else:
                print(output)
        
        except Exception as e:
            logger.error(f"Debugging failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())