#!/usr/bin/env python3
"""
Automated Security Audit Script for Q2 Platform

Performs comprehensive security checks including:
- Vulnerability scanning
- Configuration validation
- Access control verification
- Compliance reporting
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.security.policies import get_security_policy_engine
from shared.security_config import get_security_config
from shared.vault_client import VaultClient


class SecurityAuditor:
    """Automated security auditing system"""
    
    def __init__(self):
        self.project_root = project_root
        self.report_dir = self.project_root / "security_reports"
        self.report_dir.mkdir(exist_ok=True)
        
        self.policy_engine = get_security_policy_engine()
        self.security_config = get_security_config()
        
        self.audit_results = {
            "audit_timestamp": datetime.now(timezone.utc).isoformat(),
            "security_score": 0,
            "vulnerabilities": [],
            "configuration_issues": [],
            "compliance_status": {},
            "recommendations": []
        }
    
    async def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        print("🔍 Starting comprehensive security audit...")
        
        # Run all audit checks
        await self._scan_vulnerabilities()
        await self._check_configurations()
        await self._verify_access_controls()
        await self._check_secrets_management()
        await self._validate_network_security()
        await self._check_compliance()
        
        # Calculate overall security score
        self._calculate_security_score()
        
        # Generate report
        report_path = await self._generate_report()
        
        print(f"✅ Security audit completed. Report saved to: {report_path}")
        print(f"🎯 Overall Security Score: {self.audit_results['security_score']}/100")
        
        return self.audit_results
    
    async def _scan_vulnerabilities(self):
        """Scan for code vulnerabilities using multiple tools"""
        print("🔎 Scanning for vulnerabilities...")
        
        vulnerabilities = []
        
        # Run Bandit security scan
        try:
            result = subprocess.run([
                "bandit", "-r", str(self.project_root),
                "--exclude", "tests,*/tests/*",
                "-f", "json",
                "-o", "/tmp/bandit_report.json"
            ], capture_output=True, text=True, timeout=300)
            
            if os.path.exists("/tmp/bandit_report.json"):
                with open("/tmp/bandit_report.json") as f:
                    bandit_data = json.load(f)
                    
                for result in bandit_data.get("results", []):
                    vulnerabilities.append({
                        "tool": "bandit",
                        "severity": result.get("issue_severity", "UNKNOWN"),
                        "confidence": result.get("issue_confidence", "UNKNOWN"),
                        "file": result.get("filename", ""),
                        "line": result.get("line_number", 0),
                        "issue": result.get("issue_text", ""),  
                        "rule": result.get("test_name", ""),
                        "cwe": result.get("issue_cwe", {}).get("id", "")
                    })
        
        except Exception as e:
            vulnerabilities.append({
                "tool": "bandit",
                "severity": "ERROR",
                "issue": f"Failed to run bandit: {e}"
            })
        
        # Run Safety check for known vulnerabilities in dependencies
        try:
            requirements_files = list(self.project_root.rglob("requirements.txt"))
            
            for req_file in requirements_files[:5]:  # Limit to avoid timeout
                result = subprocess.run([
                    "safety", "check", "-r", str(req_file), "--json"
                ], capture_output=True, text=True, timeout=60)
                
                if result.stdout:
                    try:
                        safety_data = json.loads(result.stdout)
                        for vuln in safety_data:
                            vulnerabilities.append({
                                "tool": "safety",
                                "severity": "HIGH",
                                "package": vuln.get("package", ""),
                                "version": vuln.get("installed_version", ""),
                                "issue": vuln.get("advisory", ""),
                                "cve": vuln.get("cve", ""),
                                "file": str(req_file)
                            })
                    except json.JSONDecodeError:
                        pass
        
        except Exception as e:
            vulnerabilities.append({
                "tool": "safety",
                "severity": "ERROR", 
                "issue": f"Failed to run safety check: {e}"
            })
        
        self.audit_results["vulnerabilities"] = vulnerabilities
        print(f"Found {len(vulnerabilities)} potential vulnerabilities")
    
    async def _check_configurations(self):
        """Check security configurations"""
        print("⚙️  Checking security configurations...")
        
        issues = []
        
        # Check API security settings
        if self.security_config.api_host == "0.0.0.0":
            issues.append({
                "category": "network",
                "severity": "MEDIUM",
                "issue": "API bound to all interfaces (0.0.0.0)",
                "recommendation": "Bind to specific interface or use 127.0.0.1 for localhost"
            })
        
        # Check JWT configuration
        if len(self.security_config.jwt_secret_key) < 32:
            issues.append({
                "category": "authentication",
                "severity": "HIGH",
                "issue": "JWT secret key too short",
                "recommendation": "Use at least 32 character secret key"
            })
        
        # Check session security
        if not self.security_config.session_cookie_secure:
            issues.append({
                "category": "session",
                "severity": "MEDIUM",
                "issue": "Session cookies not marked as secure",
                "recommendation": "Enable secure cookie flag for HTTPS"
            })
        
        if not self.security_config.session_cookie_httponly:
            issues.append({
                "category": "session", 
                "severity": "HIGH",
                "issue": "Session cookies accessible via JavaScript",
                "recommendation": "Enable HttpOnly flag to prevent XSS"
            })
        
        # Check CORS configuration
        if "*" in self.security_config.cors_origins:
            issues.append({
                "category": "cors",
                "severity": "HIGH",
                "issue": "CORS allows all origins (*)",
                "recommendation": "Restrict CORS to specific domains"
            })
        
        # Check file permissions
        for config_file in self.project_root.rglob("*.env"):
            try:
                stat = config_file.stat()
                if stat.st_mode & 0o077:  # World or group readable
                    issues.append({
                        "category": "filesystem",
                        "severity": "MEDIUM",
                        "issue": f"Config file {config_file} has overly permissive permissions",
                        "recommendation": "Set permissions to 600 (owner read/write only)"
                    })
            except Exception:
                pass
        
        self.audit_results["configuration_issues"] = issues
        print(f"Found {len(issues)} configuration issues")
    
    async def _verify_access_controls(self):
        """Verify access control implementations"""
        print("🔐 Verifying access controls...")
        
        issues = []
        
        # Check role definitions
        roles = self.policy_engine.roles
        
        # Verify admin role exists and is properly configured
        if "admin" not in roles:
            issues.append({
                "category": "rbac",
                "severity": "HIGH",
                "issue": "No admin role defined",
                "recommendation": "Define admin role with appropriate permissions"
            })
        elif len(roles["admin"].permissions) < 3:
            issues.append({
                "category": "rbac", 
                "severity": "MEDIUM",
                "issue": "Admin role has insufficient permissions",
                "recommendation": "Review admin role permissions"
            })
        
        # Check for overly permissive roles
        for role_name, role in roles.items():
            if len(role.permissions) > 4 and role_name != "admin":
                issues.append({
                    "category": "rbac",
                    "severity": "MEDIUM", 
                    "issue": f"Role '{role_name}' may have too many permissions",
                    "recommendation": "Review and apply principle of least privilege"
                })
        
        # Check policy enforcement
        policies = self.policy_engine.policies
        
        if not policies:
            issues.append({
                "category": "policy",
                "severity": "HIGH",
                "issue": "No security policies defined",
                "recommendation": "Implement security policies for access control"
            })
        
        for policy_id, policy in policies.items():
            if not policy.enabled:
                issues.append({
                    "category": "policy",
                    "severity": "MEDIUM",
                    "issue": f"Security policy '{policy_id}' is disabled",
                    "recommendation": "Enable security policy or remove if not needed"
                })
        
        self.audit_results["access_control_issues"] = issues
        print(f"Found {len(issues)} access control issues")
    
    async def _check_secrets_management(self):
        """Check secrets management implementation"""
        print("🔑 Checking secrets management...")
        
        issues = []
        
        # Check for hardcoded secrets in code
        hardcoded_patterns = [
            "password=", "secret=", "key=", "token=",
            "api_key=", "private_key=", "auth="
        ]
        
        secret_files = 0
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text()
                for pattern in hardcoded_patterns:
                    if pattern in content.lower() and "example" not in content.lower():
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line.lower() and not line.strip().startswith('#'):
                                issues.append({
                                    "category": "secrets",
                                    "severity": "HIGH",
                                    "issue": f"Potential hardcoded secret in {py_file}:{i+1}",
                                    "recommendation": "Move secret to environment variable or vault"
                                })
                                secret_files += 1
                                break
            except Exception:
                pass
        
        # Check vault connectivity
        try:
            vault_client = VaultClient()
            if not vault_client.client.is_authenticated():
                issues.append({
                    "category": "vault",
                    "severity": "HIGH",
                    "issue": "Cannot authenticate with Vault",
                    "recommendation": "Check Vault configuration and credentials"
                })
        except Exception as e:
            issues.append({
                "category": "vault",
                "severity": "MEDIUM",
                "issue": f"Vault not available: {e}",
                "recommendation": "Ensure Vault is properly configured and accessible"
            })
        
        self.audit_results["secrets_management_issues"] = issues
        print(f"Found {len(issues)} secrets management issues")
    
    async def _validate_network_security(self):
        """Validate network security configurations"""
        print("🌐 Validating network security...")
        
        issues = []
        
        # Check for services binding to all interfaces
        for service_dir in ["agentQ", "managerQ", "AuthQ", "VectorStoreQ"]:
            service_path = self.project_root / service_dir
            if service_path.exists():
                for py_file in service_path.rglob("*.py"):
                    try:
                        content = py_file.read_text()
                        if '"0.0.0.0"' in content or "'0.0.0.0'" in content:
                            issues.append({
                                "category": "network",
                                "severity": "MEDIUM",
                                "issue": f"Service {service_dir} binds to all interfaces",
                                "file": str(py_file),
                                "recommendation": "Use specific interface or localhost binding"
                            })
                    except Exception:
                        pass
        
        # Check TLS/SSL configuration
        cert_files = list(self.project_root.rglob("*.crt")) + list(self.project_root.rglob("*.pem"))
        if not cert_files:
            issues.append({
                "category": "tls",
                "severity": "MEDIUM", 
                "issue": "No TLS certificates found",
                "recommendation": "Implement TLS/SSL for secure communications"
            })
        
        self.audit_results["network_security_issues"] = issues
        print(f"Found {len(issues)} network security issues")
    
    async def _check_compliance(self):
        """Check compliance with security standards"""
        print("📋 Checking compliance...")
        
        compliance = {}
        
        # Basic security compliance checklist
        checks = {
            "authentication_required": self._has_authentication(),
            "authorization_implemented": self._has_authorization(),
            "audit_logging_enabled": self._has_audit_logging(),
            "data_encryption": self._has_data_encryption(),
            "session_management": self._has_session_management(),
            "input_validation": self._has_input_validation(),
            "error_handling": self._has_error_handling(),
            "secure_headers": self._has_secure_headers()
        }
        
        passed = sum(checks.values())
        total = len(checks)
        
        compliance = {
            "checks": checks,
            "passed": passed,
            "total": total,
            "percentage": round((passed / total) * 100, 2) if total > 0 else 0
        }
        
        self.audit_results["compliance_status"] = compliance
        print(f"Compliance: {passed}/{total} checks passed ({compliance['percentage']}%)")
    
    def _has_authentication(self) -> bool:
        """Check if authentication is implemented"""
        auth_files = list(self.project_root.rglob("*auth*.py"))
        return len(auth_files) > 0
    
    def _has_authorization(self) -> bool:
        """Check if authorization is implemented"""
        return len(self.policy_engine.roles) > 1 and len(self.policy_engine.policies) > 0
    
    def _has_audit_logging(self) -> bool:
        """Check if audit logging is implemented"""
        audit_files = list(self.project_root.rglob("*audit*.py"))
        return len(audit_files) > 0
    
    def _has_data_encryption(self) -> bool:
        """Check if data encryption is implemented"""  
        security_files = list(self.project_root.rglob("*security*.py"))
        return len(security_files) > 0
    
    def _has_session_management(self) -> bool:
        """Check if session management is implemented"""
        return self.security_config.session_cookie_secure
    
    def _has_input_validation(self) -> bool:
        """Check if input validation is implemented"""
        # Look for pydantic models or validation
        validation_patterns = ["pydantic", "validator", "Field"]
        for py_file in list(self.project_root.rglob("*.py"))[:20]:  # Sample check
            try:
                content = py_file.read_text()
                if any(pattern in content for pattern in validation_patterns):
                    return True
            except Exception:
                pass
        return False
    
    def _has_error_handling(self) -> bool:
        """Check if proper error handling is implemented"""
        # Look for try-except blocks
        error_handling_count = 0
        for py_file in list(self.project_root.rglob("*.py"))[:20]:  # Sample check
            try:
                content = py_file.read_text()
                error_handling_count += content.count("try:")
            except Exception:
                pass
        return error_handling_count > 10
    
    def _has_secure_headers(self) -> bool:
        """Check if secure headers are implemented"""
        # Look for security headers in FastAPI or other web frameworks
        for py_file in list(self.project_root.rglob("*.py"))[:20]:  # Sample check
            try:
                content = py_file.read_text()
                if "X-Content-Type-Options" in content or "X-Frame-Options" in content:
                    return True
            except Exception:
                pass
        return False
    
    def _calculate_security_score(self):
        """Calculate overall security score"""
        score = 100  # Start with perfect score
        
        # Deduct points for vulnerabilities
        vulnerabilities = self.audit_results.get("vulnerabilities", [])
        for vuln in vulnerabilities:
            severity = vuln.get("severity", "UNKNOWN").upper()
            if severity == "HIGH":
                score -= 15
            elif severity == "MEDIUM":
                score -= 8
            elif severity == "LOW":
                score -= 3
        
        # Deduct points for configuration issues
        config_issues = self.audit_results.get("configuration_issues", [])
        for issue in config_issues:
            severity = issue.get("severity", "UNKNOWN").upper()
            if severity == "HIGH":
                score -= 10
            elif severity == "MEDIUM":
                score -= 5
            elif severity == "LOW":
                score -= 2
        
        # Bonus points for compliance
        compliance = self.audit_results.get("compliance_status", {})
        compliance_percentage = compliance.get("percentage", 0)
        score += int(compliance_percentage * 0.2)  # Up to 20 bonus points
        
        # Ensure score is between 0 and 100
        score = max(0, min(100, score))
        
        self.audit_results["security_score"] = score
        
        # Generate recommendations based on score
        if score < 60:
            self.audit_results["recommendations"].append(
                "CRITICAL: Security score is below acceptable threshold. Immediate action required."
            )
        elif score < 80:
            self.audit_results["recommendations"].append(
                "WARNING: Security improvements needed to meet best practices."
            )
        else:
            self.audit_results["recommendations"].append(
                "GOOD: Security posture is acceptable but continue monitoring."
            )
    
    async def _generate_report(self) -> Path:
        """Generate comprehensive security audit report"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = self.report_dir / f"security_audit_{timestamp}.json"
        
        # Add summary statistics
        self.audit_results["summary"] = {
            "total_vulnerabilities": len(self.audit_results.get("vulnerabilities", [])),
            "high_risk_vulnerabilities": len([
                v for v in self.audit_results.get("vulnerabilities", [])
                if v.get("severity", "").upper() == "HIGH"
            ]),
            "total_config_issues": len(self.audit_results.get("configuration_issues", [])),
            "compliance_percentage": self.audit_results.get("compliance_status", {}).get("percentage", 0)
        }
        
        # Write report
        with open(report_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2)
        
        # Also generate human-readable summary
        summary_path = self.report_dir / f"security_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("Q2 Platform Security Audit Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Audit Date: {self.audit_results['audit_timestamp']}\n")
            f.write(f"Security Score: {self.audit_results['security_score']}/100\n\n")
            
            summary = self.audit_results["summary"]
            f.write(f"Vulnerabilities Found: {summary['total_vulnerabilities']}\n")
            f.write(f"High-Risk Issues: {summary['high_risk_vulnerabilities']}\n")
            f.write(f"Configuration Issues: {summary['total_config_issues']}\n")
            f.write(f"Compliance Score: {summary['compliance_percentage']}%\n\n")
            
            f.write("Recommendations:\n")
            for rec in self.audit_results.get("recommendations", []):
                f.write(f"- {rec}\n")
        
        return report_path


async def main():
    """Main function to run security audit"""
    auditor = SecurityAuditor()
    
    try:
        results = await auditor.run_full_audit()
        
        # Print summary
        print("\n" + "="*50)
        print("SECURITY AUDIT SUMMARY")
        print("="*50)
        print(f"Overall Score: {results['security_score']}/100")
        print(f"Vulnerabilities: {len(results.get('vulnerabilities', []))}")
        print(f"Config Issues: {len(results.get('configuration_issues', []))}")
        print(f"Compliance: {results.get('compliance_status', {}).get('percentage', 0)}%")
        
        if results['security_score'] < 70:
            print("\n❌ Security audit FAILED - Score below 70")
            sys.exit(1)
        else:
            print("\n✅ Security audit PASSED")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Security audit failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())