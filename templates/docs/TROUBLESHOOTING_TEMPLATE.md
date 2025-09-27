# {SERVICE_NAME} Troubleshooting Guide

This guide helps diagnose and resolve common issues with the {SERVICE_NAME} service.

## Quick Diagnostics

### 1. Health Check
```bash
# Check if service is responding
curl http://localhost:{SERVICE_PORT}/health

# Expected response:
# {"service": "{SERVICE_NAME_LOWER}", "status": "healthy", "version": "1.0.0"}
```

### 2. Service Status
```bash
# Check if service is running
ps aux | grep "{SERVICE_MODULE}"

# Check port binding
lsof -i :{SERVICE_PORT}

# Check logs
tail -f logs/{SERVICE_NAME_LOWER}.log
```

### 3. Automated Diagnostics
```bash
# Run automated debugging
make debug-service SERVICE={SERVICE_NAME_LOWER}

# Full system check
make debug-services
```

## Common Issues

### Service Won't Start

#### Symptom
Service fails to start or exits immediately.

#### Possible Causes & Solutions

**1. Port Already in Use**
```bash
# Check what's using the port
lsof -i :{SERVICE_PORT}

# Solution: Kill the process or change port
export PORT=8001
```

**2. Missing Dependencies**
```bash
# Check dependencies
pip check

# Solution: Reinstall dependencies
pip install -r requirements.txt
```

**3. Configuration Issues**
```bash
# Check configuration
python -c "from {SERVICE_MODULE}.app.config import get_settings; print(get_settings())"

# Solution: Fix .env file or environment variables
cp .env.example .env
# Edit .env with correct values
```

**4. Database Connection Issues**
```bash
# Test database connectivity
python -c "import {SERVICE_MODULE}.app.database; {SERVICE_MODULE}.app.database.test_connection()"

# Solution: Check database URL and credentials
```

### Service Responding Slowly

#### Symptom
API requests taking longer than expected (>1s for simple requests).

#### Diagnostic Steps

**1. Check Resource Usage**
```bash
# Check CPU and memory
top -p $(pgrep -f {SERVICE_MODULE})

# Check Docker stats (if containerized)
docker stats <container_name>
```

**2. Check Database Performance**
```bash
# Check database connections
# Add service-specific database diagnostics

# Check slow queries
# Add service-specific query analysis
```

**3. Check Network Connectivity**
```bash
# Test Pulsar connectivity
python -c "import pulsar; client = pulsar.Client('pulsar://localhost:6650'); print('Connected')"

# Test external APIs
curl -w "@curl-format.txt" http://external-api/endpoint
```

#### Solutions

**1. Scale Resources**
```bash
# Increase memory limit (Docker)
docker run -m 2g q2/{SERVICE_NAME_LOWER}

# Increase CPU limit (Kubernetes)
# Edit k8s/deployment.yaml resources
```

**2. Optimize Database Queries**
- Add indexes for frequently queried fields
- Use connection pooling
- Implement query caching

**3. Enable Connection Pooling**
```python
# Add to configuration
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
```

### High Error Rate

#### Symptom
Service returning HTTP 5xx errors frequently.

#### Diagnostic Steps

**1. Check Error Logs**
```bash
# Filter error logs
grep "ERROR" logs/{SERVICE_NAME_LOWER}.log | tail -50

# Check structured logs
jq 'select(.level=="error")' logs/{SERVICE_NAME_LOWER}.jsonl
```

**2. Check Dependencies**
```bash
# Test Pulsar connection
make debug-infrastructure

# Test database connection
# Add service-specific dependency checks
```

**3. Check Resource Limits**
```bash
# Check if hitting memory limits
dmesg | grep "Killed process"

# Check disk space
df -h
```

#### Solutions

**1. Fix Dependency Issues**
- Restart dependent services
- Check network connectivity
- Verify credentials

**2. Increase Resource Limits**
- Add more memory/CPU
- Scale horizontally

**3. Implement Circuit Breakers**
```python
# Add circuit breaker for external calls
from circuit_breaker import CircuitBreaker

@CircuitBreaker(failure_threshold=5, recovery_timeout=30)
def call_external_service():
    # External service call
    pass
```

### Authentication/Authorization Failures

#### Symptom
Getting HTTP 401/403 errors when calling authenticated endpoints.

#### Diagnostic Steps

**1. Check Token Validity**
```bash
# Decode JWT token
echo "YOUR_TOKEN" | base64 -d

# Check token expiration
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:{SERVICE_PORT}/auth/validate
```

**2. Check Keycloak Configuration**
```bash
# Test Keycloak connectivity
curl http://keycloak:8080/auth/realms/q-platform/.well-known/openid_configuration

# Check realm and client configuration
```

#### Solutions

**1. Refresh Token**
```bash
# Get new token from Keycloak
curl -X POST http://keycloak:8080/auth/realms/q-platform/protocol/openid-connect/token \
  -d "grant_type=client_credentials" \
  -d "client_id=your-client" \
  -d "client_secret=your-secret"
```

**2. Fix Configuration**
- Verify Keycloak realm settings
- Check client permissions
- Update service configuration

### Data Consistency Issues

#### Symptom
Data not consistent between services or database.

#### Diagnostic Steps

**1. Check Pulsar Messages**
```bash
# Check topic messages
pulsar-admin topics stats persistent://public/default/{SERVICE_NAME_LOWER}-events

# Check subscription lag
pulsar-admin topics stats-internal persistent://public/default/{SERVICE_NAME_LOWER}-events
```

**2. Check Database State**
```sql
-- Check recent changes
SELECT * FROM audit_log ORDER BY created_at DESC LIMIT 10;

-- Check data integrity
-- Add service-specific integrity checks
```

#### Solutions

**1. Replay Messages**
```bash
# Reset subscription to replay messages
pulsar-admin topics reset-cursor persistent://public/default/{SERVICE_NAME_LOWER}-events \
  --subscription {SERVICE_NAME_LOWER}-consumer \
  --time 2024-01-01T00:00:00Z
```

**2. Manual Data Correction**
- Identify inconsistent records
- Apply corrective updates
- Monitor for future issues

## Performance Tuning

### Memory Optimization

**1. Monitor Memory Usage**
```bash
# Check memory allocation
python -m memory_profiler {SERVICE_MODULE}/app/main.py

# Check for memory leaks
python -m pympler.tracker
```

**2. Optimization Strategies**
- Use connection pooling
- Implement proper caching
- Optimize data structures
- Use generators for large datasets

### Database Optimization

**1. Query Performance**
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

**2. Connection Pooling**
```python
# Optimize connection pool settings
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_pre_ping": True,
    "pool_recycle": 3600,
}
```

## Monitoring Setup

### Key Metrics to Monitor

1. **Response Time**
   - P50, P95, P99 latencies
   - Target: <100ms for simple requests

2. **Error Rate**
   - HTTP 4xx/5xx error percentage
   - Target: <1% error rate

3. **Throughput**
   - Requests per second
   - Message processing rate

4. **Resource Usage**
   - CPU utilization (<80%)
   - Memory usage (<80%)
   - Database connection pool usage

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: {SERVICE_NAME_LOWER}
    rules:
      - alert: {SERVICE_NAME}HighErrorRate
        expr: rate(http_requests_total{{status=~"5.."}})
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
      
      - alert: {SERVICE_NAME}HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
```

## Emergency Procedures

### Service Recovery

**1. Quick Restart**
```bash
# Docker
docker restart {SERVICE_NAME_LOWER}-container

# Kubernetes
kubectl rollout restart deployment/{SERVICE_NAME_LOWER}

# Direct process
pkill -f {SERVICE_MODULE} && python -m {SERVICE_MODULE}.app.main
```

**2. Rollback Deployment**
```bash
# Kubernetes rollback
kubectl rollout undo deployment/{SERVICE_NAME_LOWER}

# Docker rollback
docker run q2/{SERVICE_NAME_LOWER}:previous-version
```

### Data Recovery

**1. Database Backup Restore**
```bash
# Restore from backup
pg_restore -d {DATABASE_NAME} backup_file.sql

# Point-in-time recovery
# Follow database-specific procedures
```

**2. Message Replay**
```bash
# Replay Pulsar messages from specific timestamp
pulsar-admin topics reset-cursor persistent://public/default/{SERVICE_NAME_LOWER}-events \
  --subscription {SERVICE_NAME_LOWER}-consumer \
  --time "2024-01-01T12:00:00Z"
```

## Getting Help

### Internal Resources
- [Q2 Platform Developer Guide](../DEVELOPER_GUIDE.md)
- [Service Architecture Docs](./ARCHITECTURE.md)
- [API Documentation](http://localhost:{SERVICE_PORT}/docs)

### Contact Information
- **Team:** {TEAM_NAME}
- **Slack Channel:** #{SERVICE_NAME_LOWER}-support
- **On-call:** {ON_CALL_CONTACT}

### Creating Support Tickets

When creating a support ticket, include:

1. **Issue Description**
   - What were you trying to do?
   - What happened instead?
   - When did this start?

2. **Environment Information**
   - Service version
   - Deployment environment (dev/staging/prod)
   - Related service versions

3. **Diagnostic Information**
   - Error messages and stack traces
   - Relevant log snippets
   - Output of diagnostic commands

4. **Steps to Reproduce**
   - Exact steps to reproduce the issue
   - Sample requests/data if applicable

**Template:**
```
**Issue:** Brief description

**Environment:** 
- Service: {SERVICE_NAME} v1.0.0
- Environment: production
- Time: 2024-01-01 14:30 UTC

**Steps to Reproduce:**
1. 
2. 
3. 

**Expected Behavior:**

**Actual Behavior:**

**Error Messages:**
```

**Diagnostic Output:**
```
make debug-service SERVICE={SERVICE_NAME_LOWER}
```

---

**Last Updated:** {LAST_UPDATED}  
**Maintainers:** {MAINTAINERS}