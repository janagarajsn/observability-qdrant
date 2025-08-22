import os
import json
import random
import datetime
import logging
from pathlib import Path

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
logger = logging.getLogger(__name__)

# Predefined incident templates
INCIDENT_TEMPLATES = [
    {
        "title": "Database connection timeout",
        "details": "Application could not connect to MySQL due to high latency.",
        "impact": "Login API was unavailable for ~5 minutes",
        "resolution": "Increased connection pool size and restarted DB service",
        "tags": ["database", "mysql", "api"]
    },
    {
        "title": "Pod CrashLoopBackOff",
        "details": "Kubernetes pod for user-service entered CrashLoop due to OOMKilled.",
        "impact": "User profiles were inaccessible",
        "resolution": "Increased memory limits and redeployed pod",
        "tags": ["kubernetes", "memory", "oom"]
    },
    {
        "title": "Network partition",
        "details": "One AZ lost connectivity to internal load balancer.",
        "impact": "Services in that AZ were unreachable",
        "resolution": "Traffic was shifted to healthy AZs while networking was restored",
        "tags": ["network", "az", "loadbalancer"]
    },
    {
        "title": "SSL Certificate Expired",
        "details": "Frontend services failed due to expired TLS certificate.",
        "impact": "All HTTPS traffic was rejected",
        "resolution": "Certificate was renewed and deployed",
        "tags": ["security", "ssl", "cert"]
    },
    {
        "title": "High CPU usage on API Gateway",
        "details": "API Gateway instances reached 95% CPU utilization due to traffic surge.",
        "impact": "Requests were throttled and response times increased",
        "resolution": "Auto-scaling group was expanded",
        "tags": ["cpu", "gateway", "scaling"]
    },
]

INCIDENT_DIR = "incidents"
if not os.path.exists(INCIDENT_DIR):
    os.makedirs(INCIDENT_DIR)

def generate_incident(i: int):
    """Generate a synthetic incident from templates."""
    tmpl = random.choice(INCIDENT_TEMPLATES).copy()
    incident = {
        "id": f"INC{i:09d}",
        "title": tmpl["title"],
        "details": tmpl["details"],
        "impact": tmpl["impact"],
        "resolution": tmpl["resolution"],
        "tags": tmpl["tags"],
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    return incident

def generate_batch(num=10, date_str=None):
    """Generate a batch of incidents and save to a daily file."""
    # Use today's date if not provided
    if date_str is None:
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    
    output_file = os.path.join(INCIDENT_DIR, f"incidents_{date_str}.json")
    with open(output_file, "w") as f:
        for i in range(1, num + 1):
            inc = generate_incident(i)
            f.write(json.dumps(inc) + "\n")

    logging.info(f"Generated {num} incidents for {date_str} at {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incident Generator (daily files)")
    parser.add_argument("--num", type=int, default=10, help="Number of incidents to generate")
    parser.add_argument("--date", type=str, help="Date for file name (YYYY-MM-DD)")
    args = parser.parse_args()

    generate_batch(args.num, date_str=args.date)
