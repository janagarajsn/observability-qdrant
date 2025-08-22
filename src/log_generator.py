import os
import json
import random
import uuid
import time
import logging
from datetime import datetime, timedelta

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s")
logger = logging.getLogger(__name__)

# Static log file path
STATIC_LOG_DIR = "static-logs"
if not os.path.exists(STATIC_LOG_DIR):
    os.makedirs(STATIC_LOG_DIR)

# Stream log file path
STREAM_LOG_DIR = "stream-logs"

APPLICATIONS = ["app1", "app2", "checkout", "payment", "auth"]
NAMESPACES = ["namespace-1", "namespace-2", "namespace-3", "namespace-4", "namespace-5"]
NODES = ["aks-nodepool-1", "aks-nodepool-2", "aks-nodepool-3"]

# Burst tracking
current_burst = None
burst_end_time = None

def random_embedding(dim=8):
    return [round(random.uniform(-1, 1), 3) for _ in range(dim)]

def start_burst(timestamp):
    """Start a burst pattern"""
    global current_burst, burst_end_time
    # Randomly select one of three burst types: pod crashes, scaling events, or authentication failures
    current_burst = random.choice(["pod_crash", "scale_up", "auth_failure"])
    # Set the burst to end 2-5 minutes after the current timestamp
    burst_end_time = timestamp + timedelta(minutes=random.randint(2, 5))
    # Return the selected burst type for reference
    return current_burst

def end_burst():
    """End the current burst pattern by resetting burst tracking variables"""
    global current_burst, burst_end_time
    # Reset the current burst type to None (no active burst)
    current_burst = None
    # Reset the burst end time to None (no scheduled end time)
    burst_end_time = None

def choose_log_level(cpu, memory, latency, burst=None):
    # Return ERROR level for critical conditions:
    # - Pod crash burst events
    # - High CPU usage (>90%)
    # - High memory usage (>1800MB)
    # - High latency (>700ms)
    if burst == "pod_crash" or cpu > 0.9 or memory > 1800 or latency > 700:
        return "ERROR"
    # Return WARN level for moderate issues:
    # - Scale up burst events
    # - Elevated CPU usage (>75%)
    # - Elevated memory usage (>1400MB)
    # - Elevated latency (>400ms)
    elif burst == "scale_up" or cpu > 0.75 or memory > 1400 or latency > 400:
        return "WARN"
    # Return WARN level for authentication failure burst events
    elif burst == "auth_failure":
        return "WARN"
    # Return INFO level for all other normal conditions
    return "INFO"

# Generate a log entry
def generate_log(timestamp=None):
    global current_burst, burst_end_time

    if timestamp is None:
        timestamp = datetime.utcnow()

    # Randomly start bursts
    if current_burst is None and random.random() < 0.01:
        start_burst(timestamp)

    # End burst if over
    if current_burst and timestamp > burst_end_time:
        end_burst()

    namespace = random.choice(NAMESPACES)
    app = random.choice(APPLICATIONS)
    pod = f"{app}-pod-{random.randint(1,5)}"
    node = random.choice(NODES)

    cpu = round(random.uniform(0.05, 0.98), 2)
    memory = random.randint(200, 2000)
    latency = random.randint(50, 800)

    level = choose_log_level(cpu, memory, latency, current_burst)

    # Message generation
    if current_burst == "pod_crash":
        message = f"{pod} crashed: PodCrashLoopBackOff detected in {namespace}"
    elif current_burst == "scale_up":
        message = f"Cluster scale-up event: Added {random.randint(1,3)} nodes to {namespace}"
    elif current_burst == "auth_failure":
        user_id = f"user{random.randint(1,1000)}"
        message = f"Repeated failed login attempts for user {user_id}"
    else:
        if level == "ERROR":
            message = f"Database connection timeout after 30s while querying orders table in {namespace}"
        elif level == "WARN":
            message = f"CPU usage high ({cpu*100:.0f}%) on {node} in {namespace}"
        else:
            message = f"Deployment rollout successful: {app} v1.2.{random.randint(0,9)} on {namespace}"

    return {
        "timestamp": timestamp.isoformat() + "Z",
        "namespace": namespace,
        "pod": pod,
        "container": f"{app}-container",
        "application": app,
        "cluster": "aks-demo-cluster",
        "node": node,
        "hostIP": f"40.76.{random.randint(100,255)}.{random.randint(1,255)}",
        "podIP": f"192.168.{random.randint(0,255)}.{random.randint(1,255)}",
        "traceId": str(uuid.uuid4()),
        "cpuUsage": cpu,
        "memoryUsageMB": memory,
        "latencyMs": latency,
        "screenshotEmbedding": random_embedding(),
        "logEmbedding": random_embedding(),
        "level": level,
        "message": message
    }

# Generate statis logs for the input Date string
def generate_static_logs_for_day(date_str, num_logs=1000):
    logs = []
    date = datetime.strptime(date_str, "%Y-%m-%d")
    for i in range(num_logs):
        seconds_offset = int((86400 / num_logs) * i)
        log_time = date + timedelta(seconds=seconds_offset)
        logs.append(generate_log(timestamp=log_time))

    output_file = os.path.join(STATIC_LOG_DIR, f"aks_logs_{date_str}.json")
    with open(output_file, "w") as f:
        json.dump(logs, f, indent=2)
    logger.info(f"Generated {len(logs)} logs -> {output_file}")

# Continuously generate logs in real-time
def stream_logs(interval_seconds=2, output_file=STREAM_LOG_DIR + "/stream_logs.jsonl"):
    from pathlib import Path
    """Continuously generate logs in real-time and append to a JSONL file"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        while True:
            log = generate_log()
            json_line = json.dumps(log)
            print(json_line)           # optional: still print to console
            f.write(json_line + "\n")  # append to file for ingestion
            f.flush()                  # ensure it's immediately written
            time.sleep(interval_seconds)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python log_generator.py <date|stream> [num_logs|interval_seconds]")
        sys.exit(1)

    if sys.argv[1] == "stream":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        stream_logs(interval_seconds=interval)
    else:
        date_str = sys.argv[1]
        num_logs = int(sys.argv[2]) if len(sys.argv) > 2 else 500
        generate_static_logs_for_day(date_str, num_logs=num_logs)
