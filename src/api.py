# Expose generate-logs and ingest-logs as fast api
import logging
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from log_generator import generate_logs_for_day
from ingest_logs import ingest_logs

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Observability AI POC")

# Enable CORS since we are calling this API from a frontend application running separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate-logs", description="Generate sample logs for a specific day")
def generate_logs_api(input_date: str, num_logs: int):
    try:
        file_path = generate_logs_for_day(input_date, num_logs)
        return JSONResponse(status_code=200, content={"message": "Logs generated successfully", "file_path": file_path})
    except Exception as e:
        logger.error(f"Error generating logs: {e}")
        return JSONResponse(status_code=500, content={"message": "Error generating logs"})

@app.post("/ingest-logs", description="Ingest logs into a Qdrant collection")
def ingest_logs_api(collection_name: str, background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(ingest_logs, collection_name)
        return JSONResponse(status_code=202, content={"message": "Ingestion started", "collection_name": collection_name})
    except Exception as e:
        logger.error(f"Error starting ingestion: {e}")
        return JSONResponse(status_code=500, content={"message": "Error starting ingestion"})
