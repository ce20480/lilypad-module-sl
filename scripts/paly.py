from fastapi import FastAPI, Body, BackgroundTasks, HTTPException
from pydantic import BaseModel
import subprocess
import uuid
import os
import json
import re
app = FastAPI()

class LandmarkRequest(BaseModel):
    landmarks: list[float]

# In-memory store of job statuses for the example
JOBS = {}

@app.post("/evaluate_async")
def evaluate_landmarks_async(data: LandmarkRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status": "pending", "result": None}
    
    # Convert to JSON
    landmarks_json = json.dumps(data.landmarks)
    web3_private_key = os.environ.get("WEB3_PRIVATE_KEY", "")

    command = [
        "lilypad",
        "run",
        "github.com/ce20480/lilypad-module-sl:1cd44ef8f155c8e6a379250d972de71ce77b6ea2",
        "--web3-private-key",
        web3_private_key,
        "-i",
        f'INPUT="{landmarks_json}"'
    ]

    # Add the long-running job as a background task
    background_tasks.add_task(run_lilypad_job, command, job_id)
    
    return {"job_id": job_id, "status": "started"}

def run_lilypad_job(command, job_id):
    try:
        # Run the Lilypad module in a subprocess
        result = subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
        )
        
        # If we reach here, the Lilypad CLI exited with code 0 (no subprocess.CalledProcessError).
        # Let's parse its stdout to find the path line.
        # The JS snippet uses lines[-4], so we’ll do the same here:
        lines = result.stdout.strip().split("\n")
        # if len(lines) < 4:
        #     # If there aren't enough lines, fallback to the last line
        #     path_line = lines[-1].strip()
        # else:
        #     path_line = lines[-4].strip()
        path_line = None
        count = 0

        # We'll search from the bottom up, because "open /tmp/lilypad/..." usually appears last
        for line in reversed(lines):
            line_stripped = line.strip()
            if "open /tmp/lilypad/data/downloaded-files" in line_stripped:
                path_line = line_stripped
                break
            count += 1
            if count > 10:
                break

        if not path_line:
            # No matching line found – handle the error
            # e.g., raise an exception or store a failure in JOBS
            pattern = r"open\s+(/tmp/lilypad/data/downloaded-files/\S+)"
            match = re.search(pattern, result.stdout)
            if match:
                dir_path = match.group(1)  # e.g. "/tmp/lilypad/data/downloaded-files/QmeXn..."
                file_path = os.path.join(dir_path, "outputs", "result.json")
            else:
                # handle error
                pass
        else:
        
            # The line might have something like: "open /tmp/lilypad/data/downloaded-files/xyz..."
            path_line = path_line.replace("open ", "")
        
            # Now we append /outputs/result.json to get the full path
            file_path = os.path.join(path_line, "outputs", "result.json")
        
        if not os.path.exists(file_path):
            # If the file doesn't exist, treat it as an error
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["result"] = {
                "status": "error",
                "message": f"Could not find result.json at {file_path}"
            }
            return
        
        # Read and parse the result.json
        with open(file_path, "r") as f:
            file_data = json.load(f)
        
        # The result.json might look like:
        #   {"status": "error", "message": "..."}  OR
        #   {"output": {...}, "landmarks": [...]} (with "status": "success" inside the "output" dict)
        
        if file_data.get("status") == "error":
            # If the JSON indicates an error
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["result"] = file_data
        else:
            # Otherwise assume success
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["result"] = file_data

    except subprocess.CalledProcessError as e:
        # If the subprocess returns a nonzero exit code, we store the error
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["result"] = {
            "status": "error",
            "message": e.stderr or e.stdout
        }

@app.get("/job_status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]