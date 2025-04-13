import subprocess
import os
import json

# Replace this with your real private key for actual testing
landmarks = [0.5, 0.3, 0.0, 0.53, 0.28, 0.02, 0.56, 0.25, 0.04, 0.59, 0.22, 0.05, 0.61, 0.19, 0.06, 0.49, 0.22, 0.02, 0.48, 0.18, 0.03, 0.47, 0.15, 0.04, 0.46, 0.12, 0.05, 0.46, 0.23, 0.01, 0.45, 0.19, 0.02, 0.44, 0.16, 0.03, 0.43, 0.13, 0.04, 0.43, 0.24, 0.01, 0.42, 0.21, 0.02, 0.41, 0.18, 0.03, 0.4, 0.15, 0.04, 0.4, 0.25, 0.01, 0.38, 0.23, 0.02, 0.36, 0.21, 0.03, 0.34, 0.19, 0.04]

# Serialize landmarks into a string as expected by the Lilypad module
# landmarks_json = json.loads(landmarks)

# Prepare the environment variables
env_vars = os.environ.copy()

# Lilypad command
command = [
    "lilypad",
    "run",
    "github.com/ce20480/lilypad-module-sl:1cd44ef8f155c8e6a379250d972de71ce77b6ea2",
    "-i",
    f'INPUT="{landmarks}"'
]

try:
    # Run the command and capture output
    result = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
        env=env_vars
    )

    print("✅ Lilypad command succeeded.")
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)

except subprocess.CalledProcessError as e:
    print("❌ Lilypad command failed.")
    print("--- STDOUT ---")
    print(e.stdout)
    print("--- STDERR ---")
    print(e.stderr)
except FileNotFoundError:
    print("❌ Lilypad CLI not found. Is it installed and available on your PATH?")
