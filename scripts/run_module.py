import argparse
import os
import subprocess
import sys

from config.constants import DOCKER_REPO, GITHUB_REPO, GITHUB_TAG


def run_module():
    DEMONET_PRIVATE_KEY = os.environ.get("DEMONET_PRIVATE_KEY")
    WEB3_PRIVATE_KEY = os.environ.get("WEB3_PRIVATE_KEY")


    parser = argparse.ArgumentParser(
        description="Run the Lilypad module with specified input."
    )

    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default=None,
        help="The input to be processed by the Lilypad module.",
    )

    parser.add_argument(
        "--local",
        action="store_true",
        help="Run the Lilypad module Docker image locally.",
    )

    parser.add_argument(
        "--demonet",
        action="store_true",
        help="Test the Lilypad module Docker image on Lilypad's Demonet.",
    )

    args = parser.parse_args()

    local = args.local
    demonet = args.demonet

    output_dir = os.path.abspath("./outputs")

    if local:
        command = [
            "docker",
            "run",
            "-e",
            f"INPUT={args.input}",
            "-v",
            f"{output_dir}:/outputs",
            f"{DOCKER_REPO}:latest",
        ]
    elif demonet:
        command = [
            "lilypad",
            "run",
            "--network",
            "demonet",
            f"{GITHUB_REPO}:{GITHUB_TAG}",
            "--web3-private-key",
            DEMONET_PRIVATE_KEY,
            "-i",
            f'INPUT="{args.input}"',
        ]
    else:
        command = [
            "lilypad",
            "run",
            f"{GITHUB_REPO}:{GITHUB_TAG}",
            "--web3-private-key",
            WEB3_PRIVATE_KEY,
            "-i",
            f'INPUT="{args.input}"',
        ]

    try:
        print("Executing Lilypad module...")
        result = subprocess.run(command, check=True, text=True)
        print("✅ Lilypad module executed successfully.")
        print(f"👉 {output_dir}/result.json")
        return result
    except subprocess.CalledProcessError as error:
        print(
            f"❌ Error: Module execution failed. {error}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    run_module()
