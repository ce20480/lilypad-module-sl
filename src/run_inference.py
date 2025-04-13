# standard library imports
import os
import sys
import traceback
import json

# third party imports
import numpy as np
import torch

LETTER_MAPPING = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}

def run_job(landmarks, model):
    """
    Run the job
    """
    """Combine input file with working directory to get the full path"""
    try:
        if isinstance(landmarks, str):
            landmarks = json.loads(landmarks)

        features = np.array(landmarks).astype(np.float32)
        input_tensor = torch.tensor(features).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values

        print("Predicted class:", pred_class.item(), file=sys.stderr, flush=True)
        # Get letter from prediction
        letter = LETTER_MAPPING[pred_class.item()]


        # Calculate a confidence score for evaluation (0.0-1.0)
        confidence = float(confidence.item())
        score = min(1.0, confidence / 0.7)  # Normalize to 0-1 scale

        output = {
            "letter": letter,
            "confidence": score,
            "status": "success",
        }
        print("Output:", output, file=sys.stderr, flush=True)

    except Exception as error:
        print(
            f"❌ Error running job: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        output = {
            "status": "error",
            "message": str(error),
        }

    return output


def main():
    print("Starting inference...")

    # Get input from environment variable
    input = os.environ.get("INPUT")
    print("Input:", input, file=sys.stderr, flush=True)
    if isinstance(input, str):
        landmarks = json.loads(input)
    else:
        landmarks = input

    # `/app` directory aligns with the `WORKDIR` specified in the `Dockerfile`
    model_path = "/app/models/asl_model_20250310_184213.pt"
    output = {"input": input, "status": "error"}

    try:
        # Load model checkpoint with weights_only=True
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)

        # Create model instance
        model = torch.nn.Sequential(
            *[
                torch.nn.Linear(
                    checkpoint["input_size"], checkpoint["hidden_layers"][0][0]
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    checkpoint["hidden_layers"][0][0], checkpoint["hidden_layers"][1][0]
                ),
                torch.nn.ReLU(),
                torch.nn.Linear(
                    checkpoint["hidden_layers"][1][0], checkpoint["output_size"]
                ),
            ]
        )

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        output = run_job(landmarks, model)

    except Exception as error:
        print("❌ Error during processing:", file=sys.stderr, flush=True)
        print("Error for input:", output["input"], file=sys.stderr, flush=True)

        traceback.print_exc(file=sys.stderr)
        output["error"] = str(error)

    os.makedirs("/outputs", exist_ok=True)
    output_path = "/outputs/result.json"

    try:
        with open(output_path, "w") as file:
            json.dump({"output": output, "landmarks": landmarks}, file, indent=2)
        print(
            f"✅ Successfully wrote output to {output_path}",
        )
    except Exception as error:
        print(f"❌ Error writing output file: {error}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
