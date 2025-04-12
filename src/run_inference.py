import json

# TODO: Update `../requirements.txt`.
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import sys
import traceback

import numpy as np
import pandas as pd
import torch


def run_job(input_file, model):
    """
    Run the job
    """
    """Combine input file with working directory to get the full path"""
    # input_file = os.path.join(os.getcwd(), input_file)
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file: {input_file} not found")
        # Load and preprocess input data
        if input_file.endswith(".csv"):
            # Read only the feature columns (assuming labels are in the last column)
            df = pd.read_csv(input_file)
            # Remove any non-numeric columns (like labels) if present
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            input_data = df[numeric_cols].values
        elif input_file.endswith(".npy"):
            input_data = np.load(input_file)
        else:
            raise ValueError(f"Input file: {input_file} must be .csv or .npy")

        # Ensure input data is float32
        input_data = input_data.astype(np.float32)

        # If we have multiple samples, we can either:
        # 1. Process one sample at a time:
        predictions = []
        confidences = []

        for sample in input_data:
            # Add batch dimension and convert to tensor
            input_tensor = torch.tensor(sample).unsqueeze(0)

            # Make prediction
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                pred_class = torch.argmax(probabilities, dim=1)
                confidence = torch.max(probabilities, dim=1).values

                predictions.append(pred_class.item())
                confidences.append(confidence.item())

        output = {
            "predictions": predictions,
            "confidences": confidences,
        }

        return output

    except Exception as error:
        print(
            f"❌ Error running job: {error}",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc(file=sys.stderr)
        raise


def main():
    print("Starting inference...")

    input = os.environ.get("INPUT", "/app/models/processed/x_test.csv")

    # `/app` directory aligns with the `WORKDIR` specified in the `Dockerfile`
    model_path = "/app/models/asl_model_20250310_184213.pt"

    # output = {"input": input, "status": "error"}

    try:
        # TODO: Initialize `model` and `tokenizer`.
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

        output = run_job(input, model)
        output.update(
            {
                "status": "success",
            }
        )

    except Exception as error:
        print("❌ Error during processing:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        output["error"] = str(error)

    os.makedirs("/outputs", exist_ok=True)
    output_path = "/outputs/result.json"

    try:
        with open(output_path, "w") as file:
            json.dump(output, file, indent=2)
        print(
            f"✅ Successfully wrote output to {output_path}",
        )
    except Exception as error:
        print(f"❌ Error writing output file: {error}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
