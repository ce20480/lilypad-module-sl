import os
import sys
import traceback
import json

import numpy as np
import torch
# import argparse


def run_job(landmarks, model):
    """
    Run the job
    """
    """Combine input file with working directory to get the full path"""
    print(type(landmarks[0]))
    print(landmarks)

    try:
        # landmarks_array = np.array(landmarks)
        # # Check if we have the right shape
        # if landmarks_array.size == 1:
        #     return {
        #         "success": False,
        #         "message": "Invalid landmarks format: Empty or single value"
        #     }
    
        # # Ensure we have the correct shape (21 landmarks with 3 coordinates each)
        # if len(landmarks) % 3 == 0:
        #     # If flat array, reshape to (n, 3)
        #     num_landmarks = len(landmarks_array) // 3
        #     landmarks_array = landmarks_array.reshape(num_landmarks, 3)
        # elif len(landmarks_array.shape) == 1:
        #     # If it's still a 1D array but not divisible by 3, we have a problem
        #     return {
        #         "success": False,
        #         "message": f"Invalid landmarks format: Cannot reshape array of size {landmarks_array.size} into landmarks"
        #     }

        # min_x = np.min(landmarks_array[:, 0])
        # max_x = np.max(landmarks_array[:, 0])
        # min_y = np.min(landmarks_array[:, 1])
        # max_y = np.max(landmarks_array[:, 1])
        
        # # Calculate scale factors
        # x_range = max_x - min_x
        # y_range = max_y - min_y
        
        # if x_range == 0 or y_range == 0:
        #     return landmarks_array
        
        # # Normalize coordinates to [0, 1] range
        # normalized_landmarks = landmarks_array.copy()
        # normalized_landmarks[:, 0] = (normalized_landmarks[:, 0] - min_x) / x_range
        # normalized_landmarks[:, 1] = (normalized_landmarks[:, 1] - min_y) / y_range

        # features = normalized_landmarks.flatten()

        features = np.array(landmarks).astype(np.float32)
        input_tensor = torch.tensor(features).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1).values

        letter_mapping = {
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

        # Get letter from prediction
        letter = letter_mapping[pred_class.item()]

        # Calculate a confidence score for evaluation (0.0-1.0)
        confidence = float(confidence.item())
        score = min(1.0, confidence / 0.7)  # Normalize to 0-1 scale

        output = {
            "letter": letter,
            "confidence": score,
            "status": "success",
        }

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


def main(input=False):
    print("Starting inference...")

    if not input:
        input = os.environ.get("INPUT")
    parsed_landmarks = json.loads(input)

    # `/app` directory aligns with the `WORKDIR` specified in the `Dockerfile`
    # model_path = "/app/models/asl_model_20250310_184213.pt"
    model_path = "models/asl_model_20250310_184213.pt"
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

        output = run_job(parsed_landmarks, model)

    except Exception as error:
        print("❌ Error during processing:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        output["error"] = str(error)

    os.makedirs("/outputs", exist_ok=True)
    output_path = "/outputs/result.json"

    try:
        with open(output_path, "w") as file:
            json.dump({"output": output, "landmarks": parsed_landmarks}, file, indent=2)
        print(
            f"✅ Successfully wrote output to {output_path}",
        )
    except Exception as error:
        print(f"❌ Error writing output file: {error}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    # for testing to see if the model is working in my local dev
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", "-i", type=str, required=True)
    # args = parser.parse_args()
    # main(args.input)
    main()
