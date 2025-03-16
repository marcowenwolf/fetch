#!/usr/bin/env python3

import argparse
import glob
import os
import string

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from fetch.pulsar_data import PulsarData
from fetch.model import PulsarModel

# Use GPU if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    r""" Entry point for running via command line
    Uses a pre-trained combined model to make predictions
    """
    parser = argparse.ArgumentParser(
        description="Fast Extragalactic Transient Candiate Hunter (FETCH)",
    )
    parser.add_argument(
        "-g",
        "--gpu_id",
        help="GPU ID (use -1 for CPU)",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "-c",
        "--data_dir",
        help="Directory with candidate h5s.",
        required=True,
        type=str,
        action='append'
    )
    parser.add_argument(
        "-b", "--batch_size", help="Batch size for making predictions", default=64, type=int
    )
    parser.add_argument(
        "-w", "--weights", help="Directory containing model weights", required=True
    )
    parser.add_argument(
        "-p", "--probability", help="Detection threshold", default=0.5, type=float
    )
    args = parser.parse_args()

    if args.gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"

    print(f"Using {DEVICE} for computation", flush=True)

    # Get the model and set it to eval mode
    model = PulsarModel()
    path = os.path.split(__file__)[0]
    model.load_state_dict(torch.load(f"{args.weights}/DenseNet201_DenseNet201_64.pth", weights_only=True))
    model.eval()
    model.to(DEVICE)
    
    for data_dir in args.data_dir:

        # Get all our candidate files
        cands_to_eval = glob.glob(f"{data_dir}/*h*5")

        if len(cands_to_eval) == 0:
            print(f"No candidates to evaluate in directory: {data_dir}", flush=True)
            continue

        # Setup the candidate data
        inputs = PulsarData(files=cands_to_eval)
        dataloader = DataLoader(inputs, batch_size=args.batch_size, pin_memory=True, shuffle=False)

        # Make predictions in batches
        predictions = []
        probs = []
        with torch.no_grad():
            for batch_idx, (freq_data, dm_data, labels) in enumerate(dataloader):
                freq_data = freq_data.to(DEVICE, non_blocking=True)
                dm_data = dm_data.to(DEVICE, non_blocking=True)

                predicted = model(freq_data, dm_data)

                predicted = predicted.to('cpu').numpy()
                probs.extend(predicted)
                predictions.extend(np.round(predicted >= args.probability))

        # Save the results
        print(f"Saving final results", flush=True)
        results_dict = {}
        results_dict["candidate"] = cands_to_eval
        results_dict["probability"] = probs
        results_dict["label"] = predictions

        results_file = data_dir + f"/results_full_model.csv"
        pd.DataFrame(results_dict).to_csv(results_file)
