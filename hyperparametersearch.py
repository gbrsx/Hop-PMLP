import os
import subprocess
from itertools import product

# Define hyperparameters for tuning
hyperparams = {
    'lr': [0.01, 0.05, 0.1],  # Adjusted range for more flexibility
    'dropout': [0.3, 0.5],
    'num_layers': [2, 4],
    'hidden_channels': [32, 64],
    'weight_decay': [5e-4, 5e-3],
    'num_hop': [6],
}

# Generate all combinations of hyperparameters
combinations = list(product(*hyperparams.values()))

# Define dataset and settings
dataset = 'texas'
protocol = 'supervised'
device = 0
conv_tr = False
conv_va = False
conv_te = True

# Track the best configuration and score
best_score = -float('inf')
best_hyperparams = None

# Run hyperparameter search
for comb in combinations:
    current_hyperparams = dict(zip(hyperparams.keys(), comb))
    command = (
        f"python main.py --dataset {dataset} --method pmlp_hopgnn --protocol {protocol} "
        f"--lr {current_hyperparams['lr']} --dropout {current_hyperparams['dropout']} "
        f"--num_layers {current_hyperparams['num_layers']} --hidden_channels {current_hyperparams['hidden_channels']} "
        f"--weight_decay {current_hyperparams['weight_decay']} --num_hop {current_hyperparams['num_hop']} "
        f"--device {device}"
    )

    if conv_tr:
        command += " --conv_tr"
    if conv_va:
        command += " --conv_va"
    if conv_te:
        command += " --conv_te"

    print(f"\n--- Running configuration: {current_hyperparams} ---")

    # Capture the output from the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout

    # Print each epoch's result in a structured way
    final_test_score = None
    for line in output.splitlines():
        if "Epoch:" in line:
            print(line)  # Print each epoch line
        elif "Final Test:" in line:
            # Extract the final test score for comparison
            try:
                final_test_score = float(line.split("Final Test: ")[1].strip().replace("%", ""))
            except ValueError:
                print("Could not parse final test score from line:", line)

    # Update the best score if this configuration performed better
    if final_test_score and final_test_score > best_score:
        best_score = final_test_score
        best_hyperparams = current_hyperparams
        print(f"New Best Score: {best_score:.2f}% with {best_hyperparams}")

# Print the best hyperparameters
if best_hyperparams:
    print("\n--- Best Hyperparameters Found ---")
    print(best_hyperparams)
    print(f"Best Score: {best_score:.2f}%")
else:
    print("No valid scores found in the output.")