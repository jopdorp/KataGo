import os
from load_model import load_model, Model
from modelconfigs import config_of_name
from board import Board
import torch
import torch.nn as nn
import torch.optim as optim
from features import Features
import numpy as np
from model_pytorch import ExtraOutputs
from sgfmetadata import SGFMetadata
from gamestate import GameState
import argparse

parser = argparse.ArgumentParser(description="Train a Go AI model.")
parser.add_argument('--load-model', type=str, help='Path to the model to load')
parser.add_argument('--model-config', type=str, help='Model config')
model_config = "b6c96-fson-mish-rvgl-bnh-meta"

args = parser.parse_args()

if args.model_config:
    model_config = model_config

size = 19

def get_model_outputs(pla, sgfmeta, bin_input_data, global_input_data, model):
    input_meta = None
    if sgfmeta is not None:
        metarow = sgfmeta.get_metadata_row(nextPlayer=pla, boardArea=size*size)
        input_meta = torch.tensor(metarow, dtype=torch.float32, device=model.device)
        input_meta = input_meta.reshape([1,-1])

    extra_outputs = ExtraOutputs([])

    return model(
        torch.tensor(bin_input_data, dtype=torch.float32, device=model.device),
        torch.tensor(global_input_data, dtype=torch.float32, device=model.device),
        input_meta=input_meta,
        extra_outputs=extra_outputs,
    )

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

model_path_b18c384 = os.path.join(script_dir, "b18c384nbt-humanv0.ckpt")

def generate_new_model():
    config = config_of_name[model_config]
    model = Model(config, size)
    model.initialize()
    return model

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model loading or generation
if args.load_model:
    print(f"Loading model from {args.load_model}")
    model_b6c96, swa_model_b6c96, _ = load_model(
        args.load_model,
        use_swa=False,  # or True if you want to use SWA
        device=device,
        pos_len=size,
        verbose=False
    )
else:
    print("No model path provided. Generating a new model.")
    model_b6c96 = generate_new_model()
    model_b6c96.to(device)

def print_model_dtypes(model, model_name="model"):
    print(f"Data types for {model_name}:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, dtype: {param.dtype}")


# Load the b18c384 model
model_b18c384, swa_model_b18c384, _ = load_model(
    model_path_b18c384, 
    use_swa=False, 
    device=device,
    pos_len=19, 
    verbose=False
)

print_model_dtypes(model_b18c384, "model_b18c384")

print("model config model_b6c96")
print(model_b6c96.config)

print("model config model_b18c384")
print(model_b18c384.config)

# Define a new loss function (Smooth L1 Loss)
criterion = nn.MSELoss()

# Define an optimizer
optimizer = optim.Adam(model_b6c96.parameters(), lr=1e-4)

def flatten_all_tensors(output):
    if isinstance(output, (tuple, list)):
        flattened_tensors = []
        for item in output:
            flattened_tensors.append(flatten_all_tensors(item))
        # Concatenate all the flattened tensors into a single tensor
        return torch.cat(flattened_tensors, dim=1)  # Concatenate along the second dimension
    elif isinstance(output, torch.Tensor):
        return output.flatten(start_dim=1)  # Flatten only from the second dimension onward
    else:
        raise TypeError(f"Unexpected type {type(output)} encountered in model output.")

# Function to save checkpoint
def save_checkpoint(model_state_dict, swa_model_state_dict, optimizer_state_dict, metrics_obj_state_dict, running_metrics, train_state, last_val_metrics, path):
    state_dict = {
        "model": model_state_dict,
        "optimizer": optimizer_state_dict,
        "metrics": metrics_obj_state_dict,
        "running_metrics": running_metrics,
        "train_state": train_state,
        "last_val_metrics": last_val_metrics,
        "config": model_b6c96.config,  # assuming the model has a 'config' attribute
    }

    if swa_model_state_dict is not None:
        state_dict["swa_model"] = swa_model_state_dict

    torch.save(state_dict, path + ".tmp")
    os.replace(path + ".tmp", path)

def choose_move(moves_and_probs):
    if not moves_and_probs:
        raise ValueError("moves_and_probs is empty, cannot choose a move.")

    choice = np.random.rand()  # choose random float between 0 and 1
    cumulative_prob = 0
    
    for prob in moves_and_probs:
        cumulative_prob += prob[1]
        if cumulative_prob >= choice:
            prob_loc = prob[0]
            col = (prob_loc) % (19 + 1) - 1
            row = np.floor(prob_loc / (19 + 1)) - 1
            return int((size + 1) * (row + 1) + (col + 1))

    # Fallback: if no move is selected (cumulative_prob < 1.0 due to rounding errors), return the last move
    prob_loc = moves_and_probs[-1][0]
    col = (prob_loc) % (19 + 1) - 1
    row = np.floor(prob_loc / (19 + 1)) - 1
    return int((size + 1) * (row + 1) + (col + 1))


# Training loop
num_epochs = 8000
train_state = {
    "global_step_samples": 0,  # Example, replace with actual step count
}

def new_game():
    return GameState(size, GameState.RULES_JAPANESE)

model_b6c96.train()
model_b18c384.eval()

for epoch in range(num_epochs):
    game_state = new_game()
    move = 1
    
    time_known = np.random.choice([True, False])
    periods = np.random.randint(5)
    source = np.random.randint(7)

    b_rank = np.random.randint(30)
    w_rank = np.random.randint(30)

    if source >= 5:
        time_known = False
        b_rank = np.random.randint(6)
        w_rank = np.random.randint(6)
    

    sgfmeta = {
        "inverseBRank": np.random.randint(30),
        "inverseWRank": np.random.randint(30),
        "bIsHuman": b_rank != 0,
        "wIsHuman": w_rank != 0,
        "gameIsUnrated": False,
        "gameRatednessIsUnknown": source != 2,
        "tcIsUnknown": not time_known,
        "tcIsByoYomi": time_known,
        "mainTimeSeconds": np.random.randint(1800) if time_known else 0,
        "periodTimeSeconds": np.random.randint(31) if (time_known and periods) else 0,
        "byoYomiPeriods": periods if time_known else 0,
        "gameDate": str(np.random.randint(200) + 1823) + "-06-01",
        "canadianMoves": 0,
        "source": source
    }
    
    while move is not Board.PASS_LOC: #Play one game per epoch
        sgfmeta = SGFMetadata.of_dict(sgfmeta)
        
        features = Features(model_b18c384.config, model_b18c384.pos_len)
        inputs, input_global = game_state.get_input_features(features)
        
        # Forward pass through both models
        outputs_b6c96 = get_model_outputs(game_state.board.pla, sgfmeta, inputs, input_global, model_b6c96)
        with torch.no_grad():
            outputs_b18c384 = get_model_outputs(game_state.board.pla, sgfmeta, inputs, input_global, model_b18c384)
            
        outputs_b6c96 = flatten_all_tensors(outputs_b6c96)
        outputs_b18c384 = flatten_all_tensors(outputs_b18c384)
        
        # Calculate loss
        loss = criterion(outputs_b6c96, outputs_b18c384)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update train_state (for example, by incrementing steps and rows)
        train_state["global_step_samples"] += 1  # Example update

        moves_and_probs = game_state.get_model_outputs(model_b18c384, sgfmeta)["moves_and_probs0"]
        move = choose_move(moves_and_probs)
        game_state.play(game_state.board.pla, move)

    # Print loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Define checkpoint file path
    modelname = f"{model_config}{epoch+1}_s{train_state['global_step_samples']}"
    checkpoint_path = os.path.join("checkpoints", modelname + ".ckpt")

    if epoch % 20 == 0:
        # Save the checkpoint
        save_checkpoint(
            model_b6c96.state_dict(),
            None,  # Replace with swa_model_state_dict if using SWA
            optimizer.state_dict(),
            {},  # Replace with metrics_obj_state_dict if using
            {},
            train_state,
            {},
            checkpoint_path
        )
        print(f"Checkpoint saved to {checkpoint_path}")

