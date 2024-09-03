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

def generate_new_model(config="b6c96-fson-mish-rvgl-bnh"):
    model_config = config_of_name[config]
    model = Model(model_config, size)
    model.initialize()
    return model


parser = argparse.ArgumentParser(description="Train a Go AI model.")
parser.add_argument('--load-model', type=str, help='Path to the model to load')
args = parser.parse_args()

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

# Load the b18c384 model
model_b18c384, swa_model_b18c384, _ = load_model(
    model_path_b18c384, 
    use_swa=False, 
    device=device,
    pos_len=19, 
    verbose=False
)

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
    choice = np.random.rand() # chooise random float between 0 and 1
    cumulative_prob = 0
    for prob in moves_and_probs:
        cumulative_prob += prob[1]
        if cumulative_prob >= choice:
            prob_loc = prob[0]
            col = (prob_loc) % (19 + 1) - 1
            row = np.floor(prob_loc / (19 + 1)) - 1
            return int((size + 1) * (row + 1) + (col + 1))

def post_process_outputs(board, model, model_outputs):
    outputs = model.postprocess_output(model_outputs)
    (
        policy_logits,      # N, num_policy_outputs, move
        value_logits,       # N, {win,loss,noresult}
        td_value_logits,    # N, {long, mid, short} {win,loss,noresult}
        pred_td_score,      # N, {long, mid, short}
        ownership_pretanh,  # N, 1, y, x
        pred_scoring,       # N, 1, y, x
        futurepos_pretanh,  # N, 2, y, x
        seki_logits,        # N, 4, y, x
        pred_scoremean,     # N
        pred_scorestdev,    # N
        pred_lead,          # N
        pred_variance_time, # N
        pred_shortterm_value_error, # N
        pred_shortterm_score_error, # N
        scorebelief_logits, # N, 2 * (self.pos_len*self.pos_len + EXTRA_SCORE_DISTR_RADIUS)
    ) = (x[0] for x in outputs[0]) # N = 0

    policy0 = torch.nn.functional.softmax(policy_logits[0,:],dim=0).cpu().numpy()

    moves_and_probs0 = []
    for i in range(len(policy0)):
        move = features.tensor_pos_to_loc(i,board)
        if i == len(policy0)-1:
            moves_and_probs0.append((Board.PASS_LOC,policy0[i]))
        elif board.would_be_legal(board.pla ,move):
            moves_and_probs0.append((move,policy0[i]))
    
    return moves_and_probs0

# Training loop
num_epochs = 2000
batch_size = 32  # Define a batch size for training
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
    while move is not Board.PASS_LOC: #Play one game per epoch
        sgfmeta = {
            "inverseBRank": np.random.randint(31),
            "inverseWRank": np.random.randint(31),
            "bIsHuman": True,
            "wIsHuman": True,
            "gameIsUnrated": False,
            "gameRatednessIsUnknown": False,
            "tcIsUnknown": False,
            "tcIsByoYomi": True,
            "mainTimeSeconds": np.random.randint(3001),
            "periodTimeSeconds": np.random.randint(31),
            "byoYomiPeriods": np.random.randint(11),
            "gameDate": str(np.random.randint(200) + 1823) + "-06-01",
            "source": np.random.randint(7)
        }

        sgfmeta = SGFMetadata.of_dict(sgfmeta)
        pla = np.random.randint(2)
        
        features = Features(model_b18c384.config, model_b18c384.pos_len)
        inputs, input_global = game_state.get_input_features(features)
        
        # Forward pass through both models
        outputs_b6c96 = get_model_outputs(pla, sgfmeta, inputs, input_global, model_b6c96)
        with torch.no_grad():
            outputs_b18c384 = get_model_outputs(pla, sgfmeta, inputs, input_global, model_b18c384)
            
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
    modelname = f"model_b6c96_epoch_{epoch+1}_s{train_state['global_step_samples']}"
    checkpoint_path = os.path.join("checkpoints", modelname + ".ckpt")

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
