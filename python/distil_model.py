import os
from load_model import load_model, Model
from modelconfigs import config_of_name
import torch
import torch.nn as nn
import torch.optim as optim
from features import Features
import numpy as np
from model_pytorch import ExtraOutputs
from sgfmetadata import SGFMetadata

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
        input_meta = input_meta,
        extra_outputs=extra_outputs,
    )

def get_input_features(features: Features):
    bin_input_data = np.random.rand(1, *features.bin_input_shape).astype(np.float32)
    global_input_data = np.random.rand(1, *features.global_input_shape).astype(np.float32)

    pos_len = features.pos_len
    bin_input_data = bin_input_data.reshape([1,pos_len,pos_len,-1])
    bin_input_data = np.transpose(bin_input_data,axes=(0,3,1,2))

    return bin_input_data, global_input_data

device = torch.device("cpu")

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model files
model_path_b6c96 = os.path.join(script_dir, "checkpoints/model_b6c96_epoch_4_s640_d640.ckpt")

model_path_b18c384 = os.path.join(script_dir, "b18c384nbt-humanv0.ckpt")

# Load the b6c96 model
model_b6c96, swa_model_b6c96, _ = load_model(
    model_path_b6c96, 
    use_swa=False,  # or True if you want to use SWA
    device=device,
    pos_len=19,  # Adjust if needed
    verbose=True
)

model_b6c96.to(device)

def generate_new_model(config="b6c96-fson-mish-rvgl-bnh"):
    model_config = config_of_name[config]
    model = Model(model_config,size)
    model.initialize()


# Load the b18c384 model
model_b18c384, swa_model_b18c384, _ = load_model(
    model_path_b18c384, 
    use_swa=False, 
    device=device,
    pos_len=19, 
    verbose=True
)

print("model config model_b6c96")
print(model_b6c96.config)

print("model config model_b18c384")
print(model_b18c384.config)
# Define a loss function (e.g., Mean Squared Error)
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

# Training loop
num_epochs = 2000
train_state = {
    "global_step_samples": 0,  # Example, replace with actual step count
}

for epoch in range(num_epochs):
    model_b6c96.train()
    model_b18c384.eval()

    for _ in range(100):  # Set num_batches to control how many batches you want per epoch
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
        inputs, input_global = get_input_features(features)
        
        # Forward pass through both models
        outputs_b6c96 = get_model_outputs(pla, sgfmeta, inputs, input_global, model_b6c96)
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

    # Print loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Define checkpoint file path
    modelname = f"model_b6c96_epoch_{epoch+1}_s{train_state['global_step_samples']}}"
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
