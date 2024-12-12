import torch

# Load the saved model
saved_model_path = "swin_tiny_patch4_window7_224.pth"
swin_model = torch.load(saved_model_path)

# Extract the parameters from the 'model' key
swin_model_parameters = swin_model['model']

# Add prefix to each key
prefixed_parameters = {}
prefix = "model.0."
for key, value in swin_model_parameters.items():
    new_key = prefix + key  # Add the prefix
    prefixed_parameters[new_key] = value

# Update the 'model' key with the modified parameters
swin_model['model'] = prefixed_parameters

# Save the modified model back to a file
modified_model_path = "swin_tiny_patch4_window7_224_prefixed.pth"
torch.save(swin_model, modified_model_path)

print(f"Modified model saved to {modified_model_path}")
