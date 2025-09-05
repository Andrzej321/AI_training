import pandas as pd

# defining the hyperparameter set as DataFrame

param_sets = [
    {"ID": 1, "Model": "RNN", "Seq_len": 20, "Hidden_size": 64, "Layers": 1,
     "LR": 1e-3, "Batch": 64, "Dropout": 0.2, "Optimizer": "Adam"},

    {"ID": 2, "Model": "LSTM", "Seq_len": 30, "Hidden_size": 128, "Layers": 2,
     "LR": 5e-4, "Batch": 128, "Dropout": 0.3, "Optimizer": "RMSProp"},

    {"ID": 3, "Model": "GRU", "Seq_len": 50, "Hidden_size": 64, "Layers": 3,
     "LR": 1e-3, "Batch": 64, "Dropout": 0.1, "Optimizer": "Adam"},
]

# Convert to DataFrame
df = pd.DataFrame(param_sets)

# Show the DataFrame
print(df)