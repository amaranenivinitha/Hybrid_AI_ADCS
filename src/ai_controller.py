import torch
import torch.nn as nn
import numpy as np


# ==========================================================
# UPDATED NEURAL NETWORK
# ==========================================================

class NeuralCompensator(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(6, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3)

        )

    def forward(self, x):

        return self.net(x)


# ==========================================================
# AI CONTROLLER
# ==========================================================

class AIController:

    def __init__(self, model_path):

        self.model = NeuralCompensator()

        state_dict = torch.load(
            model_path,
            map_location=torch.device("cpu")
        )

        self.model.load_state_dict(state_dict)

        self.model.eval()

        print(f"✅ Loaded AI model from {model_path}")

    # ------------------------------------------------------
    # COMPUTE AI TORQUE
    # ------------------------------------------------------

    def compute(self, q_err, w):

        x = np.concatenate([q_err, w])

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            tau = self.model(x).numpy()[0]

        return tau