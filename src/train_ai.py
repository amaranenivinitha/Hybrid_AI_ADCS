import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os

from src.space_disturbances import total_disturbance_torque


# ==========================================================
# NEURAL NETWORK
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
# TRAINING DATA GENERATION
# ==========================================================

def generate_data(num_samples=60000):

    np.random.seed(42)

    X = []
    Y = []

    I = np.diag([0.02, 0.025, 0.03])

    for _ in tqdm(range(num_samples), desc="Generating data"):

        # -----------------------------------------
        # Random quaternion error vector
        # -----------------------------------------

        q_err = np.random.uniform(
            -0.3,
            0.3,
            3
        )

        # -----------------------------------------
        # Random angular velocity
        # -----------------------------------------

        w = np.random.uniform(
            -0.08,
            0.08,
            3
        )

        # -----------------------------------------
        # Fake quaternion for disturbance model
        # -----------------------------------------

        q_fake = np.array([
            1.0,
            q_err[0],
            q_err[1],
            q_err[2]
        ])

        q_fake /= np.linalg.norm(q_fake)

        # -----------------------------------------
        # REAL ORBITAL DISTURBANCES
        # -----------------------------------------

        disturbance = total_disturbance_torque(
            q_fake,
            w,
            I
        )

        # -----------------------------------------
        # Desired compensation torque
        # AI learns to cancel disturbance
        # -----------------------------------------

        target_tau = -disturbance

        # -----------------------------------------
        # Add nonlinear residual behavior
        # -----------------------------------------

        target_tau += (
            -0.03 * w
            + 0.02 * np.tanh(3 * q_err)
        )

        # -----------------------------------------
        # Save sample
        # -----------------------------------------

        state = np.concatenate([q_err, w])

        X.append(state)

        Y.append(target_tau)

    X = np.array(X)
    Y = np.array(Y)

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32)
    )


# ==========================================================
# TRAINING
# ==========================================================

def train_model():

    model = NeuralCompensator()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    epochs = 250

    print("\nGenerating realistic orbital dataset...\n")

    X, Y = generate_data()

    print(f"\nTraining on {len(X)} samples...\n")

    for epoch in range(epochs):

        optimizer.zero_grad()

        out = model(X)

        # -----------------------------------------
        # Loss function
        # -----------------------------------------

        mse_loss = ((out - Y)**2).mean()

        smoothness_loss = 1e-3 * (out**2).mean()

        loss = mse_loss + smoothness_loss

        loss.backward()

        optimizer.step()

        if (epoch + 1) % 25 == 0:

            print(
                f"Epoch {epoch+1}: "
                f"loss = {loss.item():.8f}"
            )

    # ------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------

    os.makedirs("results", exist_ok=True)

    torch.save(
        model.state_dict(),
        "results/ai_model.pt"
    )

    print("\n✅ New orbital-trained AI model saved.")
    print("✅ File: results/ai_model.pt")


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    train_model()