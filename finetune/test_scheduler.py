import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, ExponentialLR, SequentialLR
import matplotlib.pyplot as plt

# Dummy model
model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Scheduler setup
warmup_steps = 250
total_steps = 1250
gamma = 0.999  # for exponential decay

# Linear warmup: LR from 0.1 * base_lr to base_lr
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

# Exponential decay: starts after warmup
exp_decay_scheduler = ExponentialLR(optimizer, gamma=gamma)

# Combine both using SequentialLR
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, exp_decay_scheduler],
    milestones=[warmup_steps]
)

# Track learning rates
lrs = []

for step in range(total_steps):
    # Simulate a training step
    optimizer.step()
    scheduler.step()

    # Log the current LR (assumes one param group)
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)

# Plotting
plt.figure(figsize=(8, 4))
plt.plot(range(total_steps), lrs, label="Learning Rate")
plt.axvline(warmup_steps, color='red', linestyle='--', label="Warmup Ends")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Linear Warmup + Exponential Decay")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lr.png")
