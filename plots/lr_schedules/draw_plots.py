import numpy as np
import matplotlib.pyplot as plt

# Parameters
INITIAL_LEARNING_RATE = 0.0003
DECAY_RATE = 0.1

# Function definition
def exponential_schedule(progress: float):
    progress = max(progress, 0)
    return INITIAL_LEARNING_RATE * (progress ** DECAY_RATE)

def linear_schedule(progress):
    progress = max(progress, 0)
    return progress * INITIAL_LEARNING_RATE

def constant_schedule(progress):
    return INITIAL_LEARNING_RATE

# Generate progress values (from 1 to 0) and corresponding learning rates
progress_values = np.linspace(1, 0, 1000)
learning_rates = [exponential_schedule(progress) for progress in progress_values]

# Plot exponential schedule
plt.figure(figsize=(5, 4))
plt.plot(progress_values, learning_rates, color='blue', lw=3)
plt.title('Exponential Learning Rate Schedule', fontsize=14)
plt.xlabel('Remaining Progress', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('exponential_schedule.png')
plt.close()

# Plot linear schedule
learning_rates = [linear_schedule(progress) for progress in progress_values]

plt.figure(figsize=(5, 4))
plt.plot(progress_values, learning_rates, color='blue', lw=3)
plt.title('Linear Learning Rate Schedule', fontsize=14)
plt.xlabel('Remaining Progress', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.grid(True)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('linear_schedule.png')
plt.close()

# Plot constant schedule
learning_rates = [constant_schedule(progress) for progress in progress_values]

plt.figure(figsize=(5, 4))
plt.plot(progress_values, learning_rates, color='blue', lw=3)
plt.title('Constant Learning Rate Schedule', fontsize=14)
plt.xlabel('Remaining Progress', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.ylim(0 - INITIAL_LEARNING_RATE*0.05, INITIAL_LEARNING_RATE + INITIAL_LEARNING_RATE*0.05)
plt.gca().invert_xaxis()
plt.grid(True)
plt.tight_layout()
plt.savefig('constant_schedule.png')
