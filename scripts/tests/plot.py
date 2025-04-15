import matplotlib.pyplot as plt

# Data
count = [10, 110, 210, 310, 410, 510, 610, 710, 810, 910]
tensor = [0.781, 0.733, 0.750, 0.750, 0.733, 0.717, 0.733, 0.718, 0.733, 0.734]
oop = [0.0, 0.031, 0.031, 0.047, 0.062, 0.078, 0.108, 0.125, 0.141, 0.187]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(count, tensor, marker='o', linestyle='-', label='Tensor')
plt.plot(count, oop, marker='o', linestyle='-', label='OOP')

# Adding chart labels and title
plt.xlabel('Count')
plt.ylabel('Time (s)')
plt.title('Performance Comparison: Tensor vs OOP')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()