import numpy as np

# Number of Instance
SIZE = 100
# X range [-5, 5]
RANGE = 5.0
# Noise
NOISE = 0.05
# FILE NAME
FILE_NAME = "data.csv"

# Initiate Random Weights
w1 = np.random.normal(0.0, 1.0, (3, 2))
w2 = np.random.normal(0.0, 1.0, (2, 3))

# Initiate Random X
x = np.random.random((SIZE, 3)) * 2 * RANGE - RANGE

# Compute Prediction
H = np.dot(x, w1)
F = np.dot(H, w2)

y = np.sum(F, axis=-1)

# Add noise
noise = np.random.normal(0.0, NOISE, (SIZE, ))

y += noise

print("X Shape:", x.shape)
print("Y Shape:", y.shape)

# Save
with open("data.csv", "w") as f:
    f.write(",X1,X2,X3,Y\n")

    for i in range(len(x)):
        f.write("%d,%.4f,%.4f,%.4f,%.4f\n" % (i, x[i, 0], x[i, 1], x[i, 2], y[i]))

