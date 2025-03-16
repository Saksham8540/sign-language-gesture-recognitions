import os
import numpy as np

# Load existing data
data = []
labels = []

gesture_classes = ['thumbs_up', 'peace_sign', 'fist', 'open_hand', 'pointing']# Add new gestures here

for idx, gesture in enumerate(gesture_classes):
    gesture_path = f'data/{gesture}'
    if os.path.exists(gesture_path):
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)
            sample = np.load(file_path)
            data.append(sample)
            labels.append(idx)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Save as dataset
np.save('data/X.npy', X)
np.save('data/y.npy', y)
