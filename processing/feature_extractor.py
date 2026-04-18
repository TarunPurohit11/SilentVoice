import numpy as np

def extract_features(landmarks):
    if landmarks is None:
        return None
    
    return np.array(landmarks).flatten()  # shape: (63,)