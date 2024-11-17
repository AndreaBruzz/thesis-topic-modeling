import numpy as np

def normalize(v):
    return v / np.linalg.norm(v)

def join(v1, v2):
    v1 = normalize(np.array(v1))
    v2 = normalize(np.array(v2))
    
    a2 = np.dot(v1, v2)
    
    b2 = np.sqrt(1 - a2**2)
    if b2 == 0:
        raise ValueError("Vectors are parallel; b^2 cannot be zero.")
    
    u1 = v1
    u2 = (v2 - a2 * v1) / b2

    return u1, u2
