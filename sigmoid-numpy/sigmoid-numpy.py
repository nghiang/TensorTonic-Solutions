import numpy as np
import math
import numbers
def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    return 1/(1+pow(math.e, -x))