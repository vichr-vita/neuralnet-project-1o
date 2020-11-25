import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    transform z to activations for layer l
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_inv(sig: np.ndarray) -> np.ndarray:
    """
    return z that has been transformed through sigmoid func

    loses precision for values close to 0 and 1
    """
    return np.log(sig / (1 - sig))


def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """
    return first derivative of the sigmoid function with respect to z
    """
    return sigmoid(z)*(1-sigmoid(z))


def linear(z: np.ndarray) -> np.ndarray:
    return z


def linear_prime(z: np.ndarray) -> np.ndarray:
    return np.zeros(z.shape) + 1


def reLU(z: np.ndarray) -> np.ndarray:
    return max(0.0, z)


def reLU_prime(z: np.ndarray) -> np.ndarray:
    """
    The slope for negative values is 0.0 and the slope for positive values is 1.0
    """
    return 1 if max(0.0, z) > 0 else 0

def step(z: np.ndarray) -> np.ndarray:
    return np.array([1 if x > 0 else 0 for x in z])


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_prime(z: np.ndarray) -> np.ndarray:
    return 1/np.power(np.cosh(z), 2)