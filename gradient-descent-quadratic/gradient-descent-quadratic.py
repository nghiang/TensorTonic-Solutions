def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    def derivative(x):
        return 2 * a * x + b
    for i in range(0, steps):
        x0= x0 - lr * derivative(x0)
    return x0