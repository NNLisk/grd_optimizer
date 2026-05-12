
def gradient_descent(data, grad, init_params, lr=0.1, epochs=50, tolerance=1e-6):
    params = init_params[:]

    for epoch in range(epochs):
        gradients = grad(data, params)
        params = [p - lr * g for p, g in zip(params, gradients)]
    return params
