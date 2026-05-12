import ctypes

from optimizer import gradient_descent


lib = ctypes.CDLL("./c_ml/gradients.so")
lib.compute_dw.restype = ctypes.c_double
lib.compute_db.restype = ctypes.c_double

def lin_reg(x, params):
    w, b = params
    return w * x + b


def linear_gradients(data, params):

    # derivatives of the loss function x² for all datapoints
    n = len(data)
    dw = sum(2*(lin_reg(x, params) - y) * x for x, y in data) / n
    db = sum(2*(lin_reg(x, params) - y) for x, y in data) / n

    return [dw, db]

def linear_gradients_c(data, params):
    w, b = params
    n = len(data)
    xs = (ctypes.c_double * n)(*[x for x, y in data])
    ys = (ctypes.c_double * n)(*[y for x, y in data])
    dw = lib.compute_dw(xs, ys, ctypes.c_double(w), ctypes.c_double(b), n)
    db = lib.compute_db(xs, ys, ctypes.c_double(w), ctypes.c_double(b), n)
    return [dw, db]



if __name__ == "__main__":
    data = [(1, 3), (2, 5), (3, 7), (4, 9), (5, 11)]
    init_params = [0.0, 0.0]

    print("Input data:", data)
    print("Initial params (w, b):", init_params)

    result = gradient_descent(data, linear_gradients_c, init_params, lr=0.01, epochs=50)

    print("Learned params (w, b):", [round(p, 4) for p in result])
    print(f"y = {round(result[0], 4)}x + {round(result[1], 4)}")
