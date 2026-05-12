
def lin_reg(x, params):
    w, b = params
    return w * x + b

def linear_gradients(data, params):
    n = len(data)
    dw = sum(2*(lin_reg(x, params) - y) * x for x, y in data) / n
    db = sum(2*(lin_reg(x, params) - y) for x, y in data) / n

    return [dw, db]

