from optimizer import gradient_descent
from reg import lin_reg, linear_gradients


if __name__ == "__main__":
    data = [(1, 3), (2, 5), (3, 7), (4, 9), (5, 11)]

    init_params = [0.0, 0.0] 

    print("Input data:", data)
    print("Initial params (w, b):", init_params)

    result = gradient_descent(data, linear_gradients, init_params, lr=0.01, epochs=50)

    print("Learned params (w, b):", [round(p, 4) for p in result])
    print(f"y = {round(result[0], 4)}x + {round(result[1], 4)}")