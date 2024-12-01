import numpy as np
import math

def idft_signal():
    # Input amplitude and phase
    amp = [
        20.9050074380220, 11.3137084989848, 8.65913760233915,
        8, 8.65913760233915, 11.3137084989848, 20.9050074380220
    ]
    phase = [
        1.96349540849362, 2.35619449019235, 2.74889357189107,
        -3.14159265358979, -2.74889357189107, -2.35619449019235, -1.96349540849362
    ]

    # Construct complex spectrum X[k]
    X = [
        a * (math.cos(p) + 1j * math.sin(p)) if p >= 0 else a * (math.cos(p) - 1j * math.sin(p))
        for a, p in zip(amp, phase)
    ]

    N = len(X)
    inv_X = [0 + 0j] * N  # Initialize as complex numbers

    # Compute IDFT
    for n in range(N):
        for k in range(N):
            inv_X[n] += X[k] * np.exp(1j * 2 * np.pi * k * n / N)
        inv_X[n] /= N  # Normalize

    # Extract the real part and round to integers
    real_inv_X = [round(value.real) for value in inv_X]
    return real_inv_X

# Call the function and print the result
result = idft_signal()
print(result)
