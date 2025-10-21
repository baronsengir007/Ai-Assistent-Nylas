import numpy as np
import matplotlib as mpl

print("SUCCESS: Virtual environment is working correctly!")
print(f"NumPy: {np.__version__}, Matplotlib: {mpl.__version__}")

# Test voor Ruff formatter
import matplotlib as mpl  # extra spaties

x = 1 + 2  # geen spaties rond operators


def test():  # onnodige spaties
    print("test")
