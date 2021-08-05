import numpy as np

y = np.array([100, 120, 130, 140, 150, 160, 170, 180, 190])
x = np.array([200, 205, 210, 220, 230, 250, 270, 280, 285])

errors = []
w = 0.1
b = 0.1

for epoch in range(1000):
  y_pred = w * x + b

  w = w - 0.0000001 * ((y_pred - y) * x).sum()
  b = b - 0.0000001 * (y_pred - y).sum()

  error = 0.5 * ((y_pred - y) ** 2).sum()
  errors.append(error)

  print("{0:2} w ={1:.5f}, b={2:.5f} error = {3:.5f}".format(epoch, w, b, error))

print(260 * w + b)
