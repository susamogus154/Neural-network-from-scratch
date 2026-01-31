from main import *
import numpy as np

test_network = NeuralNet([
    Layer(3, 3, 'relu'),
    Layer(3, 2, 'relu'),
    Layer(2, 1, 'linear'),
])

test_in1 = np.array([
    1.0, 2.0, 3.0
])
test_out1 = 0.20

test_in2 = np.array([
    1.5, 3.0, 4.5
])
test_out2 = 0.30

test_in3 = np.array([
    2.0, 4.0, 6.0
])
test_out3 = 0.41

for i in range(100): # training for 100 epochs
  test_network.train(x_in=np.array([test_in1, test_in2, test_in3]),
                     output=np.array([test_out1, test_out2, test_out3]),
                     alpha=1e-2)
  

test_network.test(
    x_in=np.array([test_in1, test_in2, test_in3]),
    output=np.array([test_out1, test_out2, test_out3])
)
