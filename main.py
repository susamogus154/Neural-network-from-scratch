import numpy as np
import pandas as pd
import sympy as sy

def sigmoid(z):
  return 1/(1+np.exp(-z))

def ReLu(z):
  return np.maximum(0, z)

def softmax(z):
  exp_z = np.exp(z - np.max(z))
  softmax = exp_z / exp_z.sum() #optimized to prevent overflow on large values
  return softmax

class Layer:
  def __init__(self, input_size: int, size: int, activation):
    self.parameters = np.zeros((size, input_size))

    self.bias = np.ones(size) * 0.01
    self.input_size = input_size
    self.size = size
    self.activation = activation
    self.last_pred_no_activation = np.zeros(size)
    self.last_pred = np.zeros(size)

    #weight initialization
    if activation == 'sigmoid':
      x=np.sqrt(6/(self.input_size+self.size))
      self.parameters = np.random.uniform(0, x, self.parameters.shape)
    else:
      self.parameters = np.random.normal(0, np.sqrt(2/self.input_size), self.parameters.shape)

    print("Initialized layer of size " + str(self.size) + "\n")

  def set_parameters(self, new_params):
    self.parameters = np.copy(new_params)

  def set_bias(self, new_bias):
    self.bias = np.copy(new_bias)

  def predict(self, a_in):
    """
    Make the individual layer's prediction given the previous layer's activation

    :a_in: a vector of length m (prev layer activation)
    :W: matrix n * m (n=# of nodes in this layer)
    :b: vector of length n
    """

    #print("a_in shape: " + str(a_in.shape))
    #print("W shape: " + str(self.parameters.shape))
    #print("b shape: " + str(self.bias.shape))

    # z=WX+b
    W_t = np.transpose(self.parameters)
    #print("transposed W shape: " + str(W_t.shape))
    a_out = np.matmul(a_in, W_t)
    a_out += self.bias

    self.last_pred_no_activation = np.copy(a_out) # snapshot

    # activation function
    if self.activation == 'relu':
      a_out = ReLu(a_out)
    elif self.activation == 'sigmoid':
      a_out = sigmoid(a_out)

    self.last_pred = np.copy(a_out) # snapshot

    return a_out

# CREATING NEURAL NET CLASS

class NeuralNet:
  def __init__(self, layers: list[Layer]):
    self.layers = layers
    self.num_layers = len(layers)


    self.param_adjust = [0] * self.num_layers
    self.bias_adjust = [0] * self.num_layers

    for i in range(len(layers)):
      self.param_adjust[i] = np.zeros((self.layers[i].size, self.layers[i].input_size))
      self.bias_adjust[i] = np.zeros((self.layers[i].size))

    print("Neural network created with number of layers: " + str(self.num_layers))

  def forward_prop(self, x_in): # only for one training example
    output = np.copy(x_in)

    for layer in self.layers:
      output = layer.predict(output)
    print(f"Predicted value: {output}")
    #print("\n\n\n")
    #print("Shape: " + str(output.shape))
    return output

  def backward_prop_step(self, layer_num: int, pos: int, derivative, input_data):
    """
    Given the derivative of the layer after the current one, use chain rule to
    derive activation then derive neuron (atrocious complexity)

    Passes back derivative value, not expression
    """
    # define variables
    curr_layer = self.layers[layer_num]
    activation = curr_layer.activation
    pred_pre_activation = curr_layer.last_pred_no_activation
    pred_post_activation = curr_layer.last_pred # !!!

    # derive activation
    z_val = pred_pre_activation[pos] #!!! sketchy

    z = sy.symbols('z')
    if activation == 'linear':
      derivative *= sy.diff(z)
    elif activation == 'relu':
      # print(f"Z value for layer {layer_num} and position {pos}: {z_val}")
      if z_val <= 0:
        derivative *= sy.diff(0)
      else:
        derivative *= sy.diff(z)
    elif activation == 'sigmoid':
      derivative *= sy.diff(1/(1+sy.exp(-z)))

    # print(f"Deriving activation of layer {layer_num}, position {pos}: {derivative}")
    derivative = derivative.subs({z: z_val}) #!!!?

    # derive pre-activation/neuron
      # z = w dot x+b, wrt w->x, wrt b -> 1
    self.bias_adjust[layer_num][pos] += (derivative+0)

    x = sy.symbols('x')
    derivative *= x

    if layer_num > 0:
      for i in range(curr_layer.input_size): #ASSUMING NOT INPUT LAYER !!!
        derivative = derivative.subs({x: self.layers[layer_num-1].last_pred[i]})
        # print(f"Deriving the {i}th parameter at layer {layer_num} at position {pos}")

        self.param_adjust[layer_num][pos][i] += (derivative+0)
    else:
      for i in range(input_data.shape[0]):
        derivative = derivative.subs({x: input_data[i]})

        self.param_adjust[layer_num][pos][i] += (derivative+0)


    # go to previous layer
    if layer_num > 0:
      for i in range(self.layers[layer_num-1].size):
        self.backward_prop_step(layer_num-1, i, derivative, input_data)

  def backward_prop(self, input_data, output_data, lr):
    # derive loss function (ASSUMING MSE!!!)
    yh,y = sy.symbols('yh y')
    derivative = sy.diff((yh-y)**2, yh)

    derivative = derivative.subs({y: output_data, yh: self.layers[self.num_layers-1].last_pred[0]})

    # derive layers
    for i in range(self.layers[self.num_layers-1].size):
      self.backward_prop_step(self.num_layers-1, i, derivative, input_data=input_data)

  def train(self, x_in, output: float, alpha):
    """
    Trains the model on the input data

    x_in: must be of shape [m (number of examples), data_point.shape]
    """
    loss = 0
    for i in range(x_in.shape[0]):
      y_pred = self.forward_prop(x_in[i])
      self.backward_prop(input_data=x_in[i], output_data=output[i], lr=alpha)

      loss += (y_pred-output[i])**2
    
    loss /= x_in.shape[0]
    print(f"The average loss is {loss}")

    self.adjust_weights(alpha)

  def adjust_weights(self, lr):
    for i in range(len(self.layers)):

      for j in range(self.layers[i].size): # size
        for k in range(self.layers[i].input_size): # input_size
          self.layers[i].parameters[j][k] -= self.param_adjust[i][j][k] * lr

      for j in range(self.layers[i].size): # size
        self.layers[i].bias[j] -= self.bias_adjust[i][j] * lr

      self.param_adjust[i] = np.zeros((self.layers[i].size, self.layers[i].input_size))
      self.bias_adjust[i] = np.zeros(self.layers[i].size)


  def test(self, x_in, output: float):
    """
    Trains the model on the input data

    x_in: must be of shape [m (number of examples), data_point.shape]
    """
    loss = 0
    for i in range(x_in.shape[0]):
      y_pred = self.forward_prop(x_in[i])
      loss += (y_pred-output[i])**2
    
    loss /= x_in.shape[0]
    print(f"The average loss is {loss}")

