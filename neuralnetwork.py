#  https://github.com/oreilly-japan/deep-learning-from-scratch/tree/master/common

#  https://github.com/dennybritz/nn-from-scratch/blob/master/nn_from_scratch.py
#  http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

import numpy as np

class neuralnetwork:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        # np.random.seed(0)
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))


    def calculate_loss(self, X, y):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2'] 

        a1 = np.dot(X, W1) + b1 
        z1 = np.tanh(a1) 
        a2 = np.dot(z1, W2) + b2 
        exp_scores = np.exp(a2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        # Calculating the loss 
        corect_logprobs = -np.log(probs[range(num_examples), y]) 
        data_loss = np.sum(corect_logprobs) 
        # Add regulatization term to loss (optional) 
        #data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))) 
        return 1./num_examples * data_loss 

    def predict(self, x): 
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 
        # Forward propagation 
        a1 = np.dot(x, W1) + b1 
        z1 = np.tanh(a1) 
        a2 = np.dot(z1, W2) + b2
        y = self.softmax(a2)
        return y 
        #exp_scores = np.exp(a2) 
        #probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        #return np.argmax(probs, axis=1) 

    def train(iters_num = 20000):
        for i in range(0, iters_num):
            a1 = np.dot(x, W1) + b1
            z1 = np.tanh(a1)
            a2 = np.dot(z1, W2) + b2
            y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

return grads


def build_model(nn_hdim, num_passes=20000, print_loss=False): 
 
    # Initialize the parameters to random values. We need to learn these. 
    np.random.seed(0) 
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim) 
    b1 = np.zeros((1, nn_hdim)) 
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim) 
    b2 = np.zeros((1, nn_output_dim)) 
 
    # This is what we return at the end 
    model = {} 
 
    # Gradient descent. For each batch... 
    for i in range(0, num_passes): 
 
        # Forward propagation 
        z1 = X.dot(W1) + b1 
        a1 = np.tanh(z1) 
        z2 = a1.dot(W2) + b2 
        exp_scores = np.exp(z2) 
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
 
        # Backpropagation 
        delta3 = probs 
        delta3[range(num_examples), y] -= 1 
        dW2 = (a1.T).dot(delta3) 
        db2 = np.sum(delta3, axis=0, keepdims=True) 
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2)) 
        dW1 = np.dot(X.T, delta2) 
        db1 = np.sum(delta2, axis=0) 
 
        # Add regularization terms (b1 and b2 don't have regularization terms) 
        dW2 += reg_lambda * W2 
        dW1 += reg_lambda * W1 
 
        # Gradient descent parameter update 
        W1 += -epsilon * dW1 
        b1 += -epsilon * db1 
        W2 += -epsilon * dW2 
        b2 += -epsilon * db2 
 
        # Assign new parameters to the model 
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2} 
 
        # Optionally print the loss. 
        # This is expensive because it uses the whole dataset, so we don't want to do it too often. 
        if print_loss and i % 1000 == 0: 
          print("Loss after iteration %i: %f" %(i, calculate_loss(model))) 
 
    return model 
