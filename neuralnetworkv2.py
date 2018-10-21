import numpy as np
import pickle

class neuralnetwork:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01, learning_rate = 0.12):
        self.params = {}
        np.random.seed(2)
        self.params['W1'] = weight_init_std * np.random.randn(hidden_size, input_size)
        self.params['b1'] = np.zeros((hidden_size, 1))
        self.params['W2'] = weight_init_std * np.random.randn(output_size, hidden_size)
        self.params['b2'] = np.zeros((output_size, 1))

        self.cost = []
        self.m = 0
        self.learning_rate = 0.12

        self.cache = {}
        self.trained = False

        self.input_size = input_size
        self.hidden_size = hidden_size
        self. output_size = output_size

    def set_learning_rate(learning_rate = 0.12):
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.multiply(self.sigmoid(x) , (1 - self.sigmoid(x)))

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, X): 
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 

        # Forward propagation 
        Z1 = np.dot(W1, X) + b1 
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2) 

        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        # softmax
        #exp_scores = np.exp(a2) 
        #probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        #return np.argmax(probs, axis=1) 
        #return A2

    def compute_cost(self, Y):
        A2 = self.cache["A2"]
        assert self.m != 0
        J = -1/self.m * np.sum( np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2) ) )
        
        self.cost.append(np.squeeze(J))
        return np.squeeze(J)
        

    def backward(self, X, Y):
        W1, b1, W2, b2 = self.params['W1'], self.params['b1'], self.params['W2'], self.params['b2'] 
        Z1 = self.cache['Z1']
        A1 = self.cache["A1"]
        A2 = self.cache["A2"]

        assert self.m != 0

        # Backward propagation
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / self.m
        db2 = np.sum(dZ2, axis = 1, keepdims = True) / self.m
        dZ1 = np.multiply(np.dot(W2.T, dZ2) , self.sigmoid_prime(Z1))
        dW1 = np.dot(dZ1, X.T) / self.m
        db1 = np.sum(dZ1, axis = 1, keepdims = True) / self.m

        W1 = W1 - self.learning_rate * dW1
        b1 = b1 - self.learning_rate * db1
        W2 = W2 - self.learning_rate * dW2
        b2 = b2 - self.learning_rate * db2

        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def train(self, X, Y, m, num_iterations = 40000, debug = False):
        self.m = m
        assert self.m != 0

        print("Training start. Using learing rate %f"%(self.learning_rate))
        print("="*50)

        for i in range(0, num_iterations):
            self.forward(X)
            cost = self.compute_cost(Y)
            self.backward(X, Y)

            if debug and i%500==0:
                print("cost after iteration %i: %2.8f"%(i, cost))

        print("Training is completed!")
        self.trained = True

    def predict(self, X):
        if self.trained:
            self.forward(X)
            prediction = (self.cache["A2"] > 0.5)
            print("Output is generated.")
            return prediction
        else:
            print("Please train the network first!")
            return None

    def output(self):
        if self.trained:
            out = open('nn_weights.dat', 'wb')
            pickle.dump(self.params, out)
        else:
            print("Please train the network with data first!") 

    def input(self, weight_file):
        self.params = pickle.load(weight_file)
        self.trained = True

    def getsize(self):
        return self.input_size, self.hidden_size, self.output_size,
