from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights_1 = 2 * random.random((3, 3)) - 1
        self.synaptic_weights_2 = 2 * random.random((3, 2)) - 1
        self.synaptic_weights_3 = 2 * random.random((2, 1)) - 1
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            a0 = training_set_inputs
            a1 = self.think_1(a0)
            a2 = self.think_2(a1)
            a3 = self.think_3(a2); 
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            delta3 = training_set_outputs - a3
            D3 = dot(a2.T, delta3 * self.__sigmoid_derivative(a3))
            delta2 = dot(delta3,self.synaptic_weights_3.T)*self.__sigmoid_derivative(a2)
            D2 = dot(a1.T, delta2 * self.__sigmoid_derivative(a2))
            delta1 = dot(delta2,self.synaptic_weights_2.T)*self.__sigmoid_derivative(a1)
            D1 = dot(a0.T, delta1 * self.__sigmoid_derivative(a1))
            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            

            # Adjust the weights.
            self.synaptic_weights_3 += D3
            self.synaptic_weights_2 += D2
            self.synaptic_weights_1 += D1
    # The neural network thinks.
    def think_1(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights_1))
    def think_2(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights_2))
    def think_3(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(dot(inputs, self.synaptic_weights_3))    

if __name__ == "__main__":

    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()

    print "Random starting synaptic weights: "
    print neural_network.synaptic_weights_1
    print neural_network.synaptic_weights_2
    print neural_network.synaptic_weights_3

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print neural_network.synaptic_weights_3
    print neural_network.synaptic_weights_2
    print neural_network.synaptic_weights_1
    # Test the neural network with a new situation.
    print "Considering new situation [0, 0, 1] -> ?: "
    print neural_network.think_3(neural_network.think_2(neural_network.think_1(array([0, 0, 1]))))
