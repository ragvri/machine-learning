"""
Our brain has neurons. It has dendrites which are the branches. The long portion is the axon.
The dendrites are inputs.

Our model of a neuron:
We have our input eg X1,X2,X3
These values are passed to a function which gives the weighted sum of the inputs + biases i.e sum(input*weight + bias)
Bias is important for the case when all the inputs are zero

This then gets passed through a threshold(Sigmoid/Activation) function and checks if output needs to be passed or not
depending upon if value is greater than threshold or not.
0 means value less than threshold. 1 means greater than threshold.
This might go to another input.

Output Y = f(x,w) where the w are the weights

Model of a neural network:

Consider layers of neurons. eg the first layer may have 3 neurons, the second 2, and so on.
We also have our input x1,x2,x3.
Each input is fed to all the neurons of the first layer and each connection has a unique weight.
The output from the neurons of the first layer becomes the input for the second layer and so on.

x1,x2,x3 -> Input layer
Layers inbetween ->Hidden Layers
Last layer ->Output Layer.

If we have one hidden layer, it is a regular neural network. If > one layer, then "deep" neural network.


"""

# Datasets available at : ImageNet, Wiki dumps, Tatoba, CominCrawl
