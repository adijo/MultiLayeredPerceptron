from validate import validate_mlp_architecture
import numpy as np


class MultiLayeredPerceptron(object):
    def __init__(self, layers):
        validate_mlp_architecture(layers)
        self.layers = layers

    def fprop(self, X):
        curr_input = X
        for layer in self.layers:
            curr_input = layer.forward(curr_input)
        return curr_input

    def bprop(self, y, loss, learning_rate):
        final_layer = self.layers[-1]
        final_layer_predictions = final_layer.activations
        final_layer_delta = final_layer.backward(final_layer_predictions, y)
        loss.append(final_layer.loss(final_layer_predictions, y))
        final_layer_weights = final_layer.weights

        curr_delta = final_layer_delta
        curr_weights = final_layer_weights

        for i in range(len(self.layers) - 2, -1, -1):
            curr_delta = self.layers[i].backward(curr_weights, curr_delta)
            curr_weights = self.layers[i].weights
            self.layers[i].update(learning_rate)

    def __reset_layers(self):
        for layer in self.layers:
            layer.reset()

    def train(self, dataset, epochs, learning_rate=0.0001, experiment=None):
        loss = []
        for epoch in range(epochs):
            while dataset.has_next_batch():
                X_batch, y_batch = dataset.get_next_batch()
                X_batch = X_batch.T
                y_batch = y_batch.T
                self.fprop(X_batch)
                self.bprop(y_batch, loss, learning_rate)

            print("Loss", loss[-1])
            if experiment:
                experiment.log_metric("train_loss", loss[-1])
            dataset.reset()
        return loss[-1]

    def predict(self, data):
        predictions = self.fprop(data.T)
        labels = np.argmax(predictions, axis=0)
        return labels
