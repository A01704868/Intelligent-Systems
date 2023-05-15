#include "neural_network.hh"
#include "nn_exception.hh"

NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X) {
	Matrix Z = X;
	// for every layer pass the output to the next layer
	for (auto layer : layers) {
		Z = layer->forward(Z);
	}

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
	// allocated memory if not already allocated
	dY.allocateMemoryIfNotAllocated(predictions.shape);
	// calculate cost
	Matrix error = bce_cost.dCost(predictions, target, dY);

	// reverse iteration to pass the error backwards from the last layer to the first
	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
	}
	// wait until all threads finish
	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}