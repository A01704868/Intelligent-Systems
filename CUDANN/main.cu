#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "linear_layer.hh"
#include "relu_activation.hh"
#include "sigmoid_activation.hh"
#include "nn_exception.hh"
#include "bce_cost.hh"

#include "coordinates_dataset.hh"
#include "predict.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {

	srand(time(NULL));

	CoodinatesDataset dataset(10, 21);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 10)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(10, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 901; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout << "Epoch: " << epoch
				<< ", Cost: " << cost / dataset.getNumOfBatches()
				<< std::endl;
		}
	}

	// compute accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(
		Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	std::cout << "Accuracy: " << accuracy << std::endl;
	/*
	int a = 0;
	std::cout << "Do you want to make a prediction? ";
	std::cin >> a;
	std::cout << std::endl;
	*/

	PredictCoordinates predict(10, 1);
	Matrix Y2;
	Y2 = nn.forward(predict.getBatches().at(predict.getNumOfBatches() - 1));
	Y2.copyDeviceToHost();
	std::cout << "Y2.shape.x: " << Y2.shape.x << std::endl;
	std::cout << "Prediction: " << std::endl;
	for (int i = 0; i < Y2.shape.x; i++) {
		float prediction = Y2[i] > 0.5 ? 1 : 0;
		std::cout << prediction << std::endl;
	}

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}

