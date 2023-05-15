#include "predict.hh"

#include <iostream>

using namespace std;

PredictCoordinates::PredictCoordinates(size_t batch_size, size_t number_of_batches) :
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	int x;
	int y;

	for (int i = 0; i < number_of_batches; i++) {
		batches.push_back(Matrix(Shape(batch_size, 2)));

		batches[i].allocateMemory();

		for (int k = 0; k < batch_size; k++) {
			// load features
			cout << "Enter x coordinate:";
			cin >> x;
			cout << "Enter y coordinate:";
			cin >> y;

			batches[i][k] = static_cast<float>(x);
			batches[i][batches[i].shape.x + k] = static_cast<float>(y);

		}

		batches[i].copyHostToDevice();
	}
}

std::vector<Matrix>& PredictCoordinates::getBatches() {
	return batches;
}

int PredictCoordinates::getNumOfBatches() {
	return number_of_batches;
}