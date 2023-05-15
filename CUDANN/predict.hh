#pragma once

#include "matrix.hh"

#include <vector>

class PredictCoordinates {
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;

public:

	PredictCoordinates(size_t batch_size, size_t number_of_batches);

	std::vector<Matrix>& getBatches();
	int getNumOfBatches();
};