#pragma once

#include "matrix.hh"

#include <vector>

class CoodinatesDataset {
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	CoodinatesDataset(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();

};