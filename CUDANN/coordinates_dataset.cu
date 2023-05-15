#include "coordinates_dataset.hh"

CoodinatesDataset::CoodinatesDataset(size_t batch_size, size_t number_of_batches) :
	batch_size(batch_size), number_of_batches(number_of_batches)
{
	for (int i = 0; i < number_of_batches; i++) {
		batches.push_back(Matrix(Shape(batch_size, 2)));
		targets.push_back(Matrix(Shape(batch_size, 1)));

		batches[i].allocateMemory();
		targets[i].allocateMemory();

		for (int k = 0; k < batch_size; k++) {
			// load features
			batches[i][k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
			batches[i][batches[i].shape.x + k] = static_cast<float>(rand()) / RAND_MAX - 0.5;
			
			// load targets
			if ((batches[i][k] > 0 && batches[i][batches[i].shape.x + k] > 0) || ((batches[i][k] < 0 && batches[i][batches[i].shape.x + k] < 0))) {
				targets[i][k] = 1;
			}
			else {
				targets[i][k] = 0;
			}
		}

		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}
}

int CoodinatesDataset::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& CoodinatesDataset::getBatches() {
	return batches;
}

std::vector<Matrix>& CoodinatesDataset::getTargets() {
	return targets;
}