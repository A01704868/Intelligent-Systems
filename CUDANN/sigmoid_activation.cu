#include "sigmoid_activation.hh"
#include "nn_exception.hh"
#include <iostream>

// device kernel is called by the device and executes on the device
__device__ float sigmoid(float x) {
	return 1.0f / (1 + exp(-x));
}
// sigmoid layer forward pass
// global kernel is called by host and executes on the device
__global__ void sigmoidActivationForward(float* Z, float* A,
	int Z_x_dim, int Z_y_dim) {
	// get the index for the current thread
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	// check that index is within matrix memory bounds, otherwise it will access wrong memory spaces
	if (index < Z_x_dim * Z_y_dim) {
		// store in A the result of sigmoid(Z) for the input being handled by this thread
		A[index] = sigmoid(Z[index]);
	}
}
// sigmoid layer backprop
__global__ void sigmoidActivationBackprop(float* Z, float* dA, float* dZ,
	int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		// calculate and store the partial derivative of the cost function with respect to the sigmoid function
		dZ[index] = dA[index] * sigmoid(Z[index]) * (1 - sigmoid(Z[index]));
	}
}

// Give the layer a name
SigmoidActivation::SigmoidActivation(std::string name) {
	this->name = name;
}

// WTF is this????
SigmoidActivation::~SigmoidActivation()
{ }

// run forward pass kernel
Matrix& SigmoidActivation::forward(Matrix& Z) {
	// store input for backprop
	this->Z = Z;
	// allocate memory for output matrix if not already allocated
	A.allocateMemoryIfNotAllocated(Z.shape);
	// compute the number of blocks and the size of each block, (see nvidia thread execution model)
	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	// call the forward pass kernel
	sigmoidActivationForward << <num_of_blocks, block_size >> > (Z.data_device.get(), A.data_device.get(),
		Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid forward propagation.");
	// return output
	return A;
}

Matrix& SigmoidActivation::backprop(Matrix& dA, float learning_rate) {
	// allocate memory for backprop output if not already allocated
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	// call backprop kernel
	sigmoidActivationBackprop << <num_of_blocks, block_size >> > (Z.data_device.get(), dA.data_device.get(),
		dZ.data_device.get(),
		Z.shape.x, Z.shape.y);
	NNException::throwIfDeviceErrorsOccurred("Cannot perform sigmoid back propagation");
	// return derivatives matrix
	return dZ;
}