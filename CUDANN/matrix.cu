#include "matrix.hh"
#include "nn_exception.hh"

Matrix::Matrix(size_t x_dim, size_t y_dim) :
	shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
	device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape) :
	Matrix(shape.x, shape.y)
{ }

// Allocate memory in device
void Matrix::allocateCudaMemory() {
	if (!device_allocated) { // memory has not been allocated to de device

		float* device_memory = nullptr;
		// allocate a block the size of the matrix and point to it with device_memory
		cudaMalloc(&device_memory, shape.x * shape.y * sizeof(float));
		NNException::throwIfDeviceErrorsOccurred("Cannot allocate CUDA memory for Tensor3D.");

		// turn the normal pointer into a shared pointer to make things easier
		data_device = std::shared_ptr<float>(device_memory,
			[&](float* ptr) { cudaFree(ptr); });

		// and now memory has been allocated for this object
		device_allocated = true;
	}
}

// Allocate memory in host
void Matrix::allocateHostMemory() {
	if (!host_allocated) {
		data_host = std::shared_ptr<float>(new float[shape.x * shape.y],
			[&](float* ptr) { delete[] ptr; });
		host_allocated = true;
	}
}

// Allocate memory in both host and device
void Matrix::allocateMemory() {
	allocateCudaMemory();
	allocateHostMemory();
}

/*
*  Check if memory has been allocated
*  if not allocate memory for a matrix of a given shape
*  this is for when we dont know the shape of the matrix at first
*/
void Matrix::allocateMemoryIfNotAllocated(Shape shape) {
	if (!device_allocated && !host_allocated) {
		this->shape = shape;
		allocateMemory();
	}
}


// copy matrix from host to device
void Matrix::copyHostToDevice() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_device.get(), data_host.get(), shape.x * shape.y * sizeof(float), cudaMemcpyHostToDevice);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy host data to CUDA device.");
	}
	else {
		throw NNException("Cannot copy host data to not allocated memory on device.");
	}
}

// copy matrix from device to host
void Matrix::copyDeviceToHost() {
	if (device_allocated && host_allocated) {
		cudaMemcpy(data_host.get(), data_device.get(), shape.x * shape.y * sizeof(float), cudaMemcpyDeviceToHost);
		NNException::throwIfDeviceErrorsOccurred("Cannot copy device data to host.");
	}
	else {
		throw NNException("Cannot copy device data to not allocated memory on host.");
	}
}

// overload [] operator to make accessing host data easy
float& Matrix::operator[](const int index) {
	return data_host.get()[index];
}

// overload [] operator to make accessing host data easy
const float& Matrix::operator[](const int index) const {
	return data_host.get()[index];
}