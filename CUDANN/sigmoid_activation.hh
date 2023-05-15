
#pragma once

#include "nn_layer.hh"

// Sigmoid layer class
class SigmoidActivation : public NNLayer {
private:
	Matrix A; // To store the layers output
	Matrix Z; // To store the layers input
	Matrix dZ; // To store backprop output

public:
	SigmoidActivation(std::string name);
	~SigmoidActivation();

	Matrix& forward(Matrix& Z);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};