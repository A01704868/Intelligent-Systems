#pragma once

#include <iostream>

#include "matrix.hh"

/*
*  Polymorphism
*  Interface for all layers
*  They all need forward pass and backprop
*  They also need a name
*  Note: I dont know what line 19 is saying
*/
class NNLayer {
protected:
	std::string name;

public:
	virtual ~NNLayer() = 0;

	virtual Matrix& forward(Matrix& A) = 0;
	virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;

	std::string getName() { return this->name; };

};

inline NNLayer::~NNLayer() {}