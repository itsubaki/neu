package optimizer

import "github.com/itsubaki/neu/math/matrix"

type Adam struct {
	LearningRate float64
}

func (o *Adam) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	// TODO
	out := make(map[string]matrix.Matrix)
	return out
}
