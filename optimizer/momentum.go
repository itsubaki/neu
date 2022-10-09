package optimizer

import "github.com/itsubaki/neu/math/matrix"

type Momentum struct {
	LearningRate float64
	Momentum     float64
}

func (o *Momentum) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	// TODO
	out := make(map[string]matrix.Matrix)
	return out
}
