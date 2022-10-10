package optimizer

import "github.com/itsubaki/neu/math/matrix"

type SGD struct {
	LearningRate float64
}

func (o *SGD) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	for k := range params {
		g := grads[k].Func(func(v float64) float64 { return -1.0 * o.LearningRate * v })
		params[k] = params[k].Add(g) // params[k] = params[k] - o.LearningRate * grads[k]
	}

	return params
}
