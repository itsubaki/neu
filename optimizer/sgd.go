package optimizer

import "github.com/itsubaki/neu/math/matrix"

type SGD struct {
	LearningRate float64
}

func (o *SGD) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	out := make(map[string]matrix.Matrix)
	for k := range params {
		out[k] = matrix.FuncWith(params[k], grads[k], sgd(o.LearningRate)) // params[k] = params[k] - o.LearningRate * grads[k]
	}

	return out
}

func sgd(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a - learningRate*b }
}
