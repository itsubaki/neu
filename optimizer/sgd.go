package optimizer

import "github.com/itsubaki/neu/math/matrix"

type SGD struct {
	LearningRate float64
}

func (o *SGD) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	out := make(map[string]matrix.Matrix)
	for k := range params {
		out[k] = matrix.FuncWith(params[k], grads[k], func(pk, gk float64) float64 { return pk - o.LearningRate*gk }) // params[k] = params[k] - o.LearningRate * grads[k]
	}

	return out
}
