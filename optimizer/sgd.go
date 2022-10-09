package optimizer

import "github.com/itsubaki/neu/math/matrix"

type SGD struct {
	LearningRate float64
}

func (o *SGD) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	for k := range params {
		params[k] = params[k].Sub(grads[k].Mulf64(o.LearningRate)) // params[k] = params[k] - o.LearningRate * grads[k]
	}

	return params
}
