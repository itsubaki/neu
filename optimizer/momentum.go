package optimizer

import "github.com/itsubaki/neu/math/matrix"

type Momentum struct {
	LearningRate float64
	Momentum     float64
	v            map[string]matrix.Matrix
}

func (o *Momentum) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	if o.v == nil {
		o.v = make(map[string]matrix.Matrix)
		for k, v := range params {
			o.v[k] = matrix.Zero(v.Dimension())
		}
	}

	out := make(map[string]matrix.Matrix)
	for k := range params {
		o.v[k] = matrix.FuncWith(o.v[k], grads[k], func(vk, gk float64) float64 {
			return o.Momentum*vk - o.LearningRate*gk // v[k] = momentum * v[k] - lr * grads[k]
		})
		out[k] = params[k].Add(o.v[k]) // params[k] = params[k] + v[k]
	}

	return out
}
