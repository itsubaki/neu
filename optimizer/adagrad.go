package optimizer

import (
	"math"

	"github.com/itsubaki/neu/math/matrix"
)

type AdaGrad struct {
	LearningRate float64
	h            map[string]matrix.Matrix
}

func (o *AdaGrad) Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix {
	if o.h == nil {
		o.h = make(map[string]matrix.Matrix)
		for k, v := range params {
			o.h[k] = matrix.Zero(v.Dimension())
		}
	}

	out := make(map[string]matrix.Matrix)
	for k := range params {
		o.h[k] = o.h[k].Add(grads[k].Mul(grads[k])) // h = h + grads[k] * grads[k]
		out[k] = params[k].Sub(matrix.FuncWith(grads[k], o.h[k], func(gk, hk float64) float64 {
			return o.LearningRate * gk / (math.Sqrt(hk) + 1e-7) // params[k] = params[k] - o.LearningRate * grads[k] / sqrt(hk)
		}))
	}

	return out
}
