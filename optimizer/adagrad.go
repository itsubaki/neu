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
		o.h[k] = o.h[k].Add(grads[k].Mul(grads[k]))                                // h[k] = h[k] + grads[k] * grads[k]
		out[k] = params[k].Sub(grads[k].FuncWith(o.h[k], adagrad(o.LearningRate))) // params[k] = params[k] - learningRate * grads[k] / sqrt(h[k])
	}

	return out
}

func adagrad(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return learningRate * a / (math.Sqrt(b) + 1e-7) }
}
