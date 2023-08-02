package optimizer

import (
	"math"

	"github.com/itsubaki/neu/math/matrix"
)

// AdaGrad is an optimizer that implements the AdaGrad algorithm.
type AdaGrad struct {
	LearningRate float64
	Hooks        []Hook
	h            [][]matrix.Matrix
}

// Update updates the parameters of the model.
func (o *AdaGrad) Update(m Model) [][]matrix.Matrix {
	params, grads := m.Params(), m.Grads()
	for _, h := range o.Hooks {
		grads = h(params, grads)
	}

	if len(o.h) == 0 {
		o.h = LikeZero(params)
	}

	updated := make([][]matrix.Matrix, len(params))
	for i := range params {
		updated[i] = make([]matrix.Matrix, len(params[i]))
		for j := range params[i] {
			o.h[i][j] = o.h[i][j].Add(grads[i][j].Mul(grads[i][j]))                                            // h[k] = h[k] + grads[k] * grads[k]
			updated[i][j] = params[i][j].Sub(matrix.FuncWith(grads[i][j], o.h[i][j], adagrad(o.LearningRate))) // params[k] = params[k] - o.LearningRate * grads[k]/sqrt(h[k])
		}
	}

	m.SetParams(updated)
	return updated
}

func adagrad(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return learningRate * a / (math.Sqrt(b) + 1e-7) }
}
