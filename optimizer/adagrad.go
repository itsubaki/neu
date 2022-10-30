package optimizer

import (
	"math"

	"github.com/itsubaki/neu/math/matrix"
)

type AdaGrad struct {
	LearningRate float64
	Hooks        []Hook
	h            [][]matrix.Matrix
}

func (o *AdaGrad) Update(m Model) [][]matrix.Matrix {
	params, grads := m.Params(), m.Grads()
	for _, h := range o.Hooks {
		grads = h(params, grads)
	}

	if len(o.h) == 0 {
		o.h = make([][]matrix.Matrix, 0)
		for i := range params {
			h := make([]matrix.Matrix, 0)
			for j := range params[i] {
				h = append(h, matrix.Zero(params[i][j].Dimension()))
			}

			o.h = append(o.h, h)
		}
	}

	updated := make([][]matrix.Matrix, 0)
	for i := range params {
		v := make([]matrix.Matrix, 0)
		for j := range params[i] {
			o.h[i][j] = o.h[i][j].Add(grads[i][j].Mul(grads[i][j]))                                 // h[k] = h[k] + grads[k] * grads[k]
			p := params[i][j].Sub(matrix.FuncWith(grads[i][j], o.h[i][j], adagrad(o.LearningRate))) // params[k] = params[k] - o.LearningRate * grads[k]
			v = append(v, p)
		}

		updated = append(updated, v)
	}

	for i, l := range m.Layers() {
		l.SetParams(updated[i])
	}

	return updated
}

func adagrad(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return learningRate * a / (math.Sqrt(b) + 1e-7) }
}
