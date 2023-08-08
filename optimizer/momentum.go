package optimizer

import "github.com/itsubaki/neu/math/matrix"

// Momemtum is an optimizer that the Momentum algorithm.
type Momentum struct {
	LearningRate float64
	Momentum     float64
	Hooks        []Hook
	v            [][]matrix.Matrix
}

// Update updates the parameters of the model.
func (o *Momentum) Update(m Model) [][]matrix.Matrix {
	params, grads := m.Params(), m.Grads()
	for _, h := range o.Hooks {
		grads = h(params, grads)
	}

	if len(o.v) == 0 {
		o.v = ZeroLike(params)
	}

	updated := make([][]matrix.Matrix, len(params))
	for i := range params {
		updated[i] = make([]matrix.Matrix, len(params[i]))
		for j := range params[i] {
			o.v[i][j] = matrix.F2(o.v[i][j], grads[i][j], momentum(o.Momentum, o.LearningRate)) // v[k] = momentum * v[k] - learningRate * grads[k]
			updated[i][j] = params[i][j].Add(o.v[i][j])                                         // params[k] = params[k] + v[k]
		}
	}

	m.SetParams(updated)
	return updated
}

func momentum(momentum, learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return momentum*a - learningRate*b }
}
