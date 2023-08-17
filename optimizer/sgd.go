package optimizer

import "github.com/itsubaki/neu/math/matrix"

// SGD is an optimizer that the Stochastic Gradient Descent algorithm.
type SGD struct {
	LearningRate float64
	Hooks        []Hook
}

// Update updates the parameters of the model.
func (o *SGD) Update(m Model) [][]matrix.Matrix {
	params, grads := m.Params(), m.Grads()
	for _, h := range o.Hooks {
		grads = h(params, grads)
	}

	updated := ZeroLike(params)
	for i := range params {
		for j := range params[i] {
			updated[i][j] = matrix.F2(params[i][j], grads[i][j], sgd(o.LearningRate)) // params[k] = params[k] - o.LearningRate * grads[k]
		}
	}

	m.SetParams(updated)
	return updated
}

func sgd(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a - learningRate*b }
}
