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

	updated := make([][]matrix.Matrix, 0)
	for i := range params {
		v := make([]matrix.Matrix, 0)
		for j := range params[i] {
			v = append(v, matrix.FuncWith(params[i][j], grads[i][j], sgd(o.LearningRate))) // params[k] = params[k] - o.LearningRate * grads[k]
		}

		updated = append(updated, v)
	}

	for i, l := range m.Layers() {
		l.SetParams(updated[i]...)
	}

	return updated
}

func sgd(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a - learningRate*b }
}
