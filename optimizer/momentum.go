package optimizer

import "github.com/itsubaki/neu/math/matrix"

type Momentum struct {
	LearningRate float64
	Momentum     float64
	v            [][]matrix.Matrix
}

func (o *Momentum) Update(params, grads [][]matrix.Matrix) [][]matrix.Matrix {
	if len(o.v) == 0 {
		o.v = make([][]matrix.Matrix, 0)
		for i := range params {
			v := make([]matrix.Matrix, 0)
			for j := range params[i] {
				v = append(v, matrix.Zero(params[i][j].Dimension()))
			}

			o.v = append(o.v, v)
		}
	}

	out := make([][]matrix.Matrix, 0)
	for i := range params {
		v := make([]matrix.Matrix, 0)
		for j := range params[i] {
			o.v[i][j] = matrix.FuncWith(o.v[i][j], grads[i][j], momentum(o.Momentum, o.LearningRate)) // v[k] = momentum * v[k] - learningRate * grads[k]
			p := params[i][j].Add(o.v[i][j])                                                          //  params[k] = params[k] + v[k]
			v = append(v, p)
		}

		out = append(out, v)
	}

	return out
}

func momentum(momentum, learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return momentum*a - learningRate*b }
}
