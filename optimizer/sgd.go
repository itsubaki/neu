package optimizer

import "github.com/itsubaki/neu/math/matrix"

type SGD struct {
	LearningRate float64
}

func (o *SGD) Update(params, grads [][]matrix.Matrix) [][]matrix.Matrix {
	out := make([][]matrix.Matrix, 0)
	for i := range params {
		v := make([]matrix.Matrix, 0)
		for j := range params[i] {
			v = append(v, matrix.FuncWith(params[i][j], grads[i][j], sgd(o.LearningRate))) // params[k] = params[k] - o.LearningRate * grads[k]
		}

		out = append(out, v)
	}

	return out
}

func sgd(learningRate float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a - learningRate*b }
}
