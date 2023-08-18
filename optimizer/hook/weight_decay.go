package hook

import "github.com/itsubaki/neu/math/matrix"

// Decay returns a function that applies weight decay to the gradients.
func WeightDecay(lambda float64) func(params, grads [][]matrix.Matrix) [][]matrix.Matrix {
	return func(params, grads [][]matrix.Matrix) [][]matrix.Matrix {
		out := make([][]matrix.Matrix, len(params))
		for i := range params { // layer
			out[i] = make([]matrix.Matrix, len(params[i]))
			for j := range params[i] { // W, B, ...
				out[i][j] = matrix.F2(grads[i][j], params[i][j], decay(lambda)) // grad = grad + lambda * param
			}
		}

		return out // grads
	}
}

func decay(lambda float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a + lambda*b }
}
