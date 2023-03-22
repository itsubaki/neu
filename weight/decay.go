package weight

import "github.com/itsubaki/neu/math/matrix"

// Decay returns a function that applies weight decay to the gradients.
func Decay(lambda float64) func(params, grads [][]matrix.Matrix) [][]matrix.Matrix {
	return func(params, grads [][]matrix.Matrix) [][]matrix.Matrix {
		out := make([][]matrix.Matrix, len(params))
		for i := range params { // layer
			out[i] = make([]matrix.Matrix, len(params[i]))
			for j := range params[i] { // W, B, ...
				out[i][j] = grads[i][j].Add(params[i][j].MulC(lambda)) // grad = grad + lambda * param
			}
		}

		return out // grads
	}
}
