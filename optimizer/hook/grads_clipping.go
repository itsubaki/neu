package hook

import (
	"math"

	"github.com/itsubaki/neu/math/matrix"
)

func GradsClipping(max float64) func(_, grads [][]matrix.Matrix) [][]matrix.Matrix {
	return func(_, grads [][]matrix.Matrix) [][]matrix.Matrix {
		var sum float64
		for i := range grads {
			for j := range grads[i] {
				sum += grads[i][j].Pow2().Sum()
			}
		}

		rate := max / (math.Sqrt(sum) + 1e-6)
		if rate >= 1.0 {
			return grads
		}

		out := make([][]matrix.Matrix, len(grads))
		for i := range grads {
			out[i] = make([]matrix.Matrix, len(grads[i]))
			for j := range grads[i] {
				out[i][j] = grads[i][j].MulC(rate)
			}
		}

		return out
	}
}
