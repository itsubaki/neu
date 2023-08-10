package layer

import "github.com/itsubaki/neu/math/matrix"

func ZeroLike(hs []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(hs))
	for i := 0; i < len(hs); i++ {
		out[i] = matrix.Zero(hs[i].Dimension())
	}

	return out
}

func TimeAdd(x, y []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = x[i].Add(y[i])
	}

	return out
}

func TimeMul(x, y []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = x[i].Mul(y[i])
	}

	return out
}

func TimeSum(hr []matrix.Matrix) matrix.Matrix {
	T, N, H := len(hr), len(hr[0]), len(hr[0][0])
	out := make(matrix.Matrix, N)
	for i := 0; i < N; i++ {
		out[i] = make([]float64, H)
		for j := 0; j < H; j++ {
			var sum float64
			for t := 0; t < T; t++ {
				sum += hr[t][i][j]
			}

			out[i][j] = sum
		}
	}

	return out
}
