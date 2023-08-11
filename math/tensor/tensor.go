package tensor

import (
	"math"

	"github.com/itsubaki/neu/math/matrix"
)

func ZeroLike(x []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = matrix.Zero(x[i].Dimension())
	}

	return out
}

func OneHot(ts []matrix.Matrix, size int) []matrix.Matrix {
	out := make([]matrix.Matrix, 0)
	for _, t := range ts {
		m := make(matrix.Matrix, 0)
		for _, r := range t {
			for _, v := range r {
				onehot := make([]float64, size)
				onehot[int(v)] = 1
				m = append(m, onehot)
			}
		}

		out = append(out, m)
	}

	return out
}

func Add(x, y []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = x[i].Add(y[i])
	}

	return out
}

func Mul(x, y []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := 0; i < len(x); i++ {
		out[i] = x[i].Mul(y[i])
	}

	return out
}

func Sum(hr []matrix.Matrix) matrix.Matrix {
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

func Concat(a, b []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(a))
	for t := 0; t < len(a); t++ {
		out[t] = make(matrix.Matrix, len(a[t]))

		for i := 0; i < len(a[t]); i++ {
			out[t][i] = append(out[t][i], a[t][i]...)
		}

		for i := 0; i < len(b[t]); i++ {
			out[t][i] = append(out[t][i], b[t][i]...)
		}
	}

	return out
}

func Split(dout []matrix.Matrix, H int) ([]matrix.Matrix, []matrix.Matrix) {
	a, b := make([]matrix.Matrix, len(dout)), make([]matrix.Matrix, len(dout))
	for t := range dout {
		a[t], b[t] = matrix.New(), matrix.New()
		for _, r := range dout[t] {
			a[t] = append(a[t], r[:H])
			b[t] = append(b[t], r[H:])
		}
	}

	return a, b
}

func Flatten(m []matrix.Matrix) []float64 {
	flatten := make([]float64, 0)
	for _, s := range m {
		flatten = append(flatten, matrix.Flatten(s)...)
	}

	return flatten
}

func Argmax(score []matrix.Matrix) int {
	max := math.SmallestNonzeroFloat64
	var arg int
	for i, v := range Flatten(score) {
		if v > max {
			max = v
			arg = i
		}
	}

	return arg
}
