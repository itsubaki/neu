package tensor

import "github.com/itsubaki/neu/math/matrix"

func Zero(m, n, o int) []matrix.Matrix {
	out := make([]matrix.Matrix, m)
	for i := range m {
		out[i] = matrix.Zero(n, o)
	}

	return out
}

func ZeroLike(x []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := range len(x) {
		out[i] = matrix.ZeroLike(x[i])
	}

	return out
}

func OneHot(ts []matrix.Matrix, size int) []matrix.Matrix {
	out := make([]matrix.Matrix, len(ts))
	for i, t := range ts {
		out[i] = make(matrix.Matrix, 0)
		for _, r := range t {
			for _, v := range r {
				onehot := make([]float64, size)
				onehot[int(v)] = 1
				out[i] = append(out[i], onehot)
			}
		}
	}

	return out
}

func Add(x, y []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := range len(x) {
		out[i] = x[i].Add(y[i])
	}

	return out
}

func Mul(x, y []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(x))
	for i := range len(x) {
		out[i] = x[i].Mul(y[i])
	}

	return out
}

func SumAxis0(hr []matrix.Matrix) matrix.Matrix {
	T, N, H := len(hr), len(hr[0]), len(hr[0][0])
	out := matrix.Zero(N, H)
	for i := range N {
		for j := range H {
			var sum float64
			for t := range T {
				sum += hr[t][i][j]
			}

			out[i][j] = sum
		}
	}

	return out
}

func Concat(a, b []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(a))
	for t := range len(a) {
		out[t] = make(matrix.Matrix, len(a[t]))

		for i := range len(a[t]) {
			out[t][i] = append(out[t][i], a[t][i]...)
		}

		for i := range len(b[t]) {
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

func Repeat(m matrix.Matrix, T int) []matrix.Matrix {
	out := make([]matrix.Matrix, T)
	for i := range T {
		out[i] = m
	}

	return out
}

func Flatten(m []matrix.Matrix) []float64 {
	flatten := make([]float64, 0)
	for _, s := range m {
		flatten = append(flatten, matrix.Flatten(s)...)
	}

	return flatten
}

func Argmax(score []matrix.Matrix) int {
	flatten := Flatten(score)
	max := flatten[0]

	var arg int
	for i, v := range flatten {
		if v > max {
			max = v
			arg = i
		}
	}

	return arg
}

func Reverse(m []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(m))
	for i := range len(m) {
		out[i] = m[len(m)-1-i]
	}

	return out
}
