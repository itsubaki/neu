package layer

import "github.com/itsubaki/neu/math/matrix"

func newMask(x matrix.Matrix, f func(x float64) bool) [][]bool {
	out := make([][]bool, 0)
	for i := range x {
		v := make([]bool, 0)
		for j := range x[i] {
			v = append(v, f(x[i][j]))
		}

		out = append(out, v)
	}

	return out
}

func mask(m matrix.Matrix, mask [][]bool) matrix.Matrix {
	out := make(matrix.Matrix, 0)
	for i := range m {
		v := make([]float64, 0)
		for j := range m[i] {
			if mask[i][j] {
				v = append(v, 0)
				continue
			}

			v = append(v, m[i][j])
		}

		out = append(out, v)
	}

	return out
}
