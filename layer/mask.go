package layer

import "github.com/itsubaki/neu/math/matrix"

func mask(x matrix.Matrix, f func(x float64) bool) matrix.Matrix {
	out := make(matrix.Matrix, 0)
	for i := range x {
		v := make([]float64, 0)
		for j := range x[i] {
			if f(x[i][j]) {
				v = append(v, 0)
				continue
			}

			v = append(v, 1)
		}

		out = append(out, v)
	}

	return out
}
