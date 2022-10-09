package layer

import "github.com/itsubaki/neu/math/matrix"

func mask(x matrix.Matrix, f func(x float64) bool) [][]bool {
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
