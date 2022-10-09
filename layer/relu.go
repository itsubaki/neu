package layer

import "github.com/itsubaki/neu/math/matrix"

type ReLU struct {
	mask [][]bool
}

func (l *ReLU) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.mask = mask(x)

	out := make(matrix.Matrix, 0)
	for i := range x {
		v := make([]float64, 0)
		for j := range x[i] {
			if l.mask[i][j] {
				v = append(v, 0)
				continue
			}

			v = append(v, x[i][j])
		}

		out = append(out, v)
	}

	return out
}

func (l *ReLU) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := make(matrix.Matrix, 0)
	for i := range dout {
		v := make([]float64, 0)
		for j := range dout[i] {
			if l.mask[i][j] {
				v = append(v, 0)
				continue
			}

			v = append(v, dout[i][j])
		}

		dx = append(dx, v)
	}

	return dx, matrix.New()
}

func mask(x matrix.Matrix) [][]bool {
	out := make([][]bool, 0)
	for i := range x {
		v := make([]bool, 0)
		for j := range x[i] {
			v = append(v, x[i][j] <= 0)
		}

		out = append(out, v)
	}

	return out
}
