package layer

import "github.com/itsubaki/neu/math/matrix"

type Affine struct {
	W  matrix.Matrix
	B  matrix.Matrix
	x  matrix.Matrix
	DW matrix.Matrix
	DB []float64
}

func (l *Affine) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.x = x
	return matrix.Dot(l.x, l.W).Add(l.B)
}

func (l *Affine) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.W.T())
	l.DW = matrix.Dot(l.x.T(), dout)

	n, m := dout.Shape()
	dB := make([]float64, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			dB[i] = dB[i] + dout[j][i]
		}
	}

	return dx, matrix.New()
}
