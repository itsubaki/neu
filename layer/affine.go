package layer

import "github.com/itsubaki/neu/math/matrix"

type Affine struct {
	W  matrix.Matrix
	B  matrix.Matrix
	x  matrix.Matrix
	DW matrix.Matrix
	DB matrix.Matrix
}

func (l *Affine) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.x = x
	return matrix.Dot(l.x, l.W).Add(l.B) // x.W + b
}

func (l *Affine) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.W.T())
	l.DW = matrix.Dot(l.x.T(), dout)
	l.DB = matrix.New(sumAxis1(dout))
	return dx, matrix.New()

}

func sumAxis1(m matrix.Matrix) []float64 {
	p, q := m.Dimension()

	out := make([]float64, q)
	for i := 0; i < q; i++ {
		for j := 0; j < p; j++ {
			out[i] = out[i] + m[j][i]
		}
	}

	return out
}
