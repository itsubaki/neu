package layer

import "github.com/itsubaki/neu/math/matrix"

type Affine struct {
	W  matrix.Matrix
	B  matrix.Matrix
	x  matrix.Matrix
	DW matrix.Matrix
}

func (l *Affine) Forward(x, _ []float64) []float64 {
	l.x = matrix.New(x)
	return matrix.Dot(l.x, l.W).Add(l.B)[0]
}

func (l *Affine) Backward(dout []float64) ([]float64, []float64) {
	mdout := matrix.New(dout)

	dx := matrix.Dot(mdout, l.W.T())
	l.DW = matrix.Dot(l.x.T(), mdout)
	// TODO: l.DB = np.sum(dout, axis=0)

	return dx[0], []float64{}
}
