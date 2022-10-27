package layer

import "github.com/itsubaki/neu/math/matrix"

type Affine struct {
	W  matrix.Matrix
	B  matrix.Matrix
	x  matrix.Matrix
	DW matrix.Matrix
	DB matrix.Matrix
}

func (l *Affine) Params() []matrix.Matrix     { return []matrix.Matrix{l.W, l.B} }
func (l *Affine) SetParams(p []matrix.Matrix) { l.W, l.B = p[0], p[1] }
func (l *Affine) Grads() []matrix.Matrix      { return []matrix.Matrix{l.DW, l.DB} }
func (l *Affine) SetGrads(g []matrix.Matrix)  { l.DW, l.DB = g[0], g[1] }

func (l *Affine) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.x = x
	return matrix.Dot(l.x, l.W).Add(l.B) // x.W + b
}

func (l *Affine) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.W.T())
	l.DW = matrix.Dot(l.x.T(), dout)
	l.DB = dout.SumAxis0()
	return dx, matrix.New()
}
