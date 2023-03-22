package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// Affine is a layer that performs an affine transformation.
type Affine struct {
	W, B   matrix.Matrix // params
	DW, DB matrix.Matrix // grads
	x      matrix.Matrix
}

func (l *Affine) Params() []matrix.Matrix      { return []matrix.Matrix{l.W, l.B} }
func (l *Affine) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW, l.DB} }
func (l *Affine) SetParams(p ...matrix.Matrix) { l.W, l.B = p[0], p[1] }

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

func (l *Affine) String() string {
	a, b := l.W.Dimension()
	c, d := l.B.Dimension()
	return fmt.Sprintf("%T: W(%v, %v), B(%v, %v): %v", l, a, b, c, d, a*b+c*d)
}
