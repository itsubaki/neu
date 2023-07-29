package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeAffine struct {
	W, B   matrix.Matrix // params
	DW, DB matrix.Matrix // grads
	x      []matrix.Matrix
}

func (l *TimeAffine) Params() []matrix.Matrix      { return []matrix.Matrix{l.W, l.B} }
func (l *TimeAffine) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW, l.DB} }
func (l *TimeAffine) SetParams(p ...matrix.Matrix) { l.W, l.B = p[0], p[1] }
func (l *TimeAffine) SetState(_ ...matrix.Matrix)  {}
func (l *TimeAffine) ResetState()                  {}

func (l *TimeAffine) Forward(xs, _ []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	l.x = xs

	// naive
	out := make([]matrix.Matrix, len(xs))
	for t := 0; t < len(xs); t++ {
		out[t] = matrix.Dot(xs[t], l.W).Add(l.B) // x.W + B
	}

	return out
}

func (l *TimeAffine) Backward(dout []matrix.Matrix) []matrix.Matrix {
	dx := make([]matrix.Matrix, len(dout))
	l.DW = matrix.Zero(1, 1)
	l.DB = matrix.Zero(1, 1)

	// naive
	for t := 0; t < len(dout); t++ {
		dx[t] = matrix.Dot(dout[t], l.W.T())
		l.DW = matrix.Dot(l.x[t].T(), dout[t]).Add(l.DW) // Broadcast
		l.DB = dout[t].SumAxis0().Add(l.DB)              // Broadcast
	}

	return dx
}

func (l *TimeAffine) String() string {
	a, b := l.W.Dimension()
	c, d := l.B.Dimension()
	return fmt.Sprintf("%T: W(%v, %v)*T, B(%v, %v)*T: %v*T", l, a, b, c, d, a*b+c*d)
}
