package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeAffine struct {
	W, B   matrix.Matrix // params
	DW, DB matrix.Matrix // grads
	layer  []Affine
}

func (l *TimeAffine) Params() []matrix.Matrix      { return []matrix.Matrix{l.W, l.B} }
func (l *TimeAffine) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW, l.DB} }
func (l *TimeAffine) SetParams(p ...matrix.Matrix) { l.W, l.B = p[0], p[1] }
func (l *TimeAffine) SetState(_ ...matrix.Matrix)  {}
func (l *TimeAffine) ResetState()                  {}
func (l *TimeAffine) String() string {
	a, b := l.W.Dimension()
	c, d := l.B.Dimension()
	return fmt.Sprintf("%T: W(%v, %v), B(%v, %v): %v", l, a, b, c, d, a*b+c*d)
}

func (l *TimeAffine) Forward(xs, _ []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	T := len(xs)
	l.layer = make([]Affine, T)
	out := make([]matrix.Matrix, T)

	for t := 0; t < T; t++ {
		l.layer[t] = Affine{W: l.W, B: l.B}
		out[t] = l.layer[t].Forward(xs[t], nil)
	}

	return out
}

func (l *TimeAffine) Backward(dout []matrix.Matrix) []matrix.Matrix {
	T := len(dout)
	dxs := make([]matrix.Matrix, T)
	l.DW = matrix.Zero(1, 1)
	l.DB = matrix.Zero(1, 1)

	for t := 0; t < T; t++ {
		dxs[t], _ = l.layer[t].Backward(dout[t])
		l.DW = l.layer[t].DW.Add(l.DW) // Broadcast
		l.DB = l.layer[t].DB.Add(l.DB) // Broadcast
	}

	return dxs
}
