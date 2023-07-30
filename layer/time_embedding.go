package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeEmbedding struct {
	W     matrix.Matrix // params
	DW    matrix.Matrix // grads
	layer []*Embedding
}

func (l *TimeEmbedding) Params() []matrix.Matrix      { return []matrix.Matrix{l.W} }
func (l *TimeEmbedding) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW} }
func (l *TimeEmbedding) SetParams(p ...matrix.Matrix) { l.W = p[0] }
func (l *TimeEmbedding) SetState(_ ...matrix.Matrix)  {}
func (l *TimeEmbedding) ResetState()                  {}

func (l *TimeEmbedding) Forward(xs, _ []matrix.Matrix, opts ...Opts) []matrix.Matrix {
	// xs(Time, N, D)
	T := len(xs)

	l.layer = make([]*Embedding, T)
	out := make([]matrix.Matrix, T)
	for t := 0; t < T; t++ {
		l.layer[t] = &Embedding{W: l.W}
		out[t] = l.layer[t].Forward(xs[t], nil, opts...)
	}

	return out
}

func (l *TimeEmbedding) Backward(dout []matrix.Matrix) []matrix.Matrix {
	// dhs(Time, N, H)
	T := len(dout)

	grads := matrix.Zero(1, 1)
	for t := T - 1; t > -1; t-- {
		l.layer[t].Backward(dout[t])
		grads = l.layer[t].DW.Add(grads) // Broadcast
	}

	l.DW = grads
	return nil
}

func (l *TimeEmbedding) String() string {
	a, b := l.W.Dimension()
	return fmt.Sprintf("%T: W(%v, %v)*T: %v*T", l, a, b, a*b)
}
