package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeEmbedding struct {
	W     matrix.Matrix // params
	DW    matrix.Matrix // grads
	layer []Embedding
}

func (l *TimeEmbedding) Params() []matrix.Matrix      { return []matrix.Matrix{l.W} }
func (l *TimeEmbedding) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DW} }
func (l *TimeEmbedding) SetParams(p ...matrix.Matrix) { l.W = p[0] }
func (l *TimeEmbedding) SetState(_ ...matrix.Matrix)  {}
func (l *TimeEmbedding) ResetState()                  {}
func (l *TimeEmbedding) String() string {
	a, b := l.W.Dim()
	return fmt.Sprintf("%T: W(%v, %v): %v", l, a, b, a*b)
}

func (l *TimeEmbedding) Forward(xs, _ []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	T := len(xs)                    // (7, 128, 1)
	l.layer = make([]Embedding, T)  //
	out := make([]matrix.Matrix, T) // (7, 128, 16)

	for t := 0; t < T; t++ {
		l.layer[t] = Embedding{W: l.W}
		out[t] = l.layer[t].Forward(xs[t], nil)
	}

	return out
}

func (l *TimeEmbedding) Backward(dout []matrix.Matrix) []matrix.Matrix {
	T := len(dout) // (Time, N, H)

	grad := matrix.Zero(1, 1)
	for t := 0; t < T; t++ {
		l.layer[t].Backward(dout[t])
		grad = l.layer[t].DW.Add(grad) // Broadcast
	}

	l.DW = grad
	return nil
}
