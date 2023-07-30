package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeSoftmaxWithLoss struct {
	ts []matrix.Matrix
	ys []matrix.Matrix
}

func (l *TimeSoftmaxWithLoss) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *TimeSoftmaxWithLoss) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *TimeSoftmaxWithLoss) SetParams(p ...matrix.Matrix) {}
func (l *TimeSoftmaxWithLoss) SetState(h ...matrix.Matrix)  {}
func (l *TimeSoftmaxWithLoss) ResetState()                  {}

func (l *TimeSoftmaxWithLoss) Forward(xs, ts []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	T := len(xs)
	l.ts = ts
	l.ys = make([]matrix.Matrix, T)

	// naive
	var loss float64
	for t := 0; t < T; t++ {
		l.ys[t] = Softmax(xs[t])
		loss += Loss(l.ys[t], ts[t])
	}

	return []matrix.Matrix{{{loss / float64(T)}}}
}

func (l *TimeSoftmaxWithLoss) Backward(dout []matrix.Matrix) []matrix.Matrix {
	T := len(l.ts)
	dx := make([]matrix.Matrix, T)
	dout = repeat(dout[0], T)

	// naive
	for t := T - 1; t > -1; t-- {
		size, _ := l.ts[t].Dimension()
		dx[t] = l.ys[t].Sub(l.ts[t]).Mul(dout[t]).MulC(1.0 / float64(size)) // (y - t) * dout / size
	}

	return dx
}

func (l *TimeSoftmaxWithLoss) String() string {
	return fmt.Sprintf("%T", l)
}

func repeat(m matrix.Matrix, n int) []matrix.Matrix {
	out := make([]matrix.Matrix, n)
	for i := 0; i < n; i++ {
		out[i] = m
	}

	return out
}
