package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeDropout struct {
	Ratio float64
	mask  matrix.Matrix
}

func (l *TimeDropout) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *TimeDropout) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *TimeDropout) SetParams(p ...matrix.Matrix) {}
func (l *TimeDropout) SetState(_ ...matrix.Matrix)  {}
func (l *TimeDropout) ResetState()                  {}
func (l *TimeDropout) String() string               { return fmt.Sprintf("%T: Ratio(%v)", l, l.Ratio) }

func (l *TimeDropout) Forward(xs, _ []matrix.Matrix, opts ...Opts) []matrix.Matrix {
	if len(opts) > 0 && opts[0].Train {
		T, N, D := len(xs), len(xs[0]), len(xs[0][0])
		rnd := matrix.Rand(N, D, opts[0].Source)
		msk := matrix.Mask(rnd, func(x float64) bool { return x > l.Ratio })
		l.mask = msk.MulC(1.0 / (1.0 - l.Ratio))

		out := make([]matrix.Matrix, T)
		for t := 0; t < T; t++ {
			out[t] = xs[t].Mul(l.mask)
		}

		return out
	}

	return xs
}

func (l *TimeDropout) Backward(dout []matrix.Matrix) []matrix.Matrix {
	T := len(dout)

	dx := make([]matrix.Matrix, T)
	for t := 0; t < T; t++ {
		dx[t] = dout[t].Mul(l.mask)
	}

	return dx
}
