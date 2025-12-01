package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeRNN struct {
	Wx, Wh, B    matrix.Matrix // params
	DWx, DWh, DB matrix.Matrix // grads
	h, dh        matrix.Matrix // hidden state
	layer        []RNN
	Stateful     bool
}

func (l *TimeRNN) Params() []matrix.Matrix {
	return []matrix.Matrix{
		l.Wx,
		l.Wh,
		l.B,
	}
}

func (l *TimeRNN) Grads() []matrix.Matrix {
	return []matrix.Matrix{
		l.DWx,
		l.DWh,
		l.DB,
	}
}

func (l *TimeRNN) SetParams(p ...matrix.Matrix) {
	l.Wx, l.Wh, l.B = p[0], p[1], p[2]
}

func (l *TimeRNN) SetState(h ...matrix.Matrix) {
	l.h = h[0]
}

func (l *TimeRNN) ResetState() {
	l.h = matrix.New()
}

func (l *TimeRNN) String() string {
	a, b := l.Wx.Dim()
	c, d := l.Wh.Dim()
	e, f := l.B.Dim()
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, a*b+c*d+e*f)
}

func (l *TimeRNN) Forward(xs, _ []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	T, N, H := len(xs), len(xs[0]), len(l.Wx[0]) // xs(Time, N, D), Wx(D, H)
	l.layer = make([]RNN, T)
	hs := make([]matrix.Matrix, T)

	if !l.Stateful || len(l.h) == 0 {
		l.h = matrix.Zero(N, H)
	}

	for t := range T {
		l.layer[t] = RNN{Wx: l.Wx, Wh: l.Wh, B: l.B}
		l.h = l.layer[t].Forward(xs[t], l.h)
		hs[t] = l.h
	}

	return hs
}

func (l *TimeRNN) Backward(dhs []matrix.Matrix) []matrix.Matrix {
	T, N, H := len(dhs), len(dhs[0]), len(dhs[0][0]) // dhs(Time, N, H)
	dxs := make([]matrix.Matrix, T)                  // dxs(Time, N, D)
	dh := matrix.Zero(N, H)                          // dh(N, H)

	grads := []matrix.Matrix{
		matrix.Zero(1, 1), // DWx(D, H)
		matrix.Zero(1, 1), // DWh(H, H)
		matrix.Zero(1, 1), // DB(1, H)
	}

	for t := T - 1; t > -1; t-- {
		dxs[t], dh = l.layer[t].Backward(dhs[t].Add(dh)) // dx(N, D), dh(N, H)

		// grads
		for i, g := range l.layer[t].Grads() {
			grads[i] = g.Add(grads[i]) // Broadcast
		}
	}

	l.DWx, l.DWh, l.DB = grads[0], grads[1], grads[2]
	l.dh = dh
	return dxs
}
