package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeGRU struct {
	Wx, Wh, B    matrix.Matrix // params
	DWx, DWh, DB matrix.Matrix // grads
	h, dh        matrix.Matrix // hidden state
	layer        []GRU
	Stateful     bool
}

func (l *TimeGRU) DH() matrix.Matrix {
	return l.dh
}

func (l *TimeGRU) Params() []matrix.Matrix {
	return []matrix.Matrix{
		l.Wx,
		l.Wh,
		l.B,
	}
}

func (l *TimeGRU) Grads() []matrix.Matrix {
	return []matrix.Matrix{
		l.DWx,
		l.DWh,
		l.DB,
	}
}

func (l *TimeGRU) SetParams(p ...matrix.Matrix) {
	l.Wx, l.Wh, l.B = p[0], p[1], p[2]
}

func (l *TimeGRU) SetState(h ...matrix.Matrix) {
	l.h = h[0]
}

func (l *TimeGRU) ResetState() {
	l.h = matrix.New()
}

func (l *TimeGRU) String() string {
	a, b := l.Wx.Dim()
	c, d := l.Wh.Dim()
	e, f := l.B.Dim()
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, a*b+c*d+e*f)
}

func (l *TimeGRU) Forward(xs, _ []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	T, N, H := len(xs), len(xs[0]), len(l.Wh)
	l.layer = make([]GRU, T)
	hs := make([]matrix.Matrix, T)

	if !l.Stateful || len(l.h) == 0 {
		l.h = matrix.Zero(N, H)
	}

	for t := 0; t < T; t++ {
		l.layer[t] = GRU{Wx: l.Wx, Wh: l.Wh, B: l.B}
		l.h = l.layer[t].Forward(xs[t], l.h)
		hs[t] = l.h
	}

	return hs
}

func (l *TimeGRU) Backward(dhs []matrix.Matrix) []matrix.Matrix {
	T, N, H := len(dhs), len(dhs[0]), len(dhs[0][0])
	dxs := make([]matrix.Matrix, T)
	dh := matrix.Zero(N, H)

	grads := []matrix.Matrix{
		matrix.Zero(1, 1), // DWx(D, 3H)
		matrix.Zero(1, 1), // DWh(H, 3H)
		matrix.Zero(1, 1), // DB(1, 3H)
	}

	for t := T - 1; t > -1; t-- {
		dxs[t], dh = l.layer[t].Backward(dhs[t].Add(dh))

		// grads
		for i, g := range l.layer[t].Grads() {
			grads[i] = g.Add(grads[i]) // Broadcast
		}
	}

	l.DWx, l.DWh, l.DB = grads[0], grads[1], grads[2]
	l.dh = dh
	return dxs
}
