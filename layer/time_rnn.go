package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeRNN struct {
	Wx, Wh, B     matrix.Matrix // params
	DWx, DWh, DWB matrix.Matrix // grads
	h, dh         matrix.Matrix // hidden state
	layer         []*RNN
	stateful      bool
}

func (l *TimeRNN) Params() []matrix.Matrix      { return []matrix.Matrix{l.Wx, l.Wh, l.B} }
func (l *TimeRNN) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DWx, l.DWh, l.DWB} }
func (l *TimeRNN) SetParams(p ...matrix.Matrix) { l.Wx, l.Wh, l.B = p[0], p[1], p[2] }
func (l *TimeRNN) SetState(h ...matrix.Matrix)  { l.h = h[0] }
func (l *TimeRNN) ResetState()                  { l.h = matrix.New() }

func (l *TimeRNN) Forward(xs, _ []matrix.Matrix, opts ...Opts) []matrix.Matrix {
	// xs(Time, N, D), Wx(D, H)
	T, N, H := len(xs), len(xs[0]), len(l.Wx[0])

	if !l.stateful || len(l.h) == 0 {
		l.h = matrix.Zero(N, H)
	}

	l.layer = make([]*RNN, T)
	hs := make([]matrix.Matrix, T)
	for t := 0; t < T; t++ {
		l.layer[t] = &RNN{Wx: l.Wx, Wh: l.Wh, B: l.B}
		l.h = l.layer[t].Forward(xs[t], l.h, opts...)
		hs[t] = l.h
	}

	return hs
}

func (l *TimeRNN) Backward(dhs []matrix.Matrix) []matrix.Matrix {
	// dhs(Time, N, H)
	T, N, H := len(dhs), len(dhs[0]), len(dhs[0][0])

	dxs := make([]matrix.Matrix, T) // dxs(Time, N, D)
	dh := matrix.Zero(N, H)         // dh(N, H)

	grads := make([]matrix.Matrix, 3) // DWx(D, H), DWh(H, H), DWB(1, H)
	for i := range grads {
		grads[i] = matrix.Zero(1, 1)
	}

	for t := T - 1; t > -1; t-- {
		// dx(N, D), dh(N, H)
		dxs[t], dh = l.layer[t].Backward(dhs[t].Add(dh))

		// grads
		for i, g := range l.layer[t].Grads() {
			grads[i] = g.Add(grads[i]) // Broadcast
		}
	}

	l.DWx, l.DWh, l.DWB = grads[0], grads[1], grads[2]
	l.dh = dh
	return dxs
}

func (l *TimeRNN) String() string {
	a, b := l.Wx.Dimension()
	c, d := l.Wh.Dimension()
	e, f := l.B.Dimension()
	return fmt.Sprintf("%T: Wx(%v, %v)*T, Wh(%v, %v)*T, B(%v, %v)*T: %v*T", l, a, b, c, d, e, f, a*b+c*d+e*f)
}
