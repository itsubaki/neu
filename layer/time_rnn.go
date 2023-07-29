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
	Stateful      bool
}

func (l *TimeRNN) Params() []matrix.Matrix      { return []matrix.Matrix{l.Wx, l.Wh, l.B} }
func (l *TimeRNN) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DWx, l.DWh, l.DWB} }
func (l *TimeRNN) SetParams(p ...matrix.Matrix) { l.Wx, l.Wh, l.B = p[0], p[1], p[2] }
func (l *TimeRNN) SetState(h ...matrix.Matrix)  { l.h = h[0] }
func (l *TimeRNN) ResetState()                  { l.h = matrix.New() }

func (l *TimeRNN) Forward(xs []matrix.Matrix, opts ...Opts) []matrix.Matrix {
	// xs(Time, N, D), Wx(D, H)
	T, N, H := len(xs), len(xs[0]), len(l.Wx[0])

	if !l.Stateful || len(l.h) == 0 {
		l.h = matrix.Zero(N, H)
	}

	l.layer = make([]*RNN, T)
	hs := make([]matrix.Matrix, 0)
	for t := 0; t < T; t++ {
		l.layer[t] = &RNN{Wx: l.Wx, Wh: l.Wh, B: l.B}
		l.h = l.layer[t].Forward(xs[t], l.h, opts...)
		hs = append(hs, l.h)
	}

	return hs
}

func (l *TimeRNN) Backward(dhs []matrix.Matrix) []matrix.Matrix {
	// dhs(Time, N, H)
	T, N, H := len(dhs), len(dhs[0]), len(dhs[0][0])

	// dxs(Time, N, D)
	dxs := make([]matrix.Matrix, 0)
	dhl := matrix.Zero(N, H)
	grads := make([]matrix.Matrix, 3)

	for t := T - 1; t > -1; t-- {
		// dx(N, D), dh(N, H)
		dx, dh := l.layer[t].Backward(dhs[t].Add(dhl))
		dxs, dhl = append(dxs, dx), dh

		for i, g := range l.layer[t].Grads() {
			grads[i] = grads[i].Add(g)
		}
	}

	l.DWx, l.DWh, l.DWB = grads[0], grads[1], grads[2]
	l.dh = dhl
	return dxs
}

func (l *TimeRNN) String() string {
	a, b := l.Wx.Dimension()
	c, d := l.Wh.Dimension()
	e, f := l.B.Dimension()
	return fmt.Sprintf("%T: Wx(%v, %v)*T, Wh(%v, %v)*T, B(%v, %v)*T: %v*T", l, a, b, c, d, e, f, a*b+c*d+e*f)
}
