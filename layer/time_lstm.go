package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeLSTM struct {
	Wx, Wh, B    matrix.Matrix // params
	DWx, DWh, DB matrix.Matrix // grads
	h, dh        matrix.Matrix // hidden state
	c            matrix.Matrix // cell state
	layer        []LSTM
	Stateful     bool
}

func (l *TimeLSTM) DH() matrix.Matrix            { return l.dh }
func (l *TimeLSTM) Params() []matrix.Matrix      { return []matrix.Matrix{l.Wx, l.Wh, l.B} }
func (l *TimeLSTM) Grads() []matrix.Matrix       { return []matrix.Matrix{l.DWx, l.DWh, l.DB} }
func (l *TimeLSTM) SetParams(p ...matrix.Matrix) { l.Wx, l.Wh, l.B = p[0], p[1], p[2] }
func (l *TimeLSTM) SetState(s ...matrix.Matrix) {
	if len(s) == 1 {
		s = append(s, matrix.New())
	}
	l.h, l.c = s[0], s[1]
}
func (l *TimeLSTM) ResetState() { l.h, l.c = matrix.New(), matrix.New() }
func (l *TimeLSTM) String() string {
	a, b, c, d, e, f := len(l.Wx), len(l.Wx[0]), len(l.Wh), len(l.Wh[0]), len(l.B), len(l.B[0])
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, a*b+c*d+e*f)
}

func (l *TimeLSTM) Forward(xs, _ []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	T, N, H := len(xs), len(xs[0]), len(l.Wh) // 7, 128, 128
	l.layer = make([]LSTM, T)                 //
	hs := make([]matrix.Matrix, T)            // (7, 128, 128)

	if !l.Stateful || len(l.h) == 0 {
		l.h = matrix.Zero(N, H) // (128, 128)
	}
	if !l.Stateful || len(l.c) == 0 {
		l.c = matrix.Zero(N, H) // (128, 128)
	}

	for t := 0; t < T; t++ {
		l.layer[t] = LSTM{Wx: l.Wx, Wh: l.Wh, B: l.B}  // Wx(D, 4H), Wh(H, 4H), B(1, 4H)
		l.h, l.c = l.layer[t].Forward(xs[t], l.h, l.c) // h(128, 128), c(128, 128)
		hs[t] = l.h
	}

	return hs
}

func (l *TimeLSTM) Backward(dhs []matrix.Matrix) []matrix.Matrix {
	T, N, H := len(dhs), len(dhs[0]), len(dhs[0][0]) // dhs(Time, N, H)
	dxs := make([]matrix.Matrix, T)                  // dxs(Time, N, D)
	dh := matrix.Zero(N, H)                          // dh(N, H)
	dc := matrix.Zero(N, H)                          // dc(N, H)

	grads := []matrix.Matrix{
		matrix.Zero(1, 1), // DWx(D, 4H)
		matrix.Zero(1, 1), // DWh(H, 4H)
		matrix.Zero(1, 1), // DB(1, 4H)
	}

	for t := T - 1; t > -1; t-- {
		dxs[t], dh, dc = l.layer[t].Backward(dhs[t].Add(dh), dc) // dx(N, D), dh(N, H)

		// grads
		for i, g := range l.layer[t].Grads() {
			grads[i] = g.Add(grads[i]) // Broadcast
		}
	}

	l.DWx, l.DWh, l.DB = grads[0], grads[1], grads[2]
	l.dh = dh
	return dxs
}
