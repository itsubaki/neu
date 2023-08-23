package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/tensor"
)

type TimeBiLSTM struct {
	F *TimeLSTM
	B *TimeLSTM
}

func (l *TimeBiLSTM) Params() []matrix.Matrix {
	return []matrix.Matrix{l.F.Wx, l.F.Wh, l.F.B, l.B.Wx, l.B.Wh, l.B.B}
}

func (l *TimeBiLSTM) Grads() []matrix.Matrix {
	return []matrix.Matrix{l.F.DWx, l.F.DWh, l.F.DB, l.B.DWx, l.B.DWh, l.B.DB}
}

func (l *TimeBiLSTM) SetParams(p ...matrix.Matrix) {
	l.F.SetParams(p[0], p[1], p[2])
	l.B.SetParams(p[3], p[4], p[5])
}

func (l *TimeBiLSTM) SetState(h ...matrix.Matrix) {}
func (l *TimeBiLSTM) ResetState()                 {}

func (l *TimeBiLSTM) String() string {
	a, b := l.F.Wx.Dim()
	c, d := l.F.Wh.Dim()
	e, f := l.F.B.Dim()
	return fmt.Sprintf("%T: Wx(%v, %v), Wh(%v, %v), B(%v, %v): %v", l, a, b, c, d, e, f, (a*b+c*d+e*f)*2)
}

func (l *TimeBiLSTM) Forward(xs, _ []matrix.Matrix, _ ...Opts) []matrix.Matrix {
	o1 := l.F.Forward(xs, nil)
	o2 := l.B.Forward(tensor.Reverse(xs), nil)
	return tensor.Concat(o1, tensor.Reverse(o2))
}

func (l *TimeBiLSTM) Backward(dhs []matrix.Matrix) []matrix.Matrix {
	H := len(dhs[0][0]) / 2
	do1, do2 := tensor.Split(dhs, H)
	do2r := tensor.Reverse(do2)

	dxs1 := l.F.Backward(do1)
	dxs2 := l.B.Backward(do2r)
	dxs := tensor.Add(dxs1, tensor.Reverse(dxs2))
	return dxs
}
