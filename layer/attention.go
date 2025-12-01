package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/tensor"
)

type Attention struct {
	AttentionWeight *AttentionWeight
	WeightSum       *WeightSum
}

func (l *Attention) Params() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *Attention) Grads() []matrix.Matrix {
	return make([]matrix.Matrix, 0)
}

func (l *Attention) SetParams(p ...matrix.Matrix) {
	// noop
}

func (l *Attention) String() string {
	return fmt.Sprintf("%T", l)
}

func (l *Attention) Forward(hs []matrix.Matrix, h matrix.Matrix) matrix.Matrix {
	a := l.AttentionWeight.Forward(hs, h)
	out := l.WeightSum.Forward(hs, a)
	return out
}

func (l *Attention) Backward(dout matrix.Matrix) ([]matrix.Matrix, matrix.Matrix) {
	dhs0, da := l.WeightSum.Backward(dout)
	dhs1, dh := l.AttentionWeight.Backward(da)
	dhs := tensor.Add(dhs0, dhs1)
	return dhs, dh
}
