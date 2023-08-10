package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeAttention struct {
	layer []*Attention
}

func (l *TimeAttention) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *TimeAttention) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *TimeAttention) SetParams(p ...matrix.Matrix) {}
func (l *TimeAttention) SetState(_ ...matrix.Matrix)  {}
func (l *TimeAttention) ResetState()                  {}
func (l *TimeAttention) String() string               { return fmt.Sprintf("%T", l) }

func (l *TimeAttention) Forward(hsenc, hsdec []matrix.Matrix) []matrix.Matrix {
	T := len(hsdec)
	l.layer = make([]*Attention, T)
	out := make([]matrix.Matrix, T)

	for t := 0; t < T; t++ {
		l.layer[t] = &Attention{
			AttentionWeight: &AttentionWeight{
				Softmax: &Softmax{},
			},
			WeightSum: &WeightSum{},
		}

		out[t] = l.layer[t].Forward(hsenc, hsdec[t])
	}

	return out
}

func (l *TimeAttention) Backward(dout []matrix.Matrix) ([]matrix.Matrix, []matrix.Matrix) {
	T := len(dout)
	dhsenc := ZeroLike(dout)
	dhsdec := make([]matrix.Matrix, T)

	for t := 0; t < T; t++ {
		dhs, dh := l.layer[t].Backward(dout[t])
		dhsenc = TimeAdd(dhsenc, dhs)
		dhsdec[t] = dh
	}

	return dhsenc, dhsdec
}
