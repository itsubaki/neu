package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/math/tensor"
)

type Encoder struct {
	TimeEmbedding *layer.TimeEmbedding
	TimeLSTM      *layer.TimeLSTM
	Source        randv2.Source
	hs            []matrix.Matrix
}

func NewEncoder(c *RNNLMConfig, s ...randv2.Source) *Encoder {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	// layer
	return &Encoder{
		TimeEmbedding: &layer.TimeEmbedding{
			W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
		},
		TimeLSTM: &layer.TimeLSTM{
			Wx:       matrix.Randn(D, 4*H, s[0]).MulC(c.WeightInit(D)),
			Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 4*H),
			Stateful: false,
		},
		Source: s[0],
	}
}

func (m *Encoder) Forward(xs []matrix.Matrix) matrix.Matrix {
	xs = m.TimeEmbedding.Forward(xs, nil) // (Time, N, D) (7, 128, 16)
	hs := m.TimeLSTM.Forward(xs, nil)     // (Time, N, H) (7, 128, 128)
	m.hs = hs                             // (Time, N, H)
	return hs[len(hs)-1]                  // hs[-1, N, H]
}

func (m *Encoder) Backward(dh matrix.Matrix) {
	dhs := tensor.ZeroLike(m.hs)     // (Time, N, H)
	dhs[len(m.hs)-1] = dh            // dhs[-1, N, H] = dh[N, H]
	dout := m.TimeLSTM.Backward(dhs) //
	m.TimeEmbedding.Backward(dout)
}

func (m *Encoder) Summary() []string {
	return []string{
		fmt.Sprintf("%T", m),
		m.TimeEmbedding.String(),
		m.TimeLSTM.String(),
	}
}

func (m *Encoder) Layers() []TimeLayer {
	return []TimeLayer{
		m.TimeEmbedding,
		m.TimeLSTM,
	}
}

func (l *Encoder) Params() []matrix.Matrix {
	return []matrix.Matrix{
		l.TimeEmbedding.W,
		l.TimeLSTM.Wx,
		l.TimeLSTM.Wh,
		l.TimeLSTM.B,
	}
}
func (l *Encoder) Grads() []matrix.Matrix {
	return []matrix.Matrix{
		l.TimeEmbedding.DW,
		l.TimeLSTM.DWx,
		l.TimeLSTM.DWh,
		l.TimeLSTM.DB,
	}
}

func (l *Encoder) SetParams(p ...matrix.Matrix) {
	l.TimeEmbedding.W = p[0]
	l.TimeLSTM.Wx = p[1]
	l.TimeLSTM.Wh = p[2]
	l.TimeLSTM.B = p[3]
}
