package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type EncoderConfig struct {
	VocabSize   int
	WordVecSize int
	HiddenSize  int
	WeightInit  WeightInit
}

type Encoder struct {
	TimeEmbedding TimeLayer
	TimeLSTM      TimeLayer
	Source        rand.Source
	hs            []matrix.Matrix
}

func NewEncoder(c *EncoderConfig, s ...rand.Source) *Encoder {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
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

func (m *Encoder) Forward(xs []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	xs = m.TimeEmbedding.Forward(xs, nil, opts...) // (Time, N, D)
	hs := m.TimeLSTM.Forward(xs, nil, opts...)     // (Time, N, H)
	m.hs = hs                                      // (Time, N, H)
	return []matrix.Matrix{hs[len(hs)-1]}          // hs[-1, N, H]
}

func (m *Encoder) Backward(dh []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	dhs := Zero(m.hs)
	dhs[len(m.hs)-1] = dh[0]
	dout := m.TimeLSTM.Backward(dhs)
	dout = m.TimeEmbedding.Backward(dout)
	return dout
}

func Zero(hs []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(hs))
	for i := 0; i < len(hs); i++ {
		out[i] = matrix.Zero(hs[i].Dimension())
	}

	return out
}
