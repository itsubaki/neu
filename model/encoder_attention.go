package model

import (
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
)

type AttentionEncoder struct {
	Encoder
}

func NewAttentionEncoder(c *RNNLMConfig, s ...rand.Source) *AttentionEncoder {
	return &AttentionEncoder{
		Encoder: *NewEncoder(c, s...),
	}
}

func (m *AttentionEncoder) Forward(xs []matrix.Matrix) []matrix.Matrix {
	xs = m.TimeEmbedding.Forward(xs, nil)
	hs := m.TimeLSTM.Forward(xs, nil)
	return hs
}

func (m *AttentionEncoder) Backward(dhs []matrix.Matrix) {
	dout := m.TimeLSTM.Backward(dhs)
	m.TimeEmbedding.Backward(dout)
}
