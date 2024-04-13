package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/math/matrix"
)

type AttentionEncoder struct {
	Encoder
}

func NewAttentionEncoder(c *RNNLMConfig, s ...randv2.Source) *AttentionEncoder {
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

func (m *AttentionEncoder) Summary() []string {
	return []string{
		fmt.Sprintf("%T", m),
		m.TimeEmbedding.String(),
		m.TimeLSTM.String(),
	}
}
