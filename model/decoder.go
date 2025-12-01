package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/math/tensor"
)

type Decoder struct {
	TimeEmbedding *layer.TimeEmbedding
	TimeLSTM      *layer.TimeLSTM
	TimeAffine    *layer.TimeAffine
	Source        randv2.Source
}

func NewDecoder(c *RNNLMConfig, s ...randv2.Source) *Decoder {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	return &Decoder{
		TimeEmbedding: &layer.TimeEmbedding{
			W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
		},
		TimeLSTM: &layer.TimeLSTM{
			Wx:       matrix.Randn(D, 4*H, s[0]).MulC(c.WeightInit(D)),
			Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 4*H),
			Stateful: true,
		},
		TimeAffine: &layer.TimeAffine{
			W: matrix.Randn(H, V, s[0]).MulC(c.WeightInit(H)),
			B: matrix.Zero(1, V),
		},
		Source: s[0],
	}
}

func (m *Decoder) Forward(xs []matrix.Matrix, h matrix.Matrix) []matrix.Matrix {
	m.TimeLSTM.SetState(h)                  // h(128, 128)
	out := m.TimeEmbedding.Forward(xs, nil) // out(4, 128, 16)  xs(4, 128, 1)
	out = m.TimeLSTM.Forward(out, nil)      // out(4, 128, 128) out(4, 128, 16)
	score := m.TimeAffine.Forward(out, nil) // score(4, 128, 13)
	return score
}

func (m *Decoder) Backward(dscore []matrix.Matrix) matrix.Matrix {
	dout := m.TimeAffine.Backward(dscore) // (4, 128, 128)
	dout = m.TimeLSTM.Backward(dout)      // (4, 128, 16)
	m.TimeEmbedding.Backward(dout)        //
	return m.TimeLSTM.DH()                // (128, 128)
}

func (m *Decoder) Generate(h matrix.Matrix, startID, length int) []int {
	m.TimeLSTM.SetState(h) // (1, 128)
	sampled := make([]int, 0)

	x := startID
	for range length {
		xs := []matrix.Matrix{{{float64(x)}}}

		out := m.TimeEmbedding.Forward(xs, nil) // (1, 1, 16)
		out = m.TimeLSTM.Forward(out, nil)      // (1, 1, 128)
		score := m.TimeAffine.Forward(out, nil) // (1, 1, 13)

		x = tensor.Argmax(score) // 0~12
		sampled = append(sampled, x)
	}

	return sampled
}

func (m *Decoder) Summary() []string {
	return []string{
		fmt.Sprintf("%T", m),
		m.TimeEmbedding.String(),
		m.TimeLSTM.String(),
		m.TimeAffine.String(),
	}
}

func (m *Decoder) Layers() []TimeLayer {
	return []TimeLayer{
		m.TimeEmbedding,
		m.TimeLSTM,
		m.TimeAffine,
	}
}

func (l *Decoder) Params() []matrix.Matrix {
	return []matrix.Matrix{
		l.TimeEmbedding.W,
		l.TimeLSTM.Wx,
		l.TimeLSTM.Wh,
		l.TimeLSTM.B,
		l.TimeAffine.W,
		l.TimeAffine.B,
	}
}

func (l *Decoder) Grads() []matrix.Matrix {
	return []matrix.Matrix{
		l.TimeEmbedding.DW,
		l.TimeLSTM.DWx,
		l.TimeLSTM.DWh,
		l.TimeLSTM.DB,
		l.TimeAffine.DW,
		l.TimeAffine.DB,
	}
}

func (l *Decoder) SetParams(p ...matrix.Matrix) {
	l.TimeEmbedding.W = p[0]
	l.TimeLSTM.Wx = p[1]
	l.TimeLSTM.Wh = p[2]
	l.TimeLSTM.B = p[3]
	l.TimeAffine.W = p[4]
	l.TimeAffine.B = p[5]
}
