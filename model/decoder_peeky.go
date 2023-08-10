package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type PeekyDecoder struct {
	Decoder
	H int
}

func NewPeekyDecoder(c *RNNLMConfig, s ...rand.Source) *PeekyDecoder {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	return &PeekyDecoder{
		Decoder: Decoder{
			TimeEmbedding: &layer.TimeEmbedding{
				W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
			},
			TimeLSTM: &layer.TimeLSTM{
				Wx:       matrix.Randn(H+D, 4*H, s[0]).MulC(c.WeightInit(H + D)),
				Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
				B:        matrix.Zero(1, 4*H),
				Stateful: true,
			},
			TimeAffine: &layer.TimeAffine{
				W: matrix.Randn(H+H, V, s[0]).MulC(c.WeightInit(H + H)),
				B: matrix.Zero(1, V),
			},
			Source: s[0],
		},
	}
}

func (m *PeekyDecoder) Forward(xs []matrix.Matrix, h matrix.Matrix) []matrix.Matrix {
	T, H := len(xs), len(h[0])
	hs := matrix.Repeat(h, T)
	m.TimeLSTM.SetState(h)
	m.H = H

	out := m.TimeEmbedding.Forward(xs, nil)
	out = m.TimeLSTM.Forward(Concat(hs, out), nil)
	score := m.TimeAffine.Forward(Concat(hs, out), nil)
	return score
}

func (m *PeekyDecoder) Backward(dscore []matrix.Matrix) matrix.Matrix {
	dout := m.TimeAffine.Backward(dscore)
	dout, dhs0 := Split(dout, m.H)
	dout = m.TimeLSTM.Backward(dout)
	dout, dhs1 := Split(dout, m.H)
	m.TimeEmbedding.Backward(dout)

	dhs := append(dhs0, dhs1...)
	dh := m.TimeLSTM.DH().Add(layer.TimeSum(dhs))
	return dh
}

func (m *PeekyDecoder) Generate(h matrix.Matrix, startID, length int) []int {
	H := len(h[0])
	sampled := make([]int, 0)
	peekyH := []matrix.Matrix{matrix.Reshape(h, 1, H)}
	m.TimeLSTM.SetState(h)

	sampleID := startID
	for i := 0; i < length; i++ {
		xs := []matrix.Matrix{{{float64(sampleID)}}}

		out := m.TimeEmbedding.Forward(xs, nil)
		out = m.TimeLSTM.Forward(Concat(peekyH, out), nil)
		score := m.TimeAffine.Forward(Concat(peekyH, out), nil)

		sampleID = Argmax(score)
		sampled = append(sampled, sampleID)
	}

	return sampled
}
