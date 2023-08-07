package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type PeekyDecoder struct {
	Decoder
	cacheH int
}

func NewPeekyDecoder(c *DecoderConfig, s ...rand.Source) *PeekyDecoder {
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
	m.cacheH = H

	out := m.TimeEmbedding.Forward(xs, nil)
	out = m.TimeLSTM.Forward(Concat(hs, out), nil)
	score := m.TimeAffine.Forward(Concat(hs, out), nil)
	return score
}

func (m *PeekyDecoder) Backward(dscore []matrix.Matrix) matrix.Matrix {
	dout := m.TimeAffine.Backward(dscore)
	dout, dhs0 := Split(dout, m.cacheH)
	dout = m.TimeLSTM.Backward(dout)
	dout, dhs1 := Split(dout, m.cacheH)
	m.TimeEmbedding.Backward(dout)

	dhs := append(dhs0, dhs1...)
	dh := m.TimeLSTM.DH().Add(SumAxis1(dhs))
	return dh
}

func (m *PeekyDecoder) Generate(h matrix.Matrix, startID, length int) []int {
	H := len(h[0])
	peekyH := []matrix.Matrix{matrix.Reshape(h, 1, H)}
	m.TimeLSTM.SetState(h)

	sampled := []int{startID}
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

func Concat(a, b []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(a))
	for t := 0; t < len(a); t++ {
		out[t] = make(matrix.Matrix, len(a[t]))

		for i := 0; i < len(a[t]); i++ {
			out[t][i] = append(out[t][i], a[t][i]...)
		}

		for i := 0; i < len(b[t]); i++ {
			out[t][i] = append(out[t][i], b[t][i]...)
		}
	}

	return out
}

func Split(dout []matrix.Matrix, H int) ([]matrix.Matrix, []matrix.Matrix) {
	a, b := make([]matrix.Matrix, len(dout)), make([]matrix.Matrix, len(dout))
	for t := range dout {
		a[t], b[t] = matrix.New(), matrix.New()
		for _, r := range dout[t] {
			a[t] = append(a[t], r[H:])
			b[t] = append(b[t], r[:H])
		}
	}

	return a, b
}

func SumAxis1(dhs []matrix.Matrix) matrix.Matrix {
	out := make(matrix.Matrix, len(dhs))
	for t := range dhs {
		out[t] = dhs[t].SumAxis0()[0]
	}

	return out
}
