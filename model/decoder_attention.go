package model

import (
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type AttentionDecoder struct {
	TimeEmbedding *layer.TimeEmbedding
	TimeLSTM      *layer.TimeLSTM
	TimeAttention *layer.TimeAttention
	TimeAffine    *layer.TimeAffine
	Source        rand.Source
}

func NewAttentionDecoder(c *RNNLMConfig, s ...rand.Source) *AttentionDecoder {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	return &AttentionDecoder{
		TimeEmbedding: &layer.TimeEmbedding{
			W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
		},
		TimeLSTM: &layer.TimeLSTM{
			Wx:       matrix.Randn(D, 4*H, s[0]).MulC(c.WeightInit(D)),
			Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 4*H),
			Stateful: true,
		},
		TimeAttention: &layer.TimeAttention{},
		TimeAffine: &layer.TimeAffine{
			W: matrix.Randn(2*H, V, s[0]).MulC(c.WeightInit(2 * H)),
			B: matrix.Zero(1, V),
		},
		Source: s[0],
	}
}

func (m *AttentionDecoder) Forward(xs, enchs []matrix.Matrix) []matrix.Matrix {
	m.TimeLSTM.SetState(enchs[len(enchs)-1])

	out := m.TimeEmbedding.Forward(xs, nil)
	dechs := m.TimeLSTM.Forward(out, nil)
	c := m.TimeAttention.Forward(enchs, dechs)
	concat := Concat(c, dechs)
	score := m.TimeAffine.Forward(concat, nil)
	return score
}

func (m *AttentionDecoder) Backward(dscore []matrix.Matrix) []matrix.Matrix {
	dout := m.TimeAffine.Backward(dscore)
	H := len(dout[0][0]) / 2

	dc, ddechs0 := Split(dout, H)
	denchs, ddechs1 := m.TimeAttention.Backward(dc)
	ddechs := layer.TimeAdd(ddechs0, ddechs1)
	ldout := m.TimeLSTM.Backward(ddechs)
	m.TimeEmbedding.Backward(ldout)

	denchs[len(denchs)-1] = denchs[len(denchs)-1].Add(m.TimeLSTM.DH())
	return denchs
}

func (m *AttentionDecoder) Generate(enchs []matrix.Matrix, startID, length int) []int {
	m.TimeLSTM.SetState(enchs[len(enchs)-1])
	sampled := make([]int, 0)

	x := startID
	for i := 0; i < length; i++ {
		xs := []matrix.Matrix{{{float64(x)}}}

		out := m.TimeEmbedding.Forward(xs, nil)
		dechs := m.TimeLSTM.Forward(out, nil)
		c := m.TimeAttention.Forward(enchs, dechs)
		concat := Concat(c, dechs)
		score := m.TimeAffine.Forward(concat, nil)

		x = Argmax(score)
		sampled = append(sampled, x)
	}

	return sampled
}

func (m *AttentionDecoder) Layers() []TimeLayer {
	return []TimeLayer{
		m.TimeEmbedding,
		m.TimeLSTM,
		m.TimeAffine,
	}
}

func (l *AttentionDecoder) Params() []matrix.Matrix {
	return []matrix.Matrix{
		l.TimeEmbedding.W,
		l.TimeLSTM.Wx,
		l.TimeLSTM.Wh,
		l.TimeLSTM.B,
		l.TimeAffine.W,
		l.TimeAffine.B,
	}
}

func (l *AttentionDecoder) Grads() []matrix.Matrix {
	return []matrix.Matrix{
		l.TimeEmbedding.DW,
		l.TimeLSTM.DWx,
		l.TimeLSTM.DWh,
		l.TimeLSTM.DB,
		l.TimeAffine.DW,
		l.TimeAffine.DB,
	}
}

func (l *AttentionDecoder) SetParams(p ...matrix.Matrix) {
	l.TimeEmbedding.W = p[0]
	l.TimeLSTM.Wx = p[1]
	l.TimeLSTM.Wh = p[2]
	l.TimeLSTM.B = p[3]
	l.TimeAffine.W = p[4]
	l.TimeAffine.B = p[5]
}
