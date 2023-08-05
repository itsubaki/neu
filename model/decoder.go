package model

import (
	"math"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type DecoderConfig struct {
	VocabSize   int
	WordVecSize int
	HiddenSize  int
	WeightInit  WeightInit
}

type Decoder struct {
	TimeEmbedding *layer.TimeEmbedding
	TimeLSTM      *layer.TimeLSTM
	TimeAffine    *layer.TimeAffine
	Source        rand.Source
}

func NewDecoder(c *DecoderConfig, s ...rand.Source) *Decoder {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
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

func (m *Decoder) Forward(xs, h []matrix.Matrix, opts ...layer.Opts) []matrix.Matrix {
	m.TimeLSTM.SetState(h...)
	out := m.TimeEmbedding.Forward(xs, nil, opts...)
	out = m.TimeLSTM.Forward(out, nil, opts...)
	score := m.TimeAffine.Forward(out, nil, opts...)
	return score
}

func (m *Decoder) Backward(dscore []matrix.Matrix) []matrix.Matrix {
	dout := m.TimeAffine.Backward(dscore)
	dout = m.TimeLSTM.Backward(dout)
	dout = m.TimeEmbedding.Backward(dout)
	return []matrix.Matrix{m.TimeLSTM.DH()}
}

func (m *Decoder) Generate(h matrix.Matrix, startID, length int) []int {
	m.TimeLSTM.SetState(h)

	sampled := []int{startID}
	x := startID
	for {
		if len(sampled) >= length {
			break
		}

		xs := []matrix.Matrix{matrix.New([]float64{float64(x)})}
		out := m.TimeEmbedding.Forward(xs, nil)
		out = m.TimeLSTM.Forward(out, nil)
		score := m.TimeAffine.Forward(out, nil)
		sampled = append(sampled, Argmax(score))
	}

	return sampled
}

func Argmax(score []matrix.Matrix) int {
	arg := 0
	max := math.SmallestNonzeroFloat64
	for i, v := range Flatten(score) {
		if v > max {
			max = v
			arg = i
		}
	}

	return arg
}
