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

	sampled := []int{startID}
	sampleID := startID
	for i := 0; i < length; i++ {
		xs := []matrix.Matrix{matrix.New([]float64{float64(sampleID)})}
		out := m.TimeEmbedding.Forward(xs, nil) // (1, 1, 16)
		out = m.TimeLSTM.Forward(out, nil)      // (1, 1, 128)
		score := m.TimeAffine.Forward(out, nil) // (1, 1, 13)
		sampleID = Argmax(score)                // 0~12
		sampled = append(sampled, sampleID)
	}

	return sampled
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
