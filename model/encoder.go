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
	TimeEmbedding *layer.TimeEmbedding
	TimeLSTM      *layer.TimeLSTM
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

func (m *Encoder) Forward(xs []matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	xs = m.TimeEmbedding.Forward(xs, nil, opts...) // (Time, N, D) (7, 128, 16)  gen(7, 1, 16)
	hs := m.TimeLSTM.Forward(xs, nil, opts...)     // (Time, N, H) (7, 128, 128) gen(7, 1, 128)
	m.hs = hs                                      // (Time, N, H)
	return hs[len(hs)-1]                           // hs[-1, N, H]
}

func (m *Encoder) Backward(dh matrix.Matrix) {
	dhs := Zero(m.hs)                // (Time, N, H)
	dhs[len(m.hs)-1] = dh            // dhs[-1, N, H] = dh[N, H]
	dout := m.TimeLSTM.Backward(dhs) //
	m.TimeEmbedding.Backward(dout)
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

func Zero(hs []matrix.Matrix) []matrix.Matrix {
	out := make([]matrix.Matrix, len(hs))
	for i := 0; i < len(hs); i++ {
		out[i] = matrix.Zero(hs[i].Dimension())
	}

	return out
}
