package model

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type LSTMLMConfig struct {
	RNNLMConfig
	DropoutRatio float64
}

type LSTMLM struct {
	RNNLM
}

func NewLSTMLM(c *LSTMLMConfig, s ...rand.Source) *LSTMLM {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	V, D, H := c.VocabSize, c.WordVecSize, c.HiddenSize

	// layer
	layers := []TimeLayer{
		&layer.TimeEmbedding{
			W: matrix.Randn(V, D, s[0]).MulC(1.0 / 100),
		},
		&layer.TimeDropout{
			Ratio: c.DropoutRatio,
		},
		&layer.TimeLSTM{
			Wx:       matrix.Randn(D, 4*H, s[0]).MulC(c.WeightInit(D)),
			Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 4*H),
			Stateful: true,
		},
		&layer.TimeDropout{
			Ratio: c.DropoutRatio,
		},
		&layer.TimeLSTM{
			Wx:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			Wh:       matrix.Randn(H, 4*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 4*H),
			Stateful: true,
		},
		&layer.TimeDropout{
			Ratio: c.DropoutRatio,
		},
		&layer.TimeAffine{
			W: matrix.Randn(D, V, s[0]).MulC(c.WeightInit(H)),
			B: matrix.Zero(1, V),
		},
		&layer.TimeSoftmaxWithLoss{},
	}

	return &LSTMLM{
		RNNLM{
			Layer:  layers,
			Source: s[0],
		},
	}
}

func (m *LSTMLM) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}
