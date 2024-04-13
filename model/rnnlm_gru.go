package model

import (
	"fmt"
	randv2 "math/rand/v2"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

type GRULM struct {
	RNNLM
}

func NewGRULM(c *LSTMLMConfig, s ...randv2.Source) *GRULM {
	if len(s) == 0 {
		s = append(s, rand.MustNewSource())
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
		&layer.TimeGRU{
			Wx:       matrix.Randn(D, 3*H, s[0]).MulC(c.WeightInit(D)),
			Wh:       matrix.Randn(H, 3*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 3*H),
			Stateful: true,
		},
		&layer.TimeDropout{
			Ratio: c.DropoutRatio,
		},
		&layer.TimeGRU{
			Wx:       matrix.Randn(H, 3*H, s[0]).MulC(c.WeightInit(H)),
			Wh:       matrix.Randn(H, 3*H, s[0]).MulC(c.WeightInit(H)),
			B:        matrix.Zero(1, 3*H),
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

	return &GRULM{
		RNNLM{
			Layer:  layers,
			Source: s[0],
		},
	}
}

func (m *GRULM) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}
