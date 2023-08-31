package model

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type QNetConfig struct {
	InputSize  int
	OutputSize int
	HiddenSize int
	WeightInit WeightInit
}

type QNet struct {
	Sequential
}

func NewQNet(c *QNetConfig, s ...rand.Source) *QNet {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	S, H, O := c.InputSize, c.HiddenSize, c.OutputSize

	layers := []Layer{
		&layer.Affine{
			W: matrix.Randn(S, H, s[0]).MulC(c.WeightInit(S)),
			B: matrix.Zero(1, H),
		},
		&layer.ReLU{},
		&layer.Affine{
			W: matrix.Randn(H, O, s[0]).MulC(c.WeightInit(H)),
			B: matrix.Zero(1, O),
		},
		&layer.MeanSquaredError{},
	}

	return &QNet{
		Sequential{
			Layer:  layers,
			Source: s[0],
		},
	}
}

func (m *QNet) Forward(x matrix.Matrix) matrix.Matrix {
	opts := layer.Opts{Train: true, Source: m.Source}
	return m.Predict(x, opts)
}

func (m *QNet) MeanSquaredError(y, t matrix.Matrix) matrix.Matrix {
	opts := layer.Opts{Train: true, Source: m.Source}
	return m.Layer[len(m.Layer)-1].Forward(y, t, opts)
}

func (m *QNet) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}
