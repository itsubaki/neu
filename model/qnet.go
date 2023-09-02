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

func (m *QNet) Loss(target, q matrix.Matrix) matrix.Matrix {
	return m.Layer[len(m.Layer)-1].Forward(target, q)
}

func (m *QNet) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}
