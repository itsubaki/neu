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
	HiddenSize []int
	WeightInit WeightInit
}

type QNet struct {
	Sequential
}

func NewQNet(c *QNetConfig, s ...rand.Source) *QNet {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	size := append([]int{c.InputSize}, c.HiddenSize...)
	size = append(size, c.OutputSize)

	// layer
	layers := make([]Layer, 0)
	for i := 0; i < len(size)-2; i++ {
		S, H := size[i], size[i+1]

		layers = append(layers, &layer.Affine{
			W: matrix.Randn(S, H, s[0]).MulC(c.WeightInit(S)),
			B: matrix.Zero(1, H),
		})

		layers = append(layers, &layer.ReLU{})
	}

	H, O := size[len(size)-2], size[len(size)-1]

	layers = append(layers, &layer.Affine{
		W: matrix.Randn(H, O, s[0]).MulC(c.WeightInit(H)),
		B: matrix.Zero(1, O),
	})

	layers = append(layers, &layer.MeanSquaredError{})

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

func (m *QNet) Sync(target *QNet) {
	params := make([][]matrix.Matrix, len(m.Params()))
	for i := range m.Params() {
		params[i] = make([]matrix.Matrix, len(m.Params()[i]))
		for j := range m.Params()[i] {
			params[i][j] = matrix.New(m.Params()[i][j]...)
		}
	}

	target.SetParams(params)
}
