package model

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type MLPConfig struct {
	InputSize         int
	OutputSize        int
	HiddenSize        []int
	WeightInit        WeightInit
	BatchNormMomentum float64
}

type MLP struct {
	Sequential
}

func NewMLP(c *MLPConfig, s ...rand.Source) *MLP {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	// size
	size := append([]int{c.InputSize}, c.HiddenSize...)
	size = append(size, c.OutputSize)

	// layer
	// Affine -> BatchNorm -> ReLU -> ... -> Affine -> SoftmaxWithLoss
	layers := make([]Layer, 0)
	for i := 0; i < len(size)-2; i++ {
		S, H := size[i], size[i+1]

		layers = append(layers, &layer.Affine{
			W: matrix.Randn(S, H, s[0]).MulC(c.WeightInit(S)),
			B: matrix.Zero(1, H),
		})

		layers = append(layers, &layer.BatchNorm{
			Gamma:    matrix.One(1, H),
			Beta:     matrix.Zero(1, H),
			Momentum: c.BatchNormMomentum,
		})

		layers = append(layers, &layer.ReLU{})
	}

	H, O := size[len(size)-2], size[len(size)-1]

	layers = append(layers, &layer.Affine{
		W: matrix.Randn(H, O, s[0]).MulC(c.WeightInit(H)),
		B: matrix.Zero(1, O),
	})

	layers = append(layers, &layer.SoftmaxWithLoss{}) // loss function

	// new
	return &MLP{
		Sequential{
			Layer:  layers,
			Source: s[0],
		},
	}
}

func (m *MLP) Summary() []string {
	s := []string{fmt.Sprintf("%T", m)}
	for _, l := range m.Layers() {
		s = append(s, l.String())
	}

	return s
}
