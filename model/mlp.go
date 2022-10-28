package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type MLPConfig struct {
	InputSize  int
	HiddenSize []int
	OutputSize int
	WeightInit WeightInit
}

type MLP struct {
	seq *Sequential
}

func NewMLP(c *MLPConfig) *MLP {
	// size
	size := append([]int{c.InputSize}, c.HiddenSize...)
	size = append(size, c.OutputSize)

	// layers
	layers := make([]Layer, 0) // init
	for i := 0; i < len(size)-1; i++ {
		layers = append(layers, &layer.Affine{
			W: matrix.Randn(size[i], size[i+1]).MulC(c.WeightInit(size[i])),
			B: matrix.Zero(1, size[i+1]),
		})
		layers = append(layers, &layer.ReLU{})
	}
	layers = layers[:len(layers)-1]                   // remove last ReLU
	layers = append(layers, &layer.SoftmaxWithLoss{}) // loss function

	// new
	return &MLP{
		seq: &Sequential{
			Layer: layers,
		},
	}
}

func (m *MLP) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	return m.seq.Predict(x, opts...)
}

func (m *MLP) Forward(x, t matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	return m.seq.Forward(x, t, opts...)
}

func (m *MLP) Backward(x, t matrix.Matrix) matrix.Matrix {
	return m.seq.Backward(x, t)
}

func (m *MLP) Optimize(opt Optimizer) [][]matrix.Matrix {
	return m.seq.Optimize(opt)
}

func (m *MLP) Params() [][]matrix.Matrix {
	return m.seq.Params()
}

func (m *MLP) Grads() [][]matrix.Matrix {
	return m.seq.Grads()
}
