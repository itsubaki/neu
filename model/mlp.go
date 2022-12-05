package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

type MLPConfig struct {
	InputSize    int
	OutputSize   int
	HiddenSize   []int
	WeightInit   WeightInit
	UseBatchNorm bool
}

type MLP struct {
	seq *Sequential
}

func NewMLP(c *MLPConfig) *MLP {
	// size
	size := append([]int{c.InputSize}, c.HiddenSize...)
	size = append(size, c.OutputSize)

	// layers
	// Affine -> BatchNorm -> ReLU -> ... -> Affine -> SoftmaxWithLoss
	layers := make([]Layer, 0)
	for i := 0; i < len(size)-2; i++ {
		layers = append(layers, &layer.Affine{
			W: matrix.Randn(size[i], size[i+1]).MulC(c.WeightInit(size[i])),
			B: matrix.Zero(1, size[i+1]),
		})

		if c.UseBatchNorm {
			layers = append(layers, &layer.BatchNorm{
				Gamma: matrix.One(1, size[i+1]),
				Beta:  matrix.Zero(1, size[i+1]),
			})
		}

		layers = append(layers, &layer.ReLU{})
	}

	layers = append(layers, &layer.Affine{
		W: matrix.Randn(size[len(size)-2], size[len(size)-1]).MulC(c.WeightInit(size[len(size)-2])),
		B: matrix.Zero(1, size[len(size)-1]),
	})
	layers = append(layers, &layer.SoftmaxWithLoss{}) // loss function

	// new
	return &MLP{
		seq: NewSequential(layers...),
	}
}

func (m *MLP) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	return m.seq.Predict(x, opts...)
}

func (m *MLP) Forward(x, t matrix.Matrix) matrix.Matrix {
	return m.seq.Forward(x, t)
}

func (m *MLP) Backward() matrix.Matrix {
	return m.seq.Backward()
}

func (m *MLP) Layers() []Layer {
	return m.seq.Layers()
}

func (m *MLP) Params() [][]matrix.Matrix {
	return m.seq.Params()
}

func (m *MLP) Grads() [][]matrix.Matrix {
	return m.seq.Grads()
}
