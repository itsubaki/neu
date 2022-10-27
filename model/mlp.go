package model

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/numerical"
)

type MLPConfig struct {
	InputSize         int
	HiddenSize        []int
	OutputSize        int
	WeightDecayLambda float64
	WeightInit        WeightInit
	Optimizer         Optimizer
}

type MLP struct {
	size              []int
	layer             []Layer
	last              Layer
	weightDecayLambda float64
	optimizer         Optimizer
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
	layers = layers[:len(layers)-1] // remove last ReLU

	// new
	return &MLP{
		size:              size,
		layer:             layers,
		last:              &layer.SoftmaxWithLoss{},
		weightDecayLambda: c.WeightDecayLambda,
		optimizer:         c.Optimizer,
	}
}

func (m *MLP) Predict(x matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	for _, l := range m.layer {
		x = l.Forward(x, nil, opts...)
	}

	return x
}

func (m *MLP) Loss(x, t matrix.Matrix, opts ...layer.Opts) matrix.Matrix {
	y := m.Predict(x, opts...)
	loss := m.last.Forward(y, t, opts...)
	return loss
}

func (m *MLP) Gradient(x, t matrix.Matrix) [][]matrix.Matrix {
	// forward
	m.Loss(x, t, layer.Opts{Train: true})

	// backward
	dout, _ := m.last.Backward(matrix.New([]float64{1}))
	for i := len(m.layer) - 1; i > -1; i-- {
		dout, _ = m.layer[i].Backward(dout)
	}

	// gradient
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.layer {
		grads = append(grads, l.Grads())
	}

	return grads
}

func (m *MLP) NumericalGradient(x, t matrix.Matrix) [][]matrix.Matrix {
	lossW := func(w ...float64) float64 {
		return m.Loss(x, t, layer.Opts{Train: true})[0][0]
	}

	grad := func(f func(x ...float64) float64, x matrix.Matrix) matrix.Matrix {
		out := make(matrix.Matrix, 0)
		for _, r := range x {
			out = append(out, numerical.Gradient(f, r))
		}

		return out
	}

	// gradient
	grads := make([][]matrix.Matrix, 0)
	for _, l := range m.layer {
		g := make([]matrix.Matrix, 0)
		for _, p := range l.Params() {
			g = append(g, grad(lossW, p))
		}

		grads = append(grads, g)
	}

	return grads
}

func (m *MLP) Optimize(grads [][]matrix.Matrix) {
	// params
	params := make([][]matrix.Matrix, 0)
	for _, l := range m.layer {
		params = append(params, l.Params())
	}

	// update
	updated := m.optimizer.Update(params, grads)
	for i, l := range m.layer {
		l.SetParams(updated[i])
	}
}
