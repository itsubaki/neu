package model

import (
	"fmt"

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
	params            map[string]matrix.Matrix
	layer             []Layer
	last              Layer
	weightDecayLambda float64
	optimizer         Optimizer
}

func NewMLP(c *MLPConfig) *MLP {
	// size
	size := append([]int{c.InputSize}, c.HiddenSize...)
	size = append(size, c.OutputSize)

	// params
	params := make(map[string]matrix.Matrix)
	for i := 0; i < len(size)-1; i++ {
		params[W(i)] = matrix.Randn(size[i], size[i+1])
		params[B(i)] = matrix.Zero(1, size[i+1])
	}

	// weight init
	for i := 0; i < len(size)-1; i++ {
		params[W(i)] = params[W(i)].MulC(c.WeightInit(size[i]))
	}

	// layers
	layers := make([]Layer, 0) // init
	for i := 0; i < len(size)-1; i++ {
		layers = append(layers, &layer.Affine{W: params[W(i)], B: params[B(i)]})
		layers = append(layers, &layer.ReLU{})
	}
	layers = layers[:len(layers)-1] // remove last ReLU

	// new
	return &MLP{
		size:              size,
		params:            params,
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

	// decay
	var decay float64
	for i := 0; i < len(m.size)-1; i++ {
		decay = decay + 0.5*m.weightDecayLambda*m.params[W(i)].Pow2().Sum() // decay = decay + 1/2 * lambda * sum(W**2)
	}

	return loss.AddC(decay)
}

func (m *MLP) Gradient(x, t matrix.Matrix) map[string]matrix.Matrix {
	// forward
	m.Loss(x, t, layer.Opts{Train: true})

	// backward
	dout, _ := m.last.Backward(matrix.New([]float64{1}))
	for i := len(m.layer) - 1; i > -1; i-- {
		dout, _ = m.layer[i].Backward(dout)
	}

	// gradient
	grads := make(map[string]matrix.Matrix)
	var i int
	for j := 0; j < len(m.layer); j++ {
		if affine, ok := m.layer[j].(*layer.Affine); ok {
			grads[W(i)], grads[B(i)] = affine.DW, affine.DB
			i++
		}
	}

	// decay
	for i := 0; i < len(m.size)-1; i++ {
		grads[W(i)] = matrix.FuncWith(grads[W(i)], m.params[W(i)], decay(m.weightDecayLambda)) // grads[W] + lambda * params[W]
	}

	return grads
}

func (m *MLP) NumericalGradient(x, t matrix.Matrix) map[string]matrix.Matrix {
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
	grads := make(map[string]matrix.Matrix)
	for i := 0; i < len(m.size)-1; i++ {
		grads[W(i)], grads[B(i)] = grad(lossW, m.params[W(i)]), grad(lossW, m.params[B(i)])
	}

	// decay
	for i := 0; i < len(m.size)-1; i++ {
		grads[W(i)] = matrix.FuncWith(grads[W(i)], m.params[W(i)], decay(m.weightDecayLambda)) // grads[W] + lambda * params[W]
	}

	return grads
}

func (m *MLP) Optimize(grads map[string]matrix.Matrix) {
	m.params = m.optimizer.Update(m.params, grads)

	var i int
	for j := 0; j < len(m.layer); j++ {
		if affine, ok := m.layer[j].(*layer.Affine); ok {
			affine.W, affine.B = m.params[W(i)], m.params[B(i)]
			i++
		}
	}
}

func W(i int) string { return fmt.Sprintf("W%v", i+1) }

func B(i int) string { return fmt.Sprintf("B%v", i+1) }

func decay(lambda float64) func(a, b float64) float64 {
	return func(a, b float64) float64 { return a + lambda*b }
}
