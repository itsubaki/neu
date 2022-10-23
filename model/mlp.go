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
		params[fmt.Sprintf("W%v", i+1)] = matrix.Randn(size[i], size[i+1])
		params[fmt.Sprintf("B%v", i+1)] = matrix.Zero(1, size[i+1])
	}

	// weight init
	for i := 0; i < len(size)-1; i++ {
		W := fmt.Sprintf("W%v", i+1)
		params[W] = matrix.Func(params[W], func(v float64) float64 {
			return c.WeightInit(size[i]) * v
		})
	}

	// new
	return &MLP{
		size:              size,
		params:            params,
		layer:             make([]Layer, 0),
		last:              &layer.SoftmaxWithLoss{},
		weightDecayLambda: c.WeightDecayLambda,
		optimizer:         c.Optimizer,
	}
}

func (m *MLP) Predict(x matrix.Matrix) matrix.Matrix {
	m.layer = make([]Layer, 0) // init
	for i := 0; i < len(m.size)-1; i++ {
		m.layer = append(m.layer, &layer.Affine{
			W: m.params[fmt.Sprintf("W%v", i+1)],
			B: m.params[fmt.Sprintf("B%v", i+1)],
		})
		m.layer = append(m.layer, &layer.ReLU{})
	}
	m.layer = m.layer[:len(m.layer)-1] // remove last ReLU

	for _, l := range m.layer {
		x = l.Forward(x, nil)
	}

	return x
}

func (m *MLP) Loss(x, t matrix.Matrix) matrix.Matrix {
	y := m.Predict(x)
	loss := m.last.Forward(y, t)

	// decay
	var decay float64
	for i := 0; i < len(m.size)-1; i++ {
		sumw2 := m.params[fmt.Sprintf("W%v", i+1)].Func(func(v float64) float64 {
			return v * v
		}).Sum() // sum(W**2)
		decay = decay + 0.5*m.weightDecayLambda*sumw2 // 1/2 * lambda * sum(W**2)
	}

	return loss.Func(func(v float64) float64 { return v + decay })
}

func (m *MLP) Gradient(x, t matrix.Matrix) map[string]matrix.Matrix {
	// forward
	m.Loss(x, t)

	// backward
	dout, _ := m.last.Backward(matrix.New([]float64{1}))
	for i := len(m.layer) - 1; i > -1; i-- {
		dout, _ = m.layer[i].Backward(dout)
	}

	// gradient
	grads := make(map[string]matrix.Matrix)
	var i int
	for j := 0; j < len(m.layer); j++ {
		if _, ok := m.layer[j].(*layer.Affine); !ok {
			continue
		}

		grads[fmt.Sprintf("W%v", i+1)] = m.layer[j].(*layer.Affine).DW
		grads[fmt.Sprintf("B%v", i+1)] = m.layer[j].(*layer.Affine).DB
		i++
	}

	// decay
	for i := 0; i < len(m.size)-1; i++ {
		W := fmt.Sprintf("W%v", i+1)
		grads[W] = grads[W].FuncWith(m.params[W], func(a, b float64) float64 {
			return a + m.weightDecayLambda*b // grads[W] + lambda * params[W]
		})
	}

	return grads
}

func (m *MLP) NumericalGradient(x, t matrix.Matrix) map[string]matrix.Matrix {
	lossW := func(w ...float64) float64 {
		return m.Loss(x, t)[0][0]
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
		W, B := fmt.Sprintf("W%v", i+1), fmt.Sprintf("B%v", i+1)
		grads[W] = grad(lossW, m.params[W])
		grads[B] = grad(lossW, m.params[B])
	}

	// decay
	for i := 0; i < len(m.size)-1; i++ {
		W := fmt.Sprintf("W%v", i+1)
		grads[W] = grads[W].FuncWith(m.params[W], func(a, b float64) float64 {
			return a + m.weightDecayLambda*b // grads[W] + lambda * params[W]
		})
	}

	return grads
}

func (m *MLP) Optimize(grads map[string]matrix.Matrix) {
	m.params = m.optimizer.Update(m.params, grads)
}
