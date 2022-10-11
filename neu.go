package neu

import (
	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/optimizer"
)

var (
	_ Layer = (*layer.Add)(nil)
	_ Layer = (*layer.Mul)(nil)
	_ Layer = (*layer.ReLU)(nil)
	_ Layer = (*layer.Sigmoid)(nil)
	_ Layer = (*layer.Affine)(nil)
	_ Layer = (*layer.SoftmaxWithLoss)(nil)
)

var (
	_ Optimizer = (*optimizer.SGD)(nil)
	_ Optimizer = (*optimizer.Momentum)(nil)
)

type Layer interface {
	Forward(x, y matrix.Matrix) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
}

type Optimizer interface {
	Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix
}

type Neu struct {
	params    map[string]matrix.Matrix
	layers    []Layer
	lastLayer Layer
	Optimizer Optimizer
}

func New(inSize, hiddenSize, outSize int, weightInitStd float64) *Neu {
	// params
	params := make(map[string]matrix.Matrix)
	params["W1"] = matrix.Rand(inSize, hiddenSize).Func(func(v float64) float64 { return weightInitStd * v })
	params["B1"] = matrix.Rand(1, hiddenSize)
	params["W2"] = matrix.Rand(hiddenSize, outSize).Func(func(v float64) float64 { return weightInitStd * v })
	params["B2"] = matrix.Rand(1, outSize)

	return &Neu{
		params:    params,
		layers:    make([]Layer, 0),
		lastLayer: &layer.SoftmaxWithLoss{},
		Optimizer: &optimizer.SGD{LearningRate: 0.1},
	}
}

func (n *Neu) Predict(x matrix.Matrix) matrix.Matrix {
	n.layers = []Layer{
		&layer.Affine{W: n.params["W1"], B: n.params["B1"]},
		&layer.ReLU{},
		&layer.Affine{W: n.params["W2"], B: n.params["B2"]},
	}

	for _, l := range n.layers {
		x = l.Forward(x, nil)
	}

	return x
}

func (n *Neu) Loss(x, t matrix.Matrix) matrix.Matrix {
	y := n.Predict(x)
	return n.lastLayer.Forward(y, t)
}

func (n *Neu) Gradient(x, t matrix.Matrix) map[string]matrix.Matrix {
	// forward
	n.Loss(x, t)

	// backward
	dout, _ := n.lastLayer.Backward(matrix.New([]float64{1}))
	for i := len(n.layers) - 1; i > -1; i-- {
		dout, _ = n.layers[i].Backward(dout)
	}

	grads := make(map[string]matrix.Matrix)
	grads["W1"] = n.layers[0].(*layer.Affine).DW
	grads["B1"] = n.layers[0].(*layer.Affine).DB
	grads["W2"] = n.layers[2].(*layer.Affine).DW
	grads["B2"] = n.layers[2].(*layer.Affine).DB

	return grads
}

func (n *Neu) Optimize(grads map[string]matrix.Matrix) {
	n.params = n.Optimizer.Update(n.params, grads)
}
