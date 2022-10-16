package neu

import (
	"math"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/numerical"
	"github.com/itsubaki/neu/optimizer"
)

var (
	_ Layer     = (*layer.Add)(nil)
	_ Layer     = (*layer.Mul)(nil)
	_ Layer     = (*layer.ReLU)(nil)
	_ Layer     = (*layer.Sigmoid)(nil)
	_ Layer     = (*layer.Affine)(nil)
	_ Layer     = (*layer.SoftmaxWithLoss)(nil)
	_ Optimizer = (*optimizer.SGD)(nil)
	_ Optimizer = (*optimizer.Momentum)(nil)
)

var (
	Xavier = func(prevNodeNum int) float64 { return math.Sqrt(1.0 / float64(prevNodeNum)) }
	He     = func(prevNodeNum int) float64 { return math.Sqrt(2.0 / float64(prevNodeNum)) }
	Std    = func(std float64) func(_ int) float64 { return func(_ int) float64 { return std } }
)

type Layer interface {
	Forward(x, y matrix.Matrix) matrix.Matrix
	Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix)
}

type Optimizer interface {
	Update(params, grads map[string]matrix.Matrix) map[string]matrix.Matrix
}

type WeightInit func(prevNodeNum int) float64

type Config struct {
	InputSize         int
	HiddenSize        int
	OutputSize        int
	WeightDecayLambda float64
	WeightInit        WeightInit
	Optimizer         Optimizer
}

type Neu struct {
	params            map[string]matrix.Matrix
	layer             []Layer
	last              Layer
	weightDecayLambda float64
	optimizer         Optimizer
}

func New(c *Config) *Neu {
	// params
	params := make(map[string]matrix.Matrix)
	params["W1"] = matrix.Randn(c.InputSize, c.HiddenSize)
	params["B1"] = matrix.Zero(1, c.HiddenSize)
	params["W2"] = matrix.Randn(c.HiddenSize, c.OutputSize)
	params["B2"] = matrix.Zero(1, c.OutputSize)

	// weight init
	params["W1"] = matrix.Func(params["W1"], func(v float64) float64 { return c.WeightInit(c.InputSize) * v })
	params["W2"] = matrix.Func(params["W2"], func(v float64) float64 { return c.WeightInit(c.HiddenSize) * v })

	// new
	return &Neu{
		params:            params,
		layer:             make([]Layer, 0),
		last:              &layer.SoftmaxWithLoss{},
		weightDecayLambda: c.WeightDecayLambda,
		optimizer:         c.Optimizer,
	}
}

func (n *Neu) Predict(x matrix.Matrix) matrix.Matrix {
	n.layer = []Layer{
		&layer.Affine{W: n.params["W1"], B: n.params["B1"]},
		&layer.ReLU{},
		&layer.Affine{W: n.params["W2"], B: n.params["B2"]},
	}

	for _, l := range n.layer {
		x = l.Forward(x, nil)
	}

	return x
}

func (n *Neu) Loss(x, t matrix.Matrix) matrix.Matrix {
	y := n.Predict(x)
	loss := n.last.Forward(y, t)

	// decay
	var decay float64
	for _, k := range []string{"W1", "W2"} {
		sump2 := matrix.Func(n.params[k], func(v float64) float64 { return v * v }).Sum() // sum(W**2)
		decay = decay + 0.5*n.weightDecayLambda*sump2                                     // decay = decay + (1/2 * lambda * sum(W**2))
	}

	return loss.Func(func(v float64) float64 { return v + decay })
}

func (n *Neu) Gradient(x, t matrix.Matrix) map[string]matrix.Matrix {
	// forward
	n.Loss(x, t)

	// backward
	dout, _ := n.last.Backward(matrix.New([]float64{1}))
	for i := len(n.layer) - 1; i > -1; i-- {
		dout, _ = n.layer[i].Backward(dout)
	}

	// gradient
	grads := make(map[string]matrix.Matrix)
	grads["W1"] = n.layer[0].(*layer.Affine).DW
	grads["B1"] = n.layer[0].(*layer.Affine).DB
	grads["W2"] = n.layer[2].(*layer.Affine).DW
	grads["B2"] = n.layer[2].(*layer.Affine).DB

	// decay
	grads["W1"] = matrix.FuncWith(grads["W1"], n.params["W1"], func(a, b float64) float64 { return a + n.weightDecayLambda*b })
	grads["W2"] = matrix.FuncWith(grads["W2"], n.params["W2"], func(a, b float64) float64 { return a + n.weightDecayLambda*b })

	return grads
}

func (n *Neu) NumericalGradient(x, t matrix.Matrix) map[string]matrix.Matrix {
	lossW := func(w ...float64) float64 {
		return n.Loss(x, t)[0][0]
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
	grads["W1"] = grad(lossW, n.params["W1"])
	grads["B1"] = grad(lossW, n.params["B1"])
	grads["W2"] = grad(lossW, n.params["W2"])
	grads["B2"] = grad(lossW, n.params["B2"])

	// decay
	grads["W1"] = matrix.FuncWith(grads["W1"], n.params["W1"], func(a, b float64) float64 { return a + n.weightDecayLambda*b })
	grads["W2"] = matrix.FuncWith(grads["W2"], n.params["W2"], func(a, b float64) float64 { return a + n.weightDecayLambda*b })

	return grads
}

func (n *Neu) Optimize(grads map[string]matrix.Matrix) {
	n.params = n.optimizer.Update(n.params, grads)
}

func Accuracy(y, t matrix.Matrix) float64 {
	count := func(x, y []int) int {
		var c int
		for i := range x {
			if x[i] == y[i] {
				c++
			}
		}

		return c
	}

	ymax := y.Argmax()
	tmax := t.Argmax()

	c := count(ymax, tmax)
	return float64(c) / float64(len(ymax))
}

func Random(trainSize, batchSize int) []int {
	tmp := make(map[int]bool)

	for c := 0; c < batchSize; {
		n := rand.Intn(trainSize)
		if _, ok := tmp[n]; !ok {
			tmp[n] = true
			c++
		}
	}

	out := make([]int, 0, len(tmp))
	for k := range tmp {
		out = append(out, k)
	}

	return out
}

func Batch(m matrix.Matrix, index []int) matrix.Matrix {
	out := make(matrix.Matrix, 0)
	for _, i := range index {
		out = append(out, m[i])
	}

	return out
}
