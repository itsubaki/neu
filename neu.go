package neu

import (
	"fmt"
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
	HiddenSize        []int
	OutputSize        int
	WeightDecayLambda float64
	WeightInit        WeightInit
	Optimizer         Optimizer
}

type Neu struct {
	size              []int
	params            map[string]matrix.Matrix
	layer             []Layer
	last              Layer
	weightDecayLambda float64
	optimizer         Optimizer
}

func New(c *Config) *Neu {
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
		params[fmt.Sprintf("W%v", i+1)] = matrix.Func(params[fmt.Sprintf("W%v", i+1)], func(v float64) float64 {
			return c.WeightInit(size[i]) * v
		})
	}

	// new
	return &Neu{
		size:              size,
		params:            params,
		layer:             make([]Layer, 0),
		last:              &layer.SoftmaxWithLoss{},
		weightDecayLambda: c.WeightDecayLambda,
		optimizer:         c.Optimizer,
	}
}

func (n *Neu) Predict(x matrix.Matrix) matrix.Matrix {
	n.layer = make([]Layer, 0)
	for i := 0; i < len(n.size)-1; i++ {
		n.layer = append(n.layer, &layer.Affine{W: n.params[fmt.Sprintf("W%v", i+1)], B: n.params[fmt.Sprintf("B%v", i+1)]})
		n.layer = append(n.layer, &layer.ReLU{})
	}
	n.layer = n.layer[:len(n.layer)-1] // remove last ReLu

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
	for i := 0; i < len(n.size)-1; i++ {
		sump2 := n.params[fmt.Sprintf("W%v", i+1)].Func(func(v float64) float64 { return v * v }).Sum()
		decay = decay + 0.5*n.weightDecayLambda*sump2
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
