package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/numerical"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func ExampleSequential_Optimize() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	// weight
	rand.Seed(1) // for test
	W1 := matrix.Randn(2, 3).MulC(weight.Std(0.01)(2))
	B1 := matrix.Zero(1, 3)
	W2 := matrix.Randn(3, 2).MulC(weight.Std(0.01)(3))
	B2 := matrix.Zero(1, 2)

	// model
	m := &model.Sequential{
		Layer: []model.Layer{
			&layer.Affine{W: W1, B: B1},
			&layer.ReLU{},
			&layer.Affine{W: W2, B: B2},
			&layer.SoftmaxWithLoss{},
		},
	}

	// optimizer
	opt := &optimizer.SGD{LearningRate: 0.1}

	// gradient
	for i := 0; i < 1000; i++ {
		loss := m.Forward(x, t)
		m.Backward(x, t)
		m.Optimize(opt)

		if i%200 == 0 {
			fmt.Printf("%.4f, %.4f\n", m.Predict(x), loss)
		}
	}

	// Output:
	// [[-0.0167 0.0167] [-0.0167 0.0167] [-0.0166 0.0170]], [[0.6931]]
	// [[-0.3434 0.3435] [-0.3435 0.3435] [-0.3562 0.3576]], [[0.6335]]
	// [[-0.2221 0.2222] [-0.2221 0.2221] [-0.7654 0.7733]], [[0.5452]]
	// [[-0.0421 0.0429] [-0.0437 0.0437] [-1.6559 1.6705]], [[0.4743]]
	// [[0.1295 -0.1192] [-0.0823 0.0823] [-1.8652 1.9011]], [[0.4060]]
}

func ExampleSequential_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	// weight
	rand.Seed(1) // for test
	W1 := matrix.Randn(2, 3).MulC(weight.Std(0.01)(2))
	B1 := matrix.Zero(1, 3)
	W2 := matrix.Randn(3, 2).MulC(weight.Std(0.01)(3))
	B2 := matrix.Zero(1, 2)

	// model
	m := &model.Sequential{
		Layer: []model.Layer{
			&layer.Affine{W: W1, B: B1},
			&layer.ReLU{},
			&layer.Affine{W: W2, B: B2},
			&layer.SoftmaxWithLoss{},
		},
	}

	// gradients
	m.Forward(x, t)
	m.Backward(x, t)
	grads := m.Grads()
	gradsn := numericalGrads(m, x, t)

	// check
	for i := range gradsn {
		// 10, 11 is ReLU. empty
		for j := range gradsn[i] {
			eps := gradsn[i][j].Sub(grads[i][j]).Abs().Avg() // avg(| A - B |)
			fmt.Printf("%v%v: %v\n", i, j, eps)
		}
	}

	// Output:
	// 00: 1.6658328893821156e-10
	// 01: 7.510020527910314e-13
	// 20: 2.8191449832846066e-10
	// 21: 3.3325323139932195e-08

}

func numericalGrads(m *model.Sequential, x, t matrix.Matrix) [][]matrix.Matrix {
	lossW := func(w ...float64) float64 {
		return m.Forward(x, t)[0][0]
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
	for _, l := range m.Layer {
		g := make([]matrix.Matrix, 0)
		for _, p := range l.Params() {
			g = append(g, grad(lossW, p))
		}

		grads = append(grads, g)
	}

	return grads
}
