package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/numerical"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

var (
	_ Model = (*model.Sequential)(nil)
	_ Model = (*model.MLP)(nil)
)

type Model interface {
	Forward(x, t matrix.Matrix) matrix.Matrix
	Layers() []model.Layer
}

func numericalGrads(m Model, x, t matrix.Matrix) [][]matrix.Matrix {
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
	for _, l := range m.Layers() {
		g := make([]matrix.Matrix, 0)
		for _, p := range l.Params() {
			g = append(g, grad(lossW, p))
		}

		grads = append(grads, g)
	}

	return grads
}

func ExampleSequential_gradientCheck() {
	// weight
	s := rand.NewSource(1)
	W1 := matrix.Randn(2, 3, s).MulC(weight.Std(0.01)(2))
	B1 := matrix.Zero(1, 3)
	W2 := matrix.Randn(3, 2, s).MulC(weight.Std(0.01)(3))
	B2 := matrix.Zero(1, 2)

	// model
	m := model.NewSequential(
		[]model.Layer{
			&layer.Affine{W: W1, B: B1},
			&layer.ReLU{},
			&layer.Affine{W: W2, B: B2},
			&layer.SoftmaxWithLoss{},
		},
		s,
	)

	// gradients
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	m.Forward(x, t)
	m.Backward()
	grads := m.Grads()
	gradsn := numericalGrads(m, x, t)

	// check
	for i := range gradsn {
		// 10, 11 is ReLU. empty
		for j := range gradsn[i] {
			eps := gradsn[i][j].Sub(grads[i][j]).Abs().Mean() // mean(| A - B |)
			fmt.Printf("%v%v: %v\n", i, j, eps)
		}
	}

	// Output:
	// 00: 1.6658328893821156e-10
	// 01: 7.510020527910314e-13
	// 20: 2.819144981296903e-10
	// 21: 3.332532312605441e-08

}

func ExampleSequential_Summary() {
	m := model.NewSequential(
		[]model.Layer{
			&layer.Affine{
				W: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.4, 0.5, 0.6}),
				B: matrix.New([]float64{0.1, 0.2, 0.3}),
			},
			&layer.ReLU{},
			&layer.Affine{
				W: matrix.New([]float64{0.1, 0.2, 0.3}, []float64{0.4, 0.5, 0.6}),
				B: matrix.New([]float64{0.1, 0.2, 0.3}),
			},
			&layer.SoftmaxWithLoss{},
		},
		rand.NewSource(1),
	)

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.Sequential
	//  0: *layer.Affine: W(2, 3), B(1, 3): 9
	//  1: *layer.ReLU
	//  2: *layer.Affine: W(2, 3), B(1, 3): 9
	//  3: *layer.SoftmaxWithLoss
}
