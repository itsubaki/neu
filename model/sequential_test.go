package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

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
			eps := gradsn[i][j].Sub(grads[i][j]).Abs().Avg() // avg(| A - B |)
			fmt.Printf("%v%v: %v\n", i, j, eps)
		}
	}

	// Output:
	// 00: 1.6658328893821156e-10
	// 01: 7.510020527910314e-13
	// 20: 2.819144981296903e-10
	// 21: 3.332532312605441e-08

}
