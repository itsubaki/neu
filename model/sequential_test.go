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
	m := model.NewSequential(
		&layer.Affine{W: W1, B: B1},
		&layer.ReLU{},
		&layer.Affine{W: W2, B: B2},
		&layer.SoftmaxWithLoss{},
	)

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
