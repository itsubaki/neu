package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleMLP() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	// model
	rand.Seed(1) // for test
	m := model.NewMLP(&model.MLPConfig{
		InputSize:    2,
		HiddenSize:   []int{3},
		OutputSize:   2,
		WeightInit:   weight.Std(0.01),
		UseBatchNorm: false,
	})

	loss := m.Forward(x, t)
	m.Backward()

	fmt.Printf("%.4f %v\n", loss, m.Predict(x).Argmax())

	// Output:
	// [[0.6931]] [1 0 1]

}

func ExampleMLP_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	// model
	rand.Seed(1) // for test
	m := model.NewMLP(&model.MLPConfig{
		InputSize:    2,
		HiddenSize:   []int{3},
		OutputSize:   2,
		WeightInit:   weight.Std(0.01),
		UseBatchNorm: false,
	})

	// gradients
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
	// 20: 2.8191449832846066e-10
	// 21: 3.3325323139932195e-08

}

func ExampleMLP_Params() {
	rand.Seed(1) // for test
	m := model.NewMLP(&model.MLPConfig{
		InputSize:    1,
		HiddenSize:   []int{2},
		OutputSize:   1,
		WeightInit:   weight.Std(0.01),
		UseBatchNorm: true,
	})

	for _, p := range m.Params() {
		// Affine: W1, B1
		// BatchNrom: Gamma, Beta
		// ReLU: empty
		// Affine: W2, B2
		// SoftmaxWithLoss: empty
		fmt.Println(p)
	}
	fmt.Println()

	for _, g := range m.Grads() {
		// empty
		fmt.Println(g)
	}

	// Output:
	// [[[-0.01233758177597947 -0.0012634751070237293]] [[0 0]]]
	// [[[1 1]] [[0 0]]]
	// []
	// [[[-0.005209945711531503] [0.022857191176995802]] [[0]]]
	// []
	//
	// [[] []]
	// [[] []]
	// []
	// [[] []]
	// []
}
