package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleMLP() {
	s := rand.NewSource(1)
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         2,
		OutputSize:        2,
		HiddenSize:        []int{3},
		WeightInit:        weight.Std(0.01),
		BatchNormMomentum: 0.9,
	}, s)

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	loss := m.Forward(x, t)
	m.Backward()

	fmt.Printf("%.4f %v\n", loss, m.Predict(x).Argmax())

	// Output:
	// *model.MLP
	//  0: *layer.Affine: W(2, 3), B(1, 3): 9
	//  1: *layer.BatchNorm: G(1, 3), B(1, 3): 6
	//  2: *layer.ReLU
	//  3: *layer.Affine: W(3, 2), B(1, 2): 8
	//  4: *layer.SoftmaxWithLoss
	//
	// [[0.6901]] [1 0 1]

}

func ExampleMLP_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	// model
	s := rand.NewSource(1)
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         2,
		OutputSize:        2,
		HiddenSize:        []int{3},
		WeightInit:        weight.Std(0.01),
		BatchNormMomentum: 0.9,
	}, s)

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// gradients
	m.Forward(x, t)
	m.Backward()
	grads := m.Grads()
	gradsn := numericalGrads(m, x, t)

	// check
	for i := range gradsn {
		// 20, 21 is ReLU. empty
		for j := range gradsn[i] {
			eps := gradsn[i][j].Sub(grads[i][j]).Abs().Avg() // avg(| A - B |)
			fmt.Printf("%v%v: %v\n", i, j, eps)
		}
	}

	// Output:
	// *model.MLP
	//  0: *layer.Affine: W(2, 3), B(1, 3): 9
	//  1: *layer.BatchNorm: G(1, 3), B(1, 3): 6
	//  2: *layer.ReLU
	//  3: *layer.Affine: W(3, 2), B(1, 2): 8
	//  4: *layer.SoftmaxWithLoss
	//
	// 00: 5.6597657680744024e-06
	// 01: 3.0068540250264654e-17
	// 10: 3.97423301588586e-10
	// 11: 0.0008320159209412661
	// 30: 3.985856805259017e-08
	// 31: 3.273245081925058e-08

}

func ExampleMLP_Params() {
	s := rand.NewSource(1)
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         1,
		HiddenSize:        []int{2},
		OutputSize:        1,
		WeightInit:        weight.Std(0.01),
		BatchNormMomentum: 0.9,
	}, s)

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

func ExampleMLP_rand() {
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         2,
		OutputSize:        2,
		HiddenSize:        []int{3},
		WeightInit:        weight.Std(0.01),
		BatchNormMomentum: 0.9,
	})

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// Output:
	// *model.MLP
	//  0: *layer.Affine: W(2, 3), B(1, 3): 9
	//  1: *layer.BatchNorm: G(1, 3), B(1, 3): 6
	//  2: *layer.ReLU
	//  3: *layer.Affine: W(3, 2), B(1, 2): 8
	//  4: *layer.SoftmaxWithLoss
}
