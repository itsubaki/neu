package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleMLP() {
	s := rand.Const(1)
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         2,
		OutputSize:        2,
		HiddenSize:        []int{3},
		WeightInit:        weight.Std(0.01),
		BatchNormMomentum: 0.9,
	}, s)

	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	loss := m.Forward(x, t)
	m.Backward()
	fmt.Printf("%.4f %v\n", loss, m.Predict(x).Argmax())

	// Output:
	// [[0.6975]] [0 0 0]
}

func ExampleMLP_Summary() {
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         2,
		OutputSize:        2,
		HiddenSize:        []int{3},
		WeightInit:        weight.Std(0.01),
		BatchNormMomentum: 0.9,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.MLP
	//  0: *layer.Affine: W(2, 3), B(1, 3): 9
	//  1: *layer.BatchNorm: G(1, 3), B(1, 3): 6
	//  2: *layer.ReLU
	//  3: *layer.Affine: W(3, 2), B(1, 2): 8
	//  4: *layer.SoftmaxWithLoss
}

func ExampleMLP_Layers() {
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

func ExampleMLP_Params() {
	s := rand.Const(1)
	m := model.NewMLP(&model.MLPConfig{
		InputSize:         1,
		HiddenSize:        []int{2},
		OutputSize:        1,
		WeightInit:        weight.Std(0.01),
		BatchNormMomentum: 0.9,
	}, s)

	// params
	for _, p := range m.Params() {
		fmt.Println(p)
	}
	fmt.Println()

	// grads
	for _, g := range m.Grads() {
		fmt.Println(g) // empty
	}
	fmt.Println()

	// set params
	m.SetParams(m.Grads())
	for _, p := range m.Params() {
		fmt.Println(p)
	}
	fmt.Println()

	// Output:
	// [[[-0.008024826241110656 0.00424707052949676]] [[0 0]]]
	// [[[1 1]] [[0 0]]]
	// []
	// [[[-0.004985070978632815] [-0.009872764577745819]] [[0]]]
	// []
	//
	// [[] []]
	// [[] []]
	// []
	// [[] []]
	// []
	//
	// [[] []]
	// [[] []]
	// []
	// [[] []]
	// []
}

func ExampleMLP_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})

	// model
	s := rand.Const(1)
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
			eps := gradsn[i][j].Sub(grads[i][j]).Abs().Mean() // mean(| A - B |)
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
	// 00: 0.0018878976165139967
	// 01: 9.251858538542972e-17
	// 10: 3.416015634604153e-10
	// 11: 0.0008399733189571372
	// 30: 3.21054273907014e-08
	// 31: 3.421842263706676e-08
}
