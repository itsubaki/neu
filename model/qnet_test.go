package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleQNet() {
	s := rand.Const(1)
	m := model.NewQNet(&model.QNetConfig{
		InputSize:  12,
		OutputSize: 4,
		HiddenSize: []int{100},
		WeightInit: weight.Std(0.01),
	}, s)

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	target := matrix.New([]float64{1})
	qs := m.Predict(matrix.New([]float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}))
	q := matrix.New([]float64{vector.Max(qs[0])})
	fmt.Printf("%.8f, %0.8f\n", qs, q)
	fmt.Printf("%.8f\n", m.Loss(target, q))
	fmt.Printf("%.8f\n", m.Backward())

	params, grads := m.Params(), m.Grads()
	for i, p := range params {
		for j := range p {
			a, b := params[i][j].Dim()
			c, d := grads[i][j].Dim()
			fmt.Printf("p(%3v, %3v), g(%3v,%3v)\n", a, b, c, d)
		}
	}

	// Output:
	// *model.QNet
	//  0: *layer.Affine: W(12, 100), B(1, 100): 1300
	//  1: *layer.ReLU
	//  2: *layer.Affine: W(100, 4), B(1, 4): 404
	//  3: *layer.MeanSquaredError
	// [[-0.00007664 -0.00098626 -0.00063908 -0.00086191]], [[-0.00007664]]
	// [[1.00015329]]
	// [[-0.00108430 0.00035194 0.00033155 -0.00098515 0.00012841 -0.00121666 -0.00180430 0.00153819 -0.00015329 0.00126458 -0.00082200 0.00016637]]
	// p( 12, 100), g( 12,100)
	// p(  1, 100), g(  1,100)
	// p(100,   4), g(100,  1)
	// p(  1,   4), g(  1,  1)
}

func ExampleQNet_Sync() {
	diff := func(m, t *model.QNet) []float64 {
		out := make([]float64, 0)
		for i := range m.Params() {
			for j := range m.Params()[i] {
				out = append(out, m.Params()[i][j].Sub(t.Params()[i][j]).Abs().Sum())
			}
		}

		return out
	}

	s := rand.Const(1)
	c := &model.QNetConfig{
		InputSize:  12,
		OutputSize: 4,
		HiddenSize: []int{100},
		WeightInit: weight.Std(0.01),
	}

	m := model.NewQNet(c, s)
	t := model.NewQNet(c, s)
	fmt.Println("new:", diff(m, t))

	t.Sync(m)
	fmt.Println("sync:", diff(m, t))

	m.SetParams(model.NewQNet(c, s).Params())
	fmt.Println("set:", diff(m, t))

	t.Sync(m)
	fmt.Println("sync:", diff(m, t))

	// Output:
	// new: [13.36619293822008 0 4.599586087033022 0]
	// sync: [0 0 0 0]
	// set: [13.515551906185843 0 4.263141626774541 0]
	// sync: [0 0 0 0]
}

func ExampleQNet_Summary() {
	m := model.NewQNet(&model.QNetConfig{
		InputSize:  12,
		OutputSize: 4,
		HiddenSize: []int{100},
		WeightInit: weight.Std(0.01),
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.QNet
	//  0: *layer.Affine: W(12, 100), B(1, 100): 1300
	//  1: *layer.ReLU
	//  2: *layer.Affine: W(100, 4), B(1, 4): 404
	//  3: *layer.MeanSquaredError
}
