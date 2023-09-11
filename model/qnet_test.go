package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/vector"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleQNet() {
	s := rand.NewSource(1)
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
	// [[0.00015980 -0.00057628 0.00159733 0.00016082]], [[0.00159733]]
	// [[0.99680788]]
	// [[0.00035037 -0.00028802 0.00124664 -0.00074836 0.00394256 -0.00254541 -0.00054185 -0.00220917 0.00031910 -0.00210713 -0.00066213 -0.00062087]]
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

	s := rand.NewSource(1)
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
	// new: [13.589057285251876 0 4.387607068711152 0]
	// sync: [0 0 0 0]
	// set: [13.90911363704504 0 4.797890879215467 0]
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
