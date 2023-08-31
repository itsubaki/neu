package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleQNet() {
	s := rand.NewSource(1)
	m := model.NewQNet(&model.QNetConfig{
		InputSize:  12,
		OutputSize: 4,
		HiddenSize: 100,
		WeightInit: weight.Std(0.01),
	}, s)

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	x := matrix.New([]float64{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0})
	qs := m.Forward(x)
	fmt.Printf("%.8f\n", qs)
	fmt.Printf("%.8f\n", m.MeanSquaredError(matrix.Column(qs, 3), matrix.New([]float64{1})))
	fmt.Printf("%.8f\n", m.Backward())

	// Output:
	// *model.QNet
	//  0: *layer.Affine: W(12, 100), B(1, 100): 1300
	//  1: *layer.ReLU
	//  2: *layer.Affine: W(100, 4), B(1, 4): 404
	//  3: *layer.MeanSquaredError
	// [[0.00015980 -0.00057628 0.00159733 0.00016082]]
	// [[0.99967839]]
	// [[-0.00035087 0.00028844 -0.00124843 0.00074944 -0.00394823 0.00254907 0.00054263 0.00221235 -0.00031956 0.00211016 0.00066308 0.00062176]]
}
