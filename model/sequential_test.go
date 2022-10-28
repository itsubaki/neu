package model_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
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
	m := model.Sequential{
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
		m.Optimize(opt, m.Gradient(x, t))

		if i%200 == 0 {
			fmt.Printf("%.4f\n", m.Loss(x, t))
		}
	}

	// Output:
	// [[0.6877]]
	// [[0.6335]]
	// [[0.5432]]
	// [[0.4741]]
	// [[0.4045]]
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
	m := model.Sequential{
		Layer: []model.Layer{
			&layer.Affine{W: W1, B: B1},
			&layer.ReLU{},
			&layer.Affine{W: W2, B: B2},
			&layer.SoftmaxWithLoss{},
		},
	}

	// gradient
	ngrads := m.NumericalGradient(x, t)
	grads := m.Gradient(x, t)

	// check
	for i := range ngrads {
		// 10, 11 is ReLU. empty
		for j := range ngrads[i] {
			diff := matrix.FuncWith(ngrads[i][j], grads[i][j], func(a, b float64) float64 { return math.Abs(a - b) })
			fmt.Printf("%v%v: %v\n", i, j, diff.Avg())
		}
	}

	// Output:
	// 00: 1.6658328893821156e-10
	// 01: 7.510020527910314e-13
	// 20: 2.8191449832846066e-10
	// 21: 3.3325323139932195e-08

}
