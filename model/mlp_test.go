package model_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
	"github.com/itsubaki/neu/weight"
)

func ExampleMLP_Optimize() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	_, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	m := model.NewMLP(&model.MLPConfig{
		InputSize:  inSize,
		HiddenSize: []int{hiddenSize},
		OutputSize: outSize,
		WeightInit: weight.Std(0.01),
	})

	// optimizer
	opt := &optimizer.SGD{LearningRate: 0.1}

	// gradient
	for i := 0; i < 1000; i++ {
		m.Optimize(opt, m.Gradient(x, t))

		if i%200 == 0 {
			fmt.Printf("%.4f, %.4f\n", m.Predict(x), m.Loss(x, t))
		}
	}

	// Output:
	// [[-0.0167 0.0167] [-0.0167 0.0167] [-0.0166 0.0170]], [[0.6877]]
	// [[-0.3434 0.3435] [-0.3435 0.3435] [-0.3562 0.3576]], [[0.6335]]
	// [[-0.2221 0.2222] [-0.2221 0.2221] [-0.7654 0.7733]], [[0.5432]]
	// [[-0.0421 0.0429] [-0.0437 0.0437] [-1.6559 1.6705]], [[0.4741]]
	// [[0.1295 -0.1192] [-0.0823 0.0823] [-1.8652 1.9011]], [[0.4045]]

}

func ExampleMLP_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	_, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	m := model.NewMLP(&model.MLPConfig{
		InputSize:  inSize,
		HiddenSize: []int{hiddenSize},
		OutputSize: outSize,
		WeightInit: weight.Std(0.01),
	})

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
