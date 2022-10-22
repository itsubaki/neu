package model_test

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/optimizer"
)

func Example_gradientCheck() {
	// data
	x := matrix.New([]float64{0.5, 0.5}, []float64{1, 0}, []float64{0, 1})
	t := matrix.New([]float64{1, 0}, []float64{0, 1}, []float64{0, 1})
	_, inSize := x.Dimension()
	hiddenSize, outSize := t.Dimension()

	// init
	rand.Seed(1) // for test
	m := model.NewMulti(&model.MultiConfig{
		InputSize:  inSize,
		HiddenSize: []int{hiddenSize},
		OutputSize: outSize,
		WeightInit: model.Std(0.01),
		Optimizer:  &optimizer.SGD{LearningRate: 0.1},
	})

	// gradient
	ngrads := m.NumericalGradient(x, t)
	grads := m.Gradient(x, t)

	// check
	for _, k := range []string{"W1", "W2", "B1", "B2"} {
		diff := matrix.FuncWith(ngrads[k], grads[k], func(a, b float64) float64 { return math.Abs(a - b) })
		fmt.Printf("%v: %v\n", k, diff.Avg())
	}

	// Output:
	// W1: 1.6658328893821156e-10
	// W2: 2.8191449832846066e-10
	// B1: 7.510020527910314e-13
	// B2: 3.3325323139932195e-08

}
