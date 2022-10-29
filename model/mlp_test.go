package model_test

import (
	"fmt"
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
		loss := m.Forward(x, t)
		m.Backward(x, t)
		m.Optimize(opt)

		if i%200 == 0 {
			fmt.Printf("%.4f, %.4f\n", m.Predict(x), loss)
		}
	}

	// Output:
	// [[-0.0165 0.0165] [-0.0165 0.0165] [-0.1041 0.1464]], [[0.6901]]
	// [[-0.0148 0.0148] [-0.0148 0.0148] [-2.2205 2.2730]], [[0.4659]]
	// [[-0.0052 0.0052] [-0.0052 0.0052] [-2.6585 2.7135]], [[0.4637]]
	// [[-0.0030 0.0030] [-0.0030 0.0030] [-2.8907 2.9469]], [[0.4631]]
	// [[-0.0021 0.0021] [-0.0021 0.0021] [-3.0637 3.1208]], [[0.4628]]

}

func ExampleMLP_Params() {
	rand.Seed(1) // for test
	m := model.NewMLP(&model.MLPConfig{
		InputSize:  1,
		HiddenSize: []int{2},
		OutputSize: 1,
		WeightInit: weight.Std(0.01),
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
