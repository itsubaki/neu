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
	// [[-0.0167 0.0167] [-0.0167 0.0167] [-0.0166 0.0170]], [[0.6931]]
	// [[-0.3434 0.3435] [-0.3435 0.3435] [-0.3562 0.3576]], [[0.6335]]
	// [[-0.2221 0.2222] [-0.2221 0.2221] [-0.7654 0.7733]], [[0.5452]]
	// [[-0.0421 0.0429] [-0.0437 0.0437] [-1.6559 1.6705]], [[0.4743]]
	// [[0.1295 -0.1192] [-0.0823 0.0823] [-1.8652 1.9011]], [[0.4060]]

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
		// W1, B1(0) -> empty(ReLU) -> W2, B2(0) -> empty(SoftmaxWithLoss)
		fmt.Println(p)
	}
	fmt.Println()

	for _, g := range m.Grads() {
		// empty
		fmt.Println(g)
	}

	// Output:
	// [[[-0.01233758177597947 -0.0012634751070237293]] [[0 0]]]
	// []
	// [[[-0.005209945711531503] [0.022857191176995802]] [[0]]]
	// []
	//
	// [[] []]
	// []
	// [[] []]
	// []
}
