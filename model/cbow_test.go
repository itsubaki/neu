package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
)

func ExampleCBOW() {
	// you, say, goodbye, and, I, hello, .

	// data
	contexts := []matrix.Matrix{
		{
			{1, 0, 0, 0, 0, 0, 0}, // you
			{0, 0, 1, 0, 0, 0, 0}, // goodbye
		},
	}
	targets := matrix.Matrix{
		[]float64{0, 1, 0, 0, 0, 0, 0}, // say
	}

	// model
	s := rand.Const(1)
	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  7,
		HiddenSize: 5,
	}, s)

	loss := m.Forward(contexts, targets)
	m.Backward()
	fmt.Println(loss)

	score := m.Predict([]matrix.Matrix{
		{
			{1, 0, 0, 0, 0, 0, 0}, // you
			{0, 0, 1, 0, 0, 0, 0}, // goodbye
		},
	})
	fmt.Println(score)

	// Output:
	// [[1.945926338366903]]
	// [[0.0002949561336916829 1.6764012522628663e-05 0.00014512664854319168 -0.00013693775475907596 -0.0001516469737667451 7.902275520347142e-05 -1.1785748579958962e-05]]
}

func ExampleCBOW_Summary() {
	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  7,
		HiddenSize: 5,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.CBOW
	//  0: *layer.Dot: W(7, 5): 35
	//  1: *layer.Dot: W(7, 5): 35
	//  2: *layer.Dot: W(5, 7): 35
	//  3: *layer.SoftmaxWithLoss
}

func ExampleCBOW_Layers() {
	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  7,
		HiddenSize: 5,
	})

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}

	// Output:
	// *model.CBOW
	//  0: *layer.Dot: W(7, 5): 35
	//  1: *layer.Dot: W(7, 5): 35
	//  2: *layer.Dot: W(5, 7): 35
	//  3: *layer.SoftmaxWithLoss
}

func ExampleCBOW_Params() {
	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  7,
		HiddenSize: 5,
	})

	m.SetParams(m.Grads())
	fmt.Println(m.Params())
	fmt.Println(m.Grads())

	// Output:
	// [[[]] [[]] [[]]]
	// [[[]] [[]] [[]]]
}
