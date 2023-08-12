package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

func ExampleCBOW() {
	// you, say, goodbye, and, I, hello, .

	// context data
	c := []matrix.Matrix{
		{
			{1, 0, 0, 0, 0, 0, 0}, // you
			{0, 0, 1, 0, 0, 0, 0}, // goodbye
		},
	}
	t := []matrix.Matrix{
		{
			{0, 1, 0, 0, 0, 0, 0}, // say
		},
	}

	// model
	s := rand.NewSource(1)
	m := model.NewCBOW(&model.CBOWConfig{
		VocabSize:  7,
		HiddenSize: 5,
	}, s)

	loss := m.Forward(c, t)
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
	// 1.9461398376656527
	// [[[0.00012703871969382832 -1.2392940985965779e-05 6.8815266046905e-05 0.00022359874505051858 0.00012362319821868092 0.0006585876988009539 0.0003365497538731326]]]

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
