package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleEncoder() {
	s := rand.NewSource(1)
	m := model.NewEncoder(&model.EncoderConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	// forward
	xs := []matrix.Matrix{
		// (T, 1, N) = (2, 1, 3)
		matrix.New([]float64{0.1, 0.2, 0.3}), // (1, N) = (1, 3)
		matrix.New([]float64{0.1, 0.2, 0.3}), // (1, N) = (1, 3)
	}

	hs := m.Forward(xs)
	fmt.Println(hs.Dimension()) // (N, H) = (3, 3)

	// backward
	// (N, H) = (3, 3)
	dh := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
		[]float64{0.3, 0.4, 0.5},
	)

	m.Backward(dh)

	// Output:
	// 3 3
}

func ExampleEncoder_rand() {
	model.NewEncoder(&model.EncoderConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})

	// Output:
}

func ExampleEncoder_Params() {
	encoder := model.NewEncoder(&model.EncoderConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})
	encoder.SetParams(make([]matrix.Matrix, 4)...)

	fmt.Println(encoder.Params())
	fmt.Println(encoder.Grads())

	// Output:
	// [[] [] [] []]
	// [[] [] [] []]
}
