package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleDecoder() {
	s := rand.NewSource(1)
	m := model.NewDecoder(&model.DecoderConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	// forward
	xs := []matrix.Matrix{
		// (T, N, 1) = (2, 3, 1)
		matrix.New([]float64{0.1}, []float64{0.2}, []float64{0.3}), // (N, 1) = (3, 1)
		matrix.New([]float64{0.1}, []float64{0.2}, []float64{0.3}), // (N, 1) = (3, 1)
	}

	// (N, H) = (3, 3)
	h := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
		[]float64{0.3, 0.4, 0.5},
	)

	score := m.Forward(xs, h)
	fmt.Println(len(score))
	fmt.Println(score[0].Dimension())
	fmt.Println(score[1].Dimension())

	// backward
	dout := []matrix.Matrix{
		matrix.New([]float64{0.1}, []float64{0.1}, []float64{0.1}),
		matrix.New([]float64{0.1}, []float64{0.1}, []float64{0.1}),
	}
	dh := m.Backward(dout)
	fmt.Println(dh.Dimension())

	sampeld := m.Generate(h, 1, 10)
	fmt.Println(sampeld)

	// Output:
	// 2
	// 3 3
	// 3 3
	// 3 3
	// [1 2 2 2 2 0 2 0 2 0 2]

}

func ExampleDecoder_rand() {
	model.NewDecoder(&model.DecoderConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})

	// Output:
}

func ExampleDecoder_Params() {
	decoder := model.NewDecoder(&model.DecoderConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})
	decoder.SetParams(make([]matrix.Matrix, 6)...)

	fmt.Println(decoder.Params())
	fmt.Println(decoder.Grads())

	// Output:
	// [[] [] [] [] [] []]
	// [[] [] [] [] [] []]
}
