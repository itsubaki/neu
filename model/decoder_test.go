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
		// (T, 1, N) = (2, 1, 3)
		matrix.New([]float64{0.1, 0.2, 0.3}), // (1, N) = (1, 3)
		matrix.New([]float64{0.1, 0.2, 0.3}), // (1, N) = (1, 3)
	}

	h := []matrix.Matrix{
		// (T, N, H) = (1, 3, 3)
		matrix.New(
			[]float64{0.1, 0.2, 0.3},
			[]float64{0.3, 0.4, 0.5},
			[]float64{0.3, 0.4, 0.5},
		),
	}

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
	fmt.Println(len(dh))
	fmt.Println(dh[0].Dimension())

	sampeld := m.Generate(h[0], 1, 10)
	fmt.Println(sampeld)

	// Output:
	// 2
	// 3 3
	// 3 3
	// 1
	// 3 3
	// [1 2 2 2 0 1 1 1 1 1]

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
