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
	fmt.Println(len(hs))           // 1
	fmt.Println(hs[0].Dimension()) // (N, H) = (3, 3)

	// backward
	dh := []matrix.Matrix{
		// (T, N, H) = (1, 3, 3)
		matrix.New(
			[]float64{0.1, 0.2, 0.3},
			[]float64{0.3, 0.4, 0.5},
			[]float64{0.3, 0.4, 0.5},
		),
	}
	dout := m.Backward(dh)
	fmt.Println(len(dout))

	// Output:
	// 1
	// 3 3
	// 0
}
