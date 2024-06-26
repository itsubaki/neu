package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExamplePeekyDecoder() {
	s := rand.Const(1)
	m := model.NewPeekyDecoder(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	// forward
	xs := []matrix.Matrix{
		// (T, N, 1) = (2, 3, 1)
		{{0.1}, {0.2}, {0.3}}, // (N, 1) = (3, 1)
		{{0.1}, {0.2}, {0.3}}, // (N, 1) = (3, 1)
	}

	// (N, H) = (3, 3)
	h := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
		[]float64{0.3, 0.4, 0.5},
	)

	score := m.Forward(xs, h)
	fmt.Println(len(score))
	for _, s := range score {
		fmt.Println(s.Dim())
	}

	// backward
	dout := []matrix.Matrix{
		{{0.1}, {0.1}, {0.1}},
		{{0.1}, {0.1}, {0.1}},
	}
	dh := m.Backward(dout)
	fmt.Println(dh.Dim())

	// generate
	sampeld := m.Generate(h, 1, 10)
	fmt.Println(sampeld)

	// Output:
	// 2
	// 3 3
	// 3 3
	// 3 3
	// [2 2 2 2 2 2 2 2 2 2]
}

func ExamplePeekyDecoder_Summary() {
	m := model.NewPeekyDecoder(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.PeekyDecoder
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(6, 12), Wh(3, 12), B(1, 12): 120
	//  2: *layer.TimeAffine: W(6, 3), B(1, 3): 21
}

func ExamplePeekyDecoder_Layers() {
	m := model.NewPeekyDecoder(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// Output:
	// *model.PeekyDecoder
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(6, 12), Wh(3, 12), B(1, 12): 120
	//  2: *layer.TimeAffine: W(6, 3), B(1, 3): 21
}
