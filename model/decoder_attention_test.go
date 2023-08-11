package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleAttentionDecoder() {
	m := model.NewAttentionDecoder(&model.RNNLMConfig{
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
	// *model.AttentionDecoder
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  2: *layer.TimeAffine: W(6, 3), B(1, 3): 21
}

func ExampleAttentionDecoder_rand() {
	model.NewAttentionDecoder(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})

	// Output:
}

func ExampleAttentionDecoder_Params() {
	m := model.NewAttentionDecoder(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})

	m.SetParams(make([]matrix.Matrix, 6)...)
	fmt.Println(m.Params())
	fmt.Println(m.Grads())

	// Output:
	// [[] [] [] [] [] []]
	// [[] [] [] [] [] []]
}
