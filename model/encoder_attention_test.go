package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleAttentionEncoder() {
	s := rand.NewSource(1)
	m := model.NewAttentionEncoder(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}

	// Output:
	// *model.AttentionEncoder
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
}
