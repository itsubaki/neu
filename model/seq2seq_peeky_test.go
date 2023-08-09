package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExamplePeekySeq2Seq() {
	s := rand.NewSource(1)
	m := model.NewPeekySeq2Seq(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// Output:
	// *model.PeekySeq2Seq
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  2: *layer.TimeEmbedding: W(3, 3): 9
	//  3: *layer.TimeLSTM: Wx(6, 12), Wh(3, 12), B(1, 12): 120
	//  4: *layer.TimeAffine: W(6, 3), B(1, 3): 21
	//  5: *layer.TimeSoftmaxWithLoss
}

func ExamplePeekySeq2Seq_rand() {
	m := model.NewPeekySeq2Seq(&model.RNNLMConfig{
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
	// *model.PeekySeq2Seq
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  2: *layer.TimeEmbedding: W(3, 3): 9
	//  3: *layer.TimeLSTM: Wx(6, 12), Wh(3, 12), B(1, 12): 120
	//  4: *layer.TimeAffine: W(6, 3), B(1, 3): 21
	//  5: *layer.TimeSoftmaxWithLoss
}
