package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleSeq2Seq() {
	s := rand.NewSource(1)
	m := model.NewSeq2Seq(&model.Seq2SeqConfig{
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
	// *model.Seq2Seq
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  2: *layer.TimeEmbedding: W(3, 3): 9
	//  3: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  4: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  5: *layer.TimeSoftmaxWithLoss
}

func ExampleSeq2Seq_Params() {
	m := model.NewSeq2Seq(&model.Seq2SeqConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})
	m.SetParams(m.Grads())

	fmt.Println(m.Params())
	fmt.Println(m.Grads())

	// Output:
	// [[[] [] [] []] [[] [] [] [] [] []]]
	// [[[] [] [] []] [[] [] [] [] [] []]]
}
