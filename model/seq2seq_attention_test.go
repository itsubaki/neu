package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleAttentionSeq2Seq() {
	s := rand.NewSource(1)
	m := model.NewAttentionSeq2Seq(&model.RNNLMConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	xs := []matrix.Matrix{{{0, 1, 2}}, {{0, 1, 2}}, {{0, 1, 2}}}
	ts := []matrix.Matrix{{{0, 1, 2}}, {{0, 1, 2}}, {{0, 1, 2}}}

	loss := m.Forward(xs, ts)
	m.Backward()

	fmt.Printf("%.4f\n", loss)
	fmt.Println(m.Generate(xs, 1, 10))

	// Output:
	// 1.0950
	// [0 0 0 0 0 0 0 0 0 0]
}

func ExampleAttentionSeq2Seq_Summary() {
	m := model.NewAttentionSeq2Seq(&model.RNNLMConfig{
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
	// *model.AttentionSeq2Seq
	//  0: *model.AttentionEncoder
	//  1: *layer.TimeEmbedding: W(3, 3): 9
	//  2: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  3: *model.AttentionDecoder
	//  4: *layer.TimeEmbedding: W(3, 3): 9
	//  5: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  6: *layer.TimeAttention
	//  7: *layer.TimeAffine: W(6, 3), B(1, 3): 21
	//  8: *layer.TimeSoftmaxWithLoss
}

func ExampleAttentionSeq2Seq_Layers() {
	m := model.NewAttentionSeq2Seq(&model.RNNLMConfig{
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
	// *model.AttentionSeq2Seq
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  2: *layer.TimeEmbedding: W(3, 3): 9
	//  3: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  4: *layer.TimeAffine: W(6, 3), B(1, 3): 21
	//  5: *layer.TimeSoftmaxWithLoss
}

func ExampleAttentionSeq2Seq_Params() {
	m := model.NewAttentionSeq2Seq(&model.RNNLMConfig{
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
