package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleSeq2Seq_Params() {
	seq2seq := model.NewSeq2Seq(&model.Seq2SeqConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	})
	seq2seq.SetParams(seq2seq.Grads())

	fmt.Println(seq2seq.Params())
	fmt.Println(seq2seq.Grads())

	// Output:
	// [[[] [] [] []] [[] [] [] [] [] []]]
	// [[[] [] [] []] [[] [] [] [] [] []]]
}
