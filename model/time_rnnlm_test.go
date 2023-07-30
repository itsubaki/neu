package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

func ExampleTimeRNNLM() {
	// model
	s := rand.NewSource(1)
	m := model.NewTimeRNNLM(&model.TimeRNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
	}, s)

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// data
	xs := []matrix.Matrix{
		matrix.New([]float64{0, 1, 2}),
	}
	ts := []matrix.Matrix{
		matrix.New([]float64{0, 1, 2}),
	}

	loss := m.Forward(xs, ts)
	dout := m.Backward()

	fmt.Printf("%.4f\n", loss)
	fmt.Println(dout)

	// Output:
	// *model.TimeRNNLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeRNN: Wx(3, 3), Wh(3, 3), B(1, 3): 21
	//  2: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  3: *layer.TimeSoftmaxWithLoss
	//
	// [[1.0110]]
	// []

}

func ExampleTimeRNNLM_Params() {
	s := rand.NewSource(1)
	m := model.NewTimeRNNLM(&model.TimeRNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
	}, s)

	for _, p := range m.Params() {
		// TimeEmbedding: W
		// TimeRNN: Wx, Wh, B
		// TimeAffine: W, B
		// TimeSoftmaxWithLoss: empty
		fmt.Println(p)
	}
	fmt.Println()

	for _, g := range m.Grads() {
		fmt.Println(g) // empty
	}

	// Output:
	// [[[-0.01233758177597947 -0.0012634751070237293 -0.005209945711531503] [0.022857191176995802 0.003228052526115799 0.005900672875996937] [0.0015880774017643562 0.009892020842955818 -0.007312830161774791]]]
	// [[[1.1888463930213164 2.746000213191051 1.451815213661066] [2.2496583388460167 0.9134115305775588 1.2686266290870107] [-1.8588019757833967 1.2126449744670267 0.7474331298082769]] [[1.7314032301645013 -2.6395894379107996 -0.5482585871803314] [3.2726480043311508 1.9065188889493625 -1.7194816452495298] [1.71422873258152 -1.065598334510692 -2.4855741803783573]] [[0 0 0]]]
	// [[[-3.7263976437777035 0.23789782546760918 0.7669912887511364] [-1.4654784428232501 -0.14340520573820997 0.2704190222374976] [-2.5125531754000554 0.484530333822867 -3.0118742306108865]] [[0 0 0]]]
	// []
	//
	// [[]]
	// [[] [] []]
	// [[] []]
	// []
}

func ExampleTimeRNNLM_ResetState() {
	s := rand.NewSource(1)
	m := model.NewTimeRNNLM(&model.TimeRNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
	}, s)

	m.ResetState()

	// Output:
}
