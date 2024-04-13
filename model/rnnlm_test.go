package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleRNNLM() {
	// model
	s := rand.Const(1)
	m := model.NewRNNLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	}, s)

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// data
	xs := []matrix.Matrix{{{0, 1, 2}}}
	ts := []matrix.Matrix{{{0, 1, 2}}}

	loss := m.Forward(xs, ts)
	dout := m.Backward()

	fmt.Printf("%.4f\n", loss)
	fmt.Println(dout)

	// Output:
	// *model.RNNLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeRNN: Wx(3, 3), Wh(3, 3), B(1, 3): 21
	//  2: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  3: *layer.TimeSoftmaxWithLoss
	//
	// [[[1.1069]]]
	// []

}

func ExampleRNNLM_Summary() {
	m := model.NewRNNLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.RNNLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeRNN: Wx(3, 3), Wh(3, 3), B(1, 3): 21
	//  2: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  3: *layer.TimeSoftmaxWithLoss
}

func ExampleRNNLM_Layers() {
	m := model.NewRNNLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	})

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// Output:
	// *model.RNNLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeRNN: Wx(3, 3), Wh(3, 3), B(1, 3): 21
	//  2: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  3: *layer.TimeSoftmaxWithLoss
}

func ExampleRNNLM_Params() {
	s := rand.Const(1)
	m := model.NewRNNLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	}, s)

	// params
	for _, p := range m.Params() {
		fmt.Println(p)
	}
	fmt.Println()

	// grads
	for _, g := range m.Grads() {
		fmt.Println(g) // empty
	}
	fmt.Println()

	// set params
	m.SetParams(m.Grads())
	for _, p := range m.Params() {
		fmt.Println(p)
	}
	fmt.Println()

	// Output:
	// [[[-0.008024826241110656 0.00424707052949676 -0.004985070978632815] [-0.009872764577745819 0.004770185009670911 -0.0037300956589935985] [0.01182810122110346 -0.008066822642915392 0.0010337870485847512]]]
	// [[[-0.4272036569812764 -0.051439283460071226 0.0588887761030214] [-0.11157067538737257 0.42479877444693387 -1.6415745494204534] [-0.969465128610357 -0.4261080517614557 1.0999236840248148]] [[0.18248391466570424 0.5477994083834968 -0.1579822276645198] [-0.48305059627014674 -0.4277824240418346 0.9395249505867087] [0.2701827393831761 0.49870245028370974 -0.1766569492870927]] [[0 0 0]]]
	// [[[-0.5584779273923454 0.2927783198117813 0.5136963587943321] [-1.208969787150421 -0.6633000273730952 0.19743387804132795] [-0.33298687925268927 -0.6371079759680973 -0.14198751865229117]] [[0 0 0]]]
	// []
	//
	// [[]]
	// [[] [] []]
	// [[] []]
	// []
	//
	// [[]]
	// [[] [] []]
	// [[] []]
	// []
}

func ExampleRNNLM_ResetState() {
	s := rand.Const(1)
	m := model.NewRNNLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	}, s)

	m.ResetState()

	// Output:
}
