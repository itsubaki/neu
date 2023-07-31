package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleRNNLM() {
	// model
	s := rand.NewSource(1)
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
	// *model.RNNLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeRNN: Wx(3, 3), Wh(3, 3), B(1, 3): 21
	//  2: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  3: *layer.TimeSoftmaxWithLoss
	//
	// [[1.0884]]
	// []

}

func ExampleRNNLM_Params() {
	s := rand.NewSource(1)
	m := model.NewRNNLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
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
	// [[[0.39628213100710546 0.9153334043970169 0.4839384045536887] [0.7498861129486722 0.3044705101925196 0.42287554302900354] [-0.6196006585944656 0.4042149914890089 0.24914437660275898]] [[0.5771344100548338 -0.8798631459702666 -0.1827528623934438] [1.0908826681103836 0.6355062963164542 -0.57316054841651] [0.5714095775271734 -0.3551994448368973 -0.8285247267927858]] [[0 0 0]]]
	// [[[-1.2421325479259013 0.07929927515586974 0.25566376291704546] [-0.4884928142744167 -0.04780173524606999 0.09013967407916586] [-0.8375177251333519 0.161510111274289 -1.0039580768702956]] [[0 0 0]]]
	// []
	//
	// [[]]
	// [[] [] []]
	// [[] []]
	// []
}

func ExampleRNNLM_ResetState() {
	s := rand.NewSource(1)
	m := model.NewRNNLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	}, s)

	m.ResetState()

	// Output:
}
