package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleLSTMLM() {
	// model
	s := rand.NewSource(1)
	m := model.NewLSTMLM(&model.RNNLMConfig{
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
	// *model.LSTMLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  2: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  3: *layer.TimeSoftmaxWithLoss
	//
	// [[1.0968]]
	// []

}

func ExampleLSTMLM_Params() {
	s := rand.NewSource(1)
	m := model.NewLSTMLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	}, s)

	// params
	for _, p := range m.Params() {
		// TimeEmbedding: W
		// TimeLSTM: Wx, Wh, B
		// TimeAffine: W, B
		// TimeSoftmaxWithLoss: empty
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
	// [[[-0.01233758177597947 -0.0012634751070237293 -0.005209945711531503] [0.022857191176995802 0.003228052526115799 0.005900672875996937] [0.0015880774017643562 0.009892020842955818 -0.007312830161774791]]]
	// [[[0.39628213100710546 0.9153334043970169 0.4839384045536887 0.7498861129486722 0.3044705101925196 0.42287554302900354 -0.6196006585944656 0.4042149914890089 0.24914437660275898 0.5771344100548338 -0.8798631459702666 -0.1827528623934438] [1.0908826681103836 0.6355062963164542 -0.57316054841651 0.5714095775271734 -0.3551994448368973 -0.8285247267927858 -1.2421325479259013 0.07929927515586974 0.25566376291704546 -0.4884928142744167 -0.04780173524606999 0.09013967407916586] [-0.8375177251333519 0.161510111274289 -1.0039580768702956 0.40573305268600524 0.19984338487202896 -0.6175023259590429 -0.481078214073466 0.19069266569394636 1.0079044587775612 -0.6470445829067093 0.4369380150625871 0.5353760217147717]] [[-0.8403366719913354 0.561781662287924 -0.1704416825045721 0.2945156737070663 -0.27258831347942014 0.1452267121495201 -0.04730662124662342 0.09623893143889134 0.21018154718078544 -0.9460698674055908 0.4824733508821635 0.6663954710024508] [-0.04332225549440719 -0.4448835706680916 -0.6485335795869753 -0.4481469056691629 0.38794691630310457 0.8199987497955661 -0.21291434358382558 -0.1907893007704559 -0.03697009980670319 -0.14262556026111975 0.09771708260027837 1.002068327943978] [-0.1548462888287685 -0.11883827331010591 0.6951057008096126 -0.6071792426690249 -0.28656630333453453 -0.3911684018173333 -1.0952580901247775 -1.132552176017447 -0.42460906842672264 1.146571032040675 0.05474384292921348 -0.023751097767584727]] [[0 0 0 0 0 0 0 0 0 0 0 0]]]
	// [[[-0.10275234241787481 0.6949747716557958 -0.5813276533580608] [-0.48416175718101867 0.05855543001758429 -0.810075964424542] [-1.3431649031513881 0.5625026469380159 0.14237814617887862]] [[0 0 0]]]
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

func ExampleLSTMLM_ResetState() {
	s := rand.NewSource(1)
	m := model.NewLSTMLM(&model.RNNLMConfig{
		VocabSize:   3,
		WordVecSize: 3,
		HiddenSize:  3,
		WeightInit:  weight.Xavier,
	}, s)

	m.ResetState()

	// Output:
}

func ExampleLSTMLM_rand() {
	m := model.NewLSTMLM(&model.RNNLMConfig{
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
	// *model.LSTMLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  2: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  3: *layer.TimeSoftmaxWithLoss
}
