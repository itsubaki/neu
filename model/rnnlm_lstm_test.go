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
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   3,
			WordVecSize: 3,
			HiddenSize:  3,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
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
	// *model.LSTMLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeDropout: Ratio(0.5)
	//  2: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  3: *layer.TimeDropout: Ratio(0.5)
	//  4: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  5: *layer.TimeDropout: Ratio(0.5)
	//  6: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  7: *layer.TimeSoftmaxWithLoss
	//
	// [[[1.0984]]]
	// []

}

func ExampleLSTMLM_Summary() {
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   3,
			WordVecSize: 3,
			HiddenSize:  3,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.LSTMLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeDropout: Ratio(0.5)
	//  2: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  3: *layer.TimeDropout: Ratio(0.5)
	//  4: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  5: *layer.TimeDropout: Ratio(0.5)
	//  6: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  7: *layer.TimeSoftmaxWithLoss
}

func ExampleLSTMLM_Layers() {
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   3,
			WordVecSize: 3,
			HiddenSize:  3,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	})

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// Output:
	// *model.LSTMLM
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeDropout: Ratio(0.5)
	//  2: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  3: *layer.TimeDropout: Ratio(0.5)
	//  4: *layer.TimeLSTM: Wx(3, 12), Wh(3, 12), B(1, 12): 84
	//  5: *layer.TimeDropout: Ratio(0.5)
	//  6: *layer.TimeAffine: W(3, 3), B(1, 3): 12
	//  7: *layer.TimeSoftmaxWithLoss
}

func ExampleLSTMLM_Params() {
	s := rand.NewSource(1)
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   3,
			WordVecSize: 3,
			HiddenSize:  3,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
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
		fmt.Println(p) // empty
	}
	fmt.Println()

	// Output:
	// [[[-0.01233758177597947 -0.0012634751070237293 -0.005209945711531503] [0.022857191176995802 0.003228052526115799 0.005900672875996937] [0.0015880774017643562 0.009892020842955818 -0.007312830161774791]]]
	// []
	// [[[0.39628213100710546 0.9153334043970169 0.4839384045536887 0.7498861129486722 0.3044705101925196 0.42287554302900354 -0.6196006585944656 0.4042149914890089 0.24914437660275898 0.5771344100548338 -0.8798631459702666 -0.1827528623934438] [1.0908826681103836 0.6355062963164542 -0.57316054841651 0.5714095775271734 -0.3551994448368973 -0.8285247267927858 -1.2421325479259013 0.07929927515586974 0.25566376291704546 -0.4884928142744167 -0.04780173524606999 0.09013967407916586] [-0.8375177251333519 0.161510111274289 -1.0039580768702956 0.40573305268600524 0.19984338487202896 -0.6175023259590429 -0.481078214073466 0.19069266569394636 1.0079044587775612 -0.6470445829067093 0.4369380150625871 0.5353760217147717]] [[-0.8403366719913354 0.561781662287924 -0.1704416825045721 0.2945156737070663 -0.27258831347942014 0.1452267121495201 -0.04730662124662342 0.09623893143889134 0.21018154718078544 -0.9460698674055908 0.4824733508821635 0.6663954710024508] [-0.04332225549440719 -0.4448835706680916 -0.6485335795869753 -0.4481469056691629 0.38794691630310457 0.8199987497955661 -0.21291434358382558 -0.1907893007704559 -0.03697009980670319 -0.14262556026111975 0.09771708260027837 1.002068327943978] [-0.1548462888287685 -0.11883827331010591 0.6951057008096126 -0.6071792426690249 -0.28656630333453453 -0.3911684018173333 -1.0952580901247775 -1.132552176017447 -0.42460906842672264 1.146571032040675 0.05474384292921348 -0.023751097767584727]] [[0 0 0 0 0 0 0 0 0 0 0 0]]]
	// []
	// [[[-0.10275234241787481 0.6949747716557958 -0.5813276533580608 -0.48416175718101867 0.05855543001758429 -0.810075964424542 -1.3431649031513881 0.5625026469380159 0.14237814617887862 -1.150637360799377 0.3584429333570677 0.22010540258634428] [-0.5658637077486658 0.23646603943121486 0.32971444858948 1.4608338463834516 0.21420969159622294 -0.733374975459535 -1.257937197375443 -0.24284958084556457 -0.4950788221188206 0.0012474625993624132 -0.6884749278539299 1.3757450618764155] [0.8539376862235681 0.11042754459903549 0.00160685709233085 0.005777081100111445 -0.04146096361872453 -0.2646070462849363 -1.2510741552054039 -0.27508291147356656 -0.3217625945352978 1.5768139474256384 -0.8407139792462694 0.030293937347552674]] [[-0.3571258467148583 0.4618088561017904 -0.5225071958504695 0.41877379391762865 0.20983886123064874 -0.6578189686486637 -0.35835390169975356 -0.5860819302760614 0.08283734994554123 -1.1890192755264755 -0.02939359271348786 -0.7514222098129523] [0.758651995952598 -0.6827582184211474 0.36651690254475283 0.6353951804882626 -0.20608359931826029 0.35273880603582053 -0.7003085617555934 1.0847913361187638 0.007922216727405954 -0.8937374876228238 -0.11147955811772293 -0.9153735882377443] [-1.3117518440713147 0.3893344095811681 -0.40121097551338414 -0.48763507406843587 0.13355916305386786 -0.5167880897304595 1.148844890769795 0.23911669895068022 0.34842932020466455 -0.7022620799632179 -1.2875533615094146 0.1017793985093133]] [[0 0 0 0 0 0 0 0 0 0 0 0]]]
	// []
	// [[[-1.4011627682597354 0.11870824471495119 0.13616130822492192] [0.2620565798100286 0.4029608284307762 0.5941724987423397] [0.043332356368910606 -0.5486397228538327 -0.384826756099125]] [[0 0 0]]]
	// []
	//
	// [[]]
	// []
	// [[] [] []]
	// []
	// [[] [] []]
	// []
	// [[] []]
	// []
	//
	// [[]]
	// []
	// [[] [] []]
	// []
	// [[] [] []]
	// []
	// [[] []]
	// []
}

func ExampleLSTMLM_ResetState() {
	s := rand.NewSource(1)
	m := model.NewLSTMLM(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   3,
			WordVecSize: 3,
			HiddenSize:  3,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	}, s)

	m.ResetState()

	// Output:
}
