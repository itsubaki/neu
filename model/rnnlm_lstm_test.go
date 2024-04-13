package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleLSTMLM() {
	// model
	s := rand.Const(1)
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
	// [[[1.0980]]]
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
	s := rand.Const(1)
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
	// [[[-0.008024826241110656 0.00424707052949676 -0.004985070978632815] [-0.009872764577745819 0.004770185009670911 -0.0037300956589935985] [0.01182810122110346 -0.008066822642915392 0.0010337870485847512]]]
	// []
	// [[[-0.4272036569812764 -0.051439283460071226 0.0588887761030214 -0.11157067538737257 0.42479877444693387 -1.6415745494204534 -0.969465128610357 -0.4261080517614557 1.0999236840248148 0.18248391466570424 0.5477994083834968 -0.1579822276645198] [-0.48305059627014674 -0.4277824240418346 0.9395249505867087 0.2701827393831761 0.49870245028370974 -0.1766569492870927 -0.5584779273923454 0.2927783198117813 0.5136963587943321 -1.208969787150421 -0.6633000273730952 0.19743387804132795] [-0.33298687925268927 -0.6371079759680973 -0.14198751865229117 -1.0755621793871613 -0.7671272572462577 -0.32570058481751957 0.21394140584165708 -0.7127881311002123 -0.04095649276208288 -0.03837447220745022 -0.5991932513467102 -0.017016953549988983]] [[-0.4098079648356177 0.5841891298736116 -0.19004422058496503 -0.242732946421929 0.4048379833430214 -1.7682180737738304 0.6755067469892815 -0.32934343529086985 0.9463978963432617 0.4732677890259521 -0.394499569814648 0.4616116370171369] [0.2777660186242951 1.349075724141619 0.8107818436096154 1.1018490163609667 0.3473367811842638 -0.567710991433003 -0.7937025138366497 0.4962986831556114 0.7269467789623814 -0.4460154953746029 0.4460578769593527 0.2012014993613915] [-0.8361576103759227 -0.7130719208488365 -0.09881029255254005 -0.7757896302700097 -0.41347222182325793 0.028587083590823334 0.1232570269208117 -0.932332007213963 1.0298043762384175 -0.13202137955123036 -0.5911391914783959 -0.2611930507876444]] [[0 0 0 0 0 0 0 0 0 0 0 0]]]
	// []
	// [[[-1.0024472239105173 -0.3441216747244834 0.07879031049900392 0.40288000908582394 1.0655841303826354 -0.5186219238074636 0.7027072489323264 0.08301787262134458 -0.11428923965872227 -0.06173865431500391 0.030275664806317493 -0.01723999636712435] [0.1265819057720313 0.8695189233518246 0.011076408796036552 -0.41842317384105804 0.5556506872869873 1.0390197464181663 0.9263873365425865 0.9342079944041369 -0.021776911727067375 -0.1557135679338374 0.7720266399590385 -0.7877506651931699] [0.28718707924261677 -0.30143842882104754 0.5275229308216967 0.12861783882061126 -0.35343087396823614 -0.5602651643915585 -0.6282678711526127 0.17888941017634233 -0.6329274282360814 -0.2330867835135977 -0.4122994881464416 0.047007428125805134]] [[0.5107855197317723 -0.7883129626058847 0.25327214898194494 0.3186142912951048 -0.5041809064145354 -0.666227030098143 0.26482770843250947 -1.0097702685047043 1.3034583837720501 -0.44737347452645254 0.0593915195099439 -0.04743844933980312] [1.507127970665918 -0.960931835864583 -0.17896862942487216 0.3142218616680891 0.3141319944155909 -0.6989857827487986 0.08172385743163078 -0.05773051501282179 -0.1642158153326509 -0.17816582491255425 -0.2401798921445646 -0.09361028557537397] [-0.6983283048615403 0.12024316719283198 -0.8740123395034683 -0.33627122783340485 -0.021649239399288773 0.1029164241859031 0.8999817603805487 -0.3527692697564272 0.5114098931786875 -0.2132777900223977 -0.26936672083908186 0.8492327229314398]] [[0 0 0 0 0 0 0 0 0 0 0 0]]]
	// []
	// [[[-0.7297382414751158 0.5360508782200062 -0.2318788772237097] [-0.2145482583925129 0.4558365206471918 0.41505891170370396] [0.7248410233361068 0.38321308798257114 -0.3101744738919172]] [[0 0 0]]]
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
	s := rand.Const(1)
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
