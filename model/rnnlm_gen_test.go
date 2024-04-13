package model_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/rand"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleRNNLMGen() {
	// model
	s := rand.Const(1)
	m := model.NewRNNLMGen(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   100,
			WordVecSize: 100,
			HiddenSize:  100,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	}, s)

	fmt.Println(m.Generate(0, []int{}, 10))
	fmt.Println(m.Generate(0, []int{6, 99, 42}, 10))

	// Output:
	// [49 20 18 15 24 96 48 31 94 79]
	// [58 61 2 35 17 90 71 52 40 29]
}

func ExampleRNNLMGen_Summary() {
	m := model.NewRNNLMGen(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   100,
			WordVecSize: 100,
			HiddenSize:  100,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.RNNLMGen
	//  0: *layer.TimeEmbedding: W(100, 100): 10000
	//  1: *layer.TimeDropout: Ratio(0.5)
	//  2: *layer.TimeGRU: Wx(100, 300), Wh(100, 300), B(1, 300): 60300
	//  3: *layer.TimeDropout: Ratio(0.5)
	//  4: *layer.TimeGRU: Wx(100, 300), Wh(100, 300), B(1, 300): 60300
	//  5: *layer.TimeDropout: Ratio(0.5)
	//  6: *layer.TimeAffine: W(100, 100), B(1, 100): 10100
	//  7: *layer.TimeSoftmaxWithLoss
}
