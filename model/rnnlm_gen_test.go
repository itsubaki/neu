package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleRNNLMGen() {
	// model
	s := rand.NewSource(1)
	m := model.NewRNNLMGen(&model.LSTMLMConfig{
		RNNLMConfig: model.RNNLMConfig{
			VocabSize:   100,
			WordVecSize: 100,
			HiddenSize:  100,
			WeightInit:  weight.Xavier,
		},
		DropoutRatio: 0.5,
	}, s)

	fmt.Println(m.Generate(0, []int{86, 28, 37}, 10))

	// Output:
	// [30 29 45 20 58 3 68 43 31 72]
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
	//  2: *layer.TimeLSTM: Wx(100, 400), Wh(100, 400), B(1, 400): 80400
	//  3: *layer.TimeDropout: Ratio(0.5)
	//  4: *layer.TimeLSTM: Wx(100, 400), Wh(100, 400), B(1, 400): 80400
	//  5: *layer.TimeDropout: Ratio(0.5)
	//  6: *layer.TimeAffine: W(100, 100), B(1, 100): 10100
	//  7: *layer.TimeSoftmaxWithLoss
}
