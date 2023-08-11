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
