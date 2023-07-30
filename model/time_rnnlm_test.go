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
		matrix.New([]float64{0}),
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
	// [[3.3973]]
	// []

}
