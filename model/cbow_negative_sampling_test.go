package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
)

func ExampleCBOWNegativeSampling() {
	s := rand.NewSource(1)
	m := model.NewCBOWNegativeSampling(model.CBOWNegativeSamplingConfig{
		CBOWConfig: model.CBOWConfig{
			VocabSize:  7,
			HiddenSize: 5,
		},
		Corpus:     []int{0, 1, 2, 3, 4, 1, 5, 6},
		WindowSize: 1,
		SampleSize: 2,
		Power:      0.75,
	}, s)

	contexts := matrix.New(
		[]float64{0, 2}, // you, goodbye
		[]float64{1, 3}, // say, and
		[]float64{2, 4}, // goodbye, i
		[]float64{3, 1}, // and, say
		[]float64{4, 5}, // i, hello
		[]float64{1, 6}, // say, .
	)

	target := matrix.New(
		[]float64{1}, // say
		[]float64{2}, // goodbye
		[]float64{3}, // and
		[]float64{4}, // i
		[]float64{1}, // say
		[]float64{5}, // hello
	)

	loss := m.Forward(contexts, target)
	m.Backward()
	fmt.Println(loss)

	// Output:
	// [[12.476638924761186]]
}

func ExampleCBOWNegativeSampling_Summary() {
	m := model.NewCBOWNegativeSampling(model.CBOWNegativeSamplingConfig{
		CBOWConfig: model.CBOWConfig{
			VocabSize:  7,
			HiddenSize: 5,
		},
		Corpus:     []int{0, 1, 2, 3, 4, 1, 5, 6},
		WindowSize: 1,
		SampleSize: 2,
		Power:      0.75,
	})

	fmt.Println(m.Summary()[0])
	for i, s := range m.Summary()[1:] {
		fmt.Printf("%2d: %v\n", i, s)
	}

	// Output:
	// *model.CBOWNegativeSampling
	//  0: *layer.Embedding: W(7, 5): 35
	//  1: *layer.Embedding: W(7, 5): 35
	//  2: *layer.NegativeSamplingLoss: W(7, 5)*3: 105
}

func ExampleCBOWNegativeSampling_Layers() {
	m := model.NewCBOWNegativeSampling(model.CBOWNegativeSamplingConfig{
		CBOWConfig: model.CBOWConfig{
			VocabSize:  7,
			HiddenSize: 5,
		},
		Corpus:     []int{0, 1, 2, 3, 4, 1, 5, 6},
		WindowSize: 1,
		SampleSize: 2,
		Power:      0.75,
	})

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}

	// Output:
	// *model.CBOWNegativeSampling
	//  0: *layer.Embedding: W(7, 5): 35
	//  1: *layer.Embedding: W(7, 5): 35
	//  2: *layer.NegativeSamplingLoss: W(7, 5)*3: 105
}

func ExampleCBOWNegativeSampling_Params() {
	m := model.NewCBOWNegativeSampling(model.CBOWNegativeSamplingConfig{
		CBOWConfig: model.CBOWConfig{
			VocabSize:  7,
			HiddenSize: 5,
		},
		Corpus:     []int{0, 1, 2, 3, 4, 1, 5, 6},
		WindowSize: 1,
		SampleSize: 2,
		Power:      0.75,
	})

	for _, p := range m.Params() {
		for _, m := range p {
			fmt.Println(m.Dim())
		}
	}

	m.SetParams(m.Grads())
	fmt.Println(m.Params())
	fmt.Println(m.Grads())

	// Output:
	// 7 5
	// 7 5
	// 7 5
	// 7 5
	// 7 5
	// [[[]] [[]] [[] [] []]]
	// [[[]] [[]] [[] [] []]]
}
