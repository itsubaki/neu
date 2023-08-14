package layer_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleUnigramSampler() {
	corpus := []int{0, 1, 2, 3, 4, 5, 6}
	power := 0.75
	sampleSize := 2
	sampler := layer.NewUnigramSampler(corpus, power, sampleSize)

	target := []int{0, 2, 4}
	for i, v := range sampler.NegativeSample(target, rand.NewSource(0)) {
		fmt.Printf("%v: %v\n", target[i], v)
	}

	for i, v := range sampler.NegativeSample(target) {
		fmt.Printf("%v: %v\n", target[i], len(v))
	}

	// Output:
	// 0: [6 2]
	// 2: [4 0]
	// 4: [2 1]
	// 0: 2
	// 2: 2
	// 4: 2
}

func ExampleNegativeSamplingLoss() {
	W := matrix.New(
		[]float64{0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1},
	)
	corpus := []int{0, 1, 2, 3, 4, 5, 6}
	power := 0.75
	sampleSize := 2

	l := layer.NewNegativeSamplingLoss(W, corpus, power, sampleSize)
	fmt.Println(l)

	// forward
	h := matrix.New(
		[]float64{0, 1, 2},
		[]float64{3, 4, 5},
		[]float64{6, 7, 8},
	)

	idx := matrix.New(
		[]float64{0},
		[]float64{3},
		[]float64{1},
	)

	loss := l.Forward(h, idx)
	dh, _ := l.Backward(matrix.New([]float64{1.0}))
	fmt.Println(loss)
	fmt.Println(dh)

	// Output:
	// *layer.NegativeSamplingLoss: W(3, 3)*3: 27
	// []
	// []
}

func ExampleNegativeSamplingLoss_Params() {
	W := matrix.New([]float64{0.1, 0.1, 0.1})
	corpus := []int{0, 1, 2, 3, 4, 5, 6}
	power := 0.75
	sampleSize := 2
	l := layer.NewNegativeSamplingLoss(W, corpus, power, sampleSize)

	l.SetParams(make([]matrix.Matrix, sampleSize+1)...)
	fmt.Println(l.Params())
	fmt.Println(l.Grads())

	// Output:
	// [[] [] []]
	// [[] [] []]
}
