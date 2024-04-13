package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/math/rand"
)

func ExampleUnigramSampler() {
	// p158
	corpus := []int{0, 1, 2, 3, 4, 1, 2, 3}
	power := 0.75
	sampleSize := 2
	sampler := layer.NewUnigramSampler(corpus, power, sampleSize)

	target := []int{1, 3, 0}
	for i, v := range sampler.NegativeSample(target, rand.Const(1)) {
		fmt.Printf("%v: %v\n", target[i], v)
	}

	for i, v := range sampler.NegativeSample(target) {
		fmt.Printf("%v: %v\n", target[i], len(v))
	}

	// Output:
	// 1: [2 3]
	// 3: [0 1]
	// 0: [3 3]
	// 1: 2
	// 3: 2
	// 0: 2
}

func ExampleNegativeSamplingLoss() {
	W := matrix.New(
		// (V, H) = (7, 5)
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
	)
	// you say goodbye and i say hello.
	corpus := []int{0, 1, 2, 3, 4, 1, 5, 6}
	power := 0.75
	sampleSize := 2

	s := rand.Const(1)
	l := layer.NewNegativeSamplingLoss(W, corpus, power, sampleSize, s)
	fmt.Println(l)

	// forward
	target := matrix.New(
		// (3, 1)
		[]float64{0},
		[]float64{1},
		[]float64{2},
	)

	h := matrix.New(
		// (3, 5)
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
		[]float64{0.1, 0.1, 0.1, 0.1, 0.1},
	)

	loss := l.Forward(h, target)
	dh, _ := l.Backward(matrix.New([]float64{1.0}))
	fmt.Println(loss)
	for _, r := range dh {
		fmt.Println(r)
	}

	// Output:
	// *layer.NegativeSamplingLoss: W(7, 5)*3: 105
	// [[6.316135015988275]]
	// [0.0537492189452631 0.0537492189452631 0.0537492189452631 0.0537492189452631 0.0537492189452631]
	// [0.0537492189452631 0.0537492189452631 0.0537492189452631 0.0537492189452631 0.0537492189452631]
	// [0.0537492189452631 0.0537492189452631 0.0537492189452631 0.0537492189452631 0.0537492189452631]
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
