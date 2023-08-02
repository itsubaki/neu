package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExampleRNNMLGen() {
	// model
	s := rand.NewSource(1)
	m := model.NewRNNLMGen(&model.RNNLMConfig{
		VocabSize:   100,
		WordVecSize: 100,
		HiddenSize:  100,
		WeightInit:  weight.Xavier,
	}, s)

	fmt.Println(m.Generate(0, []int{4, 25, 80}, 10))

	// Output:
	// [0 54 90 70 89 34 12 20 45 92]
}

func ExampleChoice() {
	s := rand.NewSource(1)
	p := []float64{0.1, 0.2, 0.3, 0.4}

	for i := 0; i < 10; i++ {
		fmt.Print(model.Choice(p, s))
	}

	// Output:
	// 3332230102
}

func ExampleChoice_rand() {
	p := []float64{0.1, 0.2, 0.3, 0.4}
	if model.Choice(p) < 0 {
		fmt.Println("invalid")
	}

	// Output:
}

func ExampleContains() {
	fmt.Println(model.Contains(3, []int{1, 2, 3}))
	fmt.Println(model.Contains(0, []int{1, 2, 3}))

	// Output:
	// true
	// false
}

func ExampleFlatten() {
	xs := []matrix.Matrix{
		matrix.New(
			[]float64{0, 1, 2},
			[]float64{0, 1, 2},
		),
		matrix.New(
			[]float64{3, 4, 5},
			[]float64{3, 4, 5},
		),
	}

	fmt.Println(model.Flatten(xs))

	// Output:
	// [0 1 2 0 1 2 3 4 5 3 4 5]
}
