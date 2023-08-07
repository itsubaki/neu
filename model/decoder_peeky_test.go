package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/neu/math/matrix"
	"github.com/itsubaki/neu/model"
	"github.com/itsubaki/neu/weight"
)

func ExamplePeekyDecoder() {
	s := rand.NewSource(1)
	m := model.NewPeekyDecoder(&model.DecoderConfig{
		VocabSize:   3, // V
		WordVecSize: 3, // D
		HiddenSize:  3, // H
		WeightInit:  weight.Xavier,
	}, s)

	fmt.Printf("%T\n", m)
	for i, l := range m.Layers() {
		fmt.Printf("%2d: %v\n", i, l)
	}
	fmt.Println()

	// forward
	xs := []matrix.Matrix{
		// (T, N, 1) = (2, 3, 1)
		{{0.1}, {0.2}, {0.3}}, // (N, 1) = (3, 1)
		{{0.1}, {0.2}, {0.3}}, // (N, 1) = (3, 1)
	}

	// (N, H) = (3, 3)
	h := matrix.New(
		[]float64{0.1, 0.2, 0.3},
		[]float64{0.3, 0.4, 0.5},
		[]float64{0.3, 0.4, 0.5},
	)

	score := m.Forward(xs, h)
	fmt.Println(len(score))
	for _, s := range score {
		fmt.Println(s.Dimension())
	}

	// backward
	dout := []matrix.Matrix{
		{{0.1}, {0.1}, {0.1}},
		{{0.1}, {0.1}, {0.1}},
	}
	dh := m.Backward(dout)
	fmt.Println(dh.Dimension())

	// generate
	sampeld := m.Generate(h, 1, 10)
	fmt.Println(sampeld)

	// Output:
	// *model.PeekyDecoder
	//  0: *layer.TimeEmbedding: W(3, 3): 9
	//  1: *layer.TimeLSTM: Wx(6, 12), Wh(3, 12), B(1, 12): 120
	//  2: *layer.TimeAffine: W(6, 3), B(1, 3): 21
	//
	// 2
	// 3 3
	// 3 3
	// 3 3
	// [1 0 0 0 0 0 0 0 0 0 0]
}

func ExampleConcat() {
	hs := []matrix.Matrix{
		{
			{1, 2},
			{3, 4},
		},
		{
			{5, 6},
			{7, 8},
		},
	}

	out := []matrix.Matrix{
		{
			{10, 20},
			{30, 40},
		},
		{
			{50, 60},
			{70, 80},
		},
	}

	for _, t := range model.Concat(hs, out) {
		fmt.Println(t)
	}

	// Output:
	// [[1 2 10 20] [3 4 30 40]]
	// [[5 6 50 60] [7 8 70 80]]
}

func ExampleSplit() {
	dout := []matrix.Matrix{
		{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		},
		{
			{9, 10, 11, 12},
			{13, 14, 15, 16},
		},
		{
			{17, 18, 19, 20},
			{21, 22, 23, 24},
		},
	}

	dout, dhs := model.Split(dout, 2)
	for _, r := range dout {
		fmt.Println(r)
	}
	for _, r := range dhs {
		fmt.Println(r)
	}

	// Output:
	// [[3 4] [7 8]]
	// [[11 12] [15 16]]
	// [[19 20] [23 24]]
	// [[1 2] [5 6]]
	// [[9 10] [13 14]]
	// [[17 18] [21 22]]

}

func ExampleSumAxis1() {
	dhs := []matrix.Matrix{
		{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
		},
		{
			{9, 10, 11, 12},
			{13, 14, 15, 16},
		},
		{
			{17, 18, 19, 20},
			{21, 22, 23, 24},
		},
	}

	for _, r := range model.SumAxis1(dhs) {
		fmt.Println(r)
	}

	// Output:
	// [6 8 10 12]
	// [22 24 26 28]
	// [38 40 42 44]
}
