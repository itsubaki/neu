package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleSoftmax() {
	softmax := &layer.Softmax{}
	fmt.Println(softmax)

	// forward
	x := matrix.New(
		[]float64{2, 2, 6},
		[]float64{4, 3, 3},
	)
	out := softmax.Forward(x, nil)
	for _, r := range out {
		fmt.Println(r)
	}

	// backward
	dx, _ := softmax.Backward(x)
	fmt.Println(dx)

	// Output:
	// *layer.Softmax
	// [0.017668422014048047 0.017668422014048047 0.9646631559719038]
	// [0.5761168847658291 0.21194155761708544 0.21194155761708544]
	// [[-0.0681763029644602 -0.0681763029644602 0.13635260592892173] [0.2442062198535453 -0.12210310992677276 -0.12210310992677276]]
}

func ExampleSoftmax_Params() {
	softmax := &layer.Softmax{}

	softmax.SetParams(make([]matrix.Matrix, 0)...)
	fmt.Println(softmax.Params())
	fmt.Println(softmax.Grads())

	// Output:
	// []
	// []
}
