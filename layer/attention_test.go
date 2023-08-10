package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleAttention() {
	at := &layer.Attention{}
	fmt.Println(at)

	// Output:
	// *layer.Attention
}

func ExampleAttention_Params() {
	at := &layer.Attention{}
	at.SetParams(matrix.New())

	fmt.Println(at.Params())
	fmt.Println(at.Grads())

	// Output:
	// []
	// []
}
