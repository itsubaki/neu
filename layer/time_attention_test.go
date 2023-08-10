package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeAttention() {
	at := &layer.TimeAttention{}
	fmt.Println(at)

	// Output:
	// *layer.TimeAttention
}

func ExampleTimeAttention_Params() {
	at := &layer.TimeAttention{}
	at.SetParams(matrix.New())

	fmt.Println(at)
	fmt.Println(at.Params())
	fmt.Println(at.Grads())

	// Output:
	// *layer.TimeAttention
	// []
	// []
}

func ExampleTimeAttention_state() {
	at := &layer.TimeAttention{}
	at.SetState(matrix.New())
	at.ResetState()

	// Output:
}
