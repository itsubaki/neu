package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func ExampleTimeAttention() {
	at := &layer.TimeAttention{}
	fmt.Println(at)

	// forward
	hsenc := []matrix.Matrix{
		// (T, N, H) (2, 2, 3)
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{4, 5, 6},
			{4, 5, 6},
		},
	}

	hsdec := []matrix.Matrix{
		// (T, N, H) (2, 2, 3)
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{4, 5, 6},
			{4, 5, 6},
		},
	}

	fmt.Println(at.Forward(hsenc, hsdec))

	// backward
	dout := []matrix.Matrix{
		{
			{1, 2, 3},
			{4, 5, 6},
		},
		{
			{4, 5, 6},
			{4, 5, 6},
		},
	}
	dhs, dh := at.Backward(dout)
	fmt.Println(dhs)
	fmt.Println(dh)

	// Output:
	// *layer.TimeAttention
	// [[[1.1450074365793674e-19 1.4312592989939168e-19 1.7175111614084662e-19] [8 10 12]] [[2 2.5 3] [6 7.5 9]]]
	// [[[-5.038032728796514e-18 -6.2975409312678295e-18 -7.557049133739145e-18] [8 10 12]] [[2 2.5 3] [6 7.5 9]]]
	// [[[-5.152533472454451e-18 -6.4406668611672205e-18 -7.728800249879992e-18] [0 0 0]] [[-1.288133361247227e-18 -2.576266722494454e-18 -3.864400083741681e-18] [0 0 0]]]
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
