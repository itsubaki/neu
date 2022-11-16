package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/layer"
	"github.com/itsubaki/neu/math/matrix"
)

func Example_outhw() {
	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	xh, xw := x.Dimension()
	fh, fw := 2, 2
	pad, stride := 1, 1

	fmt.Println(layer.Outhw(xh, xw, fh, fw, pad, stride))

	// Output:
	// 3 3

}

func Example_im2col() {
	x := matrix.New(
		[]float64{1, 2},
		[]float64{3, 4},
	)

	fh, fw := 2, 2
	pad, stride := 1, 1
	for _, r := range layer.Im2col(x, fh, fw, pad, stride) {
		fmt.Println(r)
	}

	// Output:
	// [0 0 0 1]
	// [0 0 1 2]
	// [0 0 2 0]
	// [0 1 0 3]
	// [1 2 3 4]
	// [2 0 4 0]
	// [0 3 0 0]
	// [3 4 0 0]
	// [4 0 0 0]

}

func Example_im2col_c2() {
	// N, C, H, W := 1, 2, 2, 2
	// pad := 1
	// [0 0 0 0] [0 0 0 0]
	// [0 1 2 0] [0 5 6 0]
	// [0 3 4 0] [0 7 8 0]
	// [0 0 0 0] [0 0 0 0]
	//
	// FC, FH, FW, stride := 2, 2, 2, 1
	// [0 0 0 0 1 2 0 3 4]
	// [0 0 0 1 2 0 3 4 0]
	// [0 1 2 0 3 4 0 0 0]
	// [1 2 0 3 4 0 0 0 0]
	// [0 0 0 0 5 6 0 7 8]
	// [0 0 0 5 6 0 7 8 0]
	// [0 5 6 0 7 8 0 0 0]
	// [5 6 0 7 8 0 0 0 0]
	//
	// transpose
	// [0 0 0 1 0 0 0 5]
	// [0 0 1 2 0 0 5 6]
	// [0 0 2 0 0 0 6 0]
	// [0 1 0 3 0 5 0 7]
	// [1 2 3 4 5 6 7 8]
	// [2 0 4 0 6 0 8 0]
	// [0 3 0 0 0 7 0 0]
	// [3 4 0 0 7 8 0 0]
	// [4 0 0 0 8 0 0 0]
}

func Example_col2im() {
	x := matrix.New(
		[]float64{0, 0, 0, 1},
		[]float64{0, 0, 1, 2},
		[]float64{0, 0, 2, 0},
		[]float64{0, 1, 0, 3},
		[]float64{1, 2, 3, 4},
		[]float64{2, 0, 4, 0},
		[]float64{0, 3, 0, 0},
		[]float64{3, 4, 0, 0},
		[]float64{4, 0, 0, 0},
	)

	xh, xw := 2, 2
	fh, fw := 2, 2
	pad, stride := 1, 1
	for _, r := range layer.Col2im(x, xh, xw, fh, fw, pad, stride) {
		fmt.Printf("%2v\n", r)
	}

	// Output:
	// [ 4  8]
	// [12 16]
}

func ExampleConvolution() {
	c := &layer.Convolution{}
	fmt.Println(c.Forward(nil, nil))

	// Output
	// nil
}
