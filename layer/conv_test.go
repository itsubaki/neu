package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

var outhw = func(xh, xw, fh, fw, pad, stride int) (int, int) {
	outh := 1 + int((xh+2*pad-fh)/stride)
	outw := 1 + int((xw+2*pad-fw)/stride)
	return outh, outw
}

func Example_outhw() {
	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	H, W := len(x), len(x[0])
	FH, FW := 2, 2
	pad, stride := 1, 1

	fmt.Println(outhw(H, W, FH, FW, pad, stride))

	// Output:
	// 3 3

}

func Example_im2col() {
	// N, C, H, W := 1, 1, 2, 2
	// pad := 1
	// [0 0 0 0]
	// [0 1 2 0]
	// [0 3 4 0]
	// [0 0 0 0]
	//
	// im2col
	// FH, FW, stride := 2, 2, 1
	// [0 0 0 1]
	// [0 0 1 2]
	// [0 0 2 0]
	// [0 1 0 3]
	// [1 2 3 4]
	// [2 0 4 0]
	// [0 3 0 0]
	// [3 4 0 0]
	// [4 0 0 0]
	//
	// FN, C, FH, FW := 1, 1, 2, 2
	// [1 2]
	// [3 4]
	//
	// matrix.Dot(col, [1 2 3 4].T())
	//
	im2col := func(x matrix.Matrix, fh, fw, pad, stride int) matrix.Matrix {
		outh, outw := outhw(len(x), len(x[0]), fh, fw, pad, stride)
		img := matrix.Padding(x, 1)

		fmt.Printf("%v %v\n", outh, outw)
		for _, r := range img {
			fmt.Println(r)
		}

		return matrix.New()
	}

	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	for _, r := range im2col(x, 2, 2, 1, 1) {
		fmt.Println(r)
	}

	// Output:
	// 3 3
	// [0 0 0 0]
	// [0 1 2 0]
	// [0 3 4 0]
	// [0 0 0 0]

}
