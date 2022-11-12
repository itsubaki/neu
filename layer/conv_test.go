package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

func Example_outhw() {
	outhw := func(xh, xw, fh, fw, pad, stride int) (int, int) {
		outh := 1 + int((xh+2*pad-fh)/stride)
		outw := 1 + int((xw+2*pad-fw)/stride)
		return outh, outw
	}

	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	H, W := len(x), len(x[0])
	FH, FW := 2, 2
	pad, stride := 1, 1
	fmt.Println(outhw(H, W, FH, FW, pad, stride))

	// Output:
	// 3 3

}

func Example_padding() {
	padding := func(x matrix.Matrix, pad int) matrix.Matrix {
		_, q := x.Dimension()
		pw := q + pad + pad // right + row + left

		// top
		out := matrix.New()
		for i := 0; i < pad; i++ {
			out = append(out, make([]float64, pw))
		}

		// right, left
		for i := range x {
			v := append(make([]float64, pad), x[i]...) // right + row
			v = append(v, make([]float64, pad)...)     // right + row + left
			out = append(out, v)
		}

		// bottom
		for i := 0; i < pad; i++ {
			out = append(out, make([]float64, pw))
		}

		return out
	}

	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	for _, v := range padding(x, 1) {
		fmt.Println(v)
	}

	// Output:
	// [0 0 0 0]
	// [0 1 2 0]
	// [0 3 4 0]
	// [0 0 0 0]

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

}
