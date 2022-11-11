package layer_test

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

func Example_filter() {
	outhw := func(xh, xw, fh, fw, pad, stride int) (int, int) {
		outh := 1 + int((xh+2*pad-fh)/stride)
		outw := 1 + int((xw+2*pad-fw)/stride)
		return outh, outw
	}

	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	fh, fw := 2, 2
	pad, stride := 1, 1
	fmt.Println(outhw(len(x), len(x[0]), fh, fw, pad, stride))

	// Output:
	// 3 3

}

func Example_padding() {
	padding := func(x matrix.Matrix, pad int) matrix.Matrix {
		_, q := x.Dimension()
		pw := q + pad*2

		// top
		out := matrix.New()
		for i := 0; i < pad; i++ {
			out = append(out, make([]float64, pw))
		}

		// right, left
		for i := range x {
			v := append(make([]float64, pad), x[i]...)
			v = append(v, make([]float64, pad)...)
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
	// [0 0 0 0]
	// [0 1 2 0]
	// [0 3 4 0]
	// [0 0 0 0]
	//
	// fh, fw := 2, 2 then
	//
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
