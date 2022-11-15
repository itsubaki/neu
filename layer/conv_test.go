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

var im2col = func(m matrix.Matrix, fh, fw, pad, stride int) matrix.Matrix {
	pickup := func(m matrix.Matrix, y, x, ymax, xmax, stride int) []float64 {
		// NOTE: double loop. no loop (m[y:ymax:stride, x:xmax:stride]) in numpy.
		out := make([]float64, 0)
		for i := y; i < ymax; i = i + stride {
			for j := x; j < xmax; j = j + stride {
				out = append(out, m[i][j])
			}
		}

		return out
	}

	// N, C, H, W := 1, 1, 2, 2
	// pad := 1
	// [0 0 0 0]
	// [0 1 2 0]
	// [0 3 4 0]
	// [0 0 0 0]
	img := matrix.Padding(m, 1)

	// FH, FW, stride := 2, 2, 1
	// [0 0 0 0 1 2 0 3 4]
	// [0 0 0 1 2 0 3 4 0]
	// [0 1 2 0 3 4 0 0 0]
	// [1 2 0 3 4 0 0 0 0]
	xh, xw := m.Dimension()
	outh, outw := outhw(xh, xw, fh, fw, pad, stride)
	out := matrix.New()
	for y := 0; y < fh; y++ {
		ymax := y + stride*outh
		for x := 0; x < fw; x++ {
			xmax := x + stride*outw
			out = append(out, pickup(img, y, x, ymax, xmax, stride))
		}
	}

	// Transpose
	// [0 0 0 1]
	// [0 0 1 2]
	// [0 0 2 0]
	// [0 1 0 3]
	// [1 2 3 4]
	// [2 0 4 0]
	// [0 3 0 0]
	// [3 4 0 0]
	// [4 0 0 0]
	return out.T()
}

func Example_outhw() {
	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	H, W := x.Dimension()
	FH, FW := 2, 2
	pad, stride := 1, 1

	fmt.Println(outhw(H, W, FH, FW, pad, stride))

	// Output:
	// 3 3

}

func Example_im2col() {
	x := matrix.New([]float64{1, 2}, []float64{3, 4})
	for _, r := range im2col(x, 2, 2, 1, 1) {
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
