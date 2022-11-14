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

var im2col = func(x matrix.Matrix, fh, fw, pad, stride int) matrix.Matrix {
	pickup := func(m matrix.Matrix, y, x, ymax, xmax, stride int) []float64 {
		out := make([]float64, 0)
		for i := y; i < ymax; i = i + stride {
			for j := x; j < xmax; j = j + stride {
				out = append(out, m[i][j])
			}
		}

		return out
	}

	outh, outw := outhw(len(x), len(x[0]), fh, fw, pad, stride)
	img := matrix.Padding(x, 1)

	out := matrix.New()
	for y := 0; y < fh; y++ {
		ymax := y + stride*outh
		for x := 0; x < fw; x++ {
			xmax := x + stride*outw
			out = append(out, pickup(img, y, x, ymax, xmax, stride))

			// row:column
			// 0:0, 0:2, 2:0, 2:2
			// 0:1, 0:3, 2:1, 2:3
			// 1:0, 1:2, 3:0, 3:2
			// 1:1, 1:3, 3:1, 3:3
			// fmt.Printf("y:%v, x:%v ~ y:%v, x:%v. stride=%v\n", y, x, ymax-1, xmax-1, stride)
		}
	}

	return out.T()
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
