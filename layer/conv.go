package layer

import "github.com/itsubaki/neu/math/matrix"

func outhw(xh, xw, fh, fw, pad, stride int) (int, int) {
	outh := 1 + int((xh+2*pad-fh)/stride)
	outw := 1 + int((xw+2*pad-fw)/stride)
	return outh, outw
}

// TODO: "チャンネル方向に複数の特徴マップがある場合、チャンネルごとに入力データとフィルターの畳み込み演算を行い、それらの結果を加算してひとつの出力を得ます。" ゼロから作るDeepLearning　p214
func im2col(im matrix.Matrix, fh, fw, pad, stride int) matrix.Matrix {
	pick := func(img matrix.Matrix, y, x, ymax, xmax, stride int) []float64 {
		// NOTE: double loop. no loop (m[y:ymax:stride, x:xmax:stride]) in numpy.
		out := make([]float64, 0)
		for i := y; i < ymax; i = i + stride {
			for j := x; j < xmax; j = j + stride {
				out = append(out, img[i][j])
			}
		}

		return out
	}

	// N, C, H, W := 1, 1, 2, 2
	// [1 2]
	// [3 4]
	//
	// pad := 1
	// [0 0 0 0]
	// [0 1 2 0]
	// [0 3 4 0]
	// [0 0 0 0]
	img := matrix.Padding(im, pad)

	// FC, FH, FW, stride := 1, 2, 2, 1
	// [0 0 0 0 1 2 0 3 4]
	// [0 0 0 1 2 0 3 4 0]
	// [0 1 2 0 3 4 0 0 0]
	// [1 2 0 3 4 0 0 0 0]
	xh, xw := im.Dimension()
	outh, outw := outhw(xh, xw, fh, fw, pad, stride)
	out := matrix.New()
	for y := 0; y < fh; y++ {
		ymax := y + stride*outh
		for x := 0; x < fw; x++ {
			xmax := x + stride*outw
			out = append(out, pick(img, y, x, ymax, xmax, stride))
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

func col2im(col matrix.Matrix, xh, xw, fh, fw, pad, stride int) matrix.Matrix {
	var ycount int
	add := func(img, colT matrix.Matrix, y, x, ymax, xmax, stride int) {
		var xcount int
		for i := y; i < ymax; i = i + stride {
			for j := x; j < xmax; j = j + stride {
				img[i][j] = img[i][j] + colT[ycount][xcount]
				xcount++
			}
		}
		ycount++
	}

	// FC, FH, FW, stride := 1, 2, 2, 1
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
	// Transpose
	// [0 0 0 0 1 2 0 3 4]
	// [0 0 0 1 2 0 3 4 0]
	// [0 1 2 0 3 4 0 0 0]
	// [1 2 0 3 4 0 0 0 0]
	colT := col.T()

	// [0  0  0 0]
	// [0  4  8 0]
	// [0 12 16 0]
	// [0  0  0 0]
	outh, outw := outhw(xh, xw, fh, fw, pad, stride)
	img := matrix.Zero(xh+2*pad+stride-1, xw+2*pad+stride-1)
	for y := 0; y < fh; y++ {
		ymax := y + stride*outh
		for x := 0; x < fw; x++ {
			xmax := x + stride*outw
			// [0 0 0]
			// [0 1 2]
			// [0 3 4]
			//
			// [0 0 0]
			// [1 2 0]
			// [3 4 0]
			//
			// [0 1 2]
			// [0 3 4]
			// [0 0 0]
			//
			// [1 2 0]
			// [3 4 0]
			// [0 0 0]
			add(img, colT, y, x, ymax, xmax, stride)
		}
	}

	// N, C, H, W := 1, 1, 2, 2
	// pad := 1
	// [ 4  8]
	// [12 16]
	return matrix.Unpadding(img, pad)
}
