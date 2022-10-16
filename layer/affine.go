package layer

import "github.com/itsubaki/neu/math/matrix"

type Affine struct {
	W  matrix.Matrix
	B  matrix.Matrix
	x  matrix.Matrix
	DW matrix.Matrix
	DB matrix.Matrix
}

func (l *Affine) Forward(x, _ matrix.Matrix) matrix.Matrix {
	l.x = x
	bB := Broadcast(l.B, len(l.x))
	return matrix.Dot(l.x, l.W).Add(bB) // x.W + b
}

func (l *Affine) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := matrix.Dot(dout, l.W.T())
	l.DW = matrix.Dot(l.x.T(), dout)
	l.DB = SumAxis0(dout)
	return dx, matrix.New()
}

func SumAxis0(m matrix.Matrix) matrix.Matrix {
	p, q := m.Dimension()

	v := make([]float64, q)
	for i := 0; i < q; i++ {
		for j := 0; j < p; j++ {
			v[i] = v[i] + m[j][i]
		}
	}

	return matrix.New(v)
}

func Broadcast(m matrix.Matrix, size int) matrix.Matrix {
	out := make(matrix.Matrix, 0)
	for {
		var fill bool
		for j := range m {
			out = append(out, m[j])
			if len(out) == size {
				fill = true
				break
			}
		}

		if fill {
			break
		}
	}

	return out
}
