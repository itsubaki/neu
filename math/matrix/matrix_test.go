package matrix

import (
	"math"
	"testing"
)

func TestInverse(t *testing.T) {
	m := New(
		[]float64{1, 2, 0, -1},
		[]float64{-1, 1, 2, 0},
		[]float64{2, 0, 1, 1},
		[]float64{1, -2, -1, 1},
	)

	inv := m.Inverse()
	im := m.Apply(inv)

	mm, nn := im.Dimension()
	for i := 0; i < mm; i++ {
		for j := 0; j < nn; j++ {
			if i == j {
				if math.Abs(im[i][j]-1.0) > 1e-13 {
					t.Errorf("[%v][%v] %v\n", i, j, im[i][j])
				}
				continue
			}
			if math.Abs(im[i][j]-0.0) > 1e-13 {
				t.Errorf("[%v][%v] %v\n", i, j, im[i][j])
			}
		}
	}
}

func TestDeterminant2x2(t *testing.T) {
	m := New(
		[]float64{3, 2},
		[]float64{5, 4},
	)

	det := m.Determinant()
	if math.Abs(det-2.0) > 1e-13 {
		t.Error(det)
	}
}

func TestDeterminant4x4(t *testing.T) {
	m := New(
		[]float64{2, -2, 4, 2},
		[]float64{2, -1, 6, 3},
		[]float64{3, -2, 12, 12},
		[]float64{-1, 3, -4, 4},
	)

	det := m.Determinant()
	if math.Abs(det-120.0) > 1e-13 {
		t.Error(det)
	}
}
