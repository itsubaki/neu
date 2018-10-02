package matrix

import "math"

type Matrix [][]float64

func (m0 Matrix) Dimension() (int, int) {
	return len(m0), len(m0[0])
}

// TODO
func (m0 Matrix) Inverse() Matrix {
	return m0
}

// TODO
func (m0 Matrix) Determinant() float64 {
	return 0
}

func (m0 Matrix) Transpose() Matrix {
	p, q := m0.Dimension()

	m2 := Matrix{}
	for i := 0; i < p; i++ {
		v := []float64{}
		for j := 0; j < q; j++ {
			v = append(v, m0[j][i])
		}
		m2 = append(m2, v)
	}

	return m2
}

func (m0 Matrix) Apply(m1 Matrix) Matrix {
	m, n := m1.Dimension()
	p, _ := m0.Dimension()

	m2 := Matrix{}
	for i := 0; i < m; i++ {
		v := []float64{}
		for j := 0; j < n; j++ {
			c := 0.0
			for k := 0; k < p; k++ {
				c = c + m1[i][k]*m0[k][j]
			}
			v = append(v, c)
		}
		m2 = append(m2, v)
	}

	return m2
}

func (m0 Matrix) IsOrthogonal(eps ...float64) bool {
	p, q := m0.Dimension()
	m := m0.Apply(m0.Transpose())

	e := Eps(eps...)
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			if i == j {
				if math.Abs(m[i][j]-1.0) > e {
					return false
				}
				continue
			}

			if math.Abs(m[i][j]) > e {
				return false
			}
		}
	}

	return true
}

func (m0 Matrix) Mul(z float64) Matrix {
	p, q := m0.Dimension()

	m := Matrix{}
	for i := 0; i < p; i++ {
		v := []float64{}
		for j := 0; j < q; j++ {
			v = append(v, z*m0[i][j])
		}
		m = append(m, v)
	}

	return m
}

func (m0 Matrix) Add(m1 Matrix) Matrix {
	p, q := m0.Dimension()

	m := Matrix{}
	for i := 0; i < p; i++ {
		v := []float64{}
		for j := 0; j < q; j++ {
			v = append(v, m0[i][j]+m1[i][j])
		}
		m = append(m, v)
	}

	return m
}

func (m0 Matrix) Sub(m1 Matrix) Matrix {
	p, q := m0.Dimension()

	m := Matrix{}
	for i := 0; i < p; i++ {
		v := []float64{}
		for j := 0; j < q; j++ {
			v = append(v, m0[i][j]-m1[i][j])
		}
		m = append(m, v)
	}

	return m
}

func (m0 Matrix) Trace() float64 {
	p, _ := m0.Dimension()

	var sum float64
	for i := 0; i < p; i++ {
		sum = sum + m0[i][i]
	}

	return sum
}

func (m0 Matrix) Equals(m1 Matrix, eps ...float64) bool {
	m, n := m0.Dimension()
	p, q := m1.Dimension()

	if m != p {
		return false
	}

	if n != q {
		return false
	}

	e := Eps(eps...)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			if math.Abs(m0[i][j]-m1[i][j]) > e {
				return false
			}
		}
	}

	return true
}

func (m0 Matrix) TensorProduct(m1 Matrix) Matrix {
	m, n := m0.Dimension()
	p, q := m1.Dimension()

	tmp := []Matrix{}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			tmp = append(tmp, m1.Mul(m0[i][j]))
		}
	}

	m2 := Matrix{}
	for l := 0; l < len(tmp); l = l + m {
		for j := 0; j < p; j++ {
			v := []float64{}
			for i := l; i < l+m; i++ {
				for k := 0; k < q; k++ {
					v = append(v, tmp[i][j][k])
				}
			}
			m2 = append(m2, v)
		}
	}

	return m2
}

func TensorProductN(m Matrix, bit ...int) Matrix {
	if len(bit) < 1 {
		return m
	}

	m0 := m
	for i := 1; i < bit[0]; i++ {
		m0 = m0.TensorProduct(m)
	}

	return m0
}

func TensorProduct(m ...Matrix) Matrix {
	m0 := m[0]
	for i := 1; i < len(m); i++ {
		m0 = m0.TensorProduct(m[i])
	}

	return m0
}

func Eps(eps ...float64) float64 {
	if len(eps) > 0 {
		return eps[0]
	}

	return 0.0
}
