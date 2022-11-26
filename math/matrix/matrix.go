package matrix

import (
	"math"
	"math/rand"
)

type Matrix [][]float64

func New(v ...[]float64) Matrix {
	out := make(Matrix, len(v))
	copy(out, v)
	return out
}

// Zero returns a matrix with all elements 0.
func Zero(m, n int) Matrix {
	out := make(Matrix, m)
	for i := 0; i < m; i++ {
		out[i] = make([]float64, n)
	}

	return out
}

// One returns a matrix with all elements 1.
func One(m, n int) Matrix {
	out := make(Matrix, m)
	for i := 0; i < m; i++ {
		v := make([]float64, 0)
		for j := 0; j < n; j++ {
			v = append(v, 1)
		}

		out[i] = v
	}

	return out
}

// Rand returns a matrix with elements that pseudo-random number in the half-open interval [0.0,1.0).
func Rand(m, n int) Matrix {
	out := make(Matrix, 0)
	for i := 0; i < m; i++ {
		v := make([]float64, 0)
		for j := 0; j < n; j++ {
			v = append(v, rand.Float64())
		}

		out = append(out, v)
	}

	return out
}

// Randn returns a matrix with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
func Randn(m, n int) Matrix {
	out := make(Matrix, 0)
	for i := 0; i < m; i++ {
		v := make([]float64, 0)
		for j := 0; j < n; j++ {
			v = append(v, rand.NormFloat64())
		}

		out = append(out, v)
	}

	return out
}

// Mask returns a matrix with elements that 1 if f() is true and 0 otherwise.
func Mask(m Matrix, f func(x float64) bool) Matrix {
	out := make(Matrix, 0)
	for i := range m {
		v := make([]float64, 0)
		for j := range m[i] {
			if f(m[i][j]) {
				v = append(v, 1)
				continue
			}

			v = append(v, 0)
		}

		out = append(out, v)
	}

	return out
}

// Batch returns a matrix with rows of the specified index.
func Batch(m Matrix, index []int) Matrix {
	out := make(Matrix, 0)
	for _, i := range index {
		out = append(out, m[i])
	}

	return out
}

func (m Matrix) Dimension() (int, int) {
	if len(m) == 0 {
		return 0, 0
	}

	return len(m), len(m[0])
}

func (m Matrix) Apply(n Matrix) Matrix {
	a, b := n.Dimension()
	_, p := m.Dimension()

	out := Matrix{}
	for i := 0; i < a; i++ {
		v := make([]float64, 0)

		for j := 0; j < p; j++ {
			var c float64
			for k := 0; k < b; k++ {
				c = c + n[i][k]*m[k][j]
			}

			v = append(v, c)
		}

		out = append(out, v)
	}

	return out
}

func (m Matrix) Dot(n Matrix) Matrix {
	return n.Apply(m)
}

func (m Matrix) Add(n Matrix) Matrix {
	return m.FuncWith(Broadcast(n, len(m)), func(a, b float64) float64 { return a + b })
}

func (m Matrix) Sub(n Matrix) Matrix {
	return m.FuncWith(Broadcast(n, len(m)), func(a, b float64) float64 { return a - b })
}

func (m Matrix) Mul(n Matrix) Matrix {
	return m.FuncWith(Broadcast(n, len(m)), func(a, b float64) float64 { return a * b })
}

func (m Matrix) Div(n Matrix) Matrix {
	return m.FuncWith(Broadcast(n, len(m)), func(a, b float64) float64 { return a / b })
}

func (m Matrix) AddC(c float64) Matrix {
	return m.Func(func(v float64) float64 { return c + v })
}

func (m Matrix) MulC(c float64) Matrix {
	return m.Func(func(v float64) float64 { return c * v })
}

func (m Matrix) Pow2() Matrix {
	return m.Func(func(v float64) float64 { return v * v })
}

func (m Matrix) Sqrt(eps float64) Matrix {
	return m.Func(func(v float64) float64 { return math.Sqrt(v + eps) })
}

func (m Matrix) Abs() Matrix {
	return m.Func(func(v float64) float64 { return math.Abs(v) })
}

func (m Matrix) Transpose() Matrix {
	p, q := m.Dimension()

	out := make(Matrix, q)
	for i := range out {
		out[i] = make([]float64, p)
	}

	for i := 0; i < q; i++ {
		for j := 0; j < p; j++ {
			out[i][j] = m[j][i]
		}
	}

	return out
}

func (m Matrix) T() Matrix {
	return m.Transpose()
}

func (m Matrix) Sum() float64 {
	var sum float64
	for i := range m {
		for j := range m[i] {
			sum = sum + m[i][j]
		}
	}

	return sum
}

func (m Matrix) Avg() float64 {
	a, b := m.Dimension()
	return m.Sum() / float64(a*b)
}

func (m Matrix) Argmax() []int {
	out := make([]int, 0)
	for i := range m {
		max := -math.MaxFloat64
		var index int
		for j := range m[i] {
			if m[i][j] > max {
				max = m[i][j]
				index = j
			}
		}

		out = append(out, index)
	}

	return out
}

func (m Matrix) SumAxis0() Matrix {
	p, q := m.Dimension()

	v := make([]float64, q)
	for i := 0; i < q; i++ {
		for j := 0; j < p; j++ {
			v[i] = v[i] + m[j][i]
		}
	}

	return New(v)
}

func (m Matrix) MeanAxis0() Matrix {
	return m.SumAxis0().MulC(1.0 / float64(len(m)))
}

func (m Matrix) MaxAxis1() []float64 {
	out := make([]float64, 0)
	for i := range m {
		max := -math.MaxFloat64
		for j := range m[i] {
			if m[i][j] > max {
				max = m[i][j]
			}
		}

		out = append(out, max)
	}

	return out
}

func (m Matrix) Func(f func(v float64) float64) Matrix {
	p, q := m.Dimension()

	out := make(Matrix, 0, p)
	for i := 0; i < p; i++ {
		v := make([]float64, 0, q)

		for j := 0; j < q; j++ {
			v = append(v, f(m[i][j]))
		}

		out = append(out, v)
	}

	return out
}

func (m Matrix) FuncWith(n Matrix, f func(a, b float64) float64) Matrix {
	p, q := m.Dimension()

	out := make(Matrix, 0, p)
	for i := 0; i < p; i++ {
		v := make([]float64, 0, q)

		for j := 0; j < q; j++ {
			v = append(v, f(m[i][j], n[i][j]))
		}

		out = append(out, v)
	}

	return out
}

func Dot(m, n Matrix) Matrix {
	return m.Dot(n)
}

func Func(m Matrix, f func(a float64) float64) Matrix {
	return m.Func(f)
}

func FuncWith(m, n Matrix, f func(a, b float64) float64) Matrix {
	return m.FuncWith(n, f)
}

func Broadcast(m Matrix, size int) Matrix {
	out := make(Matrix, 0)
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

func Padding(x Matrix, pad int) Matrix {
	_, q := x.Dimension()
	pw := pad + q + pad // right + row + left

	// top
	out := New()
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

func Unpadding(x Matrix, pad int) Matrix {
	m, n := x.Dimension()

	out := New()
	for _, r := range x[pad : m-pad] {
		out = append(out, r[pad:n-pad])
	}

	return out
}

func Flatten(x Matrix) []float64 {
	out := make([]float64, 0)
	for _, r := range x {
		out = append(out, r...)
	}

	return out
}

func Reshape(x Matrix, m, n int) Matrix {
	v := Flatten(x)

	if m < 1 {
		p, q := x.Dimension()
		m = p * q / n
	}

	if n < 1 {
		p, q := x.Dimension()
		n = p * q / m
	}

	out := New()
	for i := 0; i < m; i++ {
		begin, end := i*n, (i+1)*n
		out = append(out, v[begin:end])
	}

	return out
}
