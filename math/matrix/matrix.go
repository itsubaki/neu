package matrix

import (
	"math"
	"math/rand"
	"time"
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

func ZeroLike(m Matrix) Matrix {
	return Zero(m.Dim())
}

// One returns a matrix with all elements 1.
func One(m, n int) Matrix {
	out := make(Matrix, m)
	for i := 0; i < m; i++ {
		out[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			out[i][j] = 1
		}
	}

	return out
}

// Rand returns a matrix with elements that pseudo-random number in the half-open interval [0.0,1.0).
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Rand(m, n int, s ...rand.Source) Matrix {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	out := Zero(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[i][j] = rng.Float64()
		}
	}

	return out
}

// Randn returns a matrix with elements that normally distributed float64 in the range [-math.MaxFloat64, +math.MaxFloat64] with standard normal distribution.
// m, n is the dimension of the matrix.
// s is the source of the pseudo-random number.
func Randn(m, n int, s ...rand.Source) Matrix {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	out := Zero(m, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out[i][j] = rng.NormFloat64()
		}
	}

	return out
}

// Mask returns a matrix with elements that 1 if f() is true and 0 otherwise.
func Mask(m Matrix, f func(x float64) bool) Matrix {
	mask := ZeroLike(m)
	for i := range m {
		for j := range m[i] {
			if f(m[i][j]) {
				mask[i][j] = 1
			}
		}
	}

	return mask
}

// Batch returns a matrix with rows of the specified index.
func Batch(m Matrix, index []int) Matrix {
	out := make(Matrix, len(index))
	for i, idx := range index {
		out[i] = m[idx]
	}

	return out
}

// Column returns a matrix with the specified column.
func Column(m Matrix, j int) Matrix {
	out := make(Matrix, len(m))
	for i, r := range m {
		out[i] = []float64{r[j]}
	}

	return out
}

// From returns a matrix from a slice of slice of T.
func From[T int](x [][]T) Matrix {
	out := Zero(len(x), len(x[0]))
	for i, r := range x {
		for j, v := range r {
			out[i][j] = float64(v)
		}
	}

	return out
}

func Int(m Matrix) [][]int {
	out := make([][]int, len(m))
	for i, r := range m {
		out[i] = make([]int, len(r))
		for j, v := range r {
			out[i][j] = int(v)
		}
	}

	return out
}

func Identity(size int) Matrix {
	out := Zero(size, size)
	for i := 0; i < size; i++ {
		out[i][i] = 1
	}

	return out
}

func OneHot(x []int, size int) Matrix {
	out := Zero(len(x), size)
	for i, v := range x {
		out[i][v] = 1
	}

	return out
}

func (m Matrix) Dim() (int, int) {
	if len(m) == 0 {
		return 0, 0
	}

	return len(m), len(m[0])
}

func (m Matrix) T() Matrix {
	p, q := m.Dim()

	out := Zero(q, p)
	for i := 0; i < q; i++ {
		for j := 0; j < p; j++ {
			out[i][j] = m[j][i]
		}
	}

	return out
}

func (m Matrix) Add(n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a + b })
}

func (m Matrix) Sub(n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a - b })
}

func (m Matrix) Mul(n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a * b })
}

func (m Matrix) Div(n Matrix) Matrix {
	return F2(m, n, func(a, b float64) float64 { return a / b })
}

func (m Matrix) AddC(c float64) Matrix {
	return F(m, func(v float64) float64 { return c + v })
}

func (m Matrix) MulC(c float64) Matrix {
	return F(m, func(v float64) float64 { return c * v })
}

func (m Matrix) Sqrt(eps float64) Matrix {
	return F(m, func(v float64) float64 { return math.Sqrt(v + eps) })
}

func (m Matrix) Pow2() Matrix {
	return F(m, func(v float64) float64 { return v * v })
}

func (m Matrix) Abs() Matrix {
	return F(m, func(v float64) float64 { return math.Abs(v) })
}

// Sum returns the sum of all elements.
func (m Matrix) Sum() float64 {
	var sum float64
	for i := range m {
		for j := range m[i] {
			sum = sum + m[i][j]
		}
	}

	return sum
}

// Mean returns the average of all elements.
func (m Matrix) Mean() float64 {
	a, b := m.Dim()
	return m.Sum() / float64(a*b)
}

// Argmax returns the index of the maximum value of each row.
func (m Matrix) Argmax() []int {
	out := make([]int, len(m))
	for i := range m {
		max := m[i][0]
		for j := range m[i] {
			if m[i][j] > max {
				max = m[i][j]
				out[i] = j
			}
		}
	}

	return out
}

// SumAxis0 returns the sum of each column.
func (m Matrix) SumAxis0() []float64 {
	p, q := m.Dim()

	v := make([]float64, 0, q)
	for j := 0; j < q; j++ {
		var sum float64
		for i := 0; i < p; i++ {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return v
}

// SumAxis1 returns the sum of each row.
func (m Matrix) SumAxis1() []float64 {
	p, q := m.Dim()

	v := make([]float64, 0, p)
	for i := 0; i < p; i++ {
		var sum float64
		for j := 0; j < q; j++ {
			sum = sum + m[i][j]
		}

		v = append(v, sum)
	}

	return v
}

// MeanAxis0 returns the mean of each column.
func (m Matrix) MeanAxis0() []float64 {
	out := make([]float64, 0)
	for _, v := range m.SumAxis0() {
		out = append(out, v/float64(len(m)))
	}

	return out
}

// MaxAxis1 returns the maximum value of each row.
func (m Matrix) MaxAxis1() []float64 {
	out := make([]float64, len(m))
	for i := range m {
		out[i] = m[i][0]
		for j := range m[i] {
			if m[i][j] > out[i] {
				out[i] = m[i][j]
			}
		}
	}

	return out
}

// Broadcast returns the broadcasted matrix.
func (m Matrix) Broadcast(a, b int) Matrix {
	if len(m) == 1 && len(m[0]) == 1 {
		out := Zero(a, b)
		for i := 0; i < a; i++ {
			for j := 0; j < b; j++ {
				out[i][j] = m[0][0]
			}
		}

		return out
	}

	if len(m) == 1 {
		// b is ignored
		out := make(Matrix, a)
		for i := 0; i < a; i++ {
			out[i] = m[0]
		}

		return out
	}

	if len(m[0]) == 1 {
		// a is ignored
		out := Zero(len(m), b)
		for i := 0; i < len(m); i++ {
			for j := 0; j < b; j++ {
				out[i][j] = m[i][0]
			}
		}

		return out
	}

	return m
}

// Dot returns the dot product of m and n.
func Dot(m, n Matrix) Matrix {
	a, b := m.Dim()
	_, p := n.Dim()

	out := Zero(a, p)
	for i := 0; i < a; i++ {
		for j := 0; j < p; j++ {
			for k := 0; k < b; k++ {
				out[i][j] = out[i][j] + m[i][k]*n[k][j]
			}
		}
	}

	return out
}

// F applies a function to each element of the matrix.
func F(m Matrix, f func(a float64) float64) Matrix {
	p, q := m.Dim()

	out := Zero(p, q)
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			out[i][j] = f(m[i][j])
		}
	}

	return out
}

// F2 applies a function to each element of the matrix.
func F2(m, n Matrix, f func(a, b float64) float64) Matrix {
	p, q := m.Dim()
	n = n.Broadcast(p, q)

	out := Zero(p, q)
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			out[i][j] = f(m[i][j], n[i][j])
		}
	}

	return out
}

// F3 applies a function to each element of the matrix.
func F3(m, n, o Matrix, f func(a, b, c float64) float64) Matrix {
	p, q := m.Dim()
	n = n.Broadcast(p, q)
	o = o.Broadcast(p, q)

	out := Zero(p, q)
	for i := 0; i < p; i++ {
		for j := 0; j < q; j++ {
			out[i][j] = f(m[i][j], n[i][j], o[i][j])
		}
	}

	return out
}

// Padding returns the padded matrix.
func Padding(x Matrix, pad int) Matrix {
	_, q := x.Dim()
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

// Unpadding returns the unpadded matrix.
func Unpadding(x Matrix, pad int) Matrix {
	m, n := x.Dim()

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

// Reshape returns the matrix with the given shape.
func Reshape(x Matrix, m, n int) Matrix {
	v := Flatten(x)

	if m < 1 {
		p, q := x.Dim()
		m = p * q / n
	}

	if n < 1 {
		p, q := x.Dim()
		n = p * q / m
	}

	out := New()
	for i := 0; i < m; i++ {
		begin, end := i*n, (i+1)*n
		out = append(out, v[begin:end])
	}

	return out
}

// Split returns the matrix split into H parts.
func Split(x Matrix, H int) []Matrix {
	out := make([]Matrix, len(x[0])/H)
	for i := 0; i < len(out); i++ {
		out[i] = New()
		for _, r := range x {
			out[i] = append(out[i], r[i*H:(i+1)*H])
		}
	}

	return out
}

// HStack returns the matrix horizontally stacked.
func HStack(x ...Matrix) Matrix {
	out := Zero(len(x[0]), 0)
	for _, m := range x {
		for i, r := range m {
			out[i] = append(out[i], r...)
		}
	}

	return out
}
