package vector

import (
	"math"
	randv2 "math/rand/v2"
	"slices"

	"github.com/itsubaki/neu/math/rand"
)

func Zero(n int) []float64 {
	return make([]float64, n)
}

// rng returns a pseudo-random number generator.
func rng(s ...randv2.Source) *randv2.Rand {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	return randv2.New(s[0])
}

func Rand(n int, s ...randv2.Source) []float64 {
	g := rng(s...)

	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = g.Float64()
	}

	return out
}

func Randn(n int, s ...randv2.Source) []float64 {
	g := rng(s...)

	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = g.NormFloat64()
	}

	return out
}

func Int(v []float64) []int {
	out := make([]int, len(v))
	for i, e := range v {
		out[i] = int(e)
	}

	return out
}

func Max[T int | float64](v []T) T {
	max := v[0]
	for _, e := range v {
		if e > max {
			max = e
		}
	}

	return max
}

func Argmax(v []float64) int {
	max := v[0]

	var arg int
	for i, e := range v {
		if e > max {
			max = e
			arg = i
		}
	}

	return arg
}

func Add(v, w []float64) []float64 {
	out := make([]float64, len(v))
	for i := range v {
		out[i] = v[i] + w[i]
	}

	return out
}

func Mul(v []float64, a float64) []float64 {
	out := make([]float64, len(v))
	for i := range v {
		out[i] = v[i] * a
	}

	return out
}

func Div(v []float64, a float64) []float64 {
	out := make([]float64, len(v))
	for i := range v {
		out[i] = v[i] / a
	}

	return out
}

func Abs(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, e := range v {
		out[i] = math.Abs(e)
	}

	return out
}

func Sum(v []float64) float64 {
	var sum float64
	for _, e := range v {
		sum += e
	}

	return sum
}

func Mean(v []float64) float64 {
	return Sum(v) / float64(len(v))
}

func Pow2(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, e := range v {
		out[i] = e * e
	}

	return out
}

func Cos(x, y []float64) float64 {
	xps := math.Sqrt(Sum(Pow2(x)) + 1e-8)
	yps := math.Sqrt(Sum(Pow2(y)) + 1e-8)

	var sum float64
	for i := range x {
		sum += x[i] * y[i]
	}

	return sum / (xps * yps)
}

func Choice(p []float64, s ...randv2.Source) int {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}

	cumsum := make([]float64, len(p))
	var sum float64
	for i, prob := range p {
		sum += prob
		cumsum[i] = sum
	}

	r := randv2.New(s[0]).Float64()
	var ret int
	for i, prop := range cumsum {
		if r <= prop {
			ret = i
			break
		}
	}

	return ret
}

func Contains[T comparable](v T, s []T) bool {
	return slices.Contains(s, v)
}

func T[T any](v []T) [][]T {
	out := make([][]T, len(v))
	for i := range v {
		out[i] = []T{v[i]}
	}

	return out
}

// Shuffle shuffles the dataset.
func Shuffle[T any](x, t []T, s ...randv2.Source) ([]T, []T) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(rand.MustRead()))
	}
	g := randv2.New(s[0])

	xs, ts := make([]T, len(x)), make([]T, len(t))
	for i := range len(x) {
		xs[i], ts[i] = x[i], t[i]
	}

	for i := range len(x) {
		j := g.IntN(i + 1)

		// swap
		xs[i], xs[j] = xs[j], xs[i]
		ts[i], ts[j] = ts[j], ts[i]
	}

	return xs, ts
}

// Reverse reverses the slice.
func Reverse[T any](xs []T) []T {
	for i := range len(xs) / 2 {
		xs[i], xs[len(xs)-1-i] = xs[len(xs)-1-i], xs[i]
	}

	return xs
}

// MatchCount returns the number of matches.
func MatchCount[T comparable](x, y []T) int {
	var c int
	for i := range x {
		if x[i] == y[i] {
			c++
		}
	}

	return c
}

// Equals returns true if x and y are the same.
func Equals(x, y []int) bool {
	if len(x) != len(y) {
		return false
	}

	if MatchCount(x, y) == len(x) {
		return true
	}

	return false
}
