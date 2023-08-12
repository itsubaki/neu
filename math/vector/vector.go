package vector

import (
	"math"
	"math/rand"
	"time"
)

func Add(v, w []float64) []float64 {
	out := make([]float64, len(v))
	for i := range v {
		out[i] = v[i] + w[i]
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

func Max(v []int) int {
	max := math.MinInt
	for _, e := range v {
		if e > max {
			max = e
		}
	}

	return max
}

func Abs(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, e := range v {
		out[i] = math.Abs(e)
	}

	return out
}

func Choice(p []float64, s ...rand.Source) int {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}

	cumsum := make([]float64, len(p))
	var sum float64
	for i, prob := range p {
		sum += prob
		cumsum[i] = sum
	}

	var ret int
	r := rand.New(s[0]).Float64()
	for i, prop := range cumsum {
		if r <= prop {
			ret = i
			break
		}
	}

	return ret
}

func Contains[T comparable](v T, s []T) bool {
	for _, ss := range s {
		if v == ss {
			return true
		}
	}

	return false
}

func Transpose[T any](v []T) [][]T {
	out := make([][]T, len(v))
	for i := range v {
		out[i] = []T{v[i]}
	}

	return out
}

func T[T any](v []T) [][]T {
	return Transpose(v)
}

// Shuffle shuffles the dataset.
func Shuffle[T any](x, t []T, s ...rand.Source) ([]T, []T) {
	if len(s) == 0 {
		s = append(s, rand.NewSource(time.Now().UnixNano()))
	}
	rng := rand.New(s[0])

	xs, ts := make([]T, len(x)), make([]T, len(t))
	for i := 0; i < len(x); i++ {
		xs[i], ts[i] = x[i], t[i]
	}

	for i := 0; i < len(x); i++ {
		j := rng.Intn(i + 1)

		// swap
		xs[i], xs[j] = xs[j], xs[i]
		ts[i], ts[j] = ts[j], ts[i]
	}

	return xs, ts
}

// Reverse reverses the slice.
func Reverse[T any](xs []T) []T {
	for i := 0; i < len(xs)/2; i++ {
		xs[i], xs[len(xs)-1-i] = xs[len(xs)-1-i], xs[i]
	}

	return xs
}

// MatchCount returns the number of matches.
func MatchCount(x, y []int) int {
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
