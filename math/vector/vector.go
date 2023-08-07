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
