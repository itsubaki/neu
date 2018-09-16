package probability

import (
	"fmt"
	"math"
)

// k must be 0 or 1
func Bernoulli(p float64, k int) float64 {
	if k != 0 && k != 1 {
		panic(fmt.Sprintf("out of range. k: %v", k))
	}

	return math.Pow(p, float64(k)) * math.Pow((1.0-p), float64(1-k))
}

func Gaussian(x, mu, sigma float64) float64 {
	c := math.Sqrt(1 / (2 * math.Pi * sigma * sigma))
	e := math.Exp(-1 * (x - mu) * (x - mu) / (2 * sigma * sigma))
	return c * e
}

func Laplace(x, mu, gamma float64) float64 {
	c := 1 / (2 * gamma)
	e := math.Exp(-1 * math.Abs(x-mu) / gamma)
	return c * e
}

func LogisticSigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func Softplus(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}
