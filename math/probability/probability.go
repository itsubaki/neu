package probability

import (
	"fmt"
	"math"
)

// phi must be 0 or 1
func Bernoulli(x float64, phi int) float64 {
	if phi != 0 && phi != 1 {
		panic(fmt.Sprintf("out of range. phi: %v", phi))
	}

	return math.Pow(float64(phi), x) * math.Pow(float64((1-phi)), 1-x)
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
