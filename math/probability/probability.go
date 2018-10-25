package probability

import (
	"fmt"
	"math"

	"github.com/itsubaki/arts/math/matrix"
	"github.com/itsubaki/arts/math/vector"
)

// k must be 0 or 1
func Bernoulli(p float64, k int) float64 {
	if k != 0 && k != 1 {
		panic(fmt.Sprintf("out of range. k=%v", k))
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

func Softend(x float64) float64 {
	return math.Max(0, x)
}

// when sigma is precision matrix Beta, this is isotropic Gaussian.
func MultiVariateNormal(v, mu vector.Vector, sigma matrix.Matrix) float64 {
	n := float64(v.Dimension())
	z := math.Sqrt(1 / (math.Pow(2*math.Pi, n) * sigma.Determinant()))
	s := v.Sub(mu)
	i := -1 * (s.InnerProduct(s.Apply(sigma.Inverse()))) / 2
	return z * math.Exp(i)
}
