package probability

import (
	"math"
	"testing"
)

func TestGaussian(t *testing.T) {
	if Gaussian(-2, 0, 1) != Gaussian(2, 0, 1) {
		t.Errorf("%f: %f", Gaussian(-2, 0, 1), Gaussian(2, 0, 1))
	}

	if Gaussian(-1, 0, 1) != Gaussian(1, 0, 1) {
		t.Errorf("%f: %f", Gaussian(-2, 0, 1), Gaussian(2, 0, 1))
	}
}

func TestLaplace(t *testing.T) {
	if Laplace(-2, 0, 1) != Laplace(2, 0, 1) {
		t.Errorf("%f: %f", Laplace(-2, 0, 1), Laplace(2, 0, 1))
	}

	if Laplace(-1, 0, 1) != Laplace(1, 0, 1) {
		t.Errorf("%f: %f", Laplace(-2, 0, 1), Laplace(2, 0, 1))
	}
}

func TestLogisticSigmoid(t *testing.T) {
	if math.Abs(LogisticSigmoid(0)-0.5) > 1e-13 {
		t.Error(LogisticSigmoid(0))
	}

	if math.Abs(LogisticSigmoid(-10)) > 1e-4 {
		t.Error(LogisticSigmoid(-10))
	}

	if math.Abs(LogisticSigmoid(10)-1.0) > 1e-4 {
		t.Error(LogisticSigmoid(10))
	}
}

func TestSoftplus(t *testing.T) {
	if math.Abs(Softplus(-10)) > 1e-4 {
		t.Error(Softplus(-10))
	}

	if math.Abs(Softplus(10)-10.0) > 1e-4 {
		t.Error(Softplus(10))
	}
}
