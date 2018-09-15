package probability

import (
	"math"
	"testing"
)

func TestGauss(t *testing.T) {
	if Gauss(-2, 0, 1) != Gauss(2, 0, 1) {
		t.Errorf("%f: %f", Gauss(-2, 0, 1), Gauss(2, 0, 1))
	}

	if Gauss(-1, 0, 1) != Gauss(1, 0, 1) {
		t.Errorf("%f: %f", Gauss(-2, 0, 1), Gauss(2, 0, 1))
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

func TestSigmoid(t *testing.T) {
	if math.Abs(Sigmoid(0)-0.5) > 1e-13 {
		t.Error(Sigmoid(0))
	}

	if math.Abs(Sigmoid(-10)) > 1e-4 {
		t.Error(Sigmoid(-10))
	}

	if math.Abs(Sigmoid(10)-1.0) > 1e-4 {
		t.Error(Sigmoid(10))
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
