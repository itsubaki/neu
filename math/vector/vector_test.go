package vector

import "testing"

func TestVector(t *testing.T) {
	v := New(1, 0, 0, 0)

	if v.Norm() != 1 {
		t.Error(v.Norm())
	}

	if !v.IsUnit() {
		t.Error(v.Norm())
	}
}
