package vector

import (
	"math"

	"github.com/itsubaki/arts/math/matrix"
)

type Vector []float64

func New(z ...float64) Vector {
	v := Vector{}
	for _, zi := range z {
		v = append(v, zi)
	}
	return v
}

func (v0 Vector) Dimension() int {
	return len(v0)
}

func (v0 Vector) Clone() Vector {
	clone := Vector{}
	for i := 0; i < len(v0); i++ {
		clone = append(clone, v0[i])
	}
	return clone
}

func (v0 Vector) Add(v1 Vector) Vector {
	v2 := Vector{}
	for i := 0; i < len(v0); i++ {
		v2 = append(v2, v0[i]+v1[i])
	}
	return v2
}

func (v0 Vector) Mul(z float64) Vector {
	v2 := Vector{}
	for i, _ := range v0 {
		v2 = append(v2, z*v0[i])
	}
	return v2
}

func (v0 Vector) PNorm(p int) float64 {
	sum := 0.0
	for i := range v0 {
		sum = sum + math.Pow(math.Abs(v0[i]), float64(p))
	}

	return math.Pow(sum, float64(1/p))
}

func (v0 Vector) Norm() float64 {
	return v0.PNorm(2)
}

func (v0 Vector) IsUnit(eps ...float64) bool {
	e := matrix.Eps(eps...)
	if math.Abs(v0.Norm()-1.0) > e {
		return false
	}
	return true
}

func (v0 Vector) Apply(mat matrix.Matrix) Vector {
	v := Vector{}

	m, _ := mat.Dimension()
	for i := 0; i < m; i++ {
		tmp := 0.0
		for j := 0; j < len(v0); j++ {
			tmp = tmp + mat[i][j]*v0[j]
		}
		v = append(v, tmp)
	}

	return v
}

func (v0 Vector) Equals(v1 Vector, eps ...float64) bool {
	if len(v0) != len(v1) {
		return false
	}

	e := matrix.Eps(eps...)
	for i := 0; i < len(v0); i++ {
		if math.Abs(v0[i]-v1[i]) > e {
			return false
		}

	}
	return true
}

func (v0 Vector) TensorProduct(v1 Vector) Vector {
	v2 := Vector{}
	for i := 0; i < len(v0); i++ {
		for j := 0; j < len(v1); j++ {
			v2 = append(v2, v0[i]*v1[j])
		}
	}
	return v2
}

func TensorProductN(v0 Vector, bit ...int) Vector {
	if len(bit) < 1 {
		return v0
	}

	v1 := v0
	for i := 1; i < bit[0]; i++ {
		v1 = v1.TensorProduct(v0)
	}
	return v1
}

func TensorProduct(v0 ...Vector) Vector {
	v1 := v0[0]
	for i := 1; i < len(v0); i++ {
		v1 = v1.TensorProduct(v0[i])
	}
	return v1
}
