package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

// Add is a layer that performs an element-wise addition.
type Add struct{}

func (l *Add) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *Add) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *Add) SetParams(p ...matrix.Matrix) {}
func (l *Add) String() string               { return fmt.Sprintf("%T", l) }

func (l *Add) Forward(x, y matrix.Matrix, _ ...Opts) matrix.Matrix {
	return x.Add(y)
}

func (l *Add) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.F(x1)
	dy := dout.F(x1)
	return dx, dy
}

func x1(v float64) float64 { return 1.0 * v }
