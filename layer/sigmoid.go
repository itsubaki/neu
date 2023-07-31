package layer

import (
	"fmt"

	"github.com/itsubaki/neu/activation"
	"github.com/itsubaki/neu/math/matrix"
)

// Sigmoid is a layer that performs an element-wise sigmoid.
type Sigmoid struct {
	out matrix.Matrix
}

func (l *Sigmoid) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *Sigmoid) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *Sigmoid) SetParams(p ...matrix.Matrix) {}

func (l *Sigmoid) Forward(x, _ matrix.Matrix, _ ...Opts) matrix.Matrix {
	l.out = x.Func(activation.Sigmoid)
	return l.out
}

func (l *Sigmoid) Backward(dout matrix.Matrix) (matrix.Matrix, matrix.Matrix) {
	dx := dout.Mul(matrix.Func(l.out, dsigmoid))
	return dx, matrix.New()
}

func (l *Sigmoid) String() string {
	return fmt.Sprintf("%T", l)
}

func dsigmoid(a float64) float64 { return a * (1.0 - a) } // out * (1.0 - out)
