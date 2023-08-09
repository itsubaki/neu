package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type Attention struct {
}

func (l *Attention) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *Attention) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *Attention) SetParams(p ...matrix.Matrix) {}
func (l *Attention) String() string               { return fmt.Sprintf("%T", l) }
