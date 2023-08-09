package layer

import (
	"fmt"

	"github.com/itsubaki/neu/math/matrix"
)

type TimeAttention struct {
}

func (l *TimeAttention) Params() []matrix.Matrix      { return make([]matrix.Matrix, 0) }
func (l *TimeAttention) Grads() []matrix.Matrix       { return make([]matrix.Matrix, 0) }
func (l *TimeAttention) SetParams(p ...matrix.Matrix) {}
func (l *TimeAttention) SetState(_ ...matrix.Matrix)  {}
func (l *TimeAttention) ResetState()                  {}
func (l *TimeAttention) String() string               { return fmt.Sprintf("%T", l) }
