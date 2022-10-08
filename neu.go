package neu

import "github.com/itsubaki/neu/layer"

type Layer interface {
	Forward(x, y []float64) []float64
	Backward(dout []float64) ([]float64, []float64)
}

var (
	_ Layer = (*layer.Add)(nil)
	_ Layer = (*layer.Mul)(nil)
)
