package neu

import "github.com/itsubaki/neu/layer"

type Layer interface {
	Forward(x, y []float64) []float64
	Backwward(dout []float64) ([]float64, []float64)
}

var (
	_ Layer = (*layer.Add)(nil)
	_ Layer = (*layer.Mul)(nil)
)
