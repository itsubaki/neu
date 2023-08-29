package plot

import (
	"fmt"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// Range returns a slice of float64 values from begin to end with a specified delta.
func Range(begin, end, delta float64) []float64 {
	out := []float64{begin}
	for {
		begin = begin + delta
		if begin > end {
			break
		}

		out = append(out, begin)
	}

	return out
}

// Save saves a plot to a file.
func Save(x, y []float64, filename string) error {
	xys := make(plotter.XYs, 0)
	for i := range x {
		xys = append(xys, plotter.XY{
			X: x[i],
			Y: y[i],
		})
	}

	line, err := plotter.NewLine(xys)
	if err != nil {
		return fmt.Errorf("plotter newline: %v", err)
	}

	p := plot.New()
	p.Add(line)

	if err := p.Save(4*vg.Inch, 4*vg.Inch, filename); err != nil {
		return fmt.Errorf("save: %v", err)
	}

	return nil
}
