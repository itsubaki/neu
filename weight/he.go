package weight

import "math"

// He returns a function that returns the He initialization value.
func He(prevNodeNum int) float64 { return math.Sqrt(2.0 / float64(prevNodeNum)) }
