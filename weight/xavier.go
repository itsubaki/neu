package weight

import "math"

// Xavier returns a function that returns the Xavier initialization value.
func Xavier(prevNodeNum int) float64 { return math.Sqrt(1.0 / float64(prevNodeNum)) }

// Glorot returns a function that returns the Glorot(Xavier) initialization value.
func Glorot(prevNodeNum int) float64 { return Xavier(prevNodeNum) }
