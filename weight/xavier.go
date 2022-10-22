package weight

import "math"

func Xavier(prevNodeNum int) float64 { return math.Sqrt(1.0 / float64(prevNodeNum)) }
