package weight

// Std returns a function that returns the standard deviation.
func Std(std float64) func(_ int) float64 { return func(_ int) float64 { return std } }
