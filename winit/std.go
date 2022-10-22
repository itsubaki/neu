package winit

func Std(std float64) func(_ int) float64 { return func(_ int) float64 { return std } }
