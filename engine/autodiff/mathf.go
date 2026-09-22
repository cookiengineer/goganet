package autodiff

import "math"

func sqrtf(x float32) float32 {
	return float32(math.Sqrt(float64(x)))
}

func expf(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func logf(x float32) float32 {
	return float32(math.Log(float64(x)))
}
