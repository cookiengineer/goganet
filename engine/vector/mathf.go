package vector

import "math"

// The scalar math helpers are shared by both build configurations so that the
// transcendental functions produce identical results regardless of SIMD
// availability.

func expf(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func tanhf(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func sigmoidf(x float32) float32 {
	if x >= 0 {
		return 1 / (1 + expf(-x))
	}
	e := expf(x)
	return e / (1 + e)
}
