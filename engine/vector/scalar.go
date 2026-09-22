package vector

import "math"

// This file contains the portable scalar implementations of every kernel. It is
// compiled into every build and serves both as the fallback when AVX-512 is not
// available and as the behavioural reference for the SIMD kernels.

func scalarAdd(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

func scalarSub(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
}

func scalarMul(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
}

func scalarScale(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] * s
	}
}

func scalarAddScalar(dst, a []float32, s float32) {
	for i := range dst {
		dst[i] = a[i] + s
	}
}

func scalarAXPY(dst []float32, alpha float32, x []float32) {
	for i := range dst {
		dst[i] += alpha * x[i]
	}
}

func scalarDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func scalarSum(a []float32) float32 {
	var sum float32
	for _, v := range a {
		sum += v
	}
	return sum
}

func scalarMax(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func scalarMin(a []float32) float32 {
	m := a[0]
	for _, v := range a[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func scalarFill(dst []float32, v float32) {
	for i := range dst {
		dst[i] = v
	}
}

func scalarReLU(dst, a []float32) {
	for i, v := range a {
		if v > 0 {
			dst[i] = v
		} else {
			dst[i] = 0
		}
	}
}

func scalarLeakyReLU(dst, a []float32, slope float32) {
	for i, v := range a {
		if v > 0 {
			dst[i] = v
		} else {
			dst[i] = slope * v
		}
	}
}

// scalarGemm implements c = alpha*a*b + beta*c for row-major matrices with
// explicit leading dimensions. The k loop is ordered last so that the inner
// loop streams contiguously through both a and c.
func scalarGemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		crow := c[i*ldc : i*ldc+n]
		if beta == 0 {
			for j := range crow {
				crow[j] = 0
			}
		} else if beta != 1 {
			for j := range crow {
				crow[j] *= beta
			}
		}
		arow := a[i*lda : i*lda+k]
		for p := 0; p < k; p++ {
			aip := alpha * arow[p]
			if aip == 0 {
				continue
			}
			brow := b[p*ldb : p*ldb+n]
			for j, bv := range brow {
				crow[j] += aip * bv
			}
		}
	}
}

func scalarAdam(param, grad, m, v []float32, lr, beta1, beta2, eps float32, t int) {
	bc1 := 1 - float32(math.Pow(float64(beta1), float64(t)))
	bc2 := 1 - float32(math.Pow(float64(beta2), float64(t)))
	for i := range param {
		g := grad[i]
		m[i] = beta1*m[i] + (1-beta1)*g
		v[i] = beta2*v[i] + (1-beta2)*g*g
		mh := m[i] / bc1
		vh := v[i] / bc2
		param[i] -= lr * mh / (float32(math.Sqrt(float64(vh))) + eps)
	}
}
