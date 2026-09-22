//go:build amd64 && goexperiment.simd

package vector

import "simd/archsimd"

// init installs the AVX-512 kernels when the CPU supports them. The AVX-512
// instructions are never executed on a CPU without support, so the package
// remains safe on AVX2-only or older hardware.
func init() {
	if !archsimd.X86.AVX512() {
		return
	}
	hasAVX512 = true
	addImpl = avx512Add
	subImpl = avx512Sub
	mulImpl = avx512Mul
	scaleImpl = avx512Scale
	addScalarImpl = avx512AddScalar
	axpyImpl = avx512AXPY
	dotImpl = avx512Dot
	sumImpl = avx512Sum
	maxImpl = avx512Max
	minImpl = avx512Min
	fillImpl = avx512Fill
	reluImpl = avx512ReLU
	leakyImpl = avx512LeakyReLU
	gemmImpl = avx512Gemm
}

const avx512Lanes = 16

func hsum16(v archsimd.Float32x16) float32 {
	var buf [16]float32
	v.StoreArray(&buf)
	var s float32
	for _, x := range buf {
		s += x
	}
	return s
}

func avx512Add(dst, a, b []float32) {
	n := len(dst)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		va := archsimd.LoadFloat32x16(a[i:])
		vb := archsimd.LoadFloat32x16(b[i:])
		va.Add(vb).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + b[i]
	}
}

func avx512Sub(dst, a, b []float32) {
	n := len(dst)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		va := archsimd.LoadFloat32x16(a[i:])
		vb := archsimd.LoadFloat32x16(b[i:])
		va.Sub(vb).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] - b[i]
	}
}

func avx512Mul(dst, a, b []float32) {
	n := len(dst)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		va := archsimd.LoadFloat32x16(a[i:])
		vb := archsimd.LoadFloat32x16(b[i:])
		va.Mul(vb).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * b[i]
	}
}

func avx512Scale(dst, a []float32, s float32) {
	n := len(dst)
	vs := archsimd.BroadcastFloat32x16(s)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		archsimd.LoadFloat32x16(a[i:]).Mul(vs).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] * s
	}
}

func avx512AddScalar(dst, a []float32, s float32) {
	n := len(dst)
	vs := archsimd.BroadcastFloat32x16(s)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		archsimd.LoadFloat32x16(a[i:]).Add(vs).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = a[i] + s
	}
}

func avx512AXPY(dst []float32, alpha float32, x []float32) {
	n := len(dst)
	va := archsimd.BroadcastFloat32x16(alpha)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		vx := archsimd.LoadFloat32x16(x[i:])
		vd := archsimd.LoadFloat32x16(dst[i:])
		va.MulAdd(vx, vd).Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += alpha * x[i]
	}
}

func avx512Dot(a, b []float32) float32 {
	n := len(a)
	acc := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		va := archsimd.LoadFloat32x16(a[i:])
		vb := archsimd.LoadFloat32x16(b[i:])
		acc = va.MulAdd(vb, acc)
	}
	s := hsum16(acc)
	for ; i < n; i++ {
		s += a[i] * b[i]
	}
	return s
}

func avx512Sum(a []float32) float32 {
	n := len(a)
	acc := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		acc = acc.Add(archsimd.LoadFloat32x16(a[i:]))
	}
	s := hsum16(acc)
	for ; i < n; i++ {
		s += a[i]
	}
	return s
}

func avx512Max(a []float32) float32 {
	n := len(a)
	acc := archsimd.BroadcastFloat32x16(a[0])
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		acc = acc.Max(archsimd.LoadFloat32x16(a[i:]))
	}
	var buf [16]float32
	acc.StoreArray(&buf)
	m := buf[0]
	for _, v := range buf[1:] {
		if v > m {
			m = v
		}
	}
	for ; i < n; i++ {
		if a[i] > m {
			m = a[i]
		}
	}
	return m
}

func avx512Min(a []float32) float32 {
	n := len(a)
	acc := archsimd.BroadcastFloat32x16(a[0])
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		acc = acc.Min(archsimd.LoadFloat32x16(a[i:]))
	}
	var buf [16]float32
	acc.StoreArray(&buf)
	m := buf[0]
	for _, v := range buf[1:] {
		if v < m {
			m = v
		}
	}
	for ; i < n; i++ {
		if a[i] < m {
			m = a[i]
		}
	}
	return m
}

func avx512Fill(dst []float32, v float32) {
	n := len(dst)
	vf := archsimd.BroadcastFloat32x16(v)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		vf.Store(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = v
	}
}

func avx512ReLU(dst, a []float32) {
	n := len(dst)
	zero := archsimd.BroadcastFloat32x16(0)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		archsimd.LoadFloat32x16(a[i:]).Max(zero).Store(dst[i:])
	}
	for ; i < n; i++ {
		if a[i] > 0 {
			dst[i] = a[i]
		} else {
			dst[i] = 0
		}
	}
}

func avx512LeakyReLU(dst, a []float32, slope float32) {
	n := len(dst)
	zero := archsimd.BroadcastFloat32x16(0)
	slopeV := archsimd.BroadcastFloat32x16(slope)
	i := 0
	for ; i+avx512Lanes <= n; i += avx512Lanes {
		v := archsimd.LoadFloat32x16(a[i:])
		neg := v.Mul(slopeV)
		v.IfElse(v.Greater(zero), neg).Store(dst[i:])
	}
	for ; i < n; i++ {
		if a[i] > 0 {
			dst[i] = a[i]
		} else {
			dst[i] = slope * a[i]
		}
	}
}

// avx512Gemm vectorises the classic ijk loop over the n dimension: for every
// row of a it broadcasts one scalar and accumulates a full row of b into a full
// row of c with fused multiply-adds.
func avx512Gemm(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	for i := 0; i < m; i++ {
		crow := c[i*ldc : i*ldc+n]
		switch {
		case beta == 0:
			avx512Fill(crow, 0)
		case beta != 1:
			avx512Scale(crow, crow, beta)
		}
		arow := a[i*lda : i*lda+k]
		for p := 0; p < k; p++ {
			aip := alpha * arow[p]
			if aip == 0 {
				continue
			}
			vai := archsimd.BroadcastFloat32x16(aip)
			brow := b[p*ldb : p*ldb+n]
			j := 0
			for ; j+avx512Lanes <= n; j += avx512Lanes {
				vb := archsimd.LoadFloat32x16(brow[j:])
				vc := archsimd.LoadFloat32x16(crow[j:])
				vai.MulAdd(vb, vc).Store(crow[j:])
			}
			for ; j < n; j++ {
				crow[j] += aip * brow[j]
			}
		}
	}
}
