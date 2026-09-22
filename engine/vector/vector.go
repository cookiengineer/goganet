// Package vector provides float32 array kernels used by the GoGANet engine.
//
// The package has two implementations of every kernel: a portable scalar
// implementation that is always compiled, and an AVX-512 implementation that is
// compiled only when building for amd64 with GOEXPERIMENT=simd. The exported
// functions dispatch to the fastest implementation available at run time,
// guarded by a CPU feature check so that the AVX-512 code is never executed on
// a CPU that does not support it.
//
// Build the AVX-512 path with:
//
//	GOEXPERIMENT=simd go build ./...
//
// and query it with [HasAVX512].
package vector

// The kernel implementations are stored in function variables so that both
// build configurations can share the same dispatch layer without build tags
// leaking into the public API. The scalar implementations are compiled in
// every build; the AVX-512 implementations replace them during init when the
// CPU supports AVX-512.
var (
	hasAVX512 bool

	addImpl       = scalarAdd
	subImpl       = scalarSub
	mulImpl       = scalarMul
	scaleImpl     = scalarScale
	addScalarImpl = scalarAddScalar
	axpyImpl      = scalarAXPY
	dotImpl       = scalarDot
	sumImpl       = scalarSum
	maxImpl       = scalarMax
	minImpl       = scalarMin
	fillImpl      = scalarFill
	reluImpl      = scalarReLU
	leakyImpl     = scalarLeakyReLU
	gemmImpl      = scalarGemm
	adamImpl      = scalarAdam
)

// HasAVX512 reports whether the AVX-512 float32 kernels are active.
func HasAVX512() bool { return hasAVX512 }

// Add computes dst[i] = a[i] + b[i].
func Add(dst, a, b []float32) {
	checkSame3("vector.Add", dst, a, b)
	addImpl(dst, a, b)
}

// Sub computes dst[i] = a[i] - b[i].
func Sub(dst, a, b []float32) {
	checkSame3("vector.Sub", dst, a, b)
	subImpl(dst, a, b)
}

// Mul computes dst[i] = a[i] * b[i].
func Mul(dst, a, b []float32) {
	checkSame3("vector.Mul", dst, a, b)
	mulImpl(dst, a, b)
}

// Scale computes dst[i] = a[i] * s.
func Scale(dst, a []float32, s float32) {
	checkSame("vector.Scale", dst, a)
	scaleImpl(dst, a, s)
}

// AddScalar computes dst[i] = a[i] + s.
func AddScalar(dst, a []float32, s float32) {
	checkSame("vector.AddScalar", dst, a)
	addScalarImpl(dst, a, s)
}

// AXPY computes dst[i] += alpha * x[i].
func AXPY(dst []float32, alpha float32, x []float32) {
	checkSame("vector.AXPY", dst, x)
	axpyImpl(dst, alpha, x)
}

// Dot returns the inner product of a and b.
func Dot(a, b []float32) float32 {
	checkSame("vector.Dot", a, b)
	return dotImpl(a, b)
}

// Sum returns the sum of all elements of a.
func Sum(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return sumImpl(a)
}

// Max returns the largest element of a, or 0 for an empty slice.
func Max(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return maxImpl(a)
}

// Min returns the smallest element of a, or 0 for an empty slice.
func Min(a []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return minImpl(a)
}

// Fill sets every element of dst to v.
func Fill(dst []float32, v float32) { fillImpl(dst, v) }

// Zero clears dst.
func Zero(dst []float32) { fillImpl(dst, 0) }

// ReLU computes dst[i] = max(0, a[i]).
func ReLU(dst, a []float32) {
	checkSame("vector.ReLU", dst, a)
	reluImpl(dst, a)
}

// LeakyReLU computes dst[i] = a[i] if a[i] > 0, else slope*a[i].
func LeakyReLU(dst, a []float32, slope float32) {
	checkSame("vector.LeakyReLU", dst, a)
	leakyImpl(dst, a, slope)
}

// Sigmoid computes the logistic function element-wise.
func Sigmoid(dst, a []float32) {
	checkSame("vector.Sigmoid", dst, a)
	for i, v := range a {
		dst[i] = sigmoidf(v)
	}
}

// Tanh computes the hyperbolic tangent element-wise.
func Tanh(dst, a []float32) {
	checkSame("vector.Tanh", dst, a)
	for i, v := range a {
		dst[i] = tanhf(v)
	}
}

// Softmax computes the numerically stable softmax of a in place into dst.
func Softmax(dst, a []float32) {
	checkSame("vector.Softmax", dst, a)
	if len(a) == 0 {
		return
	}
	m := maxImpl(a)
	var sum float32
	for i, v := range a {
		e := expf(v - m)
		dst[i] = e
		sum += e
	}
	if sum == 0 {
		return
	}
	inv := 1 / sum
	scaleImpl(dst, dst, inv)
}

// GEMV computes y = A*x + y where A is an m x n row-major matrix, x has length
// n and y has length m. It accumulates into y, so callers must zero y first for
// a plain matrix-vector product.
func GEMV(y, a, x []float32, m, n int) {
	if len(a) < m*n {
		panic("vector.GEMV: matrix too small")
	}
	if len(x) < n {
		panic("vector.GEMV: vector x too small")
	}
	if len(y) < m {
		panic("vector.GEMV: vector y too small")
	}
	for i := 0; i < m; i++ {
		y[i] += dotImpl(a[i*n:i*n+n], x[:n])
	}
}

// Gemm computes c = a*b for row-major matrices where a is m x k, b is k x n and
// c is m x n. All strides are the natural row lengths.
func Gemm(c, a, b []float32, m, n, k int) {
	if len(a) < m*k {
		panic("vector.Gemm: matrix a too small")
	}
	if len(b) < k*n {
		panic("vector.Gemm: matrix b too small")
	}
	if len(c) < m*n {
		panic("vector.Gemm: matrix c too small")
	}
	gemmImpl(m, n, k, 1, a, k, b, n, 0, c, n)
}

// GemmEx is the general matrix multiply
//
//	c = alpha*a*b + beta*c
//
// for row-major matrices with explicit leading dimensions. It is the primitive
// used by the fully connected and convolutional layers.
func GemmEx(m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	gemmImpl(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Adam performs one Adam update step in place:
//
//	m = beta1*m + (1-beta1)*g
//	v = beta2*v + (1-beta2)*g*g
//	param -= lr * (m/(1-beta1^t)) / (sqrt(v/(1-beta2^t)) + eps)
//
// param, grad, m and v must all have the same length.
func Adam(param, grad, m, v []float32, lr, beta1, beta2, eps float32, t int) {
	checkSame3("vector.Adam", param, grad, m)
	if len(v) != len(param) {
		panic("vector.Adam: length mismatch")
	}
	adamImpl(param, grad, m, v, lr, beta1, beta2, eps, t)
}

func checkSame(name string, a, b []float32) {
	if len(a) != len(b) {
		panic(name + ": length mismatch")
	}
}

func checkSame3(name string, a, b, c []float32) {
	if len(a) != len(b) || len(a) != len(c) {
		panic(name + ": length mismatch")
	}
}
