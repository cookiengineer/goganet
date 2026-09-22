package vector

import (
	"math"
	"math/rand"
	"testing"
)

func randSlice(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(rand.NormFloat64())
	}
	return s
}

func approx(t *testing.T, name string, got, want float32) {
	t.Helper()
	diff := float32(math.Abs(float64(got - want)))
	scale := float32(math.Max(1, math.Abs(float64(want))))
	if diff > 1e-3*scale {
		t.Fatalf("%s: got %v, want %v (diff %v)", name, got, want, diff)
	}
}

func TestElementwiseMatchesScalar(t *testing.T) {
	const n = 1003
	a := randSlice(n)
	b := randSlice(n)

	dst := make([]float32, n)
	Add(dst, a, b)
	for i := range dst {
		approx(t, "Add", dst[i], a[i]+b[i])
	}

	Sub(dst, a, b)
	for i := range dst {
		approx(t, "Sub", dst[i], a[i]-b[i])
	}

	Mul(dst, a, b)
	for i := range dst {
		approx(t, "Mul", dst[i], a[i]*b[i])
	}

	Scale(dst, a, 2.5)
	for i := range dst {
		approx(t, "Scale", dst[i], a[i]*2.5)
	}

	AddScalar(dst, a, 0.25)
	for i := range dst {
		approx(t, "AddScalar", dst[i], a[i]+0.25)
	}

	AXPY(dst, 1.5, b)
	// dst currently holds a+0.25; add 1.5*b on top.
	for i := range dst {
		approx(t, "AXPY", dst[i], a[i]+0.25+1.5*b[i])
	}

	approx(t, "Dot", Dot(a, b), scalarDot(a, b))
	approx(t, "Sum", Sum(a), scalarSum(a))
	approx(t, "Max", Max(a), scalarMax(a))
	approx(t, "Min", Min(a), scalarMin(a))
}

func TestActivations(t *testing.T) {
	const n = 777
	a := randSlice(n)
	dst := make([]float32, n)

	ReLU(dst, a)
	for i := range dst {
		want := float32(0)
		if a[i] > 0 {
			want = a[i]
		}
		approx(t, "ReLU", dst[i], want)
	}

	LeakyReLU(dst, a, 0.2)
	for i := range dst {
		want := a[i]
		if a[i] <= 0 {
			want = 0.2 * a[i]
		}
		approx(t, "LeakyReLU", dst[i], want)
	}

	Sigmoid(dst, a)
	for i := range dst {
		approx(t, "Sigmoid", dst[i], sigmoidf(a[i]))
	}

	Tanh(dst, a)
	for i := range dst {
		approx(t, "Tanh", dst[i], tanhf(a[i]))
	}

	Softmax(dst, a)
	var sum float32
	for i := range dst {
		if dst[i] < 0 {
			t.Fatalf("Softmax produced negative value %v", dst[i])
		}
		sum += dst[i]
	}
	approx(t, "Softmax sum", sum, 1)
}

func TestGemmMatchesScalar(t *testing.T) {
	for _, dims := range [][3]int{{3, 5, 7}, {16, 16, 16}, {17, 19, 23}, {64, 33, 48}} {
		m, n, k := dims[0], dims[1], dims[2]
		a := randSlice(m * k)
		b := randSlice(k * n)
		got := make([]float32, m*n)
		want := make([]float32, m*n)

		Gemm(got, a, b, m, n, k)
		scalarGemm(m, n, k, 1, a, k, b, n, 0, want, n)

		for i := range got {
			approx(t, "Gemm", got[i], want[i])
		}
	}
}

func TestGemmExBeta(t *testing.T) {
	m, n, k := 5, 6, 4
	a := randSlice(m * k)
	b := randSlice(k * n)
	c0 := randSlice(m * n)

	got := append([]float32(nil), c0...)
	want := append([]float32(nil), c0...)
	GemmEx(m, n, k, 0.75, a, k, b, n, 0.5, got, n)
	scalarGemm(m, n, k, 0.75, a, k, b, n, 0.5, want, n)

	for i := range got {
		approx(t, "GemmEx", got[i], want[i])
	}
}

func TestGEMV(t *testing.T) {
	m, n := 11, 13
	a := randSlice(m * n)
	x := randSlice(n)
	y := make([]float32, m)
	want := make([]float32, m)
	GEMV(y, a, x, m, n)
	scalarGemm(m, 1, n, 1, a, n, x, 1, 0, want, 1)
	for i := range y {
		approx(t, "GEMV", y[i], want[i])
	}
}

func BenchmarkDot(b *testing.B) {
	a := randSlice(4096)
	c := randSlice(4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Dot(a, c)
	}
}

func BenchmarkAXPY(b *testing.B) {
	a := randSlice(4096)
	dst := make([]float32, 4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		AXPY(dst, 1.5, a)
	}
}

func BenchmarkGemm(b *testing.B) {
	m, n, k := 128, 128, 128
	a := randSlice(m * k)
	c := randSlice(k * n)
	out := make([]float32, m*n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Gemm(out, a, c, m, n, k)
	}
}
