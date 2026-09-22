// Package autodiff implements a small reverse-mode automatic differentiation
// tape for float32 vectors and matrices.
//
// Its distinguishing feature is that the reverse pass is expressed in terms of
// the same differentiable primitives as the forward pass. As a result the tape
// is second-order differentiable: differentiating a node that was itself
// produced by [Node.Grad] yields a gradient of a gradient. That property is
// required to train a Wasserstein GAN with a gradient penalty, because the
// penalty is a function of the critic's gradient with respect to its input, and
// training the critic requires the derivative of that penalty with respect to
// the critic parameters.
//
// All heavy linear algebra dispatches to the engine/vector kernels, so on
// AVX-512 hardware the tape inherits the SIMD speedups.
package autodiff

import (
	"github.com/cookiengineer/goganet/engine/vector"
)

// Shape is a two-dimensional tensor shape. A column vector has C == 1 and a row
// vector has R == 1.
type Shape struct{ R, C int }

// S is a convenience constructor for a shape.
func S(r, c int) Shape { return Shape{R: r, C: c} }

// Len returns the number of elements in the shape.
func (s Shape) Len() int { return s.R * s.C }

// Op identifies the operation that produced a node.
type Op uint8

const (
	opConst Op = iota
	opVar
	opMatMul
	opMatMulNT
	opMatMulTN
	opAdd
	opSub
	opMul
	opAddBias
	opColSum
	opBroadcastRow
	opMulRow
	opColSumSquares
	opScale
	opAddConst
	opSqrt
	opReciprocal
	opLeakyReLU
	opTanh
	opSumAll
	opMeanAll
	opExp
	opLog
	opColMax
	opBroadcastCol
	opRowSum
	opTranspose
	opReshape
	opFlatten
	opUnflatten
	opGlobalMean
	opBroadcastSpatial
	opIm2Col
	opCol2Im
)

// Node is a value in the computation graph. Nodes are immutable once
// constructed; [Node.Grad] never mutates them, it only builds new nodes that
// describe the gradient.
type Node struct {
	op     Op
	in     []*Node
	val    []float32
	s      Shape
	scalar float32
	mask   []float32
	cs     *ConvShape
	dims   [4]int // B, H, W, C for spatial ops
}

// Shape returns the node shape.
func (n *Node) Shape() Shape { return n.s }

// Value returns the forward value. The slice must not be modified.
func (n *Node) Value() []float32 { return n.val }

// Scalar returns the single element of a 1x1 node.
func (n *Node) Scalar() float32 { return n.val[0] }

func clone(s []float32) []float32 {
	out := make([]float32, len(s))
	copy(out, s)
	return out
}

// Const creates a constant node.
func Const(s Shape, data []float32) *Node {
	if len(data) != s.Len() {
		panic("autodiff.Const: shape mismatch")
	}
	return &Node{op: opConst, s: s, val: data}
}

// Var creates a leaf parameter node.
func Var(s Shape, data []float32) *Node {
	if len(data) != s.Len() {
		panic("autodiff.Var: shape mismatch")
	}
	return &Node{op: opVar, s: s, val: data}
}

// SetVar replaces the value of a variable node. It is the mechanism used by the
// optimizer to update parameters in place.
func SetVar(n *Node, data []float32) {
	if n.op != opVar {
		panic("autodiff.SetVar: not a variable")
	}
	if len(data) != n.s.Len() {
		panic("autodiff.SetVar: shape mismatch")
	}
	n.val = data
}

// Zeros creates a constant zero tensor.
func Zeros(s Shape) *Node {
	return &Node{op: opConst, s: s, val: make([]float32, s.Len())}
}

// Ones creates a constant tensor of ones.
func Ones(s Shape) *Node {
	v := make([]float32, s.Len())
	vector.Fill(v, 1)
	return &Node{op: opConst, s: s, val: v}
}

// Fill creates a constant tensor filled with v.
func Fill(s Shape, v float32) *Node {
	data := make([]float32, s.Len())
	vector.Fill(data, v)
	return &Node{op: opConst, s: s, val: data}
}

func unary(op Op, a *Node, val []float32, mask []float32) *Node {
	return &Node{op: op, in: []*Node{a}, s: a.s, val: val, mask: mask}
}

// MatMul returns a*b for a (m x k) and b (k x n).
func MatMul(a, b *Node) *Node {
	if a.s.C != b.s.R {
		panic("autodiff.MatMul: inner dimensions differ")
	}
	out := &Node{op: opMatMul, in: []*Node{a, b}, s: Shape{a.s.R, b.s.C}, val: make([]float32, a.s.R*b.s.C)}
	vector.Gemm(out.val, a.val, b.val, a.s.R, b.s.C, a.s.C)
	return out
}

// MatMulNT returns a*b^T for a (m x k) and b (n x k), giving (m x n).
func MatMulNT(a, b *Node) *Node {
	if a.s.C != b.s.C {
		panic("autodiff.MatMulNT: inner dimensions differ")
	}
	m, n, k := a.s.R, b.s.R, a.s.C
	out := &Node{op: opMatMulNT, in: []*Node{a, b}, s: Shape{m, n}, val: make([]float32, m*n)}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.val[i*n+j] = vector.Dot(a.val[i*k:i*k+k], b.val[j*k:j*k+k])
		}
	}
	return out
}

// MatMulTN returns a^T*b for a (k x m) and b (k x n), giving (m x n).
func MatMulTN(a, b *Node) *Node {
	if a.s.R != b.s.R {
		panic("autodiff.MatMulTN: inner dimensions differ")
	}
	k, m, n := a.s.R, a.s.C, b.s.C
	out := &Node{op: opMatMulTN, in: []*Node{a, b}, s: Shape{m, n}, val: make([]float32, m*n)}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum += a.val[p*a.s.C+i] * b.val[p*b.s.C+j]
			}
			out.val[i*n+j] = sum
		}
	}
	return out
}

func sameShape(a, b *Node, name string) {
	if a.s != b.s {
		panic("autodiff." + name + ": shape mismatch")
	}
}

// Add returns the element-wise sum.
func Add(a, b *Node) *Node {
	sameShape(a, b, "Add")
	out := unary(opAdd, a, make([]float32, a.s.Len()), nil)
	out.in = []*Node{a, b}
	vector.Add(out.val, a.val, b.val)
	return out
}

// Sub returns the element-wise difference.
func Sub(a, b *Node) *Node {
	sameShape(a, b, "Sub")
	out := unary(opSub, a, make([]float32, a.s.Len()), nil)
	out.in = []*Node{a, b}
	vector.Sub(out.val, a.val, b.val)
	return out
}

// Mul returns the element-wise product.
func Mul(a, b *Node) *Node {
	sameShape(a, b, "Mul")
	out := unary(opMul, a, make([]float32, a.s.Len()), nil)
	out.in = []*Node{a, b}
	vector.Mul(out.val, a.val, b.val)
	return out
}

// AddBias adds a column bias b (m x 1) to every column of a (m x n).
func AddBias(a, b *Node) *Node {
	if a.s.R != b.s.R || b.s.C != 1 {
		panic("autodiff.AddBias: shape mismatch")
	}
	m, n := a.s.R, a.s.C
	out := &Node{op: opAddBias, in: []*Node{a, b}, s: a.s, val: make([]float32, m*n)}
	for i := 0; i < m; i++ {
		bv := b.val[i]
		for j := 0; j < n; j++ {
			out.val[i*n+j] = a.val[i*n+j] + bv
		}
	}
	return out
}

// ColSum sums a (m x n) along rows, returning (1 x n).
func ColSum(a *Node) *Node {
	m, n := a.s.R, a.s.C
	out := &Node{op: opColSum, in: []*Node{a}, s: Shape{1, n}, val: make([]float32, n)}
	for i := 0; i < m; i++ {
		vector.AXPY(out.val, 1, a.val[i*n:i*n+n])
	}
	return out
}

// BroadcastRow expands a row vector (1 x n) to m copies (m x n).
func BroadcastRow(a *Node, m int) *Node {
	if a.s.R != 1 {
		panic("autodiff.BroadcastRow: expected row vector")
	}
	n := a.s.C
	out := &Node{op: opBroadcastRow, in: []*Node{a}, s: Shape{m, n}, val: make([]float32, m*n)}
	for i := 0; i < m; i++ {
		copy(out.val[i*n:i*n+n], a.val)
	}
	return out
}

// MulRow multiplies each row of a (m x n) by the row vector r (1 x n).
func MulRow(a, r *Node) *Node {
	if a.s.C != r.s.C || r.s.R != 1 {
		panic("autodiff.MulRow: shape mismatch")
	}
	m, n := a.s.R, a.s.C
	out := &Node{op: opMulRow, in: []*Node{a, r}, s: a.s, val: make([]float32, m*n)}
	for i := 0; i < m; i++ {
		vector.Mul(out.val[i*n:i*n+n], a.val[i*n:i*n+n], r.val)
	}
	return out
}

// ColSumSquares returns the per-column sum of squares of a (m x n) as (1 x n).
func ColSumSquares(a *Node) *Node {
	m, n := a.s.R, a.s.C
	out := &Node{op: opColSumSquares, in: []*Node{a}, s: Shape{1, n}, val: make([]float32, n)}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			v := a.val[i*n+j]
			out.val[j] += v * v
		}
	}
	return out
}

// Scale multiplies every element by the scalar s.
func Scale(a *Node, s float32) *Node {
	out := unary(opScale, a, make([]float32, a.s.Len()), nil)
	out.scalar = s
	vector.Scale(out.val, a.val, s)
	return out
}

// AddConst adds the scalar s to every element.
func AddConst(a *Node, s float32) *Node {
	out := unary(opAddConst, a, make([]float32, a.s.Len()), nil)
	out.scalar = s
	vector.AddScalar(out.val, a.val, s)
	return out
}

// Sqrt takes the element-wise square root.
func Sqrt(a *Node) *Node {
	out := unary(opSqrt, a, make([]float32, a.s.Len()), nil)
	for i, v := range a.val {
		out.val[i] = sqrtf(v)
	}
	return out
}

// Reciprocal computes 1/a element-wise.
func Reciprocal(a *Node) *Node {
	out := unary(opReciprocal, a, make([]float32, a.s.Len()), nil)
	for i, v := range a.val {
		out.val[i] = 1 / v
	}
	return out
}

// LeakyReLU applies the leaky rectifier with the given negative slope.
func LeakyReLU(a *Node, slope float32) *Node {
	mask := make([]float32, len(a.val))
	out := unary(opLeakyReLU, a, make([]float32, a.s.Len()), mask)
	out.scalar = slope
	vector.LeakyReLU(out.val, a.val, slope)
	for i, v := range a.val {
		if v > 0 {
			mask[i] = 1
		} else {
			mask[i] = slope
		}
	}
	return out
}

// Tanh applies the hyperbolic tangent element-wise.
func Tanh(a *Node) *Node {
	out := unary(opTanh, a, make([]float32, a.s.Len()), nil)
	vector.Tanh(out.val, a.val)
	return out
}

// Exp applies the exponential element-wise.
func Exp(a *Node) *Node {
	out := unary(opExp, a, make([]float32, a.s.Len()), nil)
	for i, v := range a.val {
		out.val[i] = expf(v)
	}
	return out
}

// Log applies the natural logarithm element-wise.
func Log(a *Node) *Node {
	out := unary(opLog, a, make([]float32, a.s.Len()), nil)
	for i, v := range a.val {
		out.val[i] = logf(v)
	}
	return out
}

// ColMax returns the per-column maximum of a (m x n) as (1 x n).
func ColMax(a *Node) *Node {
	m, n := a.s.R, a.s.C
	out := &Node{op: opColMax, in: []*Node{a}, s: Shape{1, n}, val: make([]float32, n), mask: make([]float32, m*n)}
	for j := 0; j < n; j++ {
		best := 0
		bv := a.val[j]
		for i := 1; i < m; i++ {
			if v := a.val[i*n+j]; v > bv {
				bv = v
				best = i
			}
		}
		out.val[j] = bv
		out.mask[best*n+j] = 1
	}
	return out
}

// BroadcastCol expands a row vector (1 x n) to m copies (m x n).
func BroadcastCol(a *Node, m int) *Node {
	if a.s.R != 1 {
		panic("autodiff.BroadcastCol: expected row vector")
	}
	n := a.s.C
	out := &Node{op: opBroadcastCol, in: []*Node{a}, s: Shape{m, n}, val: make([]float32, m*n)}
	for i := 0; i < m; i++ {
		copy(out.val[i*n:i*n+n], a.val)
	}
	return out
}

// RowSum sums a (m x n) along columns, returning (m x 1).
func RowSum(a *Node) *Node {
	m, n := a.s.R, a.s.C
	out := &Node{op: opRowSum, in: []*Node{a}, s: Shape{m, 1}, val: make([]float32, m)}
	for i := 0; i < m; i++ {
		out.val[i] = vector.Sum(a.val[i*n : i*n+n])
	}
	return out
}

// SumAll reduces every element to a single 1x1 scalar.
func SumAll(a *Node) *Node {
	out := &Node{op: opSumAll, in: []*Node{a}, s: Shape{1, 1}, val: []float32{vector.Sum(a.val)}}
	return out
}

// MeanAll reduces all elements to their arithmetic mean as a 1x1 scalar.
func MeanAll(a *Node) *Node {
	out := &Node{op: opMeanAll, in: []*Node{a}, s: Shape{1, 1}, val: []float32{vector.Sum(a.val) / float32(a.s.Len())}}
	return out
}

// SoftmaxCrossEntropy computes the mean categorical cross-entropy between the
// column-wise softmax of logits (k x batch) and the one-hot labels (k x batch).
func SoftmaxCrossEntropy(logits, labels *Node, batch int) *Node {
	k := logits.s.R
	shifted := Sub(logits, BroadcastRow(ColMax(logits), k))
	p := Exp(shifted)
	probs := Mul(p, BroadcastCol(Reciprocal(ColSum(p)), k))
	return Scale(SumAll(Mul(labels, Log(probs))), -1/float32(batch))
}

// Grad returns the gradient of root with respect to target. The returned node
// is itself part of the tape and may be differentiated again to obtain second
// order derivatives.
func (root *Node) Grad(target *Node) *Node {
	order := topo(root)
	grads := make(map[*Node]*Node)
	grads[root] = Ones(root.s)
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		g := grads[n]
		if g == nil {
			continue
		}
		if n == target {
			return g
		}
		n.vjp(g, grads)
	}
	return grads[target]
}

// GradMulti computes the gradients of root with respect to several targets in a
// single reverse pass. It is equivalent to calling Grad once per target but
// avoids rebuilding the backward graph repeatedly, which matters for large
// convolutional critics with a gradient penalty.
func (root *Node) GradMulti(targets []*Node) map[*Node]*Node {
	order := topo(root)
	grads := make(map[*Node]*Node, len(order))
	grads[root] = Ones(root.s)
	want := make(map[*Node]bool, len(targets))
	for _, t := range targets {
		want[t] = true
	}
	out := make(map[*Node]*Node, len(targets))
	for i := len(order) - 1; i >= 0; i-- {
		n := order[i]
		g := grads[n]
		if g == nil {
			continue
		}
		if want[n] {
			out[n] = g
			if len(out) == len(targets) {
				break
			}
			continue
		}
		n.vjp(g, grads)
	}
	return out
}

func accum(grads map[*Node]*Node, n, c *Node) {
	if e, ok := grads[n]; ok {
		grads[n] = Add(e, c)
	} else {
		grads[n] = c
	}
}

func (n *Node) vjp(g *Node, grads map[*Node]*Node) {
	switch n.op {
	case opMatMul:
		a, b := n.in[0], n.in[1]
		accum(grads, a, MatMulNT(g, b))
		accum(grads, b, MatMulTN(a, g))
	case opMatMulNT:
		a, b := n.in[0], n.in[1] // a (m x k), b (n x k)
		accum(grads, a, MatMul(g, b))
		accum(grads, b, MatMulTN(g, a))
	case opMatMulTN:
		a, b := n.in[0], n.in[1] // a (k x m), b (k x n)
		accum(grads, a, MatMulNT(b, g))
		accum(grads, b, MatMul(a, g))
	case opAdd:
		accum(grads, n.in[0], g)
		accum(grads, n.in[1], g)
	case opSub:
		accum(grads, n.in[0], g)
		accum(grads, n.in[1], Scale(g, -1))
	case opMul:
		accum(grads, n.in[0], Mul(g, n.in[1]))
		accum(grads, n.in[1], Mul(g, n.in[0]))
	case opAddBias:
		accum(grads, n.in[0], g)
		accum(grads, n.in[1], RowSum(g))
	case opColSum:
		accum(grads, n.in[0], BroadcastRow(g, n.in[0].s.R))
	case opBroadcastRow:
		accum(grads, n.in[0], ColSum(g))
	case opMulRow:
		a, r := n.in[0], n.in[1]
		accum(grads, a, Mul(g, BroadcastRow(r, a.s.R)))
		accum(grads, r, ColSum(Mul(g, a)))
	case opColSumSquares:
		a := n.in[0]
		accum(grads, a, Scale(Mul(a, BroadcastRow(g, a.s.R)), 2))
	case opScale:
		accum(grads, n.in[0], Scale(g, n.scalar))
	case opAddConst:
		accum(grads, n.in[0], g)
	case opSqrt:
		accum(grads, n.in[0], Scale(Mul(g, Reciprocal(n)), 0.5))
	case opReciprocal:
		accum(grads, n.in[0], Scale(Mul(g, Mul(n, n)), -1))
	case opLeakyReLU:
		accum(grads, n.in[0], Mul(g, Const(n.s, n.mask)))
	case opTanh:
		sq := Mul(n, n)
		accum(grads, n.in[0], Mul(g, Sub(Ones(n.s), sq)))
	case opExp:
		accum(grads, n.in[0], Mul(g, n))
	case opLog:
		accum(grads, n.in[0], Mul(g, Reciprocal(n.in[0])))
	case opColMax:
		accum(grads, n.in[0], Mul(Const(n.in[0].s, n.mask), BroadcastRow(g, n.in[0].s.R)))
	case opBroadcastCol:
		accum(grads, n.in[0], ColSum(g))
	case opRowSum:
		accum(grads, n.in[0], BroadcastCol(g, n.in[0].s.C))
	case opTranspose:
		accum(grads, n.in[0], Transpose(g))
	case opReshape:
		accum(grads, n.in[0], Reshape(g, n.in[0].s.R, n.in[0].s.C))
	case opFlatten:
		b, h, w, c := n.dims[0], n.dims[1], n.dims[2], n.dims[3]
		accum(grads, n.in[0], unflattenSpatial(g, b, h, w, c))
	case opUnflatten:
		b, h, w, c := n.dims[0], n.dims[1], n.dims[2], n.dims[3]
		accum(grads, n.in[0], FlattenSpatial(g, b, h, w, c))
	case opGlobalMean:
		b, h, w, c := n.dims[0], n.dims[1], n.dims[2], n.dims[3]
		accum(grads, n.in[0], Scale(broadcastSpatial(g, b, h, w, c), 1/float32(h*w)))
	case opBroadcastSpatial:
		b, h, w, c := n.dims[0], n.dims[1], n.dims[2], n.dims[3]
		accum(grads, n.in[0], GlobalMean(g, b, h, w, c))
	case opIm2Col:
		accum(grads, n.in[0], Col2Im(g, *n.cs))
	case opCol2Im:
		accum(grads, n.in[0], Im2Col(g, *n.cs))
	case opSumAll:
		accum(grads, n.in[0], Fill(n.in[0].s, g.val[0]))
	case opMeanAll:
		accum(grads, n.in[0], Fill(n.in[0].s, g.val[0]/float32(n.in[0].s.Len())))
	}
}

func topo(root *Node) []*Node {
	var order []*Node
	seen := make(map[*Node]bool)
	var visit func(n *Node)
	visit = func(n *Node) {
		if seen[n] {
			return
		}
		seen[n] = true
		for _, in := range n.in {
			visit(in)
		}
		order = append(order, n)
	}
	visit(root)
	return order
}
