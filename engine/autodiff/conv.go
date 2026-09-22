package autodiff

// This file adds two-dimensional convolution and transposed convolution to the
// tape. Both are built from linear primitives (im2col, col2im, matmul,
// transpose) whose vector-Jacobian products are themselves differentiable, so
// the convolutional critic can still be used with an exact gradient penalty.
//
// Feature maps use a planar layout: a node of shape (C, N) holds C channels of
// N = batch*height*width pixels, with index c*N + b*H*W + y*W + x.

// ConvShape describes one convolution. InH/InW/InC are the input spatial
// dimensions; OutH/OutW/OutC are filled in by the helpers from the weight and
// shape.
type ConvShape struct {
	B, InH, InW, InC int
	K, Stride, Pad   int
	OutH, OutW, OutC int
}

func (cs ConvShape) withOutput() ConvShape {
	cs.OutH = (cs.InH+2*cs.Pad-cs.K)/cs.Stride + 1
	cs.OutW = (cs.InW+2*cs.Pad-cs.K)/cs.Stride + 1
	return cs
}

// Transpose returns the transpose of a (m x n) matrix as (n x m).
func Transpose(a *Node) *Node {
	m, n := a.s.R, a.s.C
	out := &Node{op: opTranspose, in: []*Node{a}, s: Shape{n, m}, val: make([]float32, m*n)}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			out.val[j*m+i] = a.val[i*n+j]
		}
	}
	return out
}

// Reshape returns a node with the same elements in the given shape.
func Reshape(a *Node, r, c int) *Node {
	if r*c != a.s.Len() {
		panic("autodiff.Reshape: element count mismatch")
	}
	out := &Node{op: opReshape, in: []*Node{a}, s: Shape{r, c}, val: a.val}
	return out
}

// FlattenSpatial converts a planar feature map (C, B*H*W) into (C*H*W, B), which
// is the layout expected by the dense classifier heads.
func FlattenSpatial(a *Node, B, H, W, C int) *Node {
	if a.s.R != C || a.s.C != B*H*W {
		panic("autodiff.FlattenSpatial: shape mismatch")
	}
	out := &Node{op: opFlatten, in: []*Node{a}, s: Shape{C * H * W, B}, val: make([]float32, C*H*W*B), dims: [4]int{B, H, W, C}}
	for c := 0; c < C; c++ {
		for b := 0; b < B; b++ {
			for s := 0; s < H*W; s++ {
				out.val[(c*H*W+s)*B+b] = a.val[c*B*H*W+b*H*W+s]
			}
		}
	}
	return out
}

// unflattenSpatial is the inverse of FlattenSpatial.
func unflattenSpatial(a *Node, B, H, W, C int) *Node {
	out := &Node{op: opUnflatten, in: []*Node{a}, s: Shape{C, B * H * W}, val: make([]float32, C*H*W*B), dims: [4]int{B, H, W, C}}
	for c := 0; c < C; c++ {
		for b := 0; b < B; b++ {
			for s := 0; s < H*W; s++ {
				out.val[c*B*H*W+b*H*W+s] = a.val[(c*H*W+s)*B+b]
			}
		}
	}
	return out
}

// ToPlanar converts a feature-major batch (C*H*W, B) into a planar feature map
// (C, B*H*W). This is the bridge between the data path, which stores each sample
// contiguously, and the convolution operators.
func ToPlanar(a *Node, B, H, W, C int) *Node {
	return unflattenSpatial(a, B, H, W, C)
}

// FromPlanar is the inverse of ToPlanar.
func FromPlanar(a *Node, B, H, W, C int) *Node {
	return FlattenSpatial(a, B, H, W, C)
}

// GlobalMean averages each channel over its spatial positions, returning
// (C, B). It is used to collapse the critic feature map before the heads.
func GlobalMean(a *Node, B, H, W, C int) *Node {
	if a.s.R != C || a.s.C != B*H*W {
		panic("autodiff.GlobalMean: shape mismatch")
	}
	out := &Node{op: opGlobalMean, in: []*Node{a}, s: Shape{C, B}, val: make([]float32, C*B), dims: [4]int{B, H, W, C}}
	inv := 1 / float32(H*W)
	for c := 0; c < C; c++ {
		for b := 0; b < B; b++ {
			var sum float32
			base := c*B*H*W + b*H*W
			for s := 0; s < H*W; s++ {
				sum += a.val[base+s]
			}
			out.val[c*B+b] = sum * inv
		}
	}
	return out
}

func broadcastSpatial(a *Node, B, H, W, C int) *Node {
	out := &Node{op: opBroadcastSpatial, in: []*Node{a}, s: Shape{C, B * H * W}, val: make([]float32, C*H*W*B), dims: [4]int{B, H, W, C}}
	for c := 0; c < C; c++ {
		for b := 0; b < B; b++ {
			v := a.val[c*B+b]
			base := c*B*H*W + b*H*W
			for s := 0; s < H*W; s++ {
				out.val[base+s] = v
			}
		}
	}
	return out
}

// Im2Col extracts patches of a planar feature map (InC, B*InH*InW) into a matrix
// (InC*K*K, B*OutH*OutW). Its adjoint is Col2Im.
func Im2Col(a *Node, cs ConvShape) *Node {
	cs = cs.withOutput()
	if a.s.R != cs.InC || a.s.C != cs.B*cs.InH*cs.InW {
		panic("autodiff.Im2Col: shape mismatch")
	}
	k2 := cs.K * cs.K
	nout := cs.B * cs.OutH * cs.OutW
	out := &Node{op: opIm2Col, in: []*Node{a}, s: Shape{cs.InC * k2, nout}, val: make([]float32, cs.InC*k2*nout), cs: &cs}
	hw := cs.InH * cs.InW
	for c := 0; c < cs.InC; c++ {
		for kh := 0; kh < cs.K; kh++ {
			for kw := 0; kw < cs.K; kw++ {
				row := c*k2 + kh*cs.K + kw
				for b := 0; b < cs.B; b++ {
					for oh := 0; oh < cs.OutH; oh++ {
						ih := oh*cs.Stride - cs.Pad + kh
						if ih < 0 || ih >= cs.InH {
							continue
						}
						for ow := 0; ow < cs.OutW; ow++ {
							iw := ow*cs.Stride - cs.Pad + kw
							if iw < 0 || iw >= cs.InW {
								continue
							}
							out.val[row*nout+b*cs.OutH*cs.OutW+oh*cs.OutW+ow] = a.val[c*cs.B*hw+b*hw+ih*cs.InW+iw]
						}
					}
				}
			}
		}
	}
	return out
}

// Col2Im is the adjoint of Im2Col: it folds a patch matrix
// (InC*K*K, B*OutH*OutW) back into a feature map (InC, B*InH*InW).
func Col2Im(a *Node, cs ConvShape) *Node {
	cs = cs.withOutput()
	k2 := cs.K * cs.K
	nout := cs.B * cs.OutH * cs.OutW
	if a.s.R != cs.InC*k2 || a.s.C != nout {
		panic("autodiff.Col2Im: shape mismatch")
	}
	hw := cs.InH * cs.InW
	out := &Node{op: opCol2Im, in: []*Node{a}, s: Shape{cs.InC, cs.B * hw}, val: make([]float32, cs.InC*cs.B*hw), cs: &cs}
	for c := 0; c < cs.InC; c++ {
		for kh := 0; kh < cs.K; kh++ {
			for kw := 0; kw < cs.K; kw++ {
				row := c*k2 + kh*cs.K + kw
				for b := 0; b < cs.B; b++ {
					for oh := 0; oh < cs.OutH; oh++ {
						ih := oh*cs.Stride - cs.Pad + kh
						if ih < 0 || ih >= cs.InH {
							continue
						}
						for ow := 0; ow < cs.OutW; ow++ {
							iw := ow*cs.Stride - cs.Pad + kw
							if iw < 0 || iw >= cs.InW {
								continue
							}
							out.val[c*cs.B*hw+b*hw+ih*cs.InW+iw] += a.val[row*nout+b*cs.OutH*cs.OutW+oh*cs.OutW+ow]
						}
					}
				}
			}
		}
	}
	return out
}

// Conv2D applies a valid zero-padded convolution. x is (InC, B*InH*InW), w is
// (OutC, InC*K*K) and b is (OutC, 1). The result is (OutC, B*OutH*OutW).
func Conv2D(x, w, b *Node, cs ConvShape) *Node {
	cs.InC = x.s.R
	cs.OutC = w.s.R
	if w.s.C != cs.InC*cs.K*cs.K {
		panic("autodiff.Conv2D: weight shape mismatch")
	}
	col := Im2Col(x, cs)
	y := MatMul(w, col)
	return AddBias(y, b)
}

// Conv2DTranspose applies a strided transposed convolution (deconvolution). x is
// (InC, B*InH*InW) and w is (InC, OutC*K*K); the result is
// (OutC, B*OutH*OutW). The shape must satisfy the transposed-convolution
// relation OutH = (InH-1)*Stride - 2*Pad + K + outputPad.
func Conv2DTranspose(x, w, b *Node, cs ConvShape) *Node {
	inC := x.s.R
	outCK := w.s.C
	if w.s.R != inC {
		panic("autodiff.Conv2DTranspose: weight input channels mismatch")
	}
	outC := outCK / (cs.K * cs.K)
	// Build the adjoint convolution shape: im2col gathers from the (larger)
	// output spatial grid and produces the (smaller) input grid.
	if cs.InH != (cs.OutH-1)*cs.Stride-2*cs.Pad+cs.K {
		// cs.InH is the large deconv output; cs.OutH is the small deconv input.
		// Accept only the consistent relation.
		panic("autodiff.Conv2DTranspose: inconsistent shape")
	}
	tw := Transpose(w)   // (OutC*K*K, InC)
	col := MatMul(tw, x) // (OutC*K*K, B*OutH*OutW)
	adj := ConvShape{    // adjoint maps (OutC, B*OutH*OutW) -> (OutC, B*InH*InW)
		B: cs.B, InH: cs.InH, InW: cs.InW, InC: outC, K: cs.K, Stride: cs.Stride, Pad: cs.Pad,
	}
	y := Col2Im(col, adj)
	return AddBias(y, b)
}
