// Package wcgan implements a conditional Wasserstein GAN with gradient penalty
// (WGAN-GP) and an auxiliary classifier head, used to discriminate malicious
// from non-malicious network-traffic images.
//
// The critic exposes two heads: a scalar Wasserstein realness score and a
// categorical class head. Class conditioning is injected additively into the
// generator latent and into the critic input via learned embeddings. The
// generator is also trained with a classification loss (the GACN idea) so that
// it preserves class-discriminative structure.
//
// The gradient penalty is differentiated exactly using the second-order
// automatic differentiation tape in engine/autodiff.
package wcgan

import (
	"math"
	"math/rand/v2"

	"github.com/cookiengineer/goganet/engine/autodiff"
	"github.com/cookiengineer/goganet/engine/vector"
)

// Config describes the model shape and training hyperparameters.
type Config struct {
	InputDim      int     // flattened image length
	LatentDim     int     // generator latent dimension
	GenHidden     []int   // generator hidden layer widths
	CritHidden    []int   // critic hidden layer widths
	NumClasses    int     // number of classes (>= 2)
	Lambda        float32 // gradient penalty coefficient
	NCritic       int     // critic updates per generator update
	GenLR         float32 // generator learning rate
	CritLR        float32 // critic learning rate
	Alpha         float32 // critic classification loss weight
	Gamma         float32 // generator classification loss weight
	Beta1         float32 // Adam beta1
	Beta2         float32 // Adam beta2
	Eps           float32 // Adam epsilon
	Leak          float32 // leaky ReLU slope
	Seed          uint64  // RNG seed
	Frames        int     // image frame rows (metadata for classifiers)
	BytesPerFrame int     // image bytes per row (metadata for classifiers)
	BitPlanes     bool    // whether bit-plane channels are present
	Arch          string  // "dense" (default) or "conv"
	InputChannels int     // image channels for the conv architecture
	SeedChannels  int     // generator seed feature-map channels
	GenChannels   []int   // conv generator deconvolution widths
	CritChannels  []int   // conv critic convolution widths
	Kernel        int     // conv kernel size (default 3)
}

// WithDefaults returns a copy of the config with zero fields replaced by
// sensible defaults from the WGAN-GP literature.
func (c Config) WithDefaults() Config {
	if c.LatentDim <= 0 {
		c.LatentDim = 100
	}
	if len(c.GenHidden) == 0 {
		c.GenHidden = []int{256, 512}
	}
	if len(c.CritHidden) == 0 {
		c.CritHidden = []int{512, 256}
	}
	if c.NumClasses < 2 {
		c.NumClasses = 2
	}
	if c.Lambda == 0 {
		c.Lambda = 10
	}
	if c.NCritic <= 0 {
		c.NCritic = 5
	}
	if c.GenLR == 0 {
		c.GenLR = 1e-4
	}
	if c.CritLR == 0 {
		c.CritLR = 1e-4
	}
	if c.Alpha == 0 {
		c.Alpha = 1
	}
	if c.Gamma == 0 {
		c.Gamma = 1
	}
	if c.Beta1 == 0 {
		c.Beta1 = 0.5
	}
	if c.Beta2 == 0 {
		c.Beta2 = 0.9
	}
	if c.Eps == 0 {
		c.Eps = 1e-8
	}
	if c.Leak == 0 {
		c.Leak = 0.2
	}
	if c.Frames <= 0 {
		c.Frames = 32
	}
	if c.BytesPerFrame <= 0 {
		c.BytesPerFrame = 1480
	}
	if c.Arch == "" {
		c.Arch = "dense"
	}
	if c.InputChannels <= 0 {
		c.InputChannels = 2
	}
	if c.SeedChannels <= 0 {
		c.SeedChannels = 64
	}
	if len(c.GenChannels) == 0 {
		c.GenChannels = []int{32, 16, 8}
	}
	if len(c.CritChannels) == 0 {
		c.CritChannels = []int{16, 32, 64}
	}
	if c.Kernel <= 0 {
		c.Kernel = 3
	}
	return c
}

type dense struct {
	w, b *autodiff.Node
}

type parameter struct {
	node *autodiff.Node
	m, v []float32
}

func (p *parameter) initState() {
	p.m = make([]float32, len(p.node.Value()))
	p.v = make([]float32, len(p.node.Value()))
}

// Model is a conditional WGAN-GP with an auxiliary classifier.
type Model struct {
	cfg Config
	rng *rand.Rand

	genLayers  []*dense
	critLayers []*dense
	genEmbed   *autodiff.Node // latent class embedding (LatentDim x NumClasses)
	scoreW     *autodiff.Node
	scoreB     *autodiff.Node
	clsW       *autodiff.Node
	clsB       *autodiff.Node

	// Convolutional architecture.
	seedC, seedH, seedW int
	genSeedW            *autodiff.Node
	genSeedB            *autodiff.Node
	genDeconvW          []*autodiff.Node
	genDeconvB          []*autodiff.Node
	genFinalW           *autodiff.Node
	genFinalB           *autodiff.Node
	critConvW           []*autodiff.Node
	critConvB           []*autodiff.Node
	critOutC            int

	genParams  []*parameter
	critParams []*parameter

	step int
}

// New constructs a model with the given configuration.
func New(cfg Config) *Model {
	cfg = cfg.WithDefaults()
	if cfg.InputDim <= 0 {
		panic("wcgan.New: InputDim must be positive")
	}
	m := &Model{
		cfg: cfg,
		rng: rand.New(rand.NewPCG(cfg.Seed, cfg.Seed+0x9e3779b97f4a7c15)),
	}
	m.build()
	return m
}

// Config returns the effective configuration.
func (m *Model) Config() Config { return m.cfg }

func (m *Model) randSlice(n int, std float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32(m.rng.NormFloat64()) * std
	}
	return s
}

func (m *Model) zeros(n int) []float32 {
	return make([]float32, n)
}

func (m *Model) heLayer(in, out int) *dense {
	std := float32(math.Sqrt(2.0 / float64(in)))
	return &dense{
		w: autodiff.Var(autodiff.S(out, in), m.randSlice(out*in, std)),
		b: autodiff.Var(autodiff.S(out, 1), m.zeros(out)),
	}
}

func (m *Model) registerGen(n *autodiff.Node) *parameter {
	p := &parameter{node: n}
	p.initState()
	m.genParams = append(m.genParams, p)
	return p
}

func (m *Model) registerCrit(n *autodiff.Node) *parameter {
	p := &parameter{node: n}
	p.initState()
	m.critParams = append(m.critParams, p)
	return p
}

func (m *Model) build() {
	if m.cfg.Arch == "conv" {
		m.buildConv()
		return
	}
	m.buildDense()
}

func (m *Model) buildDense() {
	cfg := m.cfg

	m.genEmbed = autodiff.Var(autodiff.S(cfg.LatentDim, cfg.NumClasses), m.randSlice(cfg.LatentDim*cfg.NumClasses, 0.05))
	m.registerGen(m.genEmbed)

	prev := cfg.LatentDim
	for _, h := range cfg.GenHidden {
		l := m.heLayer(prev, h)
		m.genLayers = append(m.genLayers, l)
		m.registerGen(l.w)
		m.registerGen(l.b)
		prev = h
	}
	out := m.heLayer(prev, cfg.InputDim)
	m.genLayers = append(m.genLayers, out)
	m.registerGen(out.w)
	m.registerGen(out.b)

	prev = cfg.InputDim
	for _, h := range cfg.CritHidden {
		l := m.heLayer(prev, h)
		m.critLayers = append(m.critLayers, l)
		m.registerCrit(l.w)
		m.registerCrit(l.b)
		prev = h
	}

	std := float32(math.Sqrt(2.0 / float64(prev)))
	m.scoreW = autodiff.Var(autodiff.S(1, prev), m.randSlice(prev, std))
	m.scoreB = autodiff.Var(autodiff.S(1, 1), m.zeros(1))
	m.clsW = autodiff.Var(autodiff.S(cfg.NumClasses, prev), m.randSlice(cfg.NumClasses*prev, std))
	m.clsB = autodiff.Var(autodiff.S(cfg.NumClasses, 1), m.zeros(cfg.NumClasses))
	m.registerCrit(m.scoreW)
	m.registerCrit(m.scoreB)
	m.registerCrit(m.clsW)
	m.registerCrit(m.clsB)
}

// buildConv builds the convolutional architecture. Both generator and critic
// avoid BatchNorm (per WGAN-GP), using leaky ReLU throughout; the critic
// collapses its feature map with global mean pooling before the two heads.
func (m *Model) buildConv() {
	cfg := m.cfg
	k := cfg.Kernel
	n := len(cfg.GenChannels)
	if cfg.Frames%(1<<n) != 0 || cfg.BytesPerFrame%(1<<n) != 0 {
		panic("wcgan: Frames and BytesPerFrame must be divisible by 2^len(GenChannels)")
	}
	m.seedC = cfg.SeedChannels
	m.seedH = cfg.Frames >> n
	m.seedW = cfg.BytesPerFrame >> n

	m.genEmbed = autodiff.Var(autodiff.S(cfg.LatentDim, cfg.NumClasses), m.randSlice(cfg.LatentDim*cfg.NumClasses, 0.05))
	m.registerGen(m.genEmbed)

	seedOut := m.seedC * m.seedH * m.seedW
	m.genSeedW = autodiff.Var(autodiff.S(seedOut, cfg.LatentDim), m.randSlice(seedOut*cfg.LatentDim, float32(math.Sqrt(2.0/float64(cfg.LatentDim)))))
	m.genSeedB = autodiff.Var(autodiff.S(seedOut, 1), m.zeros(seedOut))
	m.registerGen(m.genSeedW)
	m.registerGen(m.genSeedB)

	prev := m.seedC
	for _, out := range cfg.GenChannels {
		w := autodiff.Var(autodiff.S(prev, out*4*4), m.randSlice(prev*out*4*4, float32(math.Sqrt(2.0/float64(prev*4*4)))))
		b := autodiff.Var(autodiff.S(out, 1), m.zeros(out))
		m.genDeconvW = append(m.genDeconvW, w)
		m.genDeconvB = append(m.genDeconvB, b)
		m.registerGen(w)
		m.registerGen(b)
		prev = out
	}
	m.genFinalW = autodiff.Var(autodiff.S(cfg.InputChannels, prev*k*k), m.randSlice(cfg.InputChannels*prev*k*k, float32(math.Sqrt(2.0/float64(prev*k*k)))))
	m.genFinalB = autodiff.Var(autodiff.S(cfg.InputChannels, 1), m.zeros(cfg.InputChannels))
	m.registerGen(m.genFinalW)
	m.registerGen(m.genFinalB)

	prev = cfg.InputChannels
	for _, out := range cfg.CritChannels {
		w := autodiff.Var(autodiff.S(out, prev*k*k), m.randSlice(out*prev*k*k, float32(math.Sqrt(2.0/float64(prev*k*k)))))
		b := autodiff.Var(autodiff.S(out, 1), m.zeros(out))
		m.critConvW = append(m.critConvW, w)
		m.critConvB = append(m.critConvB, b)
		m.registerCrit(w)
		m.registerCrit(b)
		prev = out
	}
	m.critOutC = prev

	std := float32(math.Sqrt(2.0 / float64(prev)))
	m.scoreW = autodiff.Var(autodiff.S(1, prev), m.randSlice(prev, std))
	m.scoreB = autodiff.Var(autodiff.S(1, 1), m.zeros(1))
	m.clsW = autodiff.Var(autodiff.S(cfg.NumClasses, prev), m.randSlice(cfg.NumClasses*prev, std))
	m.clsB = autodiff.Var(autodiff.S(cfg.NumClasses, 1), m.zeros(cfg.NumClasses))
	m.registerCrit(m.scoreW)
	m.registerCrit(m.scoreB)
	m.registerCrit(m.clsW)
	m.registerCrit(m.clsB)
}

func (m *Model) leaky(x *autodiff.Node) *autodiff.Node {
	return autodiff.LeakyReLU(x, m.cfg.Leak)
}

// Generate maps the latent z (LatentDim x batch) and one-hot classes
// (NumClasses x batch) to an image (InputDim x batch).
func (m *Model) Generate(z, classes *autodiff.Node) *autodiff.Node {
	if m.cfg.Arch == "conv" {
		return m.generateConv(z, classes)
	}
	x := autodiff.Add(z, autodiff.MatMul(m.genEmbed, classes))
	for i, l := range m.genLayers {
		x = autodiff.AddBias(autodiff.MatMul(l.w, x), l.b)
		if i == len(m.genLayers)-1 {
			x = autodiff.Tanh(x)
		} else {
			x = m.leaky(x)
		}
	}
	return x
}

// Critic maps an image (InputDim x batch) to a Wasserstein score
// (1 x batch) and class logits (NumClasses x batch). The class head is trained
// on real data, giving the model its discriminative classifier.
func (m *Model) Critic(x *autodiff.Node) (score, logits *autodiff.Node) {
	if m.cfg.Arch == "conv" {
		return m.criticConv(x)
	}
	h := x
	for _, l := range m.critLayers {
		h = m.leaky(autodiff.AddBias(autodiff.MatMul(l.w, h), l.b))
	}
	score = autodiff.AddBias(autodiff.MatMul(m.scoreW, h), m.scoreB)
	logits = autodiff.AddBias(autodiff.MatMul(m.clsW, h), m.clsB)
	return score, logits
}

// generateConv runs the transposed-convolution generator. It expects z and
// classes in the same layout as the dense path and returns a planar image
// (InputChannels, batch*Frames*BytesPerFrame).
func (m *Model) generateConv(z, classes *autodiff.Node) *autodiff.Node {
	cfg := m.cfg
	batch := z.Shape().C

	g := autodiff.Add(z, autodiff.MatMul(m.genEmbed, classes))
	h := autodiff.AddBias(autodiff.MatMul(m.genSeedW, g), m.genSeedB)
	h = autodiff.Reshape(h, m.seedC, batch*m.seedH*m.seedW)

	curH, curW := m.seedH, m.seedW
	for i := range cfg.GenChannels {
		cs := autodiff.ConvShape{B: batch, InH: curH * 2, InW: curW * 2, OutH: curH, OutW: curW, K: 4, Stride: 2, Pad: 1}
		h = m.leaky(autodiff.Conv2DTranspose(h, m.genDeconvW[i], m.genDeconvB[i], cs))
		curH, curW = curH*2, curW*2
	}
	cs := autodiff.ConvShape{B: batch, InH: curH, InW: curW, K: cfg.Kernel, Stride: 1, Pad: cfg.Kernel / 2}
	h = autodiff.Tanh(autodiff.Conv2D(h, m.genFinalW, m.genFinalB, cs))
	// Flatten back to the (InputDim, batch) planar layout used by the data path.
	return autodiff.FromPlanar(h, batch, cfg.Frames, cfg.BytesPerFrame, cfg.InputChannels)
}

// criticConv runs the convolutional critic. It reshapes the planar input image
// into a feature map, applies strided convolutions, and pools globally before
// the score and class heads.
func (m *Model) criticConv(x *autodiff.Node) (score, logits *autodiff.Node) {
	cfg := m.cfg
	batch := x.Shape().C
	h := autodiff.ToPlanar(x, batch, cfg.Frames, cfg.BytesPerFrame, cfg.InputChannels)

	curH, curW := cfg.Frames, cfg.BytesPerFrame
	for i := range cfg.CritChannels {
		cs := autodiff.ConvShape{B: batch, InH: curH, InW: curW, K: cfg.Kernel, Stride: 2, Pad: cfg.Kernel / 2}
		h = m.leaky(autodiff.Conv2D(h, m.critConvW[i], m.critConvB[i], cs))
		curH = (curH+2*(cfg.Kernel/2)-cfg.Kernel)/2 + 1
		curW = (curW+2*(cfg.Kernel/2)-cfg.Kernel)/2 + 1
	}
	pooled := autodiff.GlobalMean(h, batch, curH, curW, m.critOutC)
	score = autodiff.AddBias(autodiff.MatMul(m.scoreW, pooled), m.scoreB)
	logits = autodiff.AddBias(autodiff.MatMul(m.clsW, pooled), m.clsB)
	return score, logits
}

func (m *Model) update(params []*parameter, lr float32, root *autodiff.Node) {
	m.step++
	targets := make([]*autodiff.Node, len(params))
	for i, p := range params {
		targets[i] = p.node
	}
	grads := root.GradMulti(targets)
	for _, p := range params {
		g := grads[p.node]
		if g == nil {
			continue
		}
		vector.Adam(p.node.Value(), g.Value(), p.m, p.v, lr, m.cfg.Beta1, m.cfg.Beta2, m.cfg.Eps, m.step)
	}
}

// CriticStep performs one critic update on a real batch and returns the loss.
// xReal is (InputDim x batch) row-major, labels is one-hot (NumClasses x batch).
func (m *Model) CriticStep(xReal, labels []float32, batch int) float32 {
	cfg := m.cfg
	d := cfg.InputDim
	k := cfg.NumClasses

	realX := autodiff.Const(autodiff.S(d, batch), xReal)
	realC := autodiff.Const(autodiff.S(k, batch), labels)

	z := autodiff.Const(autodiff.S(cfg.LatentDim, batch), m.noise(batch))
	cGen := autodiff.Const(autodiff.S(k, batch), m.randomClassOneHot(batch))

	fakeX := m.Generate(z, cGen)

	scoreReal, logitsReal := m.Critic(realX)
	scoreFake, _ := m.Critic(fakeX)

	// Gradient penalty: interpolate real and fake images.
	eps := m.uniformRow(batch)
	oneMinus := make([]float32, batch)
	for i := range eps {
		oneMinus[i] = 1 - eps[i]
	}
	epsN := autodiff.Const(autodiff.S(1, batch), eps)
	omN := autodiff.Const(autodiff.S(1, batch), oneMinus)

	interp := autodiff.Add(
		autodiff.Mul(realX, autodiff.BroadcastCol(epsN, d)),
		autodiff.Mul(fakeX, autodiff.BroadcastCol(omN, d)),
	)
	scoreInterp, _ := m.Critic(interp)
	g := scoreInterp.Grad(interp)
	t := autodiff.AddConst(autodiff.Sqrt(autodiff.ColSumSquares(g)), -1)
	penalty := autodiff.MeanAll(autodiff.Mul(t, t))

	ce := autodiff.SoftmaxCrossEntropy(logitsReal, realC, batch)

	wass := autodiff.Sub(autodiff.MeanAll(scoreFake), autodiff.MeanAll(scoreReal))
	loss := autodiff.Add(
		autodiff.Add(wass, autodiff.Scale(penalty, cfg.Lambda)),
		autodiff.Scale(ce, cfg.Alpha),
	)

	m.update(m.critParams, cfg.CritLR, loss)
	return loss.Value()[0]
}

// GeneratorStep performs one generator update and returns the loss.
func (m *Model) GeneratorStep(batch int) float32 {
	cfg := m.cfg
	k := cfg.NumClasses

	z := autodiff.Const(autodiff.S(cfg.LatentDim, batch), m.noise(batch))
	cGen := autodiff.Const(autodiff.S(k, batch), m.randomClassOneHot(batch))

	fakeX := m.Generate(z, cGen)
	scoreFake, logitsFake := m.Critic(fakeX)

	ce := autodiff.SoftmaxCrossEntropy(logitsFake, cGen, batch)
	loss := autodiff.Add(autodiff.Scale(autodiff.MeanAll(scoreFake), -1), autodiff.Scale(ce, cfg.Gamma))

	m.update(m.genParams, cfg.GenLR, loss)
	return loss.Value()[0]
}

func (m *Model) noise(batch int) []float32 {
	z := make([]float32, m.cfg.LatentDim*batch)
	for i := range z {
		z[i] = float32(m.rng.NormFloat64())
	}
	return z
}

func (m *Model) uniformRow(n int) []float32 {
	r := make([]float32, n)
	for i := range r {
		r[i] = float32(m.rng.Float64())
	}
	return r
}

func (m *Model) randomClassOneHot(batch int) []float32 {
	c := make([]float32, m.cfg.NumClasses*batch)
	for b := 0; b < batch; b++ {
		c[m.rng.IntN(m.cfg.NumClasses)*batch+b] = 1
	}
	return c
}

// OneHot converts class indices to a one-hot (NumClasses x batch) matrix.
func OneHot(indices []int, numClasses int) []float32 {
	out := make([]float32, numClasses*len(indices))
	for b, c := range indices {
		out[c*len(indices)+b] = 1
	}
	return out
}

// Classify runs the critic forward on a batch of images and returns the class
// probabilities (NumClasses x batch) and the Wasserstein scores (batch).
func (m *Model) Classify(xReal []float32, batch int) (probs []float32, scores []float32) {
	k := m.cfg.NumClasses
	x := autodiff.Const(autodiff.S(m.cfg.InputDim, batch), xReal)
	score, logits := m.Critic(x)

	probs = make([]float32, k*batch)
	for b := 0; b < batch; b++ {
		var maxv float32 = logits.Value()[b]
		for i := 1; i < k; i++ {
			if v := logits.Value()[i*batch+b]; v > maxv {
				maxv = v
			}
		}
		var sum float32
		for i := 0; i < k; i++ {
			e := float32(math.Exp(float64(logits.Value()[i*batch+b] - maxv)))
			probs[i*batch+b] = e
			sum += e
		}
		for i := 0; i < k; i++ {
			probs[i*batch+b] /= sum
		}
	}
	scores = make([]float32, batch)
	copy(scores, score.Value())
	return probs, scores
}

// Params returns the generator and critic parameter nodes in a stable order.
// It is used by the model serialization code.
func (m *Model) Params() (gen, crit []*autodiff.Node) {
	for _, p := range m.genParams {
		gen = append(gen, p.node)
	}
	for _, p := range m.critParams {
		crit = append(crit, p.node)
	}
	return gen, crit
}
