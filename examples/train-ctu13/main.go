// Command train-ctu13 trains a per-protocol classifier across one or more
// labelled CTU-13 scenarios and writes the weights so they can be embedded by
// the classifiers package.
//
// Whole-dataset run (all extracted scenarios under datasets/ctu-13):
//
//	go run ./examples/train-ctu13 -dir datasets/ctu-13 -protocol http1 -arch conv
//
// A single scenario:
//
//	go run ./examples/train-ctu13 -dir datasets/ctu-13/4 -protocol dns
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/cookiengineer/goganet/adapter"
	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/net"
	"github.com/cookiengineer/goganet/adapter/pipeline"
	"github.com/cookiengineer/goganet/adapter/session"
	"github.com/cookiengineer/goganet/classifiers"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

type options struct {
	dir         string
	protocol    string
	frames      int
	bytesPer    int
	arch        string
	maxPackets  int
	maxSessions int
	ncritic     int
	epochs      int
	batch       int
	latent      int
	testFrac    float64
	seed        uint64
	balance     bool
	holdout     string
	out         string
	report      string
}

func main() {
	o := options{}
	flag.StringVar(&o.dir, "dir", "datasets/ctu-13", "scenario directory or root of several scenarios")
	flag.StringVar(&o.protocol, "protocol", "http1", "protocol adapter to train")
	flag.IntVar(&o.frames, "frames", 8, "frames per image")
	flag.IntVar(&o.bytesPer, "bytes", 64, "bytes per frame")
	flag.StringVar(&o.arch, "arch", "conv", "architecture: dense or conv")
	flag.IntVar(&o.maxPackets, "maxpackets", 300000, "max packets read per capture (0 = unlimited)")
	flag.IntVar(&o.maxSessions, "maxsessions", 6000, "max labelled sessions across the dataset")
	flag.IntVar(&o.ncritic, "ncritic", 2, "critic updates per generator update")
	flag.IntVar(&o.epochs, "epochs", 15, "training epochs")
	flag.IntVar(&o.batch, "batch", 32, "batch size")
	flag.IntVar(&o.latent, "latent", 32, "latent dimension")
	flag.Float64Var(&o.testFrac, "testfrac", 0.3, "held-out test fraction")
	flag.Uint64Var(&o.seed, "seed", 1, "random seed")
	flag.BoolVar(&o.balance, "balance", true, "oversample the minority class")
	flag.StringVar(&o.holdout, "holdout", "", "comma-separated scenario numbers held out entirely for testing")
	flag.StringVar(&o.out, "out", "", "output weights path (default classifiers/weights/<protocol>.ggnt)")
	flag.StringVar(&o.report, "report", "", "optional JSON metrics report path")
	flag.Parse()

	if o.out == "" {
		o.out = "classifiers/weights/" + o.protocol + ".ggnt"
	}
	if err := run(o); err != nil {
		fmt.Fprintln(os.Stderr, "train-ctu13:", err)
		os.Exit(1)
	}
}

func run(o options) error {
	a := adapter.ByName(o.protocol)
	if a == nil {
		return fmt.Errorf("unknown protocol %q", o.protocol)
	}
	keep := func(f *net.Frame) bool { return a.Match(f) }

	scenarios := discoverScenarios(o.dir)
	if len(scenarios) == 0 {
		return fmt.Errorf("no scenarios found under %q", o.dir)
	}

	var allSessions []*session.Session
	var scenarioOf []int
	for _, dir := range scenarios {
		capture, err := pipeline.FindCapture(dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", dir, err)
			continue
		}
		sessions, err := pipeline.ReadSessionsOpt(capture, pipeline.ReadOptions{
			MaxFrames:  o.frames,
			MaxPackets: o.maxPackets,
			Keep:       keep,
		})
		if err != nil {
			return fmt.Errorf("%s: %w", dir, err)
		}
		if bf, err := pipeline.FindBinetflow(dir); err == nil {
			if _, err := session.JoinBinetflowFile(sessions, bf); err != nil {
				return fmt.Errorf("%s: %w", dir, err)
			}
		}
		labelled, mal, ben := 0, 0, 0
		for _, s := range sessions {
			if !s.Labeled {
				continue
			}
			labelled++
			if s.Malicious {
				mal++
			} else {
				ben++
			}
			if len(allSessions) < o.maxSessions {
				allSessions = append(allSessions, s)
				scenarioOf = append(scenarioOf, scenarioNumber(dir))
			}
		}
		fmt.Printf("%-40s sessions=%-7d labelled=%-7d malicious=%-6d benign=%d\n",
			filepath.Base(dir), len(sessions), labelled, mal, ben)
	}

	cfg := image.Config{Frames: o.frames, BytesPerFrame: o.bytesPer}
	var x [][]float32
	var y []int
	if o.arch == "conv" {
		x, y = pipeline.DatasetPlanar(allSessions, o.protocol, cfg)
	} else {
		x, y = pipeline.Dataset(allSessions, o.protocol, cfg)
	}
	if len(x) == 0 {
		return fmt.Errorf("no %s sessions with labels", o.protocol)
	}
	mal := 0
	for _, v := range y {
		mal += v
	}
	fmt.Printf("\n%s samples: %d (malicious %d, benign %d)\n", o.protocol, len(x), mal, len(x)-mal)
	if mal == 0 || mal == len(x) {
		return fmt.Errorf("need both classes to train (malicious=%d of %d)", mal, len(x))
	}

	// Deterministic split. With -holdout, whole scenarios are held out so the
	// test set measures cross-scenario generalisation rather than leakage.
	rng := rand.New(rand.NewPCG(o.seed, o.seed+1))
	var trainIdx, testIdx []int
	hold := map[int]bool{}
	for _, part := range strings.Split(o.holdout, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(part)); err == nil {
			hold[n] = true
		}
	}
	if len(hold) > 0 {
		if len(scenarioOf) != len(x) {
			return fmt.Errorf("scenario tracking misaligned (%d vs %d)", len(scenarioOf), len(x))
		}
		for i := range x {
			if hold[scenarioOf[i]] {
				testIdx = append(testIdx, i)
			} else {
				trainIdx = append(trainIdx, i)
			}
		}
		fmt.Printf("holdout scenarios: %v -> test=%d train=%d\n", hold, len(testIdx), len(trainIdx))
	} else {
		perm := rng.Perm(len(x))
		nTest := int(float64(len(x)) * o.testFrac)
		if nTest < 1 {
			nTest = 1
		}
		if nTest > len(x)-1 {
			nTest = len(x) - 1
		}
		testIdx = perm[:nTest]
		trainIdx = append([]int(nil), perm[nTest:]...)
	}
	if len(testIdx) == 0 || len(trainIdx) == 0 {
		return fmt.Errorf("empty train or test split (train=%d test=%d)", len(trainIdx), len(testIdx))
	}

	if o.balance {
		trainIdx = oversample(y, trainIdx, rng)
	}

	modelCfg := wcgan.Config{
		InputDim:      cfg.Len(),
		LatentDim:     o.latent,
		GenHidden:     []int{64, 32},
		CritHidden:    []int{64, 32},
		NumClasses:    2,
		NCritic:       o.ncritic,
		Frames:        o.frames,
		BytesPerFrame: o.bytesPer,
		Seed:          o.seed,
	}
	if o.arch == "conv" {
		modelCfg.Arch = "conv"
		modelCfg.InputChannels = cfg.Channels()
		modelCfg.SeedChannels = 32
		modelCfg.GenChannels = []int{16, 8}
		modelCfg.CritChannels = []int{8, 16, 32}
		modelCfg.Kernel = 3
	}
	model := wcgan.New(modelCfg)
	nc := model.Config().NCritic

	bs := o.batch
	if bs > len(trainIdx) {
		bs = len(trainIdx)
	}
	if bs <= 0 {
		return fmt.Errorf("invalid batch size %d (train=%d)", bs, len(trainIdx))
	}
	for epoch := 0; epoch < o.epochs; epoch++ {
		order := rng.Perm(len(trainIdx))
		var dLoss, gLoss float32
		steps := 0
		for i := 0; i+bs <= len(order); i += bs {
			idx := make([]int, bs)
			yb := make([]int, bs)
			for k := range idx {
				idx[k] = trainIdx[order[i+k]]
				yb[k] = y[idx[k]]
			}
			xb := batchMatrix(x, idx)
			for c := 0; c < nc; c++ {
				dLoss += model.CriticStep(xb, wcgan.OneHot(yb, 2), bs)
			}
			gLoss += model.GeneratorStep(bs)
			steps++
		}
		if steps > 0 && (epoch%5 == 0 || epoch == o.epochs-1) {
			fmt.Printf("epoch %2d: d_loss=%+.4f g_loss=%+.4f\n", epoch, dLoss/float32(steps), gLoss/float32(steps))
		}
	}

	trainM, trainScoreAUC := evaluateModel(model, x, y, trainIdx)
	testM, testScoreAUC := evaluateModel(model, x, y, testIdx)
	fmt.Printf("\ntrain: acc=%.3f bal_acc=%.3f precision=%.3f recall=%.3f spec=%.3f f1=%.3f auc_prob=%.3f auc_score=%.3f\n",
		trainM.Accuracy, trainM.BalancedAccuracy, trainM.Precision, trainM.Recall, trainM.Specificity, trainM.F1, trainM.AUC, trainScoreAUC)
	fmt.Printf("test:  acc=%.3f bal_acc=%.3f precision=%.3f recall=%.3f spec=%.3f f1=%.3f auc_prob=%.3f auc_score=%.3f\n",
		testM.Accuracy, testM.BalancedAccuracy, testM.Precision, testM.Recall, testM.Specificity, testM.F1, testM.AUC, testScoreAUC)
	fmt.Printf("test confusion @thr=%.2f: TP=%d FP=%d TN=%d FN=%d\n",
		testM.Threshold, testM.Confusion.TP, testM.Confusion.FP, testM.Confusion.TN, testM.Confusion.FN)

	if err := classifiers.Save(o.out, model); err != nil {
		return fmt.Errorf("save: %w", err)
	}
	fmt.Printf("wrote %s\n", o.out)

	if o.report != "" {
		if err := writeReport(o, trainM, testM, len(x), mal); err != nil {
			return fmt.Errorf("report: %w", err)
		}
		fmt.Printf("wrote %s\n", o.report)
	}
	return nil
}

func discoverScenarios(root string) []string {
	// A scenario directory contains a binetflow or a capture directly.
	if _, err := pipeline.FindCapture(root); err == nil {
		return []string{root}
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(root, e.Name())
		if _, err := pipeline.FindCapture(dir); err == nil {
			out = append(out, dir)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		return scenarioNumber(out[i]) < scenarioNumber(out[j])
	})
	return out
}

func scenarioNumber(dir string) int {
	n, err := strconv.Atoi(filepath.Base(dir))
	if err != nil {
		return 1 << 30
	}
	return n
}

func oversample(y []int, idx []int, rng *rand.Rand) []int {
	var pos, neg []int
	for _, i := range idx {
		if y[i] == 1 {
			pos = append(pos, i)
		} else {
			neg = append(neg, i)
		}
	}
	minor, major := pos, neg
	if len(pos) > len(neg) {
		minor, major = neg, pos
	}
	if len(minor) == 0 {
		return idx
	}
	out := append([]int(nil), idx...)
	need := len(major) - len(minor)
	for k := 0; k < need; k++ {
		out = append(out, minor[rng.IntN(len(minor))])
	}
	return out
}

func batchMatrix(x [][]float32, idx []int) []float32 {
	d := len(x[0])
	out := make([]float32, d*len(idx))
	for b, i := range idx {
		row := x[i]
		for j, v := range row {
			out[j*len(idx)+b] = v
		}
	}
	return out
}

func evaluateModel(model *wcgan.Model, x [][]float32, y []int, idx []int) (classifiers.Metrics, float32) {
	probs := make([]float32, len(idx))
	scores := make([]float32, len(idx))
	labels := make([]int, len(idx))
	for k, i := range idx {
		p, s := model.Classify(x[i], 1)
		if len(p) >= 2 {
			probs[k] = p[1]
		}
		if len(s) >= 1 {
			scores[k] = s[0]
		}
		labels[k] = y[i]
	}
	return classifiers.BestThreshold(probs, labels), classifiers.AUC(scores, labels)
}

func writeReport(o options, train, test classifiers.Metrics, n, malicious int) error {
	rep := map[string]any{
		"protocol":  o.protocol,
		"arch":      o.arch,
		"frames":    o.frames,
		"bytes":     o.bytesPer,
		"samples":   n,
		"malicious": malicious,
		"test_frac": o.testFrac,
		"seed":      o.seed,
		"train":     train,
		"test":      test,
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(o.report, data, 0o644)
}
