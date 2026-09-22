// Command benchmark-ctu13 loads a trained or embedded classifier and benchmarks
// it on a held-out split of the CTU-13 dataset. It reports thresholded
// precision/recall/F1 as well as AUC for both the classifier probability and the
// critic score, which is the honest measure under heavy class imbalance.
//
//	go run ./examples/benchmark-ctu13 -weights /tmp/http1.ggnt -dir datasets/ctu-13 -protocol http1 -holdout 1,8
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
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

func main() {
	dir := flag.String("dir", "datasets/ctu-13", "scenario directory or root")
	protocol := flag.String("protocol", "http1", "protocol adapter")
	weights := flag.String("weights", "", "weights file (default: embedded)")
	maxPackets := flag.Int("maxpackets", 300000, "max packets per capture")
	maxSessions := flag.Int("maxsessions", 4000, "max labelled sessions")
	testFrac := flag.Float64("testfrac", 0.3, "held-out fraction when no holdout is given")
	seed := flag.Uint64("seed", 1, "random seed (must match training)")
	holdout := flag.String("holdout", "", "comma-separated scenario numbers held out")
	report := flag.String("report", "", "optional JSON report path")
	flag.Parse()

	if err := run(*dir, *protocol, *weights, *maxPackets, *maxSessions, *testFrac, *seed, *holdout, *report); err != nil {
		fmt.Fprintln(os.Stderr, "benchmark-ctu13:", err)
		os.Exit(1)
	}
}

func run(dir, protocol, weightsPath string, maxPackets, maxSessions int, testFrac float64, seed uint64, holdout, report string) error {
	a := adapter.ByName(protocol)
	if a == nil {
		return fmt.Errorf("unknown protocol %q", protocol)
	}

	var model *wcgan.Model
	if weightsPath != "" {
		f, err := os.Open(weightsPath)
		if err != nil {
			return err
		}
		defer f.Close()
		model, err = wcgan.Load(f)
		if err != nil {
			return err
		}
	} else {
		clf, err := classifiers.New(protocol)
		if err != nil {
			return err
		}
		model = clf.Model()
	}
	mc := model.Config()
	cfg := image.Config{Frames: mc.Frames, BytesPerFrame: mc.BytesPerFrame, BitPlanes: mc.BitPlanes}
	fmt.Printf("model: arch=%s %dx%d channels=%d input=%d classes=%d\n",
		mc.Arch, cfg.Frames, cfg.BytesPerFrame, cfg.Channels(), mc.InputDim, mc.NumClasses)

	scenarios := discover(dir)
	var allSessions []*session.Session
	var scenarioOf []int
	keep := func(f *net.Frame) bool { return a.Match(f) }
	for _, sdir := range scenarios {
		capture, err := pipeline.FindCapture(sdir)
		if err != nil {
			continue
		}
		sessions, err := pipeline.ReadSessionsOpt(capture, pipeline.ReadOptions{MaxFrames: cfg.Frames, MaxPackets: maxPackets, Keep: keep})
		if err != nil {
			return err
		}
		if bf, err := pipeline.FindBinetflow(sdir); err == nil {
			if _, err := session.JoinBinetflowFile(sessions, bf); err != nil {
				return err
			}
		}
		num, _ := strconv.Atoi(filepath.Base(sdir))
		for _, s := range sessions {
			if !s.Labeled {
				continue
			}
			if len(allSessions) < maxSessions {
				allSessions = append(allSessions, s)
				scenarioOf = append(scenarioOf, num)
			}
		}
	}

	var x [][]float32
	var y []int
	if mc.Arch == "conv" {
		x, y = pipeline.DatasetPlanar(allSessions, protocol, cfg)
	} else {
		x, y = pipeline.Dataset(allSessions, protocol, cfg)
	}
	mal := 0
	for _, v := range y {
		mal += v
	}
	fmt.Printf("dataset: %d samples (malicious %d, benign %d)\n", len(x), mal, len(x)-mal)
	if len(x) == 0 {
		return fmt.Errorf("no labelled %s sessions", protocol)
	}

	rng := rand.New(rand.NewPCG(seed, seed+1))
	hold := map[int]bool{}
	for _, part := range strings.Split(holdout, ",") {
		if n, err := strconv.Atoi(strings.TrimSpace(part)); err == nil {
			hold[n] = true
		}
	}
	var testIdx []int
	if len(hold) > 0 {
		for i := range x {
			if hold[scenarioOf[i]] {
				testIdx = append(testIdx, i)
			}
		}
		fmt.Printf("holdout scenarios: %v\n", hold)
	} else {
		perm := rng.Perm(len(x))
		nTest := int(float64(len(x)) * testFrac)
		if nTest < 1 {
			nTest = 1
		}
		testIdx = perm[:nTest]
	}
	if len(testIdx) == 0 {
		return fmt.Errorf("empty test split")
	}

	probs := make([]float32, len(testIdx))
	scores := make([]float32, len(testIdx))
	labels := make([]int, len(testIdx))
	for k, i := range testIdx {
		p, s := model.Classify(x[i], 1)
		if len(p) >= 2 {
			probs[k] = p[1]
		}
		if len(s) >= 1 {
			scores[k] = s[0]
		}
		labels[k] = y[i]
	}

	m := classifiers.Evaluate(probs, labels, 0.5)
	best := classifiers.BestThreshold(probs, labels)
	scoreAUC := classifiers.AUC(scores, labels)
	negScoreAUC := classifiers.AUC(negate(scores), labels)

	fmt.Printf("\n@0.50: acc=%.3f bal_acc=%.3f precision=%.3f recall=%.3f spec=%.3f f1=%.3f auc_prob=%.3f\n",
		m.Accuracy, m.BalancedAccuracy, m.Precision, m.Recall, m.Specificity, m.F1, m.AUC)
	fmt.Printf("best : thr=%.2f acc=%.3f bal_acc=%.3f precision=%.3f recall=%.3f spec=%.3f f1=%.3f\n",
		best.Threshold, best.Accuracy, best.BalancedAccuracy, best.Precision, best.Recall, best.Specificity, best.F1)
	fmt.Printf("auc : prob=%.3f critic_score=%.3f critic_neg_score=%.3f\n", m.AUC, scoreAUC, negScoreAUC)
	fmt.Printf("confusion: TP=%d FP=%d TN=%d FN=%d (positive=%d, n=%d)\n",
		best.Confusion.TP, best.Confusion.FP, best.Confusion.TN, best.Confusion.FN, best.Positive, best.N)

	if report != "" {
		rep := map[string]any{
			"protocol": protocol, "weights": weightsPath, "holdout": holdout,
			"at_0.5": m, "best": best, "auc_prob": m.AUC, "auc_critic": scoreAUC, "auc_critic_neg": negScoreAUC,
			"samples": len(x), "malicious": mal,
		}
		data, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(report, data, 0o644); err != nil {
			return err
		}
		fmt.Printf("wrote %s\n", report)
	}
	return nil
}

func negate(s []float32) []float32 {
	out := make([]float32, len(s))
	for i, v := range s {
		out[i] = -v
	}
	return out
}

func discover(root string) []string {
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
	return out
}
