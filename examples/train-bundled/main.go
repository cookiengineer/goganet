// Command train-bundled trains a single classifier on all available datasets
// (CTU-13 and IoT-23) and writes the model that ships as classifiers/weights/bundled.ggnt.
//
// Both datasets contain far more traffic than can be decoded in memory, so the
// trainer covers every capture from each dataset but samples a bounded number
// of packets per PCAP and sessions per capture. Increase -maxpackets,
// -maxpercapture and -maxsessions to use more of the data.
//
//	go run ./examples/train-bundled
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
	ctu           string
	iot           string
	protocol      string
	frames        int
	bytesPer      int
	arch          string
	maxPackets    int
	maxPerCapture int
	maxSessions   int
	iotMaxLog     int
	epochs        int
	batch         int
	ncritic       int
	latent        int
	testFrac      float64
	seed          uint64
	balance       bool
	holdout       string
	out           string
	report        string
}

// capture is one labelled PCAP group from either dataset.
type capture struct {
	name  string
	pcaps []string
	label string
	zeek  bool // true for Zeek conn.log.labeled, false for binetflow
}

func main() {
	o := options{}
	flag.StringVar(&o.ctu, "ctu", "datasets/ctu-13", "CTU-13 scenario root")
	flag.StringVar(&o.iot, "iot", "datasets/iot-23", "IoT-23 capture root")
	flag.StringVar(&o.protocol, "protocol", "raw", "protocol adapter to train")
	flag.IntVar(&o.frames, "frames", 8, "frames per image")
	flag.IntVar(&o.bytesPer, "bytes", 64, "bytes per frame")
	flag.StringVar(&o.arch, "arch", "conv", "architecture: dense or conv")
	flag.IntVar(&o.maxPackets, "maxpackets", 100000, "max packets read per pcap (0 = unlimited)")
	flag.IntVar(&o.maxPerCapture, "maxpercapture", 20000, "max sessions taken from one capture")
	flag.IntVar(&o.maxSessions, "maxsessions", 60000, "max labelled sessions across all datasets")
	flag.IntVar(&o.iotMaxLog, "iotmaxloglines", 5000000, "max Zeek log lines scanned per IoT-23 capture (0 = unlimited)")
	flag.IntVar(&o.epochs, "epochs", 2, "training epochs")
	flag.IntVar(&o.batch, "batch", 32, "batch size")
	flag.IntVar(&o.ncritic, "ncritic", 2, "critic updates per generator update")
	flag.IntVar(&o.latent, "latent", 32, "latent dimension")
	flag.Float64Var(&o.testFrac, "testfrac", 0.2, "held-out test fraction")
	flag.Uint64Var(&o.seed, "seed", 1, "random seed")
	flag.BoolVar(&o.balance, "balance", true, "oversample the minority class")
	flag.StringVar(&o.holdout, "holdout", "", "comma-separated capture-name substrings held out for testing")
	flag.StringVar(&o.out, "out", "classifiers/weights/bundled.ggnt", "output weights path")
	flag.StringVar(&o.report, "report", "", "optional JSON metrics report path")
	flag.Parse()

	if err := run(o); err != nil {
		fmt.Fprintln(os.Stderr, "train-bundled:", err)
		os.Exit(1)
	}
}

func run(o options) error {
	a := adapter.ByName(o.protocol)
	if a == nil {
		return fmt.Errorf("unknown protocol %q", o.protocol)
	}
	keep := func(f *net.Frame) bool { return a.Match(f) }

	captures, err := discover(o)
	if err != nil {
		return err
	}
	if len(captures) == 0 {
		return fmt.Errorf("no captures found")
	}
	fmt.Printf("captures: %d (CTU-13 + IoT-23)\n", len(captures))

	var allSessions []*session.Session
	var sourceOf []string
	for _, c := range captures {
		if len(allSessions) >= o.maxSessions {
			break
		}
		var sessions []*session.Session
		for _, pc := range c.pcaps {
			ss, err := pipeline.ReadSessionsOpt(pc, pipeline.ReadOptions{
				MaxFrames:  o.frames,
				MaxPackets: o.maxPackets,
				Keep:       keep,
			})
			if err != nil {
				fmt.Fprintf(os.Stderr, "  skip %s: %v\n", pc, err)
				continue
			}
			sessions = append(sessions, ss...)
		}
		sessions = dedupe(sessions)
		matched := 0
		if c.zeek {
			matched, err = session.JoinZeekFileLimit(sessions, c.label, o.iotMaxLog)
		} else {
			matched, err = session.JoinBinetflowFile(sessions, c.label)
		}
		if err != nil {
			return fmt.Errorf("%s: %w", c.name, err)
		}
		labelled, mal, ben, taken := 0, 0, 0, 0
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
			if taken >= o.maxPerCapture || len(allSessions) >= o.maxSessions {
				continue
			}
			allSessions = append(allSessions, s)
			sourceOf = append(sourceOf, c.name)
			taken++
		}
		fmt.Printf("%-42s pcaps=%-3d sessions=%-7d matched=%-7d used=%-7d malicious=%-6d benign=%d\n",
			c.name, len(c.pcaps), len(sessions), matched, taken, mal, ben)
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
	fmt.Printf("\nbundled samples: %d (malicious %d, benign %d)\n", len(x), mal, len(x)-mal)
	if mal == 0 || mal == len(x) {
		return fmt.Errorf("need both classes to train (malicious=%d of %d)", mal, len(x))
	}

	rng := rand.New(rand.NewPCG(o.seed, o.seed+1))
	var trainIdx, testIdx []int
	hold := splitNonEmpty(o.holdout, ",")
	if len(hold) > 0 {
		if len(sourceOf) != len(x) {
			return fmt.Errorf("source tracking misaligned (%d vs %d)", len(sourceOf), len(x))
		}
		for i := range x {
			if matchesAny(sourceOf[i], hold) {
				testIdx = append(testIdx, i)
			} else {
				trainIdx = append(trainIdx, i)
			}
		}
		fmt.Printf("holdout captures: %v -> test=%d train=%d\n", hold, len(testIdx), len(trainIdx))
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
		return fmt.Errorf("invalid batch size %d", bs)
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
		if steps > 0 {
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
		rep := map[string]any{
			"dataset": "bundled(ctu-13+iot-23)", "protocol": o.protocol, "arch": o.arch,
			"frames": o.frames, "bytes": o.bytesPer, "samples": len(x), "malicious": mal,
			"maxpackets": o.maxPackets, "maxpercapture": o.maxPerCapture, "maxsessions": o.maxSessions,
			"test_frac": o.testFrac, "seed": o.seed, "train": trainM, "test": testM,
		}
		data, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(o.report, data, 0o644); err != nil {
			return err
		}
		fmt.Printf("wrote %s\n", o.report)
	}
	return nil
}

// discover collects labelled captures from both datasets.
func discover(o options) ([]capture, error) {
	var out []capture
	// CTU-13: one scenario directory with a binetflow per capture.
	for _, dir := range ctuScenarios(o.ctu) {
		pc, err := pipeline.FindCapture(dir)
		if err != nil {
			continue
		}
		bf, err := pipeline.FindBinetflow(dir)
		if err != nil {
			continue
		}
		out = append(out, capture{name: "ctu13-" + filepath.Base(dir), pcaps: []string{pc}, label: bf, zeek: false})
	}
	// IoT-23: capture directories with a Zeek conn.log.labeled.
	iot, err := iotCaptures(o.iot)
	if err != nil {
		return nil, err
	}
	out = append(out, iot...)
	return out, nil
}

func ctuScenarios(root string) []string {
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
	sort.Slice(out, func(i, j int) bool { return scenarioNumber(out[i]) < scenarioNumber(out[j]) })
	return out
}

func scenarioNumber(dir string) int {
	n, err := strconv.Atoi(filepath.Base(dir))
	if err != nil {
		return 1 << 30
	}
	return n
}

func iotCaptures(root string) ([]capture, error) {
	if _, err := os.Stat(root); err != nil {
		return nil, nil // IoT-23 not present: train on CTU-13 alone.
	}
	var connLogs, pcaps []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		name := d.Name()
		switch {
		case strings.HasSuffix(name, "conn.log.labeled"):
			connLogs = append(connLogs, path)
		case strings.HasSuffix(name, ".pcap"), strings.HasSuffix(name, ".pcap.gz"):
			pcaps = append(pcaps, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	byDir := map[string]*capture{}
	var order []string
	for _, log := range connLogs {
		dir := filepath.Dir(log)
		if filepath.Base(dir) == "bro" {
			dir = filepath.Dir(dir)
		}
		if _, ok := byDir[dir]; !ok {
			byDir[dir] = &capture{name: "iot23-" + filepath.Base(dir), label: log, zeek: true}
			order = append(order, dir)
		}
	}
	prefix := func(dir string) string { return dir + string(os.PathSeparator) }
	var out []capture
	for _, dir := range order {
		c := byDir[dir]
		for _, p := range pcaps {
			if strings.HasPrefix(p, prefix(dir)) {
				c.pcaps = append(c.pcaps, p)
			}
		}
		if len(c.pcaps) == 0 {
			continue
		}
		sort.Strings(c.pcaps)
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out, nil
}

func dedupe(in []*session.Session) []*session.Session {
	seen := make(map[net.SessionKey]bool, len(in))
	out := in[:0]
	for _, s := range in {
		if seen[s.Key] {
			continue
		}
		seen[s.Key] = true
		out = append(out, s)
	}
	return out
}

func splitNonEmpty(s, sep string) []string {
	var out []string
	for _, p := range strings.Split(s, sep) {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func matchesAny(name string, tokens []string) bool {
	for _, t := range tokens {
		if strings.Contains(name, t) {
			return true
		}
	}
	return false
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
