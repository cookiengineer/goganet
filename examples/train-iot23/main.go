// Command train-iot23 trains a per-protocol classifier across the IoT-23
// captures and writes the weights so they can be embedded by the classifiers
// package.
//
// Download and extract the dataset first (see download-datasets.sh), then:
//
//	go run ./examples/train-iot23 -dir datasets/iot-23 -protocol http1 -arch conv
//
// IoT-23 stores one directory per capture containing one or more PCAPs and a
// Zeek conn.log.labeled file; labels are joined from the Zeek log.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
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
	epochs      int
	batch       int
	ncritic     int
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
	flag.StringVar(&o.dir, "dir", "datasets/iot-23", "root of the extracted IoT-23 dataset")
	flag.StringVar(&o.protocol, "protocol", "http1", "protocol adapter to train")
	flag.IntVar(&o.frames, "frames", 8, "frames per image")
	flag.IntVar(&o.bytesPer, "bytes", 64, "bytes per frame")
	flag.StringVar(&o.arch, "arch", "conv", "architecture: dense or conv")
	flag.IntVar(&o.maxPackets, "maxpackets", 300000, "max packets read per pcap")
	flag.IntVar(&o.maxSessions, "maxsessions", 6000, "max labelled sessions across the dataset")
	flag.IntVar(&o.epochs, "epochs", 15, "training epochs")
	flag.IntVar(&o.batch, "batch", 32, "batch size")
	flag.IntVar(&o.ncritic, "ncritic", 2, "critic updates per generator update")
	flag.IntVar(&o.latent, "latent", 32, "latent dimension")
	flag.Float64Var(&o.testFrac, "testfrac", 0.3, "held-out test fraction")
	flag.Uint64Var(&o.seed, "seed", 1, "random seed")
	flag.BoolVar(&o.balance, "balance", true, "oversample the minority class")
	flag.StringVar(&o.holdout, "holdout", "", "comma-separated capture-name substrings held out for testing")
	flag.StringVar(&o.out, "out", "", "output weights path (default classifiers/weights/iot23-<protocol>.ggnt)")
	flag.StringVar(&o.report, "report", "", "optional JSON metrics report path")
	flag.Parse()

	if o.out == "" {
		o.out = "classifiers/weights/iot23-" + o.protocol + ".ggnt"
	}
	if err := run(o); err != nil {
		fmt.Fprintln(os.Stderr, "train-iot23:", err)
		os.Exit(1)
	}
}

func run(o options) error {
	a := adapter.ByName(o.protocol)
	if a == nil {
		return fmt.Errorf("unknown protocol %q", o.protocol)
	}
	keep := func(f *net.Frame) bool { return a.Match(f) }

	captures, err := discoverIoTCaptures(o.dir)
	if err != nil {
		return err
	}
	if len(captures) == 0 {
		return fmt.Errorf("no IoT-23 captures found under %q", o.dir)
	}
	fmt.Printf("captures: %d\n", len(captures))

	var allSessions []*session.Session
	var captureOf []string
	for _, cap := range captures {
		var capSessions []*session.Session
		for _, pc := range cap.pcaps {
			ss, err := pipeline.ReadSessionsOpt(pc, pipeline.ReadOptions{
				MaxFrames:  o.frames,
				MaxPackets: o.maxPackets,
				Keep:       keep,
			})
			if err != nil {
				fmt.Fprintf(os.Stderr, "  skip %s: %v\n", pc, err)
				continue
			}
			capSessions = append(capSessions, ss...)
		}
		matched := 0
		if cap.zeek != "" {
			if matched, err = session.JoinZeekFile(capSessions, cap.zeek); err != nil {
				return fmt.Errorf("%s: %w", cap.name, err)
			}
		}
		labelled, mal, ben := 0, 0, 0
		for _, s := range capSessions {
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
				captureOf = append(captureOf, cap.name)
			}
		}
		fmt.Printf("%-40s pcaps=%-3d sessions=%-6d zeek_matched=%-6d labelled=%-6d malicious=%-5d benign=%d\n",
			cap.name, len(cap.pcaps), len(capSessions), matched, labelled, mal, ben)
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

	rng := rand.New(rand.NewPCG(o.seed, o.seed+1))
	var trainIdx, testIdx []int
	hold := splitNonEmpty(o.holdout, ",")
	if len(hold) > 0 {
		if len(captureOf) != len(x) {
			return fmt.Errorf("capture tracking misaligned (%d vs %d)", len(captureOf), len(x))
		}
		for i := range x {
			if matchesAny(captureOf[i], hold) {
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

// iotCapture is one IoT-23 capture directory.
type iotCapture struct {
	name  string
	dir   string
	pcaps []string
	zeek  string
}

// discoverIoTCaptures finds every capture under root. A capture is a directory
// holding a Zeek conn.log.labeled (possibly under a "bro" subdirectory) and one
// or more PCAPs (possibly in a subdirectory of the capture root).
func discoverIoTCaptures(root string) ([]iotCapture, error) {
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

	byDir := map[string]*iotCapture{}
	var order []string
	for _, log := range connLogs {
		dir := filepath.Dir(log)
		if filepath.Base(dir) == "bro" {
			dir = filepath.Dir(dir)
		}
		c, ok := byDir[dir]
		if !ok {
			c = &iotCapture{name: filepath.Base(dir), dir: dir, zeek: log}
			byDir[dir] = c
			order = append(order, dir)
		}
	}
	prefix := func(dir string) string { return dir + string(os.PathSeparator) }
	for _, dir := range order {
		c := byDir[dir]
		for _, p := range pcaps {
			if strings.HasPrefix(p, prefix(dir)) {
				c.pcaps = append(c.pcaps, p)
			}
		}
	}

	var out []iotCapture
	for _, dir := range order {
		c := byDir[dir]
		if len(c.pcaps) == 0 {
			fmt.Fprintf(os.Stderr, "warn: capture %s has no pcap\n", c.name)
			continue
		}
		sort.Strings(c.pcaps)
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out, nil
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

func writeReport(o options, train, test classifiers.Metrics, n, malicious int) error {
	rep := map[string]any{
		"dataset":   "iot-23",
		"protocol":  o.protocol,
		"arch":      o.arch,
		"frames":    o.frames,
		"bytes":     o.bytesPer,
		"samples":   n,
		"malicious": malicious,
		"test_frac": o.testFrac,
		"seed":      o.seed,
		"holdout":   o.holdout,
		"train":     train,
		"test":      test,
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(o.report, data, 0o644)
}
