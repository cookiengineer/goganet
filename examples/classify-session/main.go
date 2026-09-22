// Command classify-session classifies each protocol session in a capture as
// malicious or non-malicious using an embedded or file-backed classifier.
//
//	go run ./examples/classify-session -pcap capture.pcap -protocol dns
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cookiengineer/goganet/adapter"
	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/pipeline"
	"github.com/cookiengineer/goganet/classifiers"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

func main() {
	path := flag.String("pcap", "", "capture file")
	protocol := flag.String("protocol", "dns", "protocol adapter / classifier to use")
	modelPath := flag.String("model", "", "optional model file (default: embedded weights)")
	frames := flag.Int("frames", 32, "frames per image")
	flag.Parse()

	if *path == "" {
		fmt.Fprintln(os.Stderr, "usage: classify-session -pcap <file> -protocol dns")
		os.Exit(2)
	}

	var clf *classifiers.Classifier
	if *modelPath != "" {
		f, err := os.Open(*modelPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "classify-session:", err)
			os.Exit(1)
		}
		defer f.Close()
		model, err := wcgan.Load(f)
		if err != nil {
			fmt.Fprintln(os.Stderr, "classify-session:", err)
			os.Exit(1)
		}
		cfg := model.Config()
		shape := image.Config{Frames: cfg.Frames, BytesPerFrame: cfg.BytesPerFrame, BitPlanes: cfg.BitPlanes}
		clf = classifiers.Wrap(*protocol, model, shape)
	} else {
		var err error
		clf, err = classifiers.New(*protocol)
		if err != nil {
			fmt.Fprintln(os.Stderr, "classify-session:", err)
			os.Exit(1)
		}
	}

	a := adapter.ByName(*protocol)
	if a == nil {
		fmt.Fprintf(os.Stderr, "classify-session: unknown protocol %q\n", *protocol)
		os.Exit(1)
	}

	sessions, err := pipeline.ReadSessions(*path, *frames)
	if err != nil {
		fmt.Fprintln(os.Stderr, "classify-session:", err)
		os.Exit(1)
	}

	shape := clf.Shape()
	var malicious int
	classified := 0
	for _, s := range sessions {
		if len(s.Frames) == 0 || !a.Match(s.Frames[0]) {
			continue
		}
		img := pipeline.Render(s, a, shape)
		res, err := clf.Classify(img)
		if err != nil {
			fmt.Fprintln(os.Stderr, "classify-session:", err)
			os.Exit(1)
		}
		classified++
		if res.Malicious {
			malicious++
		}
		fmt.Printf("%-20s p(malicious)=%.3f score=%+.3f -> %s\n", s.Key.A, res.Probability, res.Score, verdict(res.Malicious))
		if classified >= 20 {
			break
		}
	}
	fmt.Printf("classified %d %s sessions, %d flagged malicious\n", classified, *protocol, malicious)
}

func verdict(mal bool) string {
	if mal {
		return "MALICIOUS"
	}
	return "benign"
}
