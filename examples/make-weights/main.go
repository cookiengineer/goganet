// Command make-weights trains a small classifier on synthetic separable data
// and writes it to the classifiers weights directory. It is both a minimal
// example of the training API and the generator for the embedded self-test
// fixture used by the classifiers package.
//
//	go run ./examples/make-weights -proto selftest -arch conv
//
// Real classifiers are produced by train-ctu13 from labelled captures.
package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"os"

	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/classifiers"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

func main() {
	proto := flag.String("proto", "selftest", "protocol / weight file name")
	arch := flag.String("arch", "conv", "architecture: dense or conv")
	frames := flag.Int("frames", 8, "frames per image")
	bytesPer := flag.Int("bytes", 64, "bytes per frame")
	steps := flag.Int("steps", 400, "training steps")
	out := flag.String("out", "", "output path (default classifiers/weights/<proto>.ggnt)")
	flag.Parse()

	if *out == "" {
		*out = "classifiers/weights/" + *proto + ".ggnt"
	}

	shape := image.Config{Frames: *frames, BytesPerFrame: *bytesPer}
	channels := shape.Channels()

	// Synthetic two-class planar data: class 0 is positive, class 1 negative.
	const perClass = 4
	batch := perClass * 2
	rows := shape.Len()
	x := make([]float32, rows*batch)
	y := make([]int, batch)
	rng := rand.New(rand.NewPCG(123, 456))
	for b := 0; b < batch; b++ {
		cls := b % 2
		y[b] = cls
		sign := float32(1)
		if cls == 1 {
			sign = -1
		}
		// Build one sample in planar layout (channels, frames*bytes).
		sample := make([]float32, rows)
		for c := 0; c < channels; c++ {
			for s := 0; s < shape.Frames*shape.BytesPerFrame; s++ {
				sample[c*shape.Frames*shape.BytesPerFrame+s] = sign * (0.4 + 0.1*float32(c) + 0.01*float32(s%5))
			}
		}
		for i := 0; i < rows; i++ {
			x[i*batch+b] = sample[i]
		}
		_ = rng
	}

	cfg := wcgan.Config{
		InputDim:      rows,
		LatentDim:     8,
		NumClasses:    2,
		Frames:        shape.Frames,
		BytesPerFrame: shape.BytesPerFrame,
		NCritic:       1,
		Lambda:        1,
		GenLR:         5e-4,
		CritLR:        5e-4,
	}
	if *arch == "conv" {
		cfg.Arch = "conv"
		cfg.InputChannels = channels
		cfg.SeedChannels = 16
		cfg.GenChannels = []int{8, 4}
		cfg.CritChannels = []int{8, 16}
		cfg.Kernel = 3
	} else {
		cfg.GenHidden = []int{64, 32}
		cfg.CritHidden = []int{64, 32}
	}

	model := wcgan.New(cfg)
	labels := wcgan.OneHot(y, 2)
	for step := 0; step < *steps; step++ {
		model.CriticStep(x, labels, batch)
		model.GeneratorStep(batch)
	}

	correct := 0
	for b := 0; b < batch; b++ {
		sample := make([]float32, rows)
		for i := 0; i < rows; i++ {
			sample[i] = x[i*batch+b]
		}
		probs, _ := model.Classify(sample, 1)
		pred := 0
		if probs[1] > 0.5 {
			pred = 1
		}
		if pred == y[b] {
			correct++
		}
	}
	fmt.Printf("synthetic accuracy: %.2f\n", float64(correct)/float64(batch))

	if err := classifiers.Save(*out, model); err != nil {
		fmt.Fprintln(os.Stderr, "make-weights:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s (arch=%s %dx%d channels=%d input=%d)\n", *out, *arch, shape.Frames, shape.BytesPerFrame, channels, rows)
}
