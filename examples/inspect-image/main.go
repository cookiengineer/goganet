// Command inspect-image renders the first session of a capture and prints its
// image statistics. It is a debugging aid for the adapter/image pipeline.
//
//	go run ./examples/inspect-image -pcap datasets/ctu-13/4/botnet-capture-20110815-rbot-dos.pcap
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/cookiengineer/goganet/adapter"
	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/pipeline"
)

func main() {
	path := flag.String("pcap", "", "capture file")
	protocol := flag.String("protocol", "", "protocol adapter (default: auto-detect per frame)")
	frames := flag.Int("frames", 32, "frames per image")
	bytesPer := flag.Int("bytes", 1480, "bytes per frame")
	bitplanes := flag.Bool("bitplanes", false, "include bit-plane channels")
	flag.Parse()

	if *path == "" {
		fmt.Fprintln(os.Stderr, "usage: inspect-image -pcap <file>")
		os.Exit(2)
	}
	sessions, err := pipeline.ReadSessions(*path, *frames)
	if err != nil {
		fmt.Fprintln(os.Stderr, "inspect-image:", err)
		os.Exit(1)
	}
	fmt.Printf("sessions: %d\n", len(sessions))
	if len(sessions) == 0 {
		return
	}

	s := sessions[0]
	a := adapter.ByName(*protocol)
	if *protocol == "" {
		a = adapter.For(s.Frames[0])
	}
	cfg := image.Config{Frames: *frames, BytesPerFrame: *bytesPer, BitPlanes: *bitplanes}
	img := pipeline.Render(s, a, cfg)

	name := "raw"
	if a != nil {
		name = a.Name()
	}
	fmt.Printf("session frames: %d, label: %q malicious=%v\n", len(s.Frames), s.Label, s.Malicious)
	fmt.Printf("adapter: %s\n", name)
	fmt.Printf("image: %d frames x %d bytes x %d channels (%d values)\n", img.Frames, img.Bytes, img.Channels, len(img.Data))

	present := 0
	for f := 0; f < img.Frames; f++ {
		if img.At(f, 0, 1) == 1 {
			present++
		}
	}
	fmt.Printf("present frame rows: %d/%d\n", present, img.Frames)
}
