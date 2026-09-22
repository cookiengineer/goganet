// Package pipeline ties capture reading, frame decoding, session grouping,
// protocol adaptation and image rendering together into the end-to-end
// GoGANet data path.
package pipeline

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cookiengineer/goganet/adapter"
	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/net"
	"github.com/cookiengineer/goganet/adapter/pcap"
	"github.com/cookiengineer/goganet/adapter/session"
)

// ReadSessions reads a capture file and groups its decodable frames into
// bidirectional sessions. Frames are grouped in a streaming fashion so that
// large captures do not need to be held in memory.
func ReadSessions(path string, maxFrames int) ([]*session.Session, error) {
	return ReadSessionsFiltered(path, maxFrames, nil)
}

// ReadSessionsFiltered is ReadSessions with an optional frame filter applied
// before grouping. Returning false for a frame drops it.
func ReadSessionsFiltered(path string, maxFrames int, keep func(*net.Frame) bool) ([]*session.Session, error) {
	return ReadSessionsOpt(path, ReadOptions{MaxFrames: maxFrames, Keep: keep})
}

// ReadOptions controls bounded streaming reads.
type ReadOptions struct {
	MaxFrames  int // frames retained per session
	MaxPackets int // stop after this many packets (0 = unlimited)
	Keep       func(*net.Frame) bool
}

// ReadSessionsOpt reads a capture with explicit bounds. It is suitable for very
// large captures because packets are streamed and grouped without retaining the
// whole capture.
func ReadSessionsOpt(path string, opt ReadOptions) ([]*session.Session, error) {
	r, err := pcap.Open(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	g := session.NewGrouper(opt.MaxFrames)
	read := 0
	for {
		if opt.MaxPackets > 0 && read >= opt.MaxPackets {
			break
		}
		p, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		read++
		f, derr := net.Decode(p.Link, p.Data, p.Time)
		if derr != nil {
			continue
		}
		if opt.Keep != nil && !opt.Keep(f) {
			continue
		}
		g.Add(f)
	}
	return g.Sessions(), nil
}

// Render renders one session using the given protocol adapter.
func Render(s *session.Session, a adapter.Adapter, cfg image.Config) *image.Image {
	var bytesOf func(*net.Frame) []byte
	if a != nil {
		bytesOf = a.Bytes
	}
	return image.RenderNet(cfg, s.Frames, bytesOf)
}

// Dataset renders every session belonging to the requested protocol into
// flattened images and class indices. The class index is 0 for benign and 1 for
// malicious. Sessions without a matching adapter are skipped. The images use
// the interleaved layout expected by the dense architecture.
func Dataset(sessions []*session.Session, protocol string, cfg image.Config) (x [][]float32, y []int) {
	return dataset(sessions, protocol, cfg, false)
}

// DatasetPlanar is like Dataset but returns the planar layout expected by the
// convolutional architecture.
func DatasetPlanar(sessions []*session.Session, protocol string, cfg image.Config) (x [][]float32, y []int) {
	return dataset(sessions, protocol, cfg, true)
}

func dataset(sessions []*session.Session, protocol string, cfg image.Config, planar bool) (x [][]float32, y []int) {
	a := adapter.ByName(protocol)
	if a == nil {
		return nil, nil
	}
	for _, s := range sessions {
		if len(s.Frames) == 0 || !a.Match(s.Frames[0]) {
			continue
		}
		img := Render(s, a, cfg)
		if planar {
			x = append(x, img.Planar())
		} else {
			x = append(x, img.Data)
		}
		if s.Malicious {
			y = append(y, 1)
		} else {
			y = append(y, 0)
		}
	}
	return x, y
}

// FindCapture returns the first .pcap/.pcapng/.cap file in a directory.
func FindCapture(dir string) (string, error) {
	return findWithExt(dir, []string{".pcap", ".pcapng", ".cap"})
}

// FindBinetflow returns the first .binetflow file in a directory.
func FindBinetflow(dir string) (string, error) {
	return findWithExt(dir, []string{".binetflow"})
}

func findWithExt(dir string, exts []string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		for _, ext := range exts {
			if strings.HasSuffix(strings.ToLower(e.Name()), ext) {
				return filepath.Join(dir, e.Name()), nil
			}
		}
	}
	return "", errors.New("pipeline: no matching file in " + dir)
}
