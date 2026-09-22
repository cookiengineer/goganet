// Package benchmarks contains end-to-end integration tests that exercise the
// whole GoGANet data path: capture parsing, frame decoding, session grouping,
// label joining, image rendering, WGAN-GP training and classification.
package benchmarks

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net/netip"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/pipeline"
	"github.com/cookiengineer/goganet/adapter/session"
	"github.com/cookiengineer/goganet/engine/wcgan"
)

// pcapBuilder accumulates classic pcap packets.
type pcapBuilder struct {
	buf bytes.Buffer
	ts  uint32
}

func newPCAPBuilder() *pcapBuilder {
	b := &pcapBuilder{}
	le := binary.LittleEndian
	write := func(v uint32) { binary.Write(&b.buf, le, v) }
	write16 := func(v uint16) { binary.Write(&b.buf, le, v) }
	write(0xa1b2c3d4)
	write16(2)
	write16(4)
	write(0)
	write(0)
	write(65535)
	write(1) // ethernet
	return b
}

// udpFrame builds an Ethernet/IPv4/UDP frame.
func udpFrame(src, dst string, sport, dport uint16, payload []byte) []byte {
	frame := make([]byte, 14+20+8+len(payload))
	copy(frame[0:6], []byte{0, 1, 2, 3, 4, 5})
	copy(frame[6:12], []byte{6, 7, 8, 9, 10, 11})
	binary.BigEndian.PutUint16(frame[12:14], 0x0800)
	ip := frame[14:]
	ip[0] = 0x45
	binary.BigEndian.PutUint16(ip[2:4], uint16(20+8+len(payload)))
	ip[9] = 17
	a := netip.MustParseAddr(src).As4()
	d := netip.MustParseAddr(dst).As4()
	copy(ip[12:16], a[:])
	copy(ip[16:20], d[:])
	udp := ip[20:]
	binary.BigEndian.PutUint16(udp[0:2], sport)
	binary.BigEndian.PutUint16(udp[2:4], dport)
	binary.BigEndian.PutUint16(udp[4:6], uint16(8+len(payload)))
	copy(udp[8:], payload)
	return frame
}

func (b *pcapBuilder) add(frame []byte, ts time.Time) {
	b.ts = uint32(ts.Unix())
	le := binary.LittleEndian
	binary.Write(&b.buf, le, b.ts)
	binary.Write(&b.buf, le, uint32(ts.Nanosecond()/1000))
	binary.Write(&b.buf, le, uint32(len(frame)))
	binary.Write(&b.buf, le, uint32(len(frame)))
	b.buf.Write(frame)
}

// buildDataset writes a labelled synthetic capture and binetflow to dir.
func buildDataset(t *testing.T, dir string) {
	t.Helper()
	b := newPCAPBuilder()
	base := time.Date(2011, 8, 15, 12, 0, 0, 0, time.UTC)

	type flow struct {
		src, dst     string
		sport, dport uint16
		payload      []byte
		label        string
	}
	benign := []byte{0x10, 0x20, 0x30, 0x40, 0x50}
	malicious := []byte{0xf0, 0xe0, 0xd0, 0xc0, 0xb0}
	flows := []flow{
		{"10.0.0.1", "8.8.8.8", 40000, 53, benign, "flow=Normal"},
		{"10.0.0.2", "8.8.4.4", 40001, 53, benign, "flow=Background"},
		{"192.168.1.100", "1.2.3.4", 50000, 53, malicious, "flow=From-Botnet-V42-TCP"},
		{"192.168.1.101", "5.6.7.8", 50001, 53, malicious, "flow=From-Botnet-V42-UDP"},
	}
	for i, f := range flows {
		for frame := 0; frame < 4; frame++ {
			p := append([]byte(fmt.Sprintf("dns%02d", frame)), f.payload...)
			b.add(udpFrame(f.src, f.dst, f.sport, f.dport, p), base.Add(time.Duration(i*60+frame)*time.Second))
		}
	}

	if err := os.WriteFile(filepath.Join(dir, "capture.pcap"), b.buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	var bf bytes.Buffer
	bf.WriteString("StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n")
	for i, f := range flows {
		start := base.Add(time.Duration(i*60) * time.Second).Format("2006-01-02 15:04:05.000000")
		fmt.Fprintf(&bf, "%s,10.0,udp,%s,%d,->,%s,%d,CON,0,0,4,100,50,%s\n",
			start, f.src, f.sport, f.dst, f.dport, f.label)
	}
	if err := os.WriteFile(filepath.Join(dir, "capture.binetflow"), bf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestEndToEndPipeline(t *testing.T) {
	dir := t.TempDir()
	buildDataset(t, dir)

	capture, err := pipeline.FindCapture(dir)
	if err != nil {
		t.Fatal(err)
	}
	sessions, err := pipeline.ReadSessions(capture, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(sessions) != 4 {
		t.Fatalf("expected 4 sessions, got %d", len(sessions))
	}

	bf, err := pipeline.FindBinetflow(dir)
	if err != nil {
		t.Fatal(err)
	}
	labels, err := session.LoadBinetflow(bf)
	if err != nil {
		t.Fatal(err)
	}
	if matched := session.JoinLabels(sessions, session.BuildLabelMap(labels)); matched != 4 {
		t.Fatalf("expected 4 labelled sessions, got %d", matched)
	}

	cfg := image.Config{Frames: 4, BytesPerFrame: 64}
	x, y := pipeline.Dataset(sessions, "dns", cfg)
	if len(x) != 4 {
		t.Fatalf("expected 4 dns samples, got %d", len(x))
	}
	if want := cfg.Len(); len(x[0]) != want {
		t.Fatalf("image length %d, want %d", len(x[0]), want)
	}
	var mal int
	for _, v := range y {
		mal += v
	}
	if mal != 2 {
		t.Fatalf("expected 2 malicious, got %d", mal)
	}
}

func TestTrainingConvergence(t *testing.T) {
	dir := t.TempDir()
	buildDataset(t, dir)

	capture, _ := pipeline.FindCapture(dir)
	sessions, err := pipeline.ReadSessions(capture, 8)
	if err != nil {
		t.Fatal(err)
	}
	bf, _ := pipeline.FindBinetflow(dir)
	labels, _ := session.LoadBinetflow(bf)
	session.JoinLabels(sessions, session.BuildLabelMap(labels))

	cfg := image.Config{Frames: 4, BytesPerFrame: 64}
	x, y := pipeline.Dataset(sessions, "dns", cfg)
	if len(x) < 4 {
		t.Fatalf("not enough samples: %d", len(x))
	}

	model := wcgan.New(wcgan.Config{
		InputDim:      cfg.Len(),
		LatentDim:     8,
		GenHidden:     []int{32},
		CritHidden:    []int{32},
		NumClasses:    2,
		Frames:        cfg.Frames,
		BytesPerFrame: cfg.BytesPerFrame,
		NCritic:       1,
		GenLR:         1e-3,
		CritLR:        1e-3,
	})

	batch := len(x)
	for epoch := 0; epoch < 120; epoch++ {
		xb := flatten(x, nil)
		dLoss := model.CriticStep(xb, wcgan.OneHot(y, 2), batch)
		gLoss := model.GeneratorStep(batch)
		if dLoss != dLoss || gLoss != gLoss { // NaN check
			t.Fatalf("NaN loss at epoch %d", epoch)
		}
	}

	correct := 0
	for i := range x {
		probs, _ := model.Classify(x[i], 1)
		pred := 0
		if probs[1] > 0.5 {
			pred = 1
		}
		if pred == y[i] {
			correct++
		}
	}
	acc := float64(correct) / float64(len(x))
	if acc < 0.8 {
		t.Fatalf("end-to-end accuracy too low: %.2f", acc)
	}
	t.Logf("end-to-end accuracy %.2f", acc)
}

func flatten(x [][]float32, _ []int) []float32 {
	d := len(x[0])
	out := make([]float32, d*len(x))
	for b := range x {
		for j, v := range x[b] {
			out[j*len(x)+b] = v
		}
	}
	return out
}
