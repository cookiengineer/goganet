package pipeline

import (
	"bytes"
	"encoding/binary"
	"net/netip"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/goganet/adapter"
	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/net"
)

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

func writePCAP(t *testing.T, frames [][]byte) string {
	t.Helper()
	var b bytes.Buffer
	le := binary.LittleEndian
	w := func(v uint32) { binary.Write(&b, le, v) }
	wh := func(v uint16) { binary.Write(&b, le, v) }
	w(0xa1b2c3d4)
	wh(2)
	wh(4)
	w(0)
	w(0)
	w(65535)
	w(1)
	for i, f := range frames {
		w(uint32(100 + i))
		w(0)
		w(uint32(len(f)))
		w(uint32(len(f)))
		b.Write(f)
	}
	path := filepath.Join(t.TempDir(), "capture.pcap")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestPipelineEndToEnd(t *testing.T) {
	frames := [][]byte{
		udpFrame("10.0.0.1", "8.8.8.8", 40000, 53, []byte("aaaa")),
		udpFrame("10.0.0.1", "8.8.8.8", 40000, 53, []byte("bbbb")),
		udpFrame("192.168.1.5", "1.2.3.4", 50000, 53, []byte("cccc")),
	}
	path := writePCAP(t, frames)
	if _, err := FindCapture(filepath.Dir(path)); err != nil {
		t.Fatalf("FindCapture: %v", err)
	}

	sessions, err := ReadSessions(path, 8)
	if err != nil {
		t.Fatal(err)
	}
	if len(sessions) != 2 {
		t.Fatalf("sessions = %d, want 2", len(sessions))
	}

	// MaxPackets cap: only the first packet is read.
	limited, err := ReadSessionsOpt(path, ReadOptions{MaxFrames: 8, MaxPackets: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(limited) != 1 || len(limited[0].Frames) != 1 {
		t.Fatalf("limited sessions = %d frames=%d", len(limited), len(limited[0].Frames))
	}

	// Keep filter: only packets with destination port 53 survive (all do here).
	kept, err := ReadSessionsFiltered(path, 8, func(f *net.Frame) bool { return f.DstPort == 53 || f.SrcPort == 53 })
	if err != nil || len(kept) != 2 {
		t.Fatalf("filtered = %d err=%v", len(kept), err)
	}
	dropped, err := ReadSessionsFiltered(path, 8, func(f *net.Frame) bool { return false })
	if err != nil || len(dropped) != 0 {
		t.Fatalf("dropped = %d err=%v", len(dropped), err)
	}

	// Render the first session with the DNS adapter.
	a := adapter.For(sessions[0].Frames[0])
	if a == nil || a.Name() != "dns" {
		t.Fatalf("adapter = %v", a)
	}
	cfg := image.Config{Frames: 4, BytesPerFrame: 16}
	img := Render(sessions[0], a, cfg)
	if img.At(0, 0, 1) != 1 || img.At(2, 0, 1) != 0 {
		t.Fatal("render presence wrong")
	}

	// Dataset labels: mark one benign and one malicious.
	sessions[0].Labeled, sessions[0].Malicious = true, false
	sessions[1].Labeled, sessions[1].Malicious = true, true
	x, y := Dataset(sessions, "dns", cfg)
	if len(x) != 2 || len(y) != 2 {
		t.Fatalf("dataset = %d,%d", len(x), len(y))
	}
	if y[0] != 0 || y[1] != 1 {
		t.Fatalf("labels = %v", y)
	}
	xp, yp := DatasetPlanar(sessions, "dns", cfg)
	if len(xp) != 2 || len(xp[0]) != cfg.Len() || len(yp) != 2 {
		t.Fatalf("planar dataset = %d len=%d", len(xp), len(xp[0]))
	}
	// Unknown protocol yields no data.
	if x, y := Dataset(sessions, "nope", cfg); x != nil || y != nil {
		t.Fatal("unknown protocol should return nil")
	}
}

func TestFindMissing(t *testing.T) {
	if _, err := FindCapture(t.TempDir()); err == nil {
		t.Fatal("expected missing capture error")
	}
	if _, err := FindBinetflow(t.TempDir()); err == nil {
		t.Fatal("expected missing binetflow error")
	}
}
