package session

import (
	"net/netip"
	"testing"
	"time"

	"github.com/cookiengineer/goganet/adapter/net"
)

func mkFrame(proto uint8, src, dst string, sport, dport uint16, ts int64, payload int) *net.Frame {
	return &net.Frame{
		Time:     time.Unix(ts, 0),
		Src:      netip.MustParseAddr(src),
		Dst:      netip.MustParseAddr(dst),
		SrcPort:  sport,
		DstPort:  dport,
		Protocol: proto,
		Payload:  make([]byte, payload),
	}
}

func TestGroupBidirectional(t *testing.T) {
	fwd := mkFrame(net.ProtoTCP, "10.0.0.1", "10.0.0.2", 1000, 80, 10, 5)
	rev := mkFrame(net.ProtoTCP, "10.0.0.2", "10.0.0.1", 80, 1000, 11, 7)
	other := mkFrame(net.ProtoUDP, "10.0.0.1", "10.0.0.2", 1000, 80, 12, 3)

	got := Group([]*net.Frame{fwd, rev, other}, 8)
	if len(got) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(got))
	}
	if len(got[0].Frames) != 2 {
		t.Fatalf("forward/reverse not grouped: %d", len(got[0].Frames))
	}
	if got[0].Start.Unix() != 10 || got[0].End.Unix() != 11 {
		t.Fatalf("session window = %v..%v", got[0].Start, got[0].End)
	}
	if got[0].Bytes != 12 {
		t.Fatalf("bytes = %d, want 12", got[0].Bytes)
	}
}

func TestGrouperCapAndBytes(t *testing.T) {
	g := NewGrouper(2)
	for i := 0; i < 5; i++ {
		g.Add(mkFrame(net.ProtoUDP, "10.0.0.1", "10.0.0.2", 1, 2, int64(i), 10))
	}
	s := g.Sessions()
	if len(s) != 1 {
		t.Fatalf("sessions = %d", len(s))
	}
	if len(s[0].Frames) != 2 {
		t.Fatalf("frames retained = %d, want 2", len(s[0].Frames))
	}
	if s[0].Bytes != 50 {
		t.Fatalf("bytes = %d, want 50 (all frames counted)", s[0].Bytes)
	}
}

func TestGrouperNilAndDefault(t *testing.T) {
	g := NewGrouper(0) // default 32
	g.Add(nil)
	for i := 0; i < 40; i++ {
		g.Add(mkFrame(net.ProtoUDP, "10.0.0.1", "10.0.0.2", 1, 2, int64(i), 1))
	}
	s := g.Sessions()
	if len(s[0].Frames) != 32 {
		t.Fatalf("default cap = %d, want 32", len(s[0].Frames))
	}
}

func TestGroupFirstSeenOrder(t *testing.T) {
	a := mkFrame(net.ProtoUDP, "10.0.0.1", "10.0.0.2", 1, 2, 100, 1)
	b := mkFrame(net.ProtoUDP, "10.0.0.3", "10.0.0.4", 3, 4, 50, 1)
	got := Group([]*net.Frame{a, b}, 4)
	if len(got) != 2 || got[0].Key != a.Key() || got[1].Key != b.Key() {
		t.Fatal("first-seen order not preserved")
	}
}
