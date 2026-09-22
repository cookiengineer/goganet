package adapter

import (
	"encoding/binary"
	"net/netip"
	"testing"
	"time"

	"github.com/cookiengineer/goganet/adapter/image"
	"github.com/cookiengineer/goganet/adapter/net"
)

func frameProto(proto uint8, sport, dport uint16, payload []byte) *net.Frame {
	return &net.Frame{
		Time:     time.Unix(1, 0),
		Src:      netip.MustParseAddr("10.0.0.1"),
		Dst:      netip.MustParseAddr("10.0.0.2"),
		SrcPort:  sport,
		DstPort:  dport,
		Protocol: proto,
		Payload:  payload,
	}
}

func TestAdapterMatch(t *testing.T) {
	cases := []struct {
		name string
		f    *net.Frame
		want string
	}{
		{"dns", frameProto(net.ProtoUDP, 50000, 53, []byte{0, 1}), "dns"},
		{"icmp", frameProto(net.ProtoICMP, 0, 0, []byte{8, 0}), "icmp"},
		{"snmp", frameProto(net.ProtoUDP, 50000, 161, []byte{0x30}), "snmp"},
		{"http1", frameProto(net.ProtoTCP, 50000, 80, []byte("GET / HTTP/1.1\r\n")), "http1"},
		{"http2", frameProto(net.ProtoTCP, 50000, 8080, []byte("PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")), "http2"},
		{"tls", frameProto(net.ProtoTCP, 50000, 443, []byte{0x16, 0x03, 0x01, 0x00}), "tls"},
	}
	for _, tc := range cases {
		got := For(tc.f)
		if got == nil || got.Name() != tc.want {
			t.Errorf("%s: got %v, want %s", tc.name, got, tc.want)
		}
	}
}

func TestImageRender(t *testing.T) {
	cfg := image.Config{Frames: 4, BytesPerFrame: 8}
	frames := [][]byte{
		[]byte("abcdefgh"),
		nil,
		[]byte("ij"),
	}
	im := image.Render(cfg, frames)
	if im.Frames != 4 || im.Bytes != 8 || im.Channels != 2 {
		t.Fatalf("shape %dx%dx%d", im.Frames, im.Bytes, im.Channels)
	}
	if im.At(0, 0, 1) != 1 {
		t.Fatalf("frame 0 presence = %v", im.At(0, 0, 1))
	}
	if im.At(1, 0, 1) != 0 {
		t.Fatalf("frame 1 presence = %v", im.At(1, 0, 1))
	}
	if im.At(2, 1, 1) != 1 {
		t.Fatalf("frame 2 presence = %v", im.At(2, 1, 1))
	}
	want := float32('a') / 255
	if im.At(0, 0, 0) != want {
		t.Fatalf("intensity = %v want %v", im.At(0, 0, 0), want)
	}
}

func TestRenderNetDNS(t *testing.T) {
	// Build a DNS frame via net.Frame directly.
	f := frameProto(net.ProtoUDP, 50000, 53, []byte{0x12, 0x34, 0x01, 0x00})
	a := For(f)
	if a == nil || a.Name() != "dns" {
		t.Fatalf("adapter %v", a)
	}
	im := image.RenderNet(image.Config{Frames: 32, BytesPerFrame: 1480}, []*net.Frame{f}, a.Bytes)
	if im.Frames != 32 {
		t.Fatalf("frames %d", im.Frames)
	}
	// The DNS id bytes should appear at the start of the first row.
	if got := im.At(0, 0, 0); got != float32(0x12)/255 {
		t.Fatalf("first byte %v", got)
	}
	if im.At(1, 0, 1) != 0 {
		t.Fatalf("second frame should be padding")
	}
}

var _ = binary.BigEndian
