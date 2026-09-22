package net

import (
	"encoding/binary"
	"net/netip"
	"testing"
	"time"
)

// buildUDPFrame builds an Ethernet/IPv4/UDP frame with the given ports and
// payload.
func buildUDPFrame(src, dst string, sport, dport uint16, payload []byte) []byte {
	frame := make([]byte, 14+20+8+len(payload))
	copy(frame[0:6], []byte{0, 1, 2, 3, 4, 5})
	copy(frame[6:12], []byte{6, 7, 8, 9, 10, 11})
	binary.BigEndian.PutUint16(frame[12:14], 0x0800)

	ip := frame[14:]
	ip[0] = 0x45
	binary.BigEndian.PutUint16(ip[2:4], uint16(20+8+len(payload)))
	ip[9] = ProtoUDP
	srcAddr := netip.MustParseAddr(src).As4()
	dstAddr := netip.MustParseAddr(dst).As4()
	copy(ip[12:16], srcAddr[:])
	copy(ip[16:20], dstAddr[:])

	udp := ip[20:]
	binary.BigEndian.PutUint16(udp[0:2], sport)
	binary.BigEndian.PutUint16(udp[2:4], dport)
	binary.BigEndian.PutUint16(udp[4:6], uint16(8+len(payload)))
	copy(udp[8:], payload)
	return frame
}

func TestDecodeUDP(t *testing.T) {
	frame := buildUDPFrame("10.0.0.1", "10.0.0.2", 12345, 53, []byte("dnsdata"))
	f, err := Decode(linkEthernet, frame, time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoUDP {
		t.Fatalf("proto %d", f.Protocol)
	}
	if f.SrcPort != 12345 || f.DstPort != 53 {
		t.Fatalf("ports %d -> %d", f.SrcPort, f.DstPort)
	}
	if f.Src != netip.MustParseAddr("10.0.0.1") {
		t.Fatalf("src %v", f.Src)
	}
	if string(f.Payload) != "dnsdata" {
		t.Fatalf("payload %q", f.Payload)
	}
}

func TestSessionKeySymmetric(t *testing.T) {
	a := buildUDPFrame("10.0.0.1", "10.0.0.2", 1000, 53, nil)
	b := buildUDPFrame("10.0.0.2", "10.0.0.1", 53, 1000, nil)
	fa, _ := Decode(linkEthernet, a, time.Now())
	fb, _ := Decode(linkEthernet, b, time.Now())
	if fa.Key() != fb.Key() {
		t.Fatalf("keys differ: %v vs %v", fa.Key(), fb.Key())
	}
}
