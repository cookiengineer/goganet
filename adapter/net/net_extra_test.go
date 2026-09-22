package net

import (
	"encoding/binary"
	"net/netip"
	"testing"
	"time"
)

const (
	linkEthernetTest = 1
	linkRawTest      = 101
	linkSLLTest      = 113
	linkSLL2Test     = 276
)

func eth(etherType uint16, l3 []byte) []byte {
	frame := make([]byte, 14+len(l3))
	binary.BigEndian.PutUint16(frame[12:14], etherType)
	copy(frame[14:], l3)
	return frame
}

func ipv4(proto uint8, src, dst string, fragOff uint16, l4 []byte) []byte {
	h := make([]byte, 20)
	h[0] = 0x45
	binary.BigEndian.PutUint16(h[2:4], uint16(20+len(l4)))
	binary.BigEndian.PutUint16(h[6:8], fragOff)
	h[8] = 64
	h[9] = proto
	a := netip.MustParseAddr(src).As4()
	b := netip.MustParseAddr(dst).As4()
	copy(h[12:16], a[:])
	copy(h[16:20], b[:])
	return append(h, l4...)
}

func ipv6(next uint8, src, dst string, l4 []byte) []byte {
	h := make([]byte, 40)
	h[0] = 0x60
	h[6] = next
	h[7] = 64
	a := netip.MustParseAddr(src).As16()
	b := netip.MustParseAddr(dst).As16()
	copy(h[8:24], a[:])
	copy(h[24:40], b[:])
	return append(h, l4...)
}

func tcp(sport, dport uint16, flags uint8, payload []byte) []byte {
	h := make([]byte, 20)
	binary.BigEndian.PutUint16(h[0:2], sport)
	binary.BigEndian.PutUint16(h[2:4], dport)
	h[12] = 5 << 4 // data offset 5 words
	h[13] = flags
	return append(h, payload...)
}

func TestDecodeTCP(t *testing.T) {
	frame := eth(0x0800, ipv4(ProtoTCP, "10.0.0.1", "10.0.0.2", 0, tcp(40000, 80, 0x18, []byte("hello"))))
	f, err := Decode(linkEthernetTest, frame, time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoTCP || f.SrcPort != 40000 || f.DstPort != 80 {
		t.Fatalf("frame = %v", f)
	}
	if f.TCPFlags != 0x18 {
		t.Fatalf("flags = %#x", f.TCPFlags)
	}
	if string(f.Payload) != "hello" {
		t.Fatalf("payload = %q", f.Payload)
	}
}

func TestDecodeICMP(t *testing.T) {
	icmp := []byte{8, 0, 0, 0, 0, 0, 0, 0}
	frame := eth(0x0800, ipv4(ProtoICMP, "10.0.0.1", "10.0.0.2", 0, icmp))
	f, err := Decode(linkEthernetTest, frame, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoICMP || len(f.Payload) != len(icmp) {
		t.Fatalf("icmp frame = %v", f)
	}
}

func TestDecodeIPv4Fragment(t *testing.T) {
	// Non-zero fragment offset: transport is not parsed.
	frame := eth(0x0800, ipv4(ProtoTCP, "10.0.0.1", "10.0.0.2", 0x0020, []byte{1, 2, 3, 4}))
	f, err := Decode(linkEthernetTest, frame, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoTCP || len(f.Payload) != 0 {
		t.Fatalf("fragment should not parse transport: %+v", f)
	}
}

func TestDecodeIPv6TCP(t *testing.T) {
	frame := eth(0x86dd, ipv6(ProtoTCP, "2001:db8::1", "2001:db8::2", tcp(1234, 443, 0x02, []byte("x"))))
	f, err := Decode(linkEthernetTest, frame, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoTCP || f.DstPort != 443 || string(f.Payload) != "x" {
		t.Fatalf("ipv6 tcp = %+v", f)
	}
}

func TestDecodeIPv6ExtensionHeader(t *testing.T) {
	// Hop-by-hop header (next=0) whose payload points at TCP.
	ext := make([]byte, 8)
	ext[0] = ProtoTCP
	ext[1] = 0 // length (0+1)*8 bytes
	ext = append(ext, tcp(1234, 80, 0x10, []byte("y"))...)
	frame := eth(0x86dd, ipv6(0, "2001:db8::1", "2001:db8::2", ext))
	f, err := Decode(linkEthernetTest, frame, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoTCP || f.DstPort != 80 || string(f.Payload) != "y" {
		t.Fatalf("ipv6 ext tcp = %+v", f)
	}
}

func TestDecodeVLAN(t *testing.T) {
	l3 := ipv4(ProtoUDP, "10.0.0.1", "10.0.0.2", 0, []byte{0, 1, 0, 2, 0, 8, 0, 0})
	frame := make([]byte, 14+4+len(l3))
	copy(frame[0:6], []byte{0, 1, 2, 3, 4, 5})
	copy(frame[6:12], []byte{6, 7, 8, 9, 10, 11})
	binary.BigEndian.PutUint16(frame[12:14], 0x8100)
	binary.BigEndian.PutUint16(frame[14:16], 0x0064) // TCI / VLAN 100
	binary.BigEndian.PutUint16(frame[16:18], 0x0800)
	copy(frame[18:], l3)

	f, err := Decode(linkEthernetTest, frame, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoUDP {
		t.Fatalf("vlan frame = %+v", f)
	}
}

func TestDecodeRawLink(t *testing.T) {
	l3 := ipv4(ProtoUDP, "10.0.0.1", "10.0.0.2", 0, []byte{0, 1, 0, 2, 0, 8, 0, 0})
	f, err := Decode(linkRawTest, l3, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoUDP {
		t.Fatalf("raw = %+v", f)
	}
}

func TestDecodeLinuxSLL(t *testing.T) {
	l3 := ipv4(ProtoUDP, "10.0.0.1", "10.0.0.2", 0, []byte{0, 1, 0, 2, 0, 8, 0, 0})
	sll := append(make([]byte, 16), l3...)
	f, err := Decode(linkSLLTest, sll, time.Time{})
	if err != nil || f.Protocol != ProtoUDP {
		t.Fatalf("sll = %+v err=%v", f, err)
	}
	sll2 := append(make([]byte, 20), l3...)
	f, err = Decode(linkSLL2Test, sll2, time.Time{})
	if err != nil || f.Protocol != ProtoUDP {
		t.Fatalf("sll2 = %+v err=%v", f, err)
	}
}

func TestDecodeErrors(t *testing.T) {
	if _, err := Decode(999, []byte{0x45}, time.Time{}); err == nil {
		t.Fatal("expected unsupported link error")
	}
	if _, err := Decode(linkEthernetTest, []byte{1, 2, 3}, time.Time{}); err == nil {
		t.Fatal("expected short ethernet error")
	}
	// Ethernet carrying ARP is not IP.
	arp := eth(0x0806, make([]byte, 28))
	if _, err := Decode(linkEthernetTest, arp, time.Time{}); err == nil {
		t.Fatal("expected unsupported ethertype error")
	}
	// Truncated IP header.
	if _, err := Decode(linkRawTest, []byte{0x45, 0}, time.Time{}); err == nil {
		t.Fatal("expected short ipv4 error")
	}
	// Unknown IP version.
	if _, err := Decode(linkRawTest, []byte{0x00, 0, 0, 0}, time.Time{}); err == nil {
		t.Fatal("expected unknown version error")
	}
}

func TestIPv4TotalLengthClampsPayload(t *testing.T) {
	// Trailing Ethernet padding must be excluded by the IP total length.
	udp := []byte{0, 1, 0, 2, 0, 10, 0, 0, 'a', 'b'}
	base := ipv4(ProtoUDP, "10.0.0.1", "10.0.0.2", 0, udp)
	padded := append(base, make([]byte, 20)...) // padding beyond total length
	frame := eth(0x0800, padded)
	f, err := Decode(linkEthernetTest, frame, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if string(f.Payload) != "ab" {
		t.Fatalf("payload = %q, want ab", f.Payload)
	}
}
