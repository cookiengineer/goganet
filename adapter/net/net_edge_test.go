package net

import (
	"testing"
	"time"
)

func TestDecodeIPv6FragmentHeader(t *testing.T) {
	frag := make([]byte, 8)
	frag[0] = ProtoTCP
	frag = append(frag, tcp(1234, 80, 0x10, []byte("z"))...)
	frame := eth(0x86dd, ipv6(44, "2001:db8::1", "2001:db8::2", frag))
	f, err := Decode(linkEthernetTest, frame, time.Time{})
	if err != nil {
		t.Fatal(err)
	}
	if f.Protocol != ProtoTCP || f.DstPort != 80 || string(f.Payload) != "z" {
		t.Fatalf("ipv6 fragment = %+v", f)
	}
}

func TestDecodeIPv6Truncated(t *testing.T) {
	// IPv6 header claims TCP but no transport bytes follow.
	frame := eth(0x86dd, ipv6(ProtoTCP, "2001:db8::1", "2001:db8::2", nil))
	if _, err := Decode(linkEthernetTest, frame, time.Time{}); err == nil {
		t.Fatal("expected short tcp header error")
	}
	// Truncated IPv6 header itself.
	if _, err := Decode(linkRawTest, []byte{0x60, 0, 0, 0}, time.Time{}); err == nil {
		t.Fatal("expected short ipv6 header error")
	}
}

func TestDecodeShortCookedHeaders(t *testing.T) {
	if _, err := Decode(linkSLLTest, []byte{1, 2, 3}, time.Time{}); err == nil {
		t.Fatal("expected short sll error")
	}
	if _, err := Decode(linkSLL2Test, []byte{1, 2, 3}, time.Time{}); err == nil {
		t.Fatal("expected short sll2 error")
	}
}

func TestDecodeIPv4UDPTruncated(t *testing.T) {
	// IPv4 with proto UDP but only two bytes of transport.
	h := ipv4(ProtoUDP, "10.0.0.1", "10.0.0.2", 0, []byte{0, 1})
	frame := eth(0x0800, h)
	if _, err := Decode(linkEthernetTest, frame, time.Time{}); err == nil {
		t.Fatal("expected short udp header error")
	}
}
