package adapter

import (
	"bytes"
	"encoding/binary"
	"net/netip"
	"testing"
	"time"

	"github.com/cookiengineer/goganet/adapter/net"
)

// TestDNSUDP verifies that DNS over UDP is matched and rendered without any
// length prefix, in both directions.
func TestDNSUDP(t *testing.T) {
	msg := []byte{0x12, 0x34, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00}
	dns := ByName("dns")
	for _, f := range []*net.Frame{
		frameProto(net.ProtoUDP, 50000, 53, msg), // query
		frameProto(net.ProtoUDP, 53, 50000, msg), // response
	} {
		if !dns.Match(f) {
			t.Fatalf("dns did not match UDP %d->%d", f.SrcPort, f.DstPort)
		}
		if got := For(f); got == nil || got.Name() != "dns" {
			t.Fatalf("For() = %v, want dns", got)
		}
		if !bytes.Equal(dns.Bytes(f), msg) {
			t.Fatalf("dns bytes changed for UDP: %x", dns.Bytes(f))
		}
	}
}

// TestDNSTCP verifies that the two-byte length prefix of DNS over TCP is
// stripped, while the UDP form is left untouched.
func TestDNSTCP(t *testing.T) {
	msg := []byte{0xde, 0xad, 0xbe, 0xef}
	prefixed := append([]byte{0x00, 0x04}, msg...)
	f := frameProto(net.ProtoTCP, 40000, 53, prefixed)
	dns := ByName("dns")
	if !dns.Match(f) {
		t.Fatal("dns did not match TCP")
	}
	if got := dns.Bytes(f); !bytes.Equal(got, msg) {
		t.Fatalf("TCP prefix not stripped: %x", got)
	}
	// A truncated TCP payload must not panic.
	_ = dns.Bytes(frameProto(net.ProtoTCP, 40000, 53, []byte{0x01}))
}

func TestDNSNonMatch(t *testing.T) {
	dns := ByName("dns")
	if dns.Match(frameProto(net.ProtoUDP, 50000, 123, nil)) {
		t.Fatal("dns matched NTP port")
	}
	if dns.Match(frameProto(net.ProtoICMP, 0, 0, nil)) {
		t.Fatal("dns matched ICMP")
	}
}

func TestMDNS(t *testing.T) {
	f := frameProto(net.ProtoUDP, 5353, 5353, []byte{0, 0})
	if got := For(f); got == nil || got.Name() != "dns" {
		t.Fatalf("mDNS For() = %v, want dns", got)
	}
}

func TestICMPv4v6(t *testing.T) {
	if got := For(frameProto(net.ProtoICMP, 0, 0, []byte{8, 0, 0, 0})); got == nil || got.Name() != "icmp" {
		t.Fatalf("ICMPv4 For() = %v", got)
	}
	if got := For(frameProto(net.ProtoICMPv6, 0, 0, []byte{128, 0, 0, 0})); got == nil || got.Name() != "icmp" {
		t.Fatalf("ICMPv6 For() = %v", got)
	}
}

func TestSNMP(t *testing.T) {
	snmp := ByName("snmp")
	if !snmp.Match(frameProto(net.ProtoUDP, 40000, 161, []byte{0x30, 0x82})) {
		t.Fatal("snmp did not match 161")
	}
	if !snmp.Match(frameProto(net.ProtoUDP, 162, 40000, nil)) {
		t.Fatal("snmp did not match 162")
	}
	if snmp.Match(frameProto(net.ProtoUDP, 40000, 163, nil)) {
		t.Fatal("snmp matched 163")
	}
}

func TestHTTP1Detection(t *testing.T) {
	h1 := ByName("http1")
	if !h1.Match(frameProto(net.ProtoTCP, 40000, 80, []byte("GET / HTTP/1.1\r\n"))) {
		t.Fatal("http1 port match failed")
	}
	// Method detection on a non-standard port.
	if got := For(frameProto(net.ProtoTCP, 40000, 9999, []byte("POST /x HTTP/1.1\r\n"))); got == nil || got.Name() != "http1" {
		t.Fatalf("http1 method detection = %v", got)
	}
}

func TestHTTP2Detection(t *testing.T) {
	h2 := ByName("http2")
	if !h2.Match(frameProto(net.ProtoTCP, 40000, 8080, http2Preface)) {
		t.Fatal("http2 preface not matched")
	}
	// Type 0x01 (HEADERS), length 4, on 8080 (frame + 4 payload bytes).
	frame := []byte{0x00, 0x00, 0x04, 0x01, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00}
	if got := For(frameProto(net.ProtoTCP, 40000, 8080, frame)); got == nil || got.Name() != "http2" {
		t.Fatalf("http2 frame For() = %v", got)
	}
	// A random port without the preface must not match.
	if h2.Match(frameProto(net.ProtoTCP, 40000, 9999, frame)) {
		t.Fatal("http2 matched on unrelated port")
	}
}

// TestHTTP3ShortHeader is the regression for QUIC packets after the handshake.
func TestHTTP3ShortHeader(t *testing.T) {
	short := []byte{0x40, 0x00, 0x00, 0x00, 0x01} // high bit clear (1-RTT)
	f := frameProto(net.ProtoUDP, 50000, 443, short)
	if got := For(f); got == nil || got.Name() != "http3" {
		t.Fatalf("QUIC short header For() = %v, want http3", got)
	}
	long := []byte{0xc0, 0x00, 0x00, 0x00, 0x01} // high bit set (Initial)
	if got := For(frameProto(net.ProtoUDP, 50000, 443, long)); got == nil || got.Name() != "http3" {
		t.Fatalf("QUIC long header For() = %v, want http3", got)
	}
	if IsQUICLongHeader(short) {
		t.Fatal("short header reported as long")
	}
	if !IsQUICLongHeader(long) {
		t.Fatal("long header not detected")
	}
}

func TestTLS(t *testing.T) {
	tls := ByName("tls")
	record := []byte{0x16, 0x03, 0x01, 0x00, 0x05}
	// Record-type detection on a non-standard port.
	if got := For(frameProto(net.ProtoTCP, 40000, 12345, record)); got == nil || got.Name() != "tls" {
		t.Fatalf("tls record detection = %v", got)
	}
	// Port-based detection even without a recognizable record.
	if got := For(frameProto(net.ProtoTCP, 40000, 443, []byte("garbage"))); got == nil || got.Name() != "tls" {
		t.Fatalf("tls port detection = %v", got)
	}
	if tls.Match(frameProto(net.ProtoUDP, 40000, 443, record)) {
		t.Fatal("tls matched UDP")
	}
}

func TestAdapterPrecedence(t *testing.T) {
	cases := []struct {
		f    *net.Frame
		want string
	}{
		{frameProto(net.ProtoUDP, 50000, 53, nil), "dns"},
		{frameProto(net.ProtoUDP, 50000, 161, nil), "snmp"},
		{frameProto(net.ProtoICMP, 0, 0, nil), "icmp"},
		{frameProto(net.ProtoUDP, 50000, 443, nil), "http3"},
		{frameProto(net.ProtoTCP, 50000, 443, nil), "tls"},
		{frameProto(net.ProtoTCP, 50000, 8080, http2Preface), "http2"},
		{frameProto(net.ProtoTCP, 50000, 80, []byte("GET / HTTP/1.1\r\n")), "http1"},
	}
	for _, tc := range cases {
		if got := For(tc.f); got == nil || got.Name() != tc.want {
			t.Errorf("For(%s) = %v, want %s", tc.f, got, tc.want)
		}
	}
}

func TestRawFallback(t *testing.T) {
	// A TCP flow on an unadapted port falls back to raw.
	f := frameProto(net.ProtoTCP, 40000, 12345, []byte("payload"))
	if got := For(f); got == nil || got.Name() != "raw" {
		t.Fatalf("raw fallback = %v", got)
	}
	if got := ByName("raw").Bytes(f); string(got) != "payload" {
		t.Fatalf("raw bytes = %q", got)
	}
	// No payload: the link-layer frame is used.
	g := &net.Frame{Protocol: net.ProtoUDP, SrcPort: 1, DstPort: 2, Raw: []byte("frame")}
	if got := ByName("raw").Bytes(g); string(got) != "frame" {
		t.Fatalf("raw fallback bytes = %q", got)
	}
	// Specific adapters still take precedence.
	if got := For(frameProto(net.ProtoUDP, 50000, 53, nil)); got == nil || got.Name() != "dns" {
		t.Fatalf("dns precedence lost: %v", got)
	}
}

// TestDecodeThenAdapt exercises the full link/network decode followed by
// protocol adaptation for a UDP DNS packet.
func TestDecodeThenAdapt(t *testing.T) {
	dnsMsg := []byte{0xab, 0xcd, 0x01, 0x00, 0x00, 0x01}
	frame := ethernetUDP("10.0.0.1", "8.8.8.8", 40000, 53, dnsMsg)
	f, err := net.Decode(1, frame, time.Unix(1, 0))
	if err != nil {
		t.Fatal(err)
	}
	a := For(f)
	if a == nil || a.Name() != "dns" {
		t.Fatalf("decoded adapter = %v", a)
	}
	if !bytes.Equal(a.Bytes(f), dnsMsg) {
		t.Fatalf("decoded dns bytes = %x", a.Bytes(f))
	}
	if f.Src != netip.MustParseAddr("10.0.0.1") || f.DstPort != 53 {
		t.Fatalf("decoded frame = %v", f)
	}
}

func ethernetUDP(src, dst string, sport, dport uint16, payload []byte) []byte {
	frame := make([]byte, 14+20+8+len(payload))
	copy(frame[0:6], []byte{0, 1, 2, 3, 4, 5})
	copy(frame[6:12], []byte{6, 7, 8, 9, 10, 11})
	binary.BigEndian.PutUint16(frame[12:14], 0x0800)
	ip := frame[14:]
	ip[0] = 0x45
	binary.BigEndian.PutUint16(ip[2:4], uint16(20+8+len(payload)))
	ip[9] = net.ProtoUDP
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
