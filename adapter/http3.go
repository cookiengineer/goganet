package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// http3Adapter matches QUIC (the transport of HTTP/3) on UDP port 443. QUIC is
// encrypted end to end, so the payload is treated as opaque bytes; only the
// packet shape is visible.
type http3Adapter struct{}

func (http3Adapter) Name() string { return "http3" }

func (http3Adapter) Match(f *net.Frame) bool {
	if f == nil || f.Protocol != net.ProtoUDP {
		return false
	}
	if f.SrcPort != 443 && f.DstPort != 443 && f.SrcPort != 80 && f.DstPort != 80 {
		return false
	}
	// QUIC long header has the high bit of the first byte set.
	if len(f.Payload) == 0 {
		return false
	}
	return f.Payload[0]&0x80 != 0
}

func (http3Adapter) Bytes(f *net.Frame) []byte { return f.Payload }
