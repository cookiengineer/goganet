package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// http3Adapter matches QUIC, the transport of HTTP/3. QUIC runs over UDP and is
// encrypted end to end, so the payload is treated as opaque bytes.
//
// Only the first packet of a QUIC connection carries a long header (the high
// bit of the first byte set). All subsequent packets use short headers, so the
// adapter matches on the well-known QUIC ports rather than on the header form;
// requiring the long-header bit would miss the majority of a flow.
type http3Adapter struct{}

func (http3Adapter) Name() string { return "http3" }

func (http3Adapter) Match(f *net.Frame) bool {
	if f == nil || f.Protocol != net.ProtoUDP {
		return false
	}
	return isQUICPort(f.SrcPort) || isQUICPort(f.DstPort)
}

func isQUICPort(p uint16) bool { return p == 443 || p == 8443 }

// IsQUICLongHeader reports whether a QUIC payload uses the long-header form
// (initial, handshake or 0-RTT packet). It is exposed for adapters and tests
// that want to distinguish handshake from data packets.
func IsQUICLongHeader(payload []byte) bool {
	return len(payload) > 0 && payload[0]&0x80 != 0
}

func (http3Adapter) Bytes(f *net.Frame) []byte { return f.Payload }
