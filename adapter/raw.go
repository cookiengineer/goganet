package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// rawAdapter matches any transport flow (TCP, UDP, ICMP, ICMPv6) and renders
// the transport payload, falling back to the link-layer frame when there is no
// payload. It is registered last, so it only catches frames that no specific
// protocol adapter claimed, and it is the adapter to use when training a single
// classifier across all traffic (for example the heterogeneous IoT-23
// captures).
type rawAdapter struct{}

func (rawAdapter) Name() string { return "raw" }

func (rawAdapter) Match(f *net.Frame) bool {
	if f == nil {
		return false
	}
	switch f.Protocol {
	case net.ProtoTCP, net.ProtoUDP, net.ProtoICMP, net.ProtoICMPv6:
		return true
	default:
		return false
	}
}

func (rawAdapter) Bytes(f *net.Frame) []byte {
	if len(f.Payload) > 0 {
		return f.Payload
	}
	return f.Raw
}
