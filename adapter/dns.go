package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// dnsAdapter matches DNS over UDP and TCP (port 53) as well as multicast DNS
// (UDP/TCP 5353). DNS over UDP carries no length prefix; DNS over TCP is
// prefixed with a two-byte message length that [dnsAdapter.Bytes] strips.
type dnsAdapter struct{}

func (dnsAdapter) Name() string { return "dns" }

func (dnsAdapter) Match(f *net.Frame) bool {
	if f == nil {
		return false
	}
	if f.Protocol != net.ProtoUDP && f.Protocol != net.ProtoTCP {
		return false
	}
	return isDNSPort(f.SrcPort) || isDNSPort(f.DstPort)
}

func isDNSPort(p uint16) bool { return p == 53 || p == 5353 }

func (dnsAdapter) Bytes(f *net.Frame) []byte {
	if f.Protocol == net.ProtoTCP && len(f.Payload) >= 2 {
		// DNS over TCP is length-prefixed.
		return f.Payload[2:]
	}
	// UDP DNS (and mDNS) is the raw message.
	return f.Payload
}
