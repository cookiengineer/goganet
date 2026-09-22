package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// dnsAdapter matches DNS over UDP and TCP (port 53).
type dnsAdapter struct{}

func (dnsAdapter) Name() string { return "dns" }

func (dnsAdapter) Match(f *net.Frame) bool {
	if f == nil {
		return false
	}
	if f.Protocol != net.ProtoUDP && f.Protocol != net.ProtoTCP {
		return false
	}
	return f.SrcPort == 53 || f.DstPort == 53
}

func (dnsAdapter) Bytes(f *net.Frame) []byte {
	if f.Protocol == net.ProtoTCP && len(f.Payload) >= 2 {
		// DNS over TCP is length-prefixed.
		return f.Payload[2:]
	}
	return f.Payload
}
