package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// icmpAdapter matches ICMP and ICMPv6.
type icmpAdapter struct{}

func (icmpAdapter) Name() string { return "icmp" }

func (icmpAdapter) Match(f *net.Frame) bool {
	if f == nil {
		return false
	}
	return f.Protocol == net.ProtoICMP || f.Protocol == net.ProtoICMPv6
}

func (icmpAdapter) Bytes(f *net.Frame) []byte { return f.Payload }
