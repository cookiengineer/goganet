package adapter

import "github.com/cookiengineer/goganet/adapter/net"

// snmpAdapter matches SNMP over UDP (ports 161 and 162).
type snmpAdapter struct{}

func (snmpAdapter) Name() string { return "snmp" }

func (snmpAdapter) Match(f *net.Frame) bool {
	if f == nil || f.Protocol != net.ProtoUDP {
		return false
	}
	return f.SrcPort == 161 || f.DstPort == 161 || f.SrcPort == 162 || f.DstPort == 162
}

func (snmpAdapter) Bytes(f *net.Frame) []byte { return f.Payload }
