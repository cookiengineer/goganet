package session

import (
	"bufio"
	"fmt"
	"net/netip"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cookiengineer/goganet/adapter/net"
)

// ZeekMalicious reports whether a Zeek label denotes malicious traffic. The
// IoT-23 label field may be a single value ("Malicious" / "Benign") or a value
// followed by the detailed behaviour ("Malicious   PartOfAHorizontalPortScan"),
// so only the first token is significant.
func ZeekMalicious(label string) bool {
	f := strings.Fields(label)
	return len(f) > 0 && strings.EqualFold(f[0], "Malicious")
}

// JoinZeekFile streams a Zeek conn.log.labeled file (as shipped with IoT-23)
// and labels sessions by their bidirectional flow key. It reads the "#fields"
// header to locate the ts, 5-tuple, proto and label columns, so it is tolerant
// of column reordering and of the optional detailed-label column.
//
// Matching is by flow key only; the pcap and Zeek timestamps frequently use
// different time zones, so the intervals may not overlap.
func JoinZeekFile(sessions []*Session, path string) (int, error) {
	index := make(map[net.SessionKey]*Session, len(sessions))
	for _, s := range sessions {
		if _, ok := index[s.Key]; !ok {
			index[s.Key] = s
		}
	}

	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 1<<20), 1<<20)

	var col map[string]int
	matched := 0
	for scan.Scan() {
		line := scan.Text()
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if strings.HasPrefix(line, "#fields") {
				names := strings.Fields(line)
				col = make(map[string]int, len(names))
				for i := 1; i < len(names); i++ {
					col[names[i]] = i - 1
				}
			}
			continue
		}
		if col == nil {
			continue
		}
		label, ok := zeekKeyAndLabel(col, strings.Split(line, "\t"))
		if !ok {
			continue
		}
		s := index[label.Key]
		if s == nil || s.Labeled {
			continue
		}
		s.Label = label.Raw
		s.Malicious = label.Malicious
		s.Labeled = true
		matched++
	}
	if err := scan.Err(); err != nil {
		return matched, fmt.Errorf("zeek: %w", err)
	}
	return matched, nil
}

// zeekKeyAndLabel parses the fields of one conn.log row. The returned FlowLabel
// only has Key/Raw/Malicious set; it is not exported because it is an internal
// intermediate.
func zeekKeyAndLabel(col map[string]int, fields []string) (FlowLabel, bool) {
	get := func(name string) (string, bool) {
		i, ok := col[name]
		if !ok || i < 0 || i >= len(fields) {
			return "", false
		}
		return fields[i], true
	}

	protoStr, ok := get("proto")
	if !ok {
		return FlowLabel{}, false
	}
	srcStr, ok := get("id.orig_h")
	if !ok {
		return FlowLabel{}, false
	}
	dstStr, ok := get("id.resp_h")
	if !ok {
		return FlowLabel{}, false
	}
	sportStr, ok := get("id.orig_p")
	if !ok {
		return FlowLabel{}, false
	}
	dportStr, ok := get("id.resp_p")
	if !ok {
		return FlowLabel{}, false
	}
	labelRaw, ok := get("label")
	if !ok {
		return FlowLabel{}, false
	}

	src, err := netip.ParseAddr(strings.TrimSpace(srcStr))
	if err != nil {
		return FlowLabel{}, false
	}
	dst, err := netip.ParseAddr(strings.TrimSpace(dstStr))
	if err != nil {
		return FlowLabel{}, false
	}
	sport, err := strconv.ParseUint(strings.TrimSpace(sportStr), 10, 16)
	if err != nil {
		// Zeek uses "-" for an unknown port.
		sport = 0
	}
	dport, err := strconv.ParseUint(strings.TrimSpace(dportStr), 10, 16)
	if err != nil {
		dport = 0
	}

	label := FlowLabel{
		Key:       makeKey(src, uint16(sport), dst, uint16(dport), zeekProto(protoStr)),
		Raw:       strings.TrimSpace(labelRaw),
		Malicious: ZeekMalicious(labelRaw),
	}
	if ts, ok := get("ts"); ok {
		if v, err := strconv.ParseFloat(strings.TrimSpace(ts), 64); err == nil {
			sec := int64(v)
			label.Start = time.Unix(sec, int64((v-float64(sec))*1e9))
			label.End = label.Start
		}
	}
	return label, true
}

func zeekProto(s string) uint8 {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "tcp":
		return net.ProtoTCP
	case "udp":
		return net.ProtoUDP
	case "icmp":
		return net.ProtoICMP
	case "icmp6", "icmpv6":
		return net.ProtoICMPv6
	default:
		return 0
	}
}
