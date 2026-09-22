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
// and labels sessions by their bidirectional flow key.
//
// Two label layouts occur in the wild:
//
//   - label and detailed-label are ordinary tab-separated columns; and
//   - they are appended to the line separated by spaces, so the last tab field
//     packs the tunnel_parents value together with the label.
//
// Both are handled: the 5-tuple/proto columns are read from the tab-separated
// fields (the packing only affects trailing columns), while the label is taken
// from the whitespace tokens at the end of the line. Matching is by flow key
// only, because the pcap and Zeek timestamps frequently use different time
// zones. The function stops early once every session has been labelled.
func JoinZeekFile(sessions []*Session, path string) (int, error) {
	return JoinZeekFileLimit(sessions, path, 0)
}

// JoinZeekFileLimit is JoinZeekFile with a bound on the number of lines read.
// IoT-23 conn logs can exceed 10 GB and some flows may never match, so a budget
// avoids reading an entire log to label a handful of remaining sessions.
// maxLines <= 0 means unlimited.
func JoinZeekFileLimit(sessions []*Session, path string, maxLines int) (int, error) {
	index := make(map[net.SessionKey]*Session, len(sessions))
	for _, s := range sessions {
		if _, ok := index[s.Key]; !ok {
			index[s.Key] = s
		}
	}
	total := len(index)
	if total == 0 {
		return 0, nil
	}

	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	scan := bufio.NewScanner(f)
	scan.Buffer(make([]byte, 1<<20), 1<<20)

	var col map[string]int
	hasDetailed := false
	matched := 0
	lines := 0
	for scan.Scan() {
		lines++
		if maxLines > 0 && lines > maxLines {
			break
		}
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
					if names[i] == "detailed-label" {
						hasDetailed = true
					}
				}
			}
			continue
		}
		if col == nil {
			continue
		}
		label, ok := zeekKeyAndLabel(col, strings.Split(line, "\t"), line, hasDetailed)
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
		if matched == total {
			break
		}
	}
	if err := scan.Err(); err != nil {
		return matched, fmt.Errorf("zeek: %w", err)
	}
	return matched, nil
}

// zeekKeyAndLabel parses one conn.log row. The flow key is built from the
// tab-separated columns; the label is the trailing whitespace token (or the
// second-to-last when a detailed-label column is present).
func zeekKeyAndLabel(col map[string]int, fields []string, line string, hasDetailed bool) (FlowLabel, bool) {
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

	src, err := netip.ParseAddr(strings.TrimSpace(srcStr))
	if err != nil {
		return FlowLabel{}, false
	}
	dst, err := netip.ParseAddr(strings.TrimSpace(dstStr))
	if err != nil {
		return FlowLabel{}, false
	}
	sport, _ := strconv.ParseUint(strings.TrimSpace(sportStr), 10, 16)
	dport, _ := strconv.ParseUint(strings.TrimSpace(dportStr), 10, 16)

	tail := strings.Fields(line)
	labelRaw := ""
	switch {
	case hasDetailed && len(tail) >= 2:
		labelRaw = tail[len(tail)-2]
	case len(tail) >= 1:
		labelRaw = tail[len(tail)-1]
	}

	label := FlowLabel{
		Key:       makeKey(src, uint16(sport), dst, uint16(dport), zeekProto(protoStr)),
		Raw:       labelRaw,
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
