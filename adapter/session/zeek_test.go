package session

import (
	"net/netip"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cookiengineer/goganet/adapter/net"
)

func TestZeekMalicious(t *testing.T) {
	cases := map[string]bool{
		"Malicious":                             true,
		"Malicious   PartOfAHorizontalPortScan": true,
		"Malicious   C&C":                       true,
		"Benign":                                false,
		"Benign   -":                            false,
		"-":                                     false,
		"":                                      false,
	}
	for in, want := range cases {
		if got := ZeekMalicious(in); got != want {
			t.Errorf("ZeekMalicious(%q) = %v, want %v", in, got, want)
		}
	}
}

func zeekLine(fields, values []string) string { return strings.Join(values, "\t") }

func TestJoinZeekFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "conn.log.labeled")
	fields := []string{
		"ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
		"proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
		"label", "detailed-label",
	}
	lines := []string{
		"#separator \\x09",
		"#fields\t" + strings.Join(fields, "\t"),
		"#types\t" + strings.Join(make([]string, len(fields)), "\t"),
		zeekLine(fields, []string{"1528123456.123456", "C1", "10.0.0.1", "40000", "8.8.8.8", "53", "udp", "dns", "0.1", "100", "200", "SF", "Benign", "-"}),
		zeekLine(fields, []string{"1528123500.000000", "C2", "192.168.1.5", "50000", "1.2.3.4", "1234", "tcp", "-", "1.0", "100", "200", "SF", "Malicious", "PartOfAHorizontalPortScan"}),
		"", // blank line
		"garbage",
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	benign := &Session{Key: makeKey(netip.MustParseAddr("10.0.0.1"), 40000, netip.MustParseAddr("8.8.8.8"), 53, net.ProtoUDP)}
	mal := &Session{Key: makeKey(netip.MustParseAddr("192.168.1.5"), 50000, netip.MustParseAddr("1.2.3.4"), 1234, net.ProtoTCP)}
	unmatched := &Session{Key: makeKey(netip.MustParseAddr("9.9.9.9"), 1, netip.MustParseAddr("8.8.4.4"), 2, net.ProtoUDP)}

	n, err := JoinZeekFile([]*Session{benign, mal, unmatched}, path)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("matched = %d, want 2", n)
	}
	if !benign.Labeled || benign.Malicious || benign.Label != "Benign" {
		t.Fatalf("benign = %+v", benign)
	}
	if !mal.Labeled || !mal.Malicious || mal.Label != "Malicious" {
		t.Fatalf("mal = %+v", mal)
	}
	if unmatched.Labeled {
		t.Fatal("unmatched session should not be labelled")
	}
}

func TestJoinZeekFileNoHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "conn.log.labeled")
	if err := os.WriteFile(path, []byte("1.0\tC1\t10.0.0.1\t1\t10.0.0.2\t2\tudp\t-\t1\t1\t1\tSF\tBenign\t-\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Session{Key: makeKey(netip.MustParseAddr("10.0.0.1"), 1, netip.MustParseAddr("10.0.0.2"), 2, net.ProtoUDP)}
	n, err := JoinZeekFile([]*Session{s}, path)
	if err != nil {
		t.Fatal(err)
	}
	if n != 0 || s.Labeled {
		t.Fatalf("row without #fields should be ignored (n=%d labelled=%v)", n, s.Labeled)
	}
}
