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

func TestJoinZeekSpacePackedLabel(t *testing.T) {
	// Mimics the real IoT-23 layout: the trailing "label   detailed-label"
	// columns are appended with spaces rather than tabs.
	names := []string{
		"ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
		"proto", "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
		"local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
		"orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents",
	}
	header := "#fields\t" + strings.Join(names, "\t") + "   label   detailed-label"
	values := []string{
		"1528123456.123456", "C1", "10.0.0.1", "40000", "1.2.3.4", "1234",
		"tcp", "-", "1", "1", "1", "SF", "-", "-", "0", "S", "1", "60", "0", "0", "-",
	}
	row := strings.Join(values, "\t") + "   malicious   PartOfAHorizontalPortScan"

	dir := t.TempDir()
	path := filepath.Join(dir, "conn.log.labeled")
	if err := os.WriteFile(path, []byte(header+"\n"+row+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Session{Key: makeKey(netip.MustParseAddr("10.0.0.1"), 40000, netip.MustParseAddr("1.2.3.4"), 1234, net.ProtoTCP)}
	n, err := JoinZeekFile([]*Session{s}, path)
	if err != nil {
		t.Fatal(err)
	}
	if n != 1 || !s.Labeled || !s.Malicious || s.Label != "malicious" {
		t.Fatalf("space-packed label not parsed: n=%d session=%+v", n, s)
	}
}

func TestJoinZeekFileLimit(t *testing.T) {
	names := []string{"ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "label"}
	header := "#fields\t" + strings.Join(names, "\t")
	other := "1.0\tC1\t10.9.9.9\t1\t10.9.9.10\t2\tudp\tBenign"
	match := "2.0\tC2\t10.0.0.1\t40000\t8.8.8.8\t53\tudp\tBenign"
	dir := t.TempDir()
	path := filepath.Join(dir, "conn.log.labeled")
	if err := os.WriteFile(path, []byte(strings.Join([]string{header, other, match}, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	s := &Session{Key: makeKey(netip.MustParseAddr("10.0.0.1"), 40000, netip.MustParseAddr("8.8.8.8"), 53, net.ProtoUDP)}

	// Budget of two lines only reaches the header and the first row.
	if n, err := JoinZeekFileLimit([]*Session{s}, path, 2); err != nil || n != 0 {
		t.Fatalf("limited join n=%d err=%v", n, err)
	}
	if n, err := JoinZeekFileLimit([]*Session{s}, path, 0); err != nil || n != 1 {
		t.Fatalf("unlimited join n=%d err=%v", n, err)
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
