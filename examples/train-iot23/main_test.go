package main

import (
	"os"
	"path/filepath"
	"testing"
)

func touch(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDiscoverIoTCaptures(t *testing.T) {
	root := t.TempDir()

	// Capture A: PCAP at the capture root, Zeek log under bro/.
	touch(t, filepath.Join(root, "CTU-IoT-Malware-Capture-1-1", "a.pcap"))
	touch(t, filepath.Join(root, "CTU-IoT-Malware-Capture-1-1", "bro", "conn.log.labeled"))
	// Capture B: PCAP and log side by side.
	touch(t, filepath.Join(root, "CTU-Honeypot-Capture-1", "b.pcap"))
	touch(t, filepath.Join(root, "CTU-Honeypot-Capture-1", "conn.log.labeled"))
	// Noise that must be ignored.
	touch(t, filepath.Join(root, "notes.txt"))
	if err := os.MkdirAll(filepath.Join(root, "empty"), 0o755); err != nil {
		t.Fatal(err)
	}

	caps, err := discoverIoTCaptures(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(caps) != 2 {
		t.Fatalf("captures = %+v", caps)
	}
	// Sorted by name: CTU-Honeypot-Capture-1 before CTU-IoT-Malware-Capture-1-1.
	if caps[0].name != "CTU-Honeypot-Capture-1" || caps[1].name != "CTU-IoT-Malware-Capture-1-1" {
		t.Fatalf("order = %q, %q", caps[0].name, caps[1].name)
	}
	for _, c := range caps {
		if len(c.pcaps) != 1 || c.zeek == "" {
			t.Fatalf("capture %q pcaps=%v zeek=%q", c.name, c.pcaps, c.zeek)
		}
	}
}

func TestSplitNonEmpty(t *testing.T) {
	got := splitNonEmpty(" a , ,b,", ",")
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("splitNonEmpty = %v", got)
	}
	if len(splitNonEmpty("", ",")) != 0 {
		t.Fatal("empty input should yield no tokens")
	}
}

func TestMatchesAny(t *testing.T) {
	if !matchesAny("CTU-IoT-Malware-Capture-1-1", []string{"Capture-1"}) {
		t.Fatal("substring should match")
	}
	if matchesAny("CTU-IoT-Malware-Capture-2-1", []string{"Capture-1"}) {
		t.Fatal("unrelated capture should not match")
	}
}
