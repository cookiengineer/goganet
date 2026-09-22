package main

import (
	"archive/tar"
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func addFile(t *testing.T, tw *tar.Writer, name, data string) {
	t.Helper()
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o644, Size: int64(len(data)), Typeflag: tar.TypeReg}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write([]byte(data)); err != nil {
		t.Fatal(err)
	}
}

func addDir(t *testing.T, tw *tar.Writer, name string) {
	t.Helper()
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Typeflag: tar.TypeDir}); err != nil {
		t.Fatal(err)
	}
}

func buildTar(t *testing.T) []byte {
	t.Helper()
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	addDir(t, tw, "CTU-13-Dataset/4/")
	addFile(t, tw, "CTU-13-Dataset/4/capture.pcap", "PCAPDATA")
	addFile(t, tw, "CTU-13-Dataset/4/sub/readme.txt", "README")
	addFile(t, tw, "CTU-13-Dataset/5/other.pcap", "OTHER")
	// Path traversal attempt must be ignored.
	addFile(t, tw, "CTU-13-Dataset/4/../evil.txt", "bad")
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestExtractTar(t *testing.T) {
	out := t.TempDir()
	if err := extractTar(bytes.NewReader(buildTar(t)), "4", out); err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(filepath.Join(out, "4", "capture.pcap"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "PCAPDATA" {
		t.Fatalf("capture = %q", got)
	}
	sub, err := os.ReadFile(filepath.Join(out, "4", "sub", "readme.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(sub) != "README" {
		t.Fatalf("nested file = %q", sub)
	}
	if _, err := os.Stat(filepath.Join(out, "5")); !os.IsNotExist(err) {
		t.Fatal("scenario 5 should not have been extracted")
	}
	// The traversal target must not exist anywhere.
	if _, err := os.Stat(filepath.Join(out, "evil.txt")); !os.IsNotExist(err) {
		t.Fatal("path traversal was not blocked")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(out), "evil.txt")); !os.IsNotExist(err) {
		t.Fatal("path traversal escaped the output directory")
	}
}

func TestExtractTarMissingScenario(t *testing.T) {
	out := t.TempDir()
	if err := extractTar(bytes.NewReader(buildTar(t)), "9", out); err == nil {
		t.Fatal("expected error for missing scenario")
	}
}
