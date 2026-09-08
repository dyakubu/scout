package commands

import (
	"strings"
	"testing"
)

func TestFileURI_Basic(t *testing.T) {
	// filepath.VolumeName only recognizes a drive letter on GOOS=windows,
	// so this exercises the non-Windows path on this machine; the Windows
	// drive-letter branch (the leading "/" before "C:") needs verifying
	// on an actual Windows run - it can't be exercised by compiling for a
	// different GOOS here.
	got := fileURI("/Users/me/Screenshot Sep 2.png")
	want := "file:///Users/me/Screenshot%20Sep%202.png"
	if got != want {
		t.Errorf("fileURI = %q, want %q", got, want)
	}
}

func TestFileURI_NoSpaces(t *testing.T) {
	got := fileURI("/Users/me/gen_pos_1.png")
	want := "file:///Users/me/gen_pos_1.png"
	if got != want {
		t.Errorf("fileURI = %q, want %q", got, want)
	}
}

func TestHyperlink_NotATerminal(t *testing.T) {
	// go test's stdout is never a TTY, so this exercises the fallback
	// path that must leave piped/redirected output untouched.
	path := "/Users/me/Screenshot Sep 2.png"
	got := hyperlink(path, path)
	if got != path {
		t.Errorf("hyperlink (non-terminal) = %q, want the plain text unchanged (%q)", got, path)
	}
	if strings.ContainsAny(got, "\x1b") {
		t.Error("hyperlink (non-terminal) contains an escape byte - must not when stdout isn't a terminal")
	}
}

func TestFileURI_MatchesFilepathToSlash(t *testing.T) {
	// Sanity check that the URI's path segment, once unescaped, round-trips
	// back to the same slash-separated form filepath.ToSlash produces.
	in := "/a/b c/d.txt"
	got := fileURI(in)
	if !strings.HasPrefix(got, "file://") {
		t.Fatalf("fileURI(%q) = %q, want a file:// URI", in, got)
	}
	if !strings.Contains(got, "%20") {
		t.Errorf("fileURI(%q) = %q, want the space percent-encoded", in, got)
	}
}
