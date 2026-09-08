package commands

import (
	"net/url"
	"os"
	"path/filepath"

	"golang.org/x/term"
)

// hyperlink wraps text in an OSC 8 terminal hyperlink pointing at path, if
// stdout looks like a terminal that might render one - otherwise text is
// returned unchanged, so piped/redirected output (e.g. `scout find | grep
// ...`) never gets escape codes mixed into it.
//
// This exists because terminal auto-linking heuristics (iTerm2's semantic
// history, similar features elsewhere) generally can't tell where a path
// containing spaces ends without quotes, so a file like
// "Screenshot Sep 2.png" never gets linkified by them while one with no
// spaces does - not a random inconsistency, a real limitation of guessing
// link targets from plain text. OSC 8 sidesteps that entirely: the
// terminal is told exactly what the link target is, not left to infer it.
func hyperlink(path, text string) string {
	if !term.IsTerminal(int(os.Stdout.Fd())) {
		return text
	}

	return "\x1b]8;;" + fileURI(path) + "\x1b\\" + text + "\x1b]8;;\x1b\\"
}

// fileURI builds a file:// URI for path. Handles Windows' drive-letter
// form (file:///C:/Users/...): filepath.ToSlash converts backslashes to
// forward slashes (a no-op on non-Windows), and the file URI scheme
// requires a leading "/" before a drive letter that filepath itself
// doesn't add.
func fileURI(path string) string {
	p := filepath.ToSlash(path)
	if filepath.VolumeName(path) != "" {
		p = "/" + p
	}

	u := url.URL{Scheme: "file", Path: p}
	return u.String()
}
