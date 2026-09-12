package commands

import (
	"strings"
	"testing"
	"time"

	"github.com/dyakubu/scout/indexer"
)

// Everything is a no-op on a nil printer, which is what stdout being a
// pipe produces - the index command calls these without checking.
func TestProgressPrinter_NilIsSafe(t *testing.T) {
	var p *progressPrinter

	p.start()
	p.update(indexer.Progress{Current: "/some/file.txt", FilesIndexed: 3})
	p.finish()
}

func TestElapsed(t *testing.T) {
	tests := []struct {
		d    time.Duration
		want string
	}{
		{900 * time.Millisecond, " 0.9s"},
		{12 * time.Second, "12.0s"},
		{90 * time.Second, "1m30s"},
		{3725 * time.Second, "62m05s"},
	}

	for _, tt := range tests {
		if got := elapsed(tt.d); got != tt.want {
			t.Errorf("elapsed(%v) = %q, want %q", tt.d, got, tt.want)
		}
	}
}

// The line rewrites itself, so it must stay inside the terminal width -
// wrapping would leave the tail of the previous line on screen.
func TestProgressPrinter_TruncatesToWidth(t *testing.T) {
	var out strings.Builder
	p := &progressPrinter{
		out:     &out,
		width:   40,
		started: time.Now(),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}

	p.update(indexer.Progress{
		Current:        "/a/very/long/path/that/keeps/going/" + strings.Repeat("x", 200) + ".txt",
		FilesIndexed:   12,
		ChunksEmbedded: 3456,
	})
	p.draw()

	line := strings.TrimPrefix(out.String(), "\r")
	if len(line) >= p.width {
		t.Errorf("drew %d chars into a %d-wide terminal:\n%q", len(line), p.width, line)
	}
	if !strings.Contains(line, "indexed 12") {
		t.Errorf("line %q lost the counts", line)
	}
}

// Before anything has been picked up there's nothing worth showing, and
// drawing would leave a line to erase for a run that indexes nothing.
func TestProgressPrinter_SilentUntilWorkStarts(t *testing.T) {
	var out strings.Builder
	p := &progressPrinter{out: &out, width: 80, started: time.Now()}

	p.draw()

	if out.Len() != 0 {
		t.Errorf("drew %q before any file was picked up", out.String())
	}
}
