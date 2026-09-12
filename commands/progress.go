package commands

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/dyakubu/scout/indexer"
	"golang.org/x/term"
)

// progressInterval is how often the line is redrawn. Updates arrive per
// file, which is far too often on a directory of small files and not often
// enough on one large one, so the ticker decides instead: it keeps the
// elapsed time moving while a single slow file is being embedded.
const progressInterval = 150 * time.Millisecond

// progressPrinter draws a single line that rewrites itself as indexing
// proceeds, and erases itself before the summary is printed.
//
// It writes nothing when stdout isn't a terminal, so `scout index | tee`
// stays free of carriage returns and escape codes - the same check
// hyperlink makes for OSC 8 links.
type progressPrinter struct {
	out     io.Writer
	width   int
	started time.Time

	mu      sync.Mutex
	latest  indexer.Progress
	printed bool
	stop    chan struct{}
	done    chan struct{}
}

// newProgressPrinter returns a printer, or nil if stdout isn't a terminal.
// A nil *progressPrinter's methods are all no-ops, so callers don't have
// to check.
func newProgressPrinter() *progressPrinter {
	fd := int(os.Stdout.Fd())
	if !term.IsTerminal(fd) {
		return nil
	}

	width, _, err := term.GetSize(fd)
	if err != nil || width <= 0 {
		width = 80
	}

	return &progressPrinter{
		out:     os.Stdout,
		width:   width,
		started: time.Now(),
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}
}

// start begins redrawing until stop is called.
func (p *progressPrinter) start() {
	if p == nil {
		return
	}

	go func() {
		defer close(p.done)

		ticker := time.NewTicker(progressInterval)
		defer ticker.Stop()

		for {
			select {
			case <-p.stop:
				return
			case <-ticker.C:
				p.draw()
			}
		}
	}()
}

// update records the latest snapshot. Called from the indexer's workers,
// so it only stores - drawing happens on the ticker.
func (p *progressPrinter) update(progress indexer.Progress) {
	if p == nil {
		return
	}

	p.mu.Lock()
	p.latest = progress
	p.mu.Unlock()
}

// finish stops the redraw loop and clears the line, leaving the cursor
// where the summary should start.
func (p *progressPrinter) finish() {
	if p == nil {
		return
	}

	close(p.stop)
	<-p.done

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.printed {
		fmt.Fprintf(p.out, "\r%s\r", strings.Repeat(" ", p.width))
	}
}

// elapsed formats a duration for a line that redraws several times a
// second: seconds while it's short, minutes once it isn't, and never so
// precise that the digits flicker.
func elapsed(d time.Duration) string {
	if d < time.Minute {
		return fmt.Sprintf("%4.1fs", d.Seconds())
	}
	return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
}

func (p *progressPrinter) draw() {
	p.mu.Lock()
	defer p.mu.Unlock()

	progress := p.latest
	if progress.Current == "" && progress.FilesIndexed == 0 && progress.Skipped == 0 {
		// Nothing has been picked up yet - the walk is still starting.
		return
	}

	indexed := progress.FilesIndexed + progress.MediaIndexed
	line := fmt.Sprintf("  %s  indexed %d, %d chunks",
		elapsed(time.Since(p.started)), indexed, progress.ChunksEmbedded)

	if progress.Skipped > 0 {
		line += fmt.Sprintf(", %d skipped", progress.Skipped)
	}
	if progress.Errors > 0 {
		line += fmt.Sprintf(", %d error(s)", progress.Errors)
	}
	if progress.Current != "" {
		line += "  " + filepath.Base(progress.Current)
	}

	// Truncated to the terminal width so a long filename can't wrap, which
	// would leave the previous line behind on the next redraw.
	if len(line) > p.width-1 {
		line = line[:p.width-1]
	}

	fmt.Fprintf(p.out, "\r%-*s", p.width-1, line)
	p.printed = true
}
