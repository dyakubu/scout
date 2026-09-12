package commands

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	"github.com/dyakubu/scout/app"
	"github.com/dyakubu/scout/cli"
	"github.com/dyakubu/scout/indexer"
)

func Index(ctx context.Context, args cli.ParsedArgs, deps app.Dependencies) error {

	if len(args.Positional) > 1 {
		return fmt.Errorf("index accepts at most one path, got %d: %v", len(args.Positional), args.Positional)
	}

	dir := "."
	if len(args.Positional) == 1 {
		dir = args.Positional[0]
	}

	dir, err := cli.ExpandHome(dir)

	if err != nil {
		return err
	}

	recursive, err := boolFlag(args, "recursive", true)

	if err != nil {
		return err
	}

	absPath, err := filepath.Abs(dir)

	if err != nil {
		return fmt.Errorf("Unable to resolve path %v. Error: %v", dir, err.Error())
	}

	// Only wired up for a terminal: newProgressPrinter returns nil
	// otherwise, and leaving Progress unset saves the indexer building a
	// snapshot per file that nothing would read.
	progress := newProgressPrinter()
	if progress != nil {
		deps.FileIndexer.Progress = progress.update
		progress.start()
	}

	stats, err := deps.FileIndexer.IndexDirectory(absPath, recursive)

	progress.finish()

	// An error with nothing visited means IndexDirectory failed before any
	// work happened - the path doesn't exist, or can't be read - and a
	// summary would misleadingly suggest a real, empty run.
	if err == nil || stats.SawAnything() {
		printIndexSummary(stats)
	}

	return err
}

func printIndexSummary(stats indexer.IndexStats) {
	fmt.Printf("Indexed %d file(s), %d chunk(s) embedded, in %s\n",
		stats.FilesIndexed, stats.ChunksEmbedded, stats.Elapsed.Round(10*time.Millisecond))

	if stats.FilesUnchanged > 0 {
		fmt.Printf("  %d file(s) unchanged, skipped\n", stats.FilesUnchanged)
	}
	if stats.FilesTooLarge > 0 {
		fmt.Printf("  %d file(s) skipped (exceeds max file size)\n", stats.FilesTooLarge)
	}
	if stats.FilesFiltered > 0 {
		fmt.Printf("  %d file(s) excluded by extension/ignore rules\n", stats.FilesFiltered)
	}
	if stats.PathsUnreadable > 0 {
		fmt.Printf("  %d path(s) skipped (unreadable - see log)\n", stats.PathsUnreadable)
	}
	if stats.Errors > 0 {
		fmt.Printf("  %d error(s) - see above\n", stats.Errors)
	}
	if stats.MediaFilesIndexed > 0 {
		fmt.Printf("  %d media file(s) embedded\n", stats.MediaFilesIndexed)
	}
	if stats.MediaFilesFiltered > 0 {
		fmt.Printf("  %d media file(s) skipped (no media worker configured)\n", stats.MediaFilesFiltered)
	}
}
