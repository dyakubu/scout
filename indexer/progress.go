package indexer

// Progress is a snapshot of a run in flight, reported as files are picked
// up and finished.
type Progress struct {
	// Current is the file being worked on, empty once the walk is done.
	// With several workers running it's whichever one most recently
	// started, which is enough to show that something is moving.
	Current string

	FilesIndexed   int
	MediaIndexed   int
	ChunksEmbedded int
	Skipped        int
	Errors         int
}

// report sends a snapshot to the configured callback, if there is one.
// Called from every file worker, so a callback has to be safe to call
// concurrently.
func (fi *FileIndexer) report(stats *statsAccumulator, current string) {
	if fi.Progress == nil {
		return
	}

	fi.Progress(Progress{
		Current:        current,
		FilesIndexed:   int(stats.filesIndexed.Load()),
		MediaIndexed:   int(stats.mediaFilesIndexed.Load()),
		ChunksEmbedded: int(stats.chunksEmbedded.Load()),
		Skipped:        int(stats.filesUnchanged.Load() + stats.filesTooLarge.Load() + stats.filesFiltered.Load() + stats.mediaFilesFiltered.Load()),
		Errors:         int(stats.errors.Load()),
	})
}
