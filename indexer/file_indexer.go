package indexer

import (
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dyakubu/scout/config"
	"github.com/dyakubu/scout/embedder"
)

const (
	// chunkSize is the rune ceiling on one chunk. The model's real limit
	// is a token count, enforced separately (see FileIndexer.chunkLimits),
	// which is what most chunks of dense content actually hit first.
	chunkSize = 500

	// writeGroupSize bounds how many chunks of one file are embedded and
	// written together. Keeping this bounded (rather than processing a
	// whole file as one unit) keeps memory and per-transaction size
	// independent of file size, and means a failure partway through a huge
	// file doesn't lose everything already embedded.
	writeGroupSize = 32

	fileWorkers  = 4
	writerBuffer = 64

	// mediaQueueSize bounds how many discovered media files can be waiting
	// for the single media goroutine before the walk blocks on sending to
	// it. Small on purpose: the media worker subprocess behind
	// FileIndexer.MediaEmbedder only usefully does one thing at a time
	// (see mediaworker.Client), so this is a queue in front of a single
	// consumer, not a pool to size up.
	mediaQueueSize = 16
)

type FileIndexer struct {
	Db          *sql.DB
	Embedder    embedder.Embedder
	IndexConfig config.IndexConfig

	// MediaConfig and MediaEmbedder are both optional together: a nil
	// MediaEmbedder means no media support is available (no Python
	// worker/model configured), in which case matching media files are
	// counted as filtered rather than erroring the run.
	MediaConfig   config.MediaConfig
	MediaEmbedder embedder.MediaEmbedder

	Logger *log.Logger
}

type ProcessFileResult struct {
	// Skipped is true when the file exceeded the configured max size.
	Skipped bool
	// Unchanged is true when the file's mtime matched what's already
	// indexed, so it was left untouched.
	Unchanged         bool
	ChunksEmbedded    int
	FailedChunkErrors []error
}

// IndexStats summarizes one IndexDirectory run.
type IndexStats struct {
	FilesIndexed   int
	FilesUnchanged int
	FilesTooLarge  int
	FilesFiltered  int
	ChunksEmbedded int

	// MediaFilesIndexed/MediaFilesFiltered mirror FilesIndexed/FilesFiltered
	// for media files. Skipped-too-large and skipped-unchanged media files
	// share FilesTooLarge/FilesUnchanged above rather than getting their
	// own counters, since those checks mean the same thing regardless of
	// file kind. MediaFilesFiltered specifically counts media-extension
	// files that had no MediaEmbedder available to handle them.
	MediaFilesIndexed  int
	MediaFilesFiltered int

	// PathsUnreadable counts directories and files the walk couldn't read
	// at all - permission denied, or removed while the walk was running.
	// They're skipped rather than failing the run.
	PathsUnreadable int

	Errors  int
	Elapsed time.Duration
}

// SawAnything reports whether the run visited anything at all. Elapsed is
// always set, so a zero-value comparison can't tell an empty directory
// apart from a walk that failed before it started.
func (s IndexStats) SawAnything() bool {
	return s.FilesIndexed+s.FilesUnchanged+s.FilesTooLarge+s.FilesFiltered+
		s.MediaFilesIndexed+s.MediaFilesFiltered+s.PathsUnreadable+s.Errors > 0
}

// statsAccumulator is IndexStats' concurrency-safe counterpart, updated
// from both the (single-threaded) walk callback and the (concurrent) file
// workers.
type statsAccumulator struct {
	filesIndexed   atomic.Int64
	filesUnchanged atomic.Int64
	filesTooLarge  atomic.Int64
	filesFiltered  atomic.Int64
	chunksEmbedded atomic.Int64

	mediaFilesIndexed  atomic.Int64
	mediaFilesFiltered atomic.Int64

	pathsUnreadable atomic.Int64

	errors atomic.Int64
}

func (s *statsAccumulator) result(elapsed time.Duration) IndexStats {
	return IndexStats{
		FilesIndexed:       int(s.filesIndexed.Load()),
		FilesUnchanged:     int(s.filesUnchanged.Load()),
		FilesTooLarge:      int(s.filesTooLarge.Load()),
		FilesFiltered:      int(s.filesFiltered.Load()),
		ChunksEmbedded:     int(s.chunksEmbedded.Load()),
		MediaFilesIndexed:  int(s.mediaFilesIndexed.Load()),
		MediaFilesFiltered: int(s.mediaFilesFiltered.Load()),
		PathsUnreadable:    int(s.pathsUnreadable.Load()),
		Errors:             int(s.errors.Load()),
		Elapsed:            elapsed,
	}
}

func NewFileIndexer(db *sql.DB, embedder embedder.Embedder, indexConfig config.IndexConfig, logger *log.Logger) (*FileIndexer, error) {
	if db == nil {
		return nil, errors.New("db is nil. cannot initialize file indexer")
	}

	if embedder == nil {
		return nil, errors.New("embedder is nil. cannot initialize file indexer")
	}

	if logger == nil {
		return nil, errors.New("logger is nil. cannot initialize file indexer")
	}

	return &FileIndexer{
		Db:          db,
		Embedder:    embedder,
		IndexConfig: indexConfig,
		Logger:      logger,
	}, nil
}

// shouldSkipDir reports whether a directory should be pruned from the walk
// entirely: its name is an exact match in IndexConfig.IgnoreDirs, it
// contains one of IndexConfig.IgnoreDirMarkers (for directories
// identifiable by a marker file but not by a fixed name, e.g. a Python
// virtualenv - always identifiable by pyvenv.cfg regardless of what the
// venv directory itself is called), or it's excluded by the root
// .gitignore.
func (fi *FileIndexer) shouldSkipDir(rootDir, path, name string, gitignoreRules []gitignoreRule) bool {
	// Dot-prefixed only, so Windows' hidden attribute isn't covered - see
	// AppData in the shipped IgnoreDirs. The root is exempt: naming it is
	// intent.
	if !fi.IndexConfig.IndexHiddenDirs && path != rootDir && strings.HasPrefix(name, ".") {
		return true
	}

	// Globs, since junk directories are often version-stamped. A pattern
	// with no metacharacters compares exactly; a malformed one falls back
	// to its literal name.
	for _, pattern := range fi.IndexConfig.IgnoreDirs {
		matched, err := filepath.Match(pattern, name)
		if err != nil {
			matched = pattern == name
		}
		if matched {
			return true
		}
	}

	for _, marker := range fi.IndexConfig.IgnoreDirMarkers {
		if _, err := os.Stat(filepath.Join(path, marker)); err == nil {
			return true
		}
	}

	return matchesGitignore(gitignoreRules, rootDir, path, true)
}

// fileKind is the result of classifying one file during a directory walk.
type fileKind int

const (
	// fileKindNone means the file is excluded (gitignore, ignore pattern,
	// or an extension in neither allowed-extensions list) and should be
	// skipped entirely.
	fileKindNone fileKind = iota
	fileKindText
	fileKindMedia
)

// classifyFile reports how a file should be routed during a directory
// walk: to the text pipeline, the media pipeline, or excluded entirely.
// IgnorePatterns and the root .gitignore apply the same way regardless of
// kind; which of IndexConfig.AllowedExtensions or
// MediaConfig.AllowedExtensions matches the file's extension decides text
// vs. media (checked in that order, so a name landing in both lists -
// which none do by default - would be treated as text).
func (fi *FileIndexer) classifyFile(rootDir, path, name string, gitignoreRules []gitignoreRule) fileKind {
	if matchesGitignore(gitignoreRules, rootDir, path, false) {
		return fileKindNone
	}

	for _, pattern := range fi.IndexConfig.IgnorePatterns {
		if matched, _ := filepath.Match(pattern, name); matched {
			return fileKindNone
		}
	}

	ext := strings.ToLower(filepath.Ext(name))

	for _, allowed := range fi.IndexConfig.AllowedExtensions {
		if ext == strings.ToLower(allowed) {
			return fileKindText
		}
	}

	for _, allowed := range fi.MediaConfig.AllowedExtensions {
		if ext == strings.ToLower(allowed) {
			return fileKindMedia
		}
	}

	return fileKindNone
}

// maxFileSizeBytes returns the size ceiling for path in bytes, or 0 for no
// limit. An extension listed in MaxFileSizeMBByType uses that value,
// otherwise the MaxFileSizeMB fallback.
//
// The limit is on the file, not on the text extracted from it. Formats
// that carry their own media - a PDF's page images, a docx's pictures -
// are mostly container, so one ceiling for everything is either too small
// for them or too large for source files.
func (fi *FileIndexer) maxFileSizeBytes(path string) int64 {
	limit := fi.IndexConfig.MaxFileSizeMB

	if ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), "."); ext != "" {
		if override, ok := fi.IndexConfig.MaxFileSizeMBByType[ext]; ok {
			limit = override
		}
	}

	return int64(limit) * 1024 * 1024
}

// IndexDirectory walks a directory and processes each file, skipping
// directories and files per IndexConfig. If recursive is false, only files
// directly in dir are processed - subdirectories are not descended into.
// dir is resolved through any symlinks first - a symlinked directory
// passed as the root would otherwise be treated as an unrecognized file
// (WalkDir's root Lstat reports the link, not what it points to) and
// silently skipped instead of indexed.
func (fi *FileIndexer) IndexDirectory(dir string, recursive bool) (IndexStats, error) {
	start := time.Now()

	resolved, err := filepath.EvalSymlinks(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return IndexStats{}, fmt.Errorf("path does not exist: %s", dir)
		}
		return IndexStats{}, fmt.Errorf("resolving path %s: %w", dir, err)
	}
	dir = resolved

	gitignoreRules, err := loadGitignore(dir)
	if err != nil {
		fi.Logger.Printf("reading .gitignore in %s: %v (continuing without it)", dir, err)
	}

	fi.Logger.Printf("index start: dir=%s recursive=%v", dir, recursive)

	var stats statsAccumulator

	files := make(chan string)
	mediaJobs := make(chan string, mediaQueueSize)
	errCh := make(chan error, 1)

	// All file workers emit through this single writer, which is the only
	// goroutine that ever touches the database.
	writer := newDBWriter(fi.Db, writerBuffer)
	writer.start()

	var wg sync.WaitGroup

	// Start a bounded number of workers so a directory with many files
	// does not create an unbounded number of goroutines.
	for i := 0; i < fileWorkers; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for path := range files {
				res, err := fi.processFile(path, writer)
				if err != nil {
					stats.errors.Add(1)
					// errCh only ever surfaces the first error from a run
					// (see below) - every error is logged here regardless,
					// so none are silently lost.
					fi.Logger.Printf("error processing %s: %v", path, err)
					select {
					case errCh <- err:
					default:
					}
					continue
				}

				switch {
				case res.Skipped:
					stats.filesTooLarge.Add(1)
				case res.Unchanged:
					stats.filesUnchanged.Add(1)
				default:
					stats.filesIndexed.Add(1)
				}
				stats.chunksEmbedded.Add(int64(res.ChunksEmbedded))

				if len(res.FailedChunkErrors) > 0 {
					stats.errors.Add(1)
					joined := errors.Join(res.FailedChunkErrors...)
					fi.Logger.Printf("error processing %s: %v", path, joined)
					select {
					case errCh <- fmt.Errorf("%s: %w", path, joined):
					default:
					}
				}
			}
		}()
	}

	// Exactly one goroutine for media, distinct from the fileWorkers pool
	// above: the worker subprocess behind MediaEmbedder only usefully
	// handles one request at a time (see mediaworker.Client), so letting
	// general workers race to call it would buy no parallelism while
	// risking a slow media file (video, eventually) tying up a slot that
	// should be processing text files.
	wg.Add(1)

	go func() {
		defer wg.Done()

		for path := range mediaJobs {
			res, err := fi.processMediaFile(path, writer)
			if err != nil {
				stats.errors.Add(1)
				fi.Logger.Printf("error processing %s: %v", path, err)
				select {
				case errCh <- err:
				default:
				}
				continue
			}

			switch {
			case res.Skipped:
				stats.filesTooLarge.Add(1)
			case res.Unchanged:
				stats.filesUnchanged.Add(1)
			default:
				stats.mediaFilesIndexed.Add(1)
			}
		}
	}()

	walkErr := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// Failing on the directory being indexed is fatal - there's
			// nothing to walk, and an empty summary would hide why.
			if path == dir {
				return err
			}

			// Deeper down, an unreadable path is skipped so the rest of
			// the tree still gets indexed. Indexing a home directory
			// reaches plenty of these: macOS denies access to ~/.Trash
			// and much of ~/Library unless the terminal has been granted
			// Full Disk Access, and a single one of them should not abort
			// the whole run.
			stats.pathsUnreadable.Add(1)
			fi.Logger.Printf("skipping %s: %v", path, err)

			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		if d.IsDir() {
			if path != dir {
				if !recursive {
					return filepath.SkipDir
				}
				if fi.shouldSkipDir(dir, path, d.Name(), gitignoreRules) {
					return filepath.SkipDir
				}
			}
			return nil
		}

		switch fi.classifyFile(dir, path, d.Name(), gitignoreRules) {
		case fileKindText:
			files <- path
		case fileKindMedia:
			if fi.MediaEmbedder == nil {
				stats.mediaFilesFiltered.Add(1)
				return nil
			}
			mediaJobs <- path
		default:
			stats.filesFiltered.Add(1)
		}

		return nil
	})

	close(files)
	close(mediaJobs)
	wg.Wait()

	writerErr := writer.close()

	result := stats.result(time.Since(start))

	fi.Logger.Printf("index complete: dir=%s indexed=%d unchanged=%d too_large=%d filtered=%d chunks_embedded=%d media_indexed=%d media_filtered=%d errors=%d elapsed=%s",
		dir, result.FilesIndexed, result.FilesUnchanged, result.FilesTooLarge, result.FilesFiltered,
		result.ChunksEmbedded, result.MediaFilesIndexed, result.MediaFilesFiltered, result.Errors, result.Elapsed)

	if walkErr != nil {
		return result, walkErr
	}

	select {
	case err := <-errCh:
		return result, err
	default:
		return result, writerErr
	}
}

// processFile skips path entirely if it's too large or unchanged since the
// last index (see maxFileSizeBytes and isUnchanged); otherwise it reads and
// chunks it, then embeds and emits its chunks in bounded groups (see
// writeGroupSize) for the single db writer to persist. A file with no
// chunks is still emitted once, empty, so it's still recorded in files.
func (fi *FileIndexer) processFile(path string, emitter Emitter) (ProcessFileResult, error) {
	processStart := time.Now()

	info, err := os.Stat(path)
	if err != nil {
		return ProcessFileResult{}, err
	}

	if maxSize := fi.maxFileSizeBytes(path); maxSize > 0 && info.Size() > maxSize {
		fi.Logger.Printf("skipping %s: %d bytes exceeds the %d MB limit for its type", path, info.Size(), maxSize/(1024*1024))
		return ProcessFileResult{Skipped: true}, nil
	}

	modifiedAt := info.ModTime().Unix()

	unchanged, err := fi.isUnchanged(path, modifiedAt)
	if err != nil {
		return ProcessFileResult{}, err
	}
	if unchanged {
		return ProcessFileResult{Unchanged: true}, nil
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return ProcessFileResult{}, err
	}

	fileHashSum := sha256.Sum256(raw)
	fileHash := fileHashSum[:]

	content, err := extractText(path, raw)
	if err != nil {
		return ProcessFileResult{}, err
	}

	chunks, err := ChunkText(content, fi.chunkLimits())
	if err != nil {
		return ProcessFileResult{}, err
	}

	res := ProcessFileResult{}

	if len(chunks) == 0 {
		if err := emitter.Emit(FileRecord{Path: path, ModifiedAt: modifiedAt, FileHash: fileHash}); err != nil {
			res.FailedChunkErrors = append(res.FailedChunkErrors, fmt.Errorf("recording empty file: %w", err))
		}
		return res, nil
	}

	modelID := fi.Embedder.ModelID()

	var embedElapsed time.Duration

	for start := 0; start < len(chunks); start += writeGroupSize {
		end := min(start+writeGroupSize, len(chunks))
		group := chunks[start:end]

		texts := make([]string, len(group))
		for i, chunk := range group {
			texts[i] = chunk.content
		}

		embedStart := time.Now()
		embeddings, err := fi.Embedder.Embed(texts)
		embedElapsed += time.Since(embedStart)
		if err != nil {
			res.FailedChunkErrors = append(
				res.FailedChunkErrors,
				fmt.Errorf("embedding chunks %d-%d: %w", group[0].index, group[len(group)-1].index, err),
			)
			continue
		}

		chunkRecords := make([]ChunkRecord, len(group))
		for i, chunk := range group {
			contentHashSum := sha256.Sum256([]byte(chunk.content))

			chunkRecords[i] = ChunkRecord{
				ChunkIndex:     chunk.index,
				Content:        chunk.content,
				ContentHash:    contentHashSum[:],
				StartLine:      chunk.startLine,
				EndLine:        chunk.endLine,
				EmbeddingModel: modelID,
				Embedding:      embeddings[i],
			}
		}

		if err := emitter.Emit(FileRecord{
			Path:       path,
			ModifiedAt: modifiedAt,
			FileHash:   fileHash,
			Chunks:     chunkRecords,
		}); err != nil {
			res.FailedChunkErrors = append(
				res.FailedChunkErrors,
				fmt.Errorf("writing chunks %d-%d: %w", group[0].index, group[len(group)-1].index, err),
			)
			continue
		}

		res.ChunksEmbedded += len(chunkRecords)
	}

	fi.Logger.Printf("processed %s: chunks=%d embed=%s total=%s",
		path, res.ChunksEmbedded, embedElapsed, time.Since(processStart))

	return res, nil
}

// processMediaFile embeds path via fi.MediaEmbedder and emits the result,
// reusing the same skip-if-too-large/skip-if-unchanged checks processFile
// uses for text. Only called when fi.MediaEmbedder is non-nil - the walk
// step in IndexDirectory filters out media files before this is reached
// otherwise. An image is a single embedded unit (frame_index 0, no
// timestamps), unlike a future video's multiple sampled frames.
func (fi *FileIndexer) processMediaFile(path string, emitter Emitter) (ProcessFileResult, error) {
	processStart := time.Now()

	info, err := os.Stat(path)
	if err != nil {
		return ProcessFileResult{}, err
	}

	if maxSize := fi.maxFileSizeBytes(path); maxSize > 0 && info.Size() > maxSize {
		fi.Logger.Printf("skipping %s: %d bytes exceeds the %d MB limit for its type", path, info.Size(), maxSize/(1024*1024))
		return ProcessFileResult{Skipped: true}, nil
	}

	modifiedAt := info.ModTime().Unix()

	unchanged, err := fi.isUnchanged(path, modifiedAt)
	if err != nil {
		return ProcessFileResult{}, err
	}
	if unchanged {
		return ProcessFileResult{Unchanged: true}, nil
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		return ProcessFileResult{}, err
	}

	fileHashSum := sha256.Sum256(raw)
	fileHash := fileHashSum[:]

	embedStart := time.Now()
	vector, err := fi.MediaEmbedder.EmbedImage(path)
	embedElapsed := time.Since(embedStart)
	if err != nil {
		return ProcessFileResult{}, fmt.Errorf("embedding image: %w", err)
	}

	record := MediaRecord{
		Path:       path,
		ModifiedAt: modifiedAt,
		FileHash:   fileHash,
		Embeddings: []MediaEmbeddingRecord{{
			FrameIndex:     0,
			ContentHash:    fileHash,
			EmbeddingModel: fi.MediaEmbedder.ModelID(),
			Embedding:      vector,
		}},
	}

	if err := emitter.EmitMedia(record); err != nil {
		return ProcessFileResult{}, fmt.Errorf("writing media embedding: %w", err)
	}

	fi.Logger.Printf("processed %s: media embed=%s total=%s", path, embedElapsed, time.Since(processStart))

	return ProcessFileResult{}, nil
}

// chunkLimits bounds a chunk by both the rune ceiling and the embedding
// model's token budget, so no chunk is ever handed to the model with more
// text than it will actually read.
func (fi *FileIndexer) chunkLimits() ChunkLimits {
	return ChunkLimits{
		MaxRunes:    chunkSize,
		MaxTokens:   fi.Embedder.TokenBudget(),
		CountTokens: fi.Embedder.CountTokens,
	}
}

// isUnchanged reports whether path is already indexed with this exact
// modification time. This is a plain read, safe to run concurrently across
// file workers - only writes need to go through the single dbWriter.
func (fi *FileIndexer) isUnchanged(path string, modifiedAt int64) (bool, error) {
	var stored int64

	err := fi.Db.QueryRow(`SELECT modified_at FROM files WHERE path = ?`, path).Scan(&stored)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}

	return stored == modifiedAt, nil
}

func ReadFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

// extractText returns the text to chunk and embed for path. raw is path's
// already-read file content, reused directly for every type except PDFs,
// which need their own extraction step to turn PDF structure into plain
// text.
func extractText(path string, raw []byte) (string, error) {
	if strings.ToLower(filepath.Ext(path)) == ".pdf" {
		return extractPDFText(path)
	}

	return string(raw), nil
}
