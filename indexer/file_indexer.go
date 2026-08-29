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
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/dyakubu/scout/config"
	"github.com/dyakubu/scout/embedder"
)

const (
	chunkSize = 500

	// writeGroupSize bounds how many chunks of one file are embedded and
	// written together. Keeping this bounded (rather than processing a
	// whole file as one unit) keeps memory and per-transaction size
	// independent of file size, and means a failure partway through a huge
	// file doesn't lose everything already embedded.
	writeGroupSize = 32

	fileWorkers  = 4
	writerBuffer = 64
)

type FileIndexer struct {
	Db          *sql.DB
	Embedder    embedder.Embedder
	IndexConfig config.IndexConfig
	Logger      *log.Logger
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
	Errors         int
	Elapsed        time.Duration
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
	errors         atomic.Int64
}

func (s *statsAccumulator) result(elapsed time.Duration) IndexStats {
	return IndexStats{
		FilesIndexed:   int(s.filesIndexed.Load()),
		FilesUnchanged: int(s.filesUnchanged.Load()),
		FilesTooLarge:  int(s.filesTooLarge.Load()),
		FilesFiltered:  int(s.filesFiltered.Load()),
		ChunksEmbedded: int(s.chunksEmbedded.Load()),
		Errors:         int(s.errors.Load()),
		Elapsed:        elapsed,
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
	if slices.Contains(fi.IndexConfig.IgnoreDirs, name) {
		return true
	}

	for _, marker := range fi.IndexConfig.IgnoreDirMarkers {
		if _, err := os.Stat(filepath.Join(path, marker)); err == nil {
			return true
		}
	}

	return matchesGitignore(gitignoreRules, rootDir, path, true)
}

// shouldIndexFile reports whether a file passes IndexConfig.IgnorePatterns
// and IndexConfig.AllowedExtensions, and isn't excluded by the root
// .gitignore.
func (fi *FileIndexer) shouldIndexFile(rootDir, path, name string, gitignoreRules []gitignoreRule) bool {
	if matchesGitignore(gitignoreRules, rootDir, path, false) {
		return false
	}

	for _, pattern := range fi.IndexConfig.IgnorePatterns {
		if matched, _ := filepath.Match(pattern, name); matched {
			return false
		}
	}

	ext := strings.ToLower(filepath.Ext(name))
	for _, allowed := range fi.IndexConfig.AllowedExtensions {
		if ext == strings.ToLower(allowed) {
			return true
		}
	}

	return false
}

// maxFileSizeBytes returns the configured max file size in bytes, or 0 if
// MaxFileSizeMB is unset, meaning no limit.
func (fi *FileIndexer) maxFileSizeBytes() int64 {
	return int64(fi.IndexConfig.MaxFileSizeMB) * 1024 * 1024
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

	walkErr := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
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

		if !fi.shouldIndexFile(dir, path, d.Name(), gitignoreRules) {
			stats.filesFiltered.Add(1)
			return nil
		}

		files <- path
		return nil
	})

	close(files)
	wg.Wait()

	writerErr := writer.close()

	result := stats.result(time.Since(start))

	fi.Logger.Printf("index complete: dir=%s indexed=%d unchanged=%d too_large=%d filtered=%d chunks_embedded=%d errors=%d elapsed=%s",
		dir, result.FilesIndexed, result.FilesUnchanged, result.FilesTooLarge, result.FilesFiltered,
		result.ChunksEmbedded, result.Errors, result.Elapsed)

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

	if maxSize := fi.maxFileSizeBytes(); maxSize > 0 && info.Size() > maxSize {
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

	chunks, err := ChunkText(content, chunkSize)
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
