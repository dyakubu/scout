package indexer

import (
	"database/sql"
	"fmt"

	// Importing this package registers a build of SQLite (via the ncruces
	// driver) with the sqlite-vec extension compiled in, and provides
	// SerializeFloat32 for encoding embeddings into the BLOB format it
	// expects.
	vecembed "github.com/asg017/sqlite-vec-go-bindings/ncruces"
)

// ChunkRecord is a single chunk, embedded and ready to be persisted.
type ChunkRecord struct {
	ChunkIndex     int
	Content        string
	ContentHash    []byte
	StartLine      int
	EndLine        int
	EmbeddingModel string
	Embedding      []float32
}

// FileRecord is one file's metadata plus a batch of its embedded chunks,
// persisted together as a single write. A file with no chunks (e.g. empty)
// is still emitted with an empty Chunks slice, so it's still recorded in
// files. A large file is split across multiple FileRecords sharing the
// same Path, so no single write has to hold an entire file's chunks in
// memory or in one transaction.
type FileRecord struct {
	Path       string
	ModifiedAt int64
	FileHash   []byte

	Chunks []ChunkRecord
}

// MediaEmbeddingRecord is a single embedded "unit" of a media file - the
// whole file for a still image, or one sampled frame for a future video.
type MediaEmbeddingRecord struct {
	FrameIndex     int
	StartMS        *int64
	EndMS          *int64
	ContentHash    []byte
	EmbeddingModel string
	Embedding      []float32
}

// MediaRecord is one file's metadata plus its media embeddings, persisted
// together as a single write - the media equivalent of FileRecord.
type MediaRecord struct {
	Path       string
	ModifiedAt int64
	FileHash   []byte

	Embeddings []MediaEmbeddingRecord
}

// Emitter accepts file and media records for persistence. Implementations
// must be safe for concurrent calls to Emit/EmitMedia from multiple
// goroutines.
type Emitter interface {
	Emit(record FileRecord) error
	EmitMedia(record MediaRecord) error
}

// writeRequest is a tagged union of the two record kinds dbWriter accepts,
// so both flow through the single writer goroutine over one channel rather
// than requiring two independently-drained channels (which would allow
// them to race against each other for no benefit, since the db.Exec calls
// still have to happen one at a time either way).
type writeRequest struct {
	file  *FileRecord
	media *MediaRecord
}

// dbWriter is the single writer to the database. File workers call
// Emit/EmitMedia concurrently, which just queues the record on a channel;
// one background goroutine drains that channel and performs all the
// db.Exec calls, so the database only ever sees a single writer at a time.
type dbWriter struct {
	db       *sql.DB
	requests chan writeRequest
	done     chan struct{}
	errCh    chan error

	// Tracks which files' old chunks/media embeddings have already been
	// cleared this run, keyed by files.id. A file can arrive across
	// multiple FileRecords (see FileRecord), so clearing has to happen
	// exactly once, on the first record seen for that file, not once per
	// record. Shared between text and media records - a given file_id is
	// one or the other, never both, so no cross-talk.
	clearedFiles map[int64]bool
}

func newDBWriter(db *sql.DB, bufferSize int) *dbWriter {
	return &dbWriter{
		db:           db,
		requests:     make(chan writeRequest, bufferSize),
		done:         make(chan struct{}),
		errCh:        make(chan error, 1),
		clearedFiles: make(map[int64]bool),
	}
}

// start launches the single writer goroutine. Must be called before
// Emit/EmitMedia.
func (w *dbWriter) start() {
	go func() {
		defer close(w.done)

		for req := range w.requests {
			var err error
			switch {
			case req.file != nil:
				err = w.writeFile(*req.file)
			case req.media != nil:
				err = w.writeMedia(*req.media)
			}

			if err != nil {
				select {
				case w.errCh <- err:
				default:
				}
			}
		}
	}()
}

// Emit queues a file record for the writer goroutine. Safe to call
// concurrently.
func (w *dbWriter) Emit(record FileRecord) error {
	w.requests <- writeRequest{file: &record}
	return nil
}

// EmitMedia queues a media record for the writer goroutine. Safe to call
// concurrently.
func (w *dbWriter) EmitMedia(record MediaRecord) error {
	w.requests <- writeRequest{media: &record}
	return nil
}

// close stops accepting new records, waits for the writer goroutine to
// drain the queue, and returns the first write error encountered, if any.
func (w *dbWriter) close() error {
	close(w.requests)
	<-w.done

	select {
	case err := <-w.errCh:
		return err
	default:
		return nil
	}
}

func (w *dbWriter) writeFile(record FileRecord) error {
	tx, err := w.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	// Upsert on path so reindexing the same file doesn't create duplicate
	// files rows; harmless to repeat across multiple records for the same
	// file, since it's idempotent.
	var fileID int64
	err = tx.QueryRow(
		`INSERT INTO files (path, modified_at, file_hash, status, last_error)
		 VALUES (?, ?, ?, 'ok', NULL)
		 ON CONFLICT(path) DO UPDATE SET
		   modified_at = excluded.modified_at,
		   file_hash = excluded.file_hash,
		   status = 'ok',
		   last_error = NULL
		 RETURNING id`,
		record.Path, record.ModifiedAt, record.FileHash,
	).Scan(&fileID)
	if err != nil {
		return fmt.Errorf("upserting file: %w", err)
	}

	if !w.clearedFiles[fileID] {
		// vec_chunks has no real foreign key to chunks (sqlite-vec's vec0
		// tables don't support them), so its rows must be deleted
		// explicitly here rather than relying on ON DELETE CASCADE.
		if _, err := tx.Exec(`DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)`, fileID); err != nil {
			return fmt.Errorf("clearing old vectors: %w", err)
		}
		if _, err := tx.Exec(`DELETE FROM chunks WHERE file_id = ?`, fileID); err != nil {
			return fmt.Errorf("clearing old chunks: %w", err)
		}
		w.clearedFiles[fileID] = true
	}

	for _, chunk := range record.Chunks {
		res, err := tx.Exec(
			`INSERT INTO chunks (file_id, chunk_index, content, content_hash, start_line, end_line, embedding_model)
			 VALUES (?, ?, ?, ?, ?, ?, ?)`,
			fileID, chunk.ChunkIndex, chunk.Content, chunk.ContentHash,
			chunk.StartLine, chunk.EndLine, chunk.EmbeddingModel,
		)
		if err != nil {
			return fmt.Errorf("inserting chunk %d: %w", chunk.ChunkIndex, err)
		}

		chunkID, err := res.LastInsertId()
		if err != nil {
			return fmt.Errorf("reading chunk id: %w", err)
		}

		embeddingBlob, err := vecembed.SerializeFloat32(chunk.Embedding)
		if err != nil {
			return fmt.Errorf("serializing embedding: %w", err)
		}

		if _, err := tx.Exec(`INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)`, chunkID, embeddingBlob); err != nil {
			return fmt.Errorf("inserting vector for chunk %d: %w", chunk.ChunkIndex, err)
		}
	}

	return tx.Commit()
}

func (w *dbWriter) writeMedia(record MediaRecord) error {
	tx, err := w.db.Begin()
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback()

	// Same upsert-on-path pattern as writeFile - files is shared between
	// text and media records.
	var fileID int64
	err = tx.QueryRow(
		`INSERT INTO files (path, modified_at, file_hash, status, last_error)
		 VALUES (?, ?, ?, 'ok', NULL)
		 ON CONFLICT(path) DO UPDATE SET
		   modified_at = excluded.modified_at,
		   file_hash = excluded.file_hash,
		   status = 'ok',
		   last_error = NULL
		 RETURNING id`,
		record.Path, record.ModifiedAt, record.FileHash,
	).Scan(&fileID)
	if err != nil {
		return fmt.Errorf("upserting file: %w", err)
	}

	if !w.clearedFiles[fileID] {
		// vec_media has no real foreign key to media_embeddings (sqlite-vec's
		// vec0 tables don't support them), so its rows must be deleted
		// explicitly here rather than relying on ON DELETE CASCADE.
		if _, err := tx.Exec(`DELETE FROM vec_media WHERE rowid IN (SELECT id FROM media_embeddings WHERE file_id = ?)`, fileID); err != nil {
			return fmt.Errorf("clearing old media vectors: %w", err)
		}
		if _, err := tx.Exec(`DELETE FROM media_embeddings WHERE file_id = ?`, fileID); err != nil {
			return fmt.Errorf("clearing old media embeddings: %w", err)
		}
		w.clearedFiles[fileID] = true
	}

	for _, embedding := range record.Embeddings {
		res, err := tx.Exec(
			`INSERT INTO media_embeddings (file_id, frame_index, start_ms, end_ms, content_hash, embedding_model)
			 VALUES (?, ?, ?, ?, ?, ?)`,
			fileID, embedding.FrameIndex, embedding.StartMS, embedding.EndMS,
			embedding.ContentHash, embedding.EmbeddingModel,
		)
		if err != nil {
			return fmt.Errorf("inserting media embedding %d: %w", embedding.FrameIndex, err)
		}

		embeddingID, err := res.LastInsertId()
		if err != nil {
			return fmt.Errorf("reading media embedding id: %w", err)
		}

		embeddingBlob, err := vecembed.SerializeFloat32(embedding.Embedding)
		if err != nil {
			return fmt.Errorf("serializing embedding: %w", err)
		}

		if _, err := tx.Exec(`INSERT INTO vec_media (rowid, embedding) VALUES (?, ?)`, embeddingID, embeddingBlob); err != nil {
			return fmt.Errorf("inserting vector for media embedding %d: %w", embedding.FrameIndex, err)
		}
	}

	return tx.Commit()
}
