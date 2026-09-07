package search

import (
	"database/sql"
	"errors"
	"io"
	"log"
	"path/filepath"
	"strings"
	"testing"
	"time"

	vecembed "github.com/asg017/sqlite-vec-go-bindings/ncruces"
	"github.com/dyakubu/scout/config"
	scoutdb "github.com/dyakubu/scout/db"
	"github.com/dyakubu/scout/embedder/embeddertest"

	_ "github.com/ncruces/go-sqlite3/driver"
)

const (
	textDim  = 384
	mediaDim = 512
)

func newTestDB(t *testing.T) *sql.DB {
	t.Helper()

	path := filepath.Join(t.TempDir(), "test.db")
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		t.Fatalf("opening db: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	if err := scoutdb.InitDB(db); err != nil {
		t.Fatalf("InitDB: %v", err)
	}

	return db
}

func insertFile(t *testing.T, db *sql.DB, path string) int64 {
	t.Helper()

	res, err := db.Exec(
		`INSERT INTO files (path, modified_at, file_hash, status) VALUES (?, ?, ?, 'ok')`,
		path, time.Now().Unix(), []byte("filehash"),
	)
	if err != nil {
		t.Fatalf("inserting file %s: %v", path, err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		t.Fatalf("file id: %v", err)
	}
	return id
}

func insertChunk(t *testing.T, db *sql.DB, fileID int64, chunkIndex int, content, embeddingModel string, vec []float32) {
	t.Helper()

	res, err := db.Exec(
		`INSERT INTO chunks (file_id, chunk_index, content, content_hash, start_line, end_line, embedding_model)
		 VALUES (?, ?, ?, ?, ?, ?, ?)`,
		fileID, chunkIndex, content, []byte("chunkhash"), chunkIndex+1, chunkIndex+2, embeddingModel,
	)
	if err != nil {
		t.Fatalf("inserting chunk: %v", err)
	}
	chunkID, err := res.LastInsertId()
	if err != nil {
		t.Fatalf("chunk id: %v", err)
	}

	blob, err := vecembed.SerializeFloat32(vec)
	if err != nil {
		t.Fatalf("serializing vector: %v", err)
	}
	if _, err := db.Exec(`INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)`, chunkID, blob); err != nil {
		t.Fatalf("inserting vec_chunks row: %v", err)
	}
}

func insertMedia(t *testing.T, db *sql.DB, fileID int64, frameIndex int, embeddingModel string, vec []float32) {
	t.Helper()

	res, err := db.Exec(
		`INSERT INTO media_embeddings (file_id, frame_index, content_hash, embedding_model) VALUES (?, ?, ?, ?)`,
		fileID, frameIndex, []byte("mediahash"), embeddingModel,
	)
	if err != nil {
		t.Fatalf("inserting media_embeddings: %v", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		t.Fatalf("media_embeddings id: %v", err)
	}

	blob, err := vecembed.SerializeFloat32(vec)
	if err != nil {
		t.Fatalf("serializing vector: %v", err)
	}
	if _, err := db.Exec(`INSERT INTO vec_media (rowid, embedding) VALUES (?, ?)`, id, blob); err != nil {
		t.Fatalf("inserting vec_media row: %v", err)
	}
}

func newLogger(t *testing.T) *log.Logger {
	t.Helper()
	return log.New(io.Discard, "", 0)
}

func TestSearch_RanksByDistance(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"query": embeddertest.UnitVector(textDim, 0)},
	}

	fileA := insertFile(t, db, "/repo/a.go")
	fileB := insertFile(t, db, "/repo/b.go")
	insertChunk(t, db, fileA, 0, "matches the query", emb.ModelID(), embeddertest.UnitVector(textDim, 0))
	insertChunk(t, db, fileB, 0, "unrelated content", emb.ModelID(), embeddertest.UnitVector(textDim, 1))

	s, err := NewSearcher(db, emb, nil, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	results, err := s.Search("query", Options{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results.Files) != 2 {
		t.Fatalf("len(Files) = %d, want 2", len(results.Files))
	}
	if results.Files[0].Path != "/repo/a.go" {
		t.Errorf("Files[0].Path = %q, want the exact-match file first", results.Files[0].Path)
	}
	if results.Files[0].Score <= results.Files[1].Score {
		t.Errorf("Files[0].Score (%.3f) should be > Files[1].Score (%.3f)", results.Files[0].Score, results.Files[1].Score)
	}
}

func TestSearch_FiltersStaleEmbeddingModel(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"query": embeddertest.UnitVector(textDim, 0)},
		Model:   "current-model",
	}

	file := insertFile(t, db, "/repo/stale.go")
	insertChunk(t, db, file, 0, "stale embedding", "old-model", embeddertest.UnitVector(textDim, 0))

	s, err := NewSearcher(db, emb, nil, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	results, err := s.Search("query", Options{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results.Files) != 0 {
		t.Errorf("len(Files) = %d, want 0 (a chunk embedded by a different model must be excluded)", len(results.Files))
	}
}

func TestSearch_RespectsMax(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"query": embeddertest.UnitVector(textDim, 0)},
	}

	for i := 0; i < 5; i++ {
		file := insertFile(t, db, filepath.Join("/repo", string(rune('a'+i))+".go"))
		insertChunk(t, db, file, 0, "content", emb.ModelID(), embeddertest.UnitVector(textDim, 0))
	}

	s, err := NewSearcher(db, emb, nil, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	results, err := s.Search("query", Options{Max: 2})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results.Files) != 2 {
		t.Errorf("len(Files) = %d, want 2", len(results.Files))
	}
}

func TestSearch_DefaultMaxFromSearchConfig(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"query": embeddertest.UnitVector(textDim, 0)},
	}

	for i := 0; i < 5; i++ {
		file := insertFile(t, db, filepath.Join("/repo", string(rune('a'+i))+".go"))
		insertChunk(t, db, file, 0, "content", emb.ModelID(), embeddertest.UnitVector(textDim, 0))
	}

	s, err := NewSearcher(db, emb, nil, config.SearchConfig{MaxResults: 3}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	// Options.Max left at 0 - Search must fall back to the Searcher's
	// configured default, not the package's own fallbackMax (5).
	results, err := s.Search("query", Options{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results.Files) != 3 {
		t.Errorf("len(Files) = %d, want 3 (from SearchConfig.MaxResults)", len(results.Files))
	}
}

func TestSearch_RestrictFiltersByPath(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"query": embeddertest.UnitVector(textDim, 0)},
	}

	inside := insertFile(t, db, "/repo/src/a.go")
	// "/repo/src-old" is NOT under "/repo/src" - a naive strings.HasPrefix
	// without the separator check would wrongly include it.
	sibling := insertFile(t, db, "/repo/src-old/b.go")
	insertChunk(t, db, inside, 0, "inside src", emb.ModelID(), embeddertest.UnitVector(textDim, 0))
	insertChunk(t, db, sibling, 0, "inside src-old", emb.ModelID(), embeddertest.UnitVector(textDim, 0))

	s, err := NewSearcher(db, emb, nil, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	results, err := s.Search("query", Options{Restrict: "/repo/src"})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results.Files) != 1 {
		t.Fatalf("len(Files) = %d, want 1", len(results.Files))
	}
	if results.Files[0].Path != "/repo/src/a.go" {
		t.Errorf("Files[0].Path = %q, want /repo/src/a.go (not the src-old sibling)", results.Files[0].Path)
	}
}

func TestSearch_EmbedErrorPropagates(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{Err: errors.New("boom")}

	s, err := NewSearcher(db, emb, nil, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	if _, err := s.Search("query", Options{}); err == nil {
		t.Fatal("Search: expected an error from a failing Embedder, got none")
	} else if !strings.Contains(err.Error(), "boom") {
		t.Errorf("error = %q, want it to wrap the underlying \"boom\" error", err.Error())
	}
}

func TestSearch_NoMediaEmbedder_MediaEmpty(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"query": embeddertest.UnitVector(textDim, 0)},
	}

	// A media row exists, but with no MediaEmbedder configured, Search
	// must never attempt to query it.
	file := insertFile(t, db, "/repo/photo.jpg")
	insertMedia(t, db, file, 0, "clip-model", embeddertest.UnitVector(mediaDim, 0))

	s, err := NewSearcher(db, emb, nil, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	results, err := s.Search("query", Options{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results.Media) != 0 {
		t.Errorf("len(Media) = %d, want 0 (no MediaEmbedder configured)", len(results.Media))
	}
}

func TestSearch_MediaResults(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"a red circle": embeddertest.UnitVector(textDim, 0)},
	}
	mediaEmb := &embeddertest.MediaEmbedder{
		TextVectors: map[string][]float32{"a red circle": embeddertest.UnitVector(mediaDim, 0)},
	}

	matching := insertFile(t, db, "/repo/circle.jpg")
	unrelated := insertFile(t, db, "/repo/blue.jpg")
	insertMedia(t, db, matching, 0, mediaEmb.ModelID(), embeddertest.UnitVector(mediaDim, 0))
	insertMedia(t, db, unrelated, 0, mediaEmb.ModelID(), embeddertest.UnitVector(mediaDim, 1))

	s, err := NewSearcher(db, emb, mediaEmb, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	results, err := s.Search("a red circle", Options{})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results.Media) != 2 {
		t.Fatalf("len(Media) = %d, want 2", len(results.Media))
	}
	if results.Media[0].Path != "/repo/circle.jpg" {
		t.Errorf("Media[0].Path = %q, want the exact-match image first", results.Media[0].Path)
	}
	if results.Media[0].Score <= results.Media[1].Score {
		t.Errorf("Media[0].Score (%.3f) should be > Media[1].Score (%.3f)", results.Media[0].Score, results.Media[1].Score)
	}
}

func TestSearch_MediaMaxRespected(t *testing.T) {
	db := newTestDB(t)
	emb := &embeddertest.Embedder{
		Vectors: map[string][]float32{"query": embeddertest.UnitVector(textDim, 0)},
	}
	mediaEmb := &embeddertest.MediaEmbedder{
		TextVectors: map[string][]float32{"query": embeddertest.UnitVector(mediaDim, 0)},
	}

	for i := 0; i < 4; i++ {
		file := insertFile(t, db, filepath.Join("/repo", string(rune('a'+i))+".jpg"))
		insertMedia(t, db, file, 0, mediaEmb.ModelID(), embeddertest.UnitVector(mediaDim, 0))
	}

	s, err := NewSearcher(db, emb, mediaEmb, config.SearchConfig{}, newLogger(t))
	if err != nil {
		t.Fatalf("NewSearcher: %v", err)
	}

	results, err := s.Search("query", Options{MediaMax: 1})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results.Media) != 1 {
		t.Errorf("len(Media) = %d, want 1", len(results.Media))
	}
}

func TestNewSearcher_RequiresDbAndEmbedder(t *testing.T) {
	if _, err := NewSearcher(nil, &embeddertest.Embedder{}, nil, config.SearchConfig{}, newLogger(t)); err == nil {
		t.Error("NewSearcher with nil db: expected an error, got none")
	}

	db := newTestDB(t)
	if _, err := NewSearcher(db, nil, nil, config.SearchConfig{}, newLogger(t)); err == nil {
		t.Error("NewSearcher with nil embedder: expected an error, got none")
	}
}
