package search

import (
	"database/sql"
	"errors"
	"fmt"
	"log"
	"path/filepath"
	"strings"
	"time"

	vecembed "github.com/asg017/sqlite-vec-go-bindings/ncruces"
	"github.com/dyakubu/scout/embedder"
)

const (
	defaultMax = 5

	// candidatePoolSize bounds how many nearest neighbors are pulled from
	// vec_chunks before Restrict/model filtering is applied in Go (vec0
	// tables here carry no metadata columns of their own, so filtering by
	// path or embedding_model can't happen inside the KNN scan itself).
	// A narrow --restrict can still return fewer than Max results if the
	// true matches within it fall outside this pool - acceptable at
	// personal-corpus scale, not a general pre-filtered ANN solution.
	candidatePoolSize = 200
)

type Searcher struct {
	Db       *sql.DB
	Embedder embedder.Embedder
	Logger   *log.Logger
}

// Options configures a single Search call.
type Options struct {
	// Max is the number of results to return. Defaults to 5 if <= 0.
	Max int

	// Restrict, if non-empty, limits results to files under this
	// directory (itself included). Resolved to an absolute path before
	// matching.
	Restrict string
}

// Result is one matched chunk, ranked by similarity to the query.
type Result struct {
	Path      string
	StartLine int
	EndLine   int
	Content   string
	Score     float64
}

func NewSearcher(db *sql.DB, embedder embedder.Embedder, logger *log.Logger) (*Searcher, error) {
	if db == nil {
		return nil, errors.New("db is nil. cannot initialize searcher")
	}

	if embedder == nil {
		return nil, errors.New("embedder is nil. cannot initialize searcher")
	}

	if logger == nil {
		return nil, errors.New("logger is nil. cannot initialize searcher")
	}

	return &Searcher{Db: db, Embedder: embedder, Logger: logger}, nil
}

// Search embeds query and returns the most similar indexed chunks, ranked
// highest score first.
func (s *Searcher) Search(query string, opts Options) ([]Result, error) {
	start := time.Now()

	max := opts.Max
	if max <= 0 {
		max = defaultMax
	}

	var restrictPrefix string
	if opts.Restrict != "" {
		abs, err := filepath.Abs(opts.Restrict)
		if err != nil {
			return nil, fmt.Errorf("resolving --restrict path: %w", err)
		}
		restrictPrefix = abs
	}

	embedStart := time.Now()
	embeddings, err := s.Embedder.Embed([]string{query})
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}
	embedElapsed := time.Since(embedStart)

	queryBlob, err := vecembed.SerializeFloat32(embeddings[0])
	if err != nil {
		return nil, fmt.Errorf("serializing query embedding: %w", err)
	}

	modelID := s.Embedder.ModelID()

	queryStart := time.Now()

	rows, err := s.Db.Query(`
		SELECT c.content, c.start_line, c.end_line, f.path, v.distance, c.embedding_model
		FROM (
			SELECT rowid, distance FROM vec_chunks
			WHERE embedding MATCH ?
			ORDER BY distance
			LIMIT ?
		) v
		JOIN chunks c ON c.id = v.rowid
		JOIN files f ON f.id = c.file_id
		ORDER BY v.distance
	`, queryBlob, candidatePoolSize)
	if err != nil {
		return nil, fmt.Errorf("querying vec_chunks: %w", err)
	}
	defer rows.Close()

	var results []Result

	for rows.Next() && len(results) < max {
		var (
			content        string
			startLine      int
			endLine        int
			path           string
			distance       float64
			embeddingModel string
		)

		if err := rows.Scan(&content, &startLine, &endLine, &path, &distance, &embeddingModel); err != nil {
			return nil, fmt.Errorf("reading search result: %w", err)
		}

		// A chunk embedded by a different model lives in an incomparable
		// vector space - including it would mix rankings that were never
		// meant to be compared, so it's excluded rather than shown with a
		// misleading score.
		if embeddingModel != modelID {
			continue
		}

		if restrictPrefix != "" && !underDir(path, restrictPrefix) {
			continue
		}

		results = append(results, Result{
			Path:      path,
			StartLine: startLine,
			EndLine:   endLine,
			Content:   content,
			// Embeddings are L2-normalized, so for unit vectors
			// ||a-b||^2 = 2 - 2*cos(a,b), giving cos(a,b) = 1 - distance^2/2.
			Score: 1 - (distance*distance)/2,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("reading search results: %w", err)
	}

	queryElapsed := time.Since(queryStart)

	s.Logger.Printf("search: query=%q embed=%s query_exec=%s results=%d total=%s",
		query, embedElapsed, queryElapsed, len(results), time.Since(start))

	return results, nil
}

// underDir reports whether path is dir itself or falls under it, avoiding
// a naive prefix match incorrectly treating "/src-old" as under "/src".
func underDir(path, dir string) bool {
	if path == dir {
		return true
	}
	return strings.HasPrefix(path, dir+string(filepath.Separator))
}
