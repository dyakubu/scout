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
	"github.com/dyakubu/scout/config"
	"github.com/dyakubu/scout/embedder"
)

const (
	// fallbackMax is used when SearchConfig's configured defaults are
	// unset (e.g. an existing config file predating the [search] section
	// decodes them as zero) - see NewSearcher.
	fallbackMax = 5

	// candidatePoolSize bounds how many nearest neighbors are pulled from
	// vec_chunks/vec_media before Restrict/model filtering is applied in
	// Go (vec0 tables here carry no metadata columns of their own, so
	// filtering by path or embedding_model can't happen inside the KNN
	// scan itself). A narrow --restrict can still return fewer than Max
	// results if the true matches within it fall outside this pool -
	// acceptable at personal-corpus scale, not a general pre-filtered ANN
	// solution.
	candidatePoolSize = 200
)

type Searcher struct {
	Db            *sql.DB
	Embedder      embedder.Embedder
	MediaEmbedder embedder.MediaEmbedder // optional; nil means media search is skipped entirely
	Logger        *log.Logger

	defaultMax      int
	defaultMediaMax int
}

// Options configures a single Search call.
type Options struct {
	// Max is the number of file results to return. Defaults to the
	// Searcher's configured default if <= 0.
	Max int

	// MediaMax is the number of media results to return. Defaults to the
	// Searcher's configured default if <= 0. Meaningless if the Searcher
	// has no MediaEmbedder configured.
	MediaMax int

	// Restrict, if non-empty, limits results to files under this
	// directory (itself included). Resolved to an absolute path before
	// matching. Applies to both file and media results.
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

// MediaResult is one matched media embedding, ranked by similarity to the
// query. There's no content snippet the way Result has one - there's no
// text to show for an image. FrameIndex is always 0 for a still image;
// it's meaningful once video support samples multiple frames per file.
type MediaResult struct {
	Path       string
	FrameIndex int
	Score      float64
}

// Results is the outcome of one Search call: two independent result sets
// that are never merged or compared against each other, since they come
// from two separate embedding spaces (the text embedder's vs. CLIP's).
type Results struct {
	Files []Result
	Media []MediaResult
}

// NewSearcher constructs a Searcher. mediaEmbedder may be nil, meaning
// media search is unavailable and Search will always return an empty
// Media result set. searchConfig's fields are the configured default
// result counts (config.SearchConfig); a zero value (e.g. from a config
// file predating the [search] section) falls back to fallbackMax rather
// than returning zero results by default.
func NewSearcher(db *sql.DB, embedder embedder.Embedder, mediaEmbedder embedder.MediaEmbedder, searchConfig config.SearchConfig, logger *log.Logger) (*Searcher, error) {
	if db == nil {
		return nil, errors.New("db is nil. cannot initialize searcher")
	}

	if embedder == nil {
		return nil, errors.New("embedder is nil. cannot initialize searcher")
	}

	if logger == nil {
		return nil, errors.New("logger is nil. cannot initialize searcher")
	}

	defaultMax := searchConfig.MaxResults
	if defaultMax <= 0 {
		defaultMax = fallbackMax
	}

	defaultMediaMax := searchConfig.MaxMediaResults
	if defaultMediaMax <= 0 {
		defaultMediaMax = fallbackMax
	}

	return &Searcher{
		Db:              db,
		Embedder:        embedder,
		MediaEmbedder:   mediaEmbedder,
		Logger:          logger,
		defaultMax:      defaultMax,
		defaultMediaMax: defaultMediaMax,
	}, nil
}

// Search embeds query and returns the most similar indexed chunks and, if
// a MediaEmbedder is configured, the most similar indexed media - two
// independent result sets, ranked and returned separately.
func (s *Searcher) Search(query string, opts Options) (Results, error) {
	start := time.Now()

	var restrictPrefix string
	if opts.Restrict != "" {
		abs, err := filepath.Abs(opts.Restrict)
		if err != nil {
			return Results{}, fmt.Errorf("resolving --restrict path: %w", err)
		}
		restrictPrefix = abs
	}

	max := opts.Max
	if max <= 0 {
		max = s.defaultMax
	}

	files, embedElapsed, queryElapsed, err := s.searchFiles(query, max, restrictPrefix)
	if err != nil {
		return Results{}, err
	}

	var media []MediaResult
	var mediaEmbedElapsed, mediaQueryElapsed time.Duration
	if s.MediaEmbedder != nil {
		mediaMax := opts.MediaMax
		if mediaMax <= 0 {
			mediaMax = s.defaultMediaMax
		}

		media, mediaEmbedElapsed, mediaQueryElapsed, err = s.searchMedia(query, mediaMax, restrictPrefix)
		if err != nil {
			// Media is additive, so a media failure costs the media
			// results and nothing else - returning the error here would
			// throw away the text results this search already has in
			// hand, turning "no images matched" into "no search at all".
			// The worker dying mid-run is the case that matters: a model
			// directory missing its files fails the worker's eager load
			// (see media/clip.py), which only surfaces here, on the first
			// job sent to it.
			s.Logger.Printf("media search unavailable, returning text results only: %v", err)
			media = nil
		}
	}

	s.Logger.Printf("search: query=%q embed=%s query_exec=%s results=%d media_embed=%s media_query_exec=%s media_results=%d total=%s",
		query, embedElapsed, queryElapsed, len(files), mediaEmbedElapsed, mediaQueryElapsed, len(media), time.Since(start))

	return Results{Files: files, Media: media}, nil
}

// searchFiles embeds query with s.Embedder and returns the most similar
// chunks from vec_chunks.
func (s *Searcher) searchFiles(query string, max int, restrictPrefix string) ([]Result, time.Duration, time.Duration, error) {
	embedStart := time.Now()
	embeddings, err := s.Embedder.Embed([]string{query})
	if err != nil {
		return nil, 0, 0, fmt.Errorf("embedding query: %w", err)
	}
	embedElapsed := time.Since(embedStart)

	queryBlob, err := vecembed.SerializeFloat32(embeddings[0])
	if err != nil {
		return nil, 0, 0, fmt.Errorf("serializing query embedding: %w", err)
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
		return nil, 0, 0, fmt.Errorf("querying vec_chunks: %w", err)
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
			return nil, 0, 0, fmt.Errorf("reading search result: %w", err)
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
		return nil, 0, 0, fmt.Errorf("reading search results: %w", err)
	}

	return results, embedElapsed, time.Since(queryStart), nil
}

// searchMedia embeds query with s.MediaEmbedder.EmbedText - into the same
// space EmbedImage's results live in - and returns the most similar media
// embeddings from vec_media. Only called when s.MediaEmbedder is non-nil.
func (s *Searcher) searchMedia(query string, max int, restrictPrefix string) ([]MediaResult, time.Duration, time.Duration, error) {
	embedStart := time.Now()
	queryEmbedding, err := s.MediaEmbedder.EmbedText(query)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("embedding query for media search: %w", err)
	}
	embedElapsed := time.Since(embedStart)

	queryBlob, err := vecembed.SerializeFloat32(queryEmbedding)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("serializing media query embedding: %w", err)
	}

	modelID := s.MediaEmbedder.ModelID()

	queryStart := time.Now()

	rows, err := s.Db.Query(`
		SELECT m.frame_index, f.path, v.distance, m.embedding_model
		FROM (
			SELECT rowid, distance FROM vec_media
			WHERE embedding MATCH ?
			ORDER BY distance
			LIMIT ?
		) v
		JOIN media_embeddings m ON m.id = v.rowid
		JOIN files f ON f.id = m.file_id
		ORDER BY v.distance
	`, queryBlob, candidatePoolSize)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("querying vec_media: %w", err)
	}
	defer rows.Close()

	var results []MediaResult

	for rows.Next() && len(results) < max {
		var (
			frameIndex     int
			path           string
			distance       float64
			embeddingModel string
		)

		if err := rows.Scan(&frameIndex, &path, &distance, &embeddingModel); err != nil {
			return nil, 0, 0, fmt.Errorf("reading media search result: %w", err)
		}

		if embeddingModel != modelID {
			continue
		}

		if restrictPrefix != "" && !underDir(path, restrictPrefix) {
			continue
		}

		results = append(results, MediaResult{
			Path:       path,
			FrameIndex: frameIndex,
			Score:      1 - (distance*distance)/2,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, 0, 0, fmt.Errorf("reading media search results: %w", err)
	}

	return results, embedElapsed, time.Since(queryStart), nil
}

// underDir reports whether path is dir itself or falls under it, avoiding
// a naive prefix match incorrectly treating "/src-old" as under "/src".
func underDir(path, dir string) bool {
	if path == dir {
		return true
	}
	return strings.HasPrefix(path, dir+string(filepath.Separator))
}
