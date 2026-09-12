package indexer

import (
	"database/sql"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/dyakubu/scout/config"
	scoutdb "github.com/dyakubu/scout/db"
	"github.com/dyakubu/scout/embedder/embeddertest"

	_ "github.com/ncruces/go-sqlite3/driver"
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

func newTestIndexer(t *testing.T) *FileIndexer {
	t.Helper()
	return &FileIndexer{
		Db: newTestDB(t),
		IndexConfig: config.IndexConfig{
			AllowedExtensions: []string{".txt", ".go"},
			IgnorePatterns:    []string{".DS_Store", "*.tmp"},
			IgnoreDirs:        []string{"node_modules", ".git"},
			IgnoreDirMarkers:  []string{"pyvenv.cfg"},
		},
		MediaConfig: config.MediaConfig{
			AllowedExtensions: []string{".jpg", ".png"},
		},
		Logger: log.New(os.Stderr, "", 0),
	}
}

func TestClassifyFile(t *testing.T) {
	fi := newTestIndexer(t)

	tests := []struct {
		name string
		want fileKind
	}{
		{"notes.txt", fileKindText},
		{"main.go", fileKindText},
		{"NOTES.TXT", fileKindText}, // extension matching is case-insensitive
		{"photo.jpg", fileKindMedia},
		{"photo.PNG", fileKindMedia},
		{"song.mp3", fileKindNone},  // not in either allowed-extensions list
		{".DS_Store", fileKindNone}, // matches an ignore pattern
		{"cache.tmp", fileKindNone}, // matches the "*.tmp" glob pattern
	}

	for _, tt := range tests {
		got := fi.classifyFile("/root", filepath.Join("/root", tt.name), tt.name, nil)
		if got != tt.want {
			t.Errorf("classifyFile(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestClassifyFile_GitignoreExcludes(t *testing.T) {
	fi := newTestIndexer(t)

	rules := []gitignoreRule{{pattern: "secret.txt"}}

	got := fi.classifyFile("/root", "/root/secret.txt", "secret.txt", rules)
	if got != fileKindNone {
		t.Errorf("classifyFile with a matching gitignore rule = %v, want fileKindNone", got)
	}

	// A different file in the same allowed-extensions list must still
	// classify normally - the gitignore rule shouldn't over-match.
	got = fi.classifyFile("/root", "/root/notes.txt", "notes.txt", rules)
	if got != fileKindText {
		t.Errorf("classifyFile(notes.txt) = %v, want fileKindText", got)
	}
}

func TestShouldSkipDir_ExactNameMatch(t *testing.T) {
	fi := newTestIndexer(t)

	if !fi.shouldSkipDir("/root", "/root/node_modules", "node_modules", nil) {
		t.Error("shouldSkipDir(node_modules) = false, want true")
	}
	if fi.shouldSkipDir("/root", "/root/src", "src", nil) {
		t.Error("shouldSkipDir(src) = true, want false")
	}
}

func TestShouldSkipDir_Marker(t *testing.T) {
	fi := newTestIndexer(t)

	venv := filepath.Join(t.TempDir(), "some-venv-name")
	if err := os.MkdirAll(venv, 0o755); err != nil {
		t.Fatalf("creating venv dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(venv, "pyvenv.cfg"), []byte(""), 0o644); err != nil {
		t.Fatalf("creating marker file: %v", err)
	}

	if !fi.shouldSkipDir(filepath.Dir(venv), venv, filepath.Base(venv), nil) {
		t.Error("shouldSkipDir with a pyvenv.cfg marker present = false, want true")
	}
}

func TestShouldSkipDir_Gitignore(t *testing.T) {
	fi := newTestIndexer(t)

	rules := []gitignoreRule{{pattern: "build", dirOnly: true}}

	if !fi.shouldSkipDir("/root", "/root/build", "build", rules) {
		t.Error("shouldSkipDir(build) with a matching dirOnly gitignore rule = false, want true")
	}
}

func TestIsUnchanged(t *testing.T) {
	fi := newTestIndexer(t)

	unchanged, err := fi.isUnchanged("/nonexistent.txt", 12345)
	if err != nil {
		t.Fatalf("isUnchanged (no row): %v", err)
	}
	if unchanged {
		t.Error("isUnchanged for a path with no files row = true, want false")
	}

	if _, err := fi.Db.Exec(
		`INSERT INTO files (path, modified_at, file_hash, status) VALUES (?, ?, ?, 'ok')`,
		"/tracked.txt", int64(1000), []byte("hash"),
	); err != nil {
		t.Fatalf("seeding files row: %v", err)
	}

	unchanged, err = fi.isUnchanged("/tracked.txt", 1000)
	if err != nil {
		t.Fatalf("isUnchanged (same mtime): %v", err)
	}
	if !unchanged {
		t.Error("isUnchanged with a matching modified_at = false, want true")
	}

	unchanged, err = fi.isUnchanged("/tracked.txt", 2000)
	if err != nil {
		t.Fatalf("isUnchanged (different mtime): %v", err)
	}
	if unchanged {
		t.Error("isUnchanged with a different modified_at = true, want false")
	}
}

// ChunkText itself is deliberately not covered in depth here - it's
// already flagged (issues #3, #8) for replacement with boundary/token-
// aware chunking, so a detailed spec against its current fixed-length
// behavior would mostly lock in something already known to be wrong. This
// is just enough to catch a crash or a badly broken boundary.
func TestChunkText_Basics(t *testing.T) {
	chunks, err := ChunkText("", 10)
	if err != nil {
		t.Fatalf("ChunkText(\"\"): %v", err)
	}
	if len(chunks) != 0 {
		t.Errorf("ChunkText(\"\") returned %d chunks, want 0", len(chunks))
	}

	chunks, err = ChunkText("hello", 10)
	if err != nil {
		t.Fatalf("ChunkText: %v", err)
	}
	if len(chunks) != 1 || chunks[0].content != "hello" {
		t.Errorf("ChunkText(\"hello\", 10) = %+v, want a single chunk containing \"hello\"", chunks)
	}

	// Multi-byte runes must not be split mid-codepoint.
	chunks, err = ChunkText("héllo wörld", 5)
	if err != nil {
		t.Fatalf("ChunkText (unicode): %v", err)
	}
	var rebuilt string
	for _, c := range chunks {
		rebuilt += c.content
	}
	if rebuilt != "héllo wörld" {
		t.Errorf("rebuilt chunks = %q, want the original text intact", rebuilt)
	}

	if _, err := ChunkText("text", 0); err == nil {
		t.Error("ChunkText with chunkSize=0: expected an error, got none")
	}
}

// lockDir makes dir unreadable for the rest of the test and restores it
// afterwards. It skips the test where that can't be arranged: as root,
// whose access checks are bypassed, and on Windows, where os.Chmod only
// toggles the read-only attribute and a directory stays listable.
func lockDir(t *testing.T, dir string) {
	t.Helper()

	if runtime.GOOS == "windows" {
		t.Skip("os.Chmod cannot make a directory unreadable on Windows")
	}
	if os.Getuid() == 0 {
		t.Skip("running as root, which can read a 0000 directory anyway")
	}

	if err := os.Chmod(dir, 0o000); err != nil {
		t.Fatalf("chmod %s: %v", dir, err)
	}
	t.Cleanup(func() { os.Chmod(dir, 0o755) })
}

// TestIndexDirectory_SkipsUnreadableDirs covers a walk that meets a
// directory it has no permission to read. Indexing a home directory hits
// this routinely on macOS (~/.Trash, much of ~/Library), and one such
// directory used to abort the entire run.
func TestIndexDirectory_SkipsUnreadableDirs(t *testing.T) {
	root := t.TempDir()

	// Named so the unreadable directory is walked before "zzz", proving
	// the walk carries on past it rather than stopping there.
	for _, dir := range []string{"aaa", "zzz"} {
		if err := os.Mkdir(filepath.Join(root, dir), 0o755); err != nil {
			t.Fatalf("creating %s: %v", dir, err)
		}
		path := filepath.Join(root, dir, "notes.txt")
		if err := os.WriteFile(path, []byte(dir+" contents"), 0o644); err != nil {
			t.Fatalf("writing %s: %v", path, err)
		}
	}

	locked := filepath.Join(root, "mmm-locked")
	if err := os.Mkdir(locked, 0o755); err != nil {
		t.Fatalf("creating locked dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(locked, "secret.txt"), []byte("secret"), 0o644); err != nil {
		t.Fatalf("writing into locked dir: %v", err)
	}
	lockDir(t, locked)

	fi := newTestIndexer(t)
	fi.Embedder = &embeddertest.Embedder{
		Vectors: map[string][]float32{
			"aaa contents": embeddertest.UnitVector(384, 0),
			"zzz contents": embeddertest.UnitVector(384, 1),
		},
	}

	stats, err := fi.IndexDirectory(root, true)
	if err != nil {
		t.Fatalf("IndexDirectory: %v", err)
	}

	if stats.FilesIndexed != 2 {
		t.Errorf("FilesIndexed = %d, want 2 (both readable files, either side of the locked dir)", stats.FilesIndexed)
	}
	if stats.PathsUnreadable != 1 {
		t.Errorf("PathsUnreadable = %d, want 1", stats.PathsUnreadable)
	}
	if stats.Errors != 0 {
		t.Errorf("Errors = %d, want 0 - an unreadable directory is skipped, not an error", stats.Errors)
	}
}

// TestIndexDirectory_UnreadableRootIsAnError is the other half: scout can
// skip what it stumbles into, but being pointed at a directory it can't
// read has to fail rather than report an empty run.
func TestIndexDirectory_UnreadableRootIsAnError(t *testing.T) {
	root := filepath.Join(t.TempDir(), "locked")
	if err := os.Mkdir(root, 0o755); err != nil {
		t.Fatalf("creating locked dir: %v", err)
	}
	lockDir(t, root)

	fi := newTestIndexer(t)
	fi.Embedder = &embeddertest.Embedder{}

	stats, err := fi.IndexDirectory(root, true)
	if err == nil {
		t.Fatal("IndexDirectory: want an error for an unreadable root, got nil")
	}
	if stats.SawAnything() {
		t.Errorf("stats report work that never happened: %+v", stats)
	}
}
