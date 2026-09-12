package config

import (
	"os"
	"path/filepath"
	"testing"
)

// sandboxHome points os.UserConfigDir() (and therefore every path this
// package resolves) at a fresh temp directory, isolating each test from
// the real user's config and from each other. This needs every env var
// os.UserConfigDir() might consult, not just $HOME - it resolves
// differently per OS: macOS always uses $HOME (which is why HOME alone
// was enough there), Linux/BSD prefer $XDG_CONFIG_HOME over $HOME/.config
// if it's set at all (so a CI runner with XDG_CONFIG_HOME already set
// would otherwise make every test share that one real, unsandboxed
// directory instead of getting its own), and Windows ignores HOME
// entirely in favor of %AppData%.
func sandboxHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("XDG_CONFIG_HOME", "")
	t.Setenv("AppData", home)
	return home
}

func TestLoad_WritesDefaultOnFirstRun(t *testing.T) {
	sandboxHome(t)

	path, err := Path()
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("config file already exists before Load: %v", err)
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if _, err := os.Stat(path); err != nil {
		t.Errorf("config file not written by Load: %v", err)
	}

	if cfg.Embedder.BatchSize != 32 {
		t.Errorf("Embedder.BatchSize = %d, want 32", cfg.Embedder.BatchSize)
	}
	if cfg.Index.MaxFileSizeMB != 2 {
		t.Errorf("Index.MaxFileSizeMB = %d, want 2", cfg.Index.MaxFileSizeMB)
	}
	if cfg.Search.MaxResults != 5 {
		t.Errorf("Search.MaxResults = %d, want 5", cfg.Search.MaxResults)
	}
	if cfg.Search.MaxMediaResults != 5 {
		t.Errorf("Search.MaxMediaResults = %d, want 5", cfg.Search.MaxMediaResults)
	}
	execDir, err := execDir()
	if err != nil {
		t.Fatalf("execDir: %v", err)
	}
	if want := filepath.Join(execDir, "models", "media"); cfg.Media.ModelDir != want {
		t.Errorf("Media.ModelDir = %q, want %q (the model ships alongside the binary)", cfg.Media.ModelDir, want)
	}
}

func TestLoad_DBPathResolvedRelativeToConfigDir(t *testing.T) {
	sandboxHome(t)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	path, err := Path()
	if err != nil {
		t.Fatalf("Path: %v", err)
	}

	want := filepath.Join(filepath.Dir(path), "scout.db")
	if cfg.DB.Path != want {
		t.Errorf("DB.Path = %q, want %q", cfg.DB.Path, want)
	}
}

func TestLoad_EmbedderPathsResolvedRelativeToExecDir(t *testing.T) {
	sandboxHome(t)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	dir, err := execDir()
	if err != nil {
		t.Fatalf("execDir: %v", err)
	}

	want := filepath.Join(dir, "models", "model_qint8_avx512_vnni.onnx")
	if cfg.Embedder.ModelPath != want {
		t.Errorf("Embedder.ModelPath = %q, want %q", cfg.Embedder.ModelPath, want)
	}
	if !filepath.IsAbs(cfg.Embedder.TokenizerPath) {
		t.Errorf("Embedder.TokenizerPath = %q, want an absolute path", cfg.Embedder.TokenizerPath)
	}
	if !filepath.IsAbs(cfg.Embedder.OrtLibraryPath) {
		t.Errorf("Embedder.OrtLibraryPath = %q, want an absolute path", cfg.Embedder.OrtLibraryPath)
	}
}

func TestLoad_PreservesExistingConfig(t *testing.T) {
	sandboxHome(t)

	// Force the default to be written first...
	if _, err := Load(); err != nil {
		t.Fatalf("Load (seed): %v", err)
	}

	path, err := Path()
	if err != nil {
		t.Fatalf("Path: %v", err)
	}

	// ...then hand-edit it, simulating a user customization.
	custom := "[embedder]\nbatch_size = 999\n\n[db]\npath = \"custom.db\"\n"
	if err := os.WriteFile(path, []byte(custom), 0o644); err != nil {
		t.Fatalf("writing custom config: %v", err)
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load (custom): %v", err)
	}

	if cfg.Embedder.BatchSize != 999 {
		t.Errorf("Embedder.BatchSize = %d, want 999 (Load must not overwrite an existing file)", cfg.Embedder.BatchSize)
	}

	if filepath.Base(cfg.DB.Path) != "custom.db" {
		t.Errorf("DB.Path = %q, want a path ending in custom.db", cfg.DB.Path)
	}
}

func TestLoad_ConfigPredatingSearchSection(t *testing.T) {
	sandboxHome(t)

	path, err := Path()
	if err != nil {
		t.Fatalf("Path: %v", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("creating config dir: %v", err)
	}

	// A config file with no [search] section at all, as if written before
	// that section existed.
	old := "[embedder]\nbatch_size = 32\n"
	if err := os.WriteFile(path, []byte(old), 0o644); err != nil {
		t.Fatalf("writing old-style config: %v", err)
	}

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Documents the real, sometimes-surprising behavior: loadRaw only ever
	// writes the embedded default for a config file that doesn't exist -
	// it never backfills a new section into one that already exists. This
	// is exactly what search.NewSearcher's <=0-means-fallback logic exists
	// to handle.
	if cfg.Search.MaxResults != 0 {
		t.Errorf("Search.MaxResults = %d, want 0 (a config predating [search] has no value to decode)", cfg.Search.MaxResults)
	}
}

func TestResolveMediaPaths_EmptyStaysEmpty(t *testing.T) {
	cfg := MediaConfig{ModelDir: ""}
	if err := resolveMediaPaths(&cfg); err != nil {
		t.Fatalf("resolveMediaPaths: %v", err)
	}
	if cfg.ModelDir != "" {
		t.Errorf("ModelDir = %q, want empty to stay empty (not resolved to execDir itself)", cfg.ModelDir)
	}
}

func TestResolveMediaPaths_AbsoluteUnchanged(t *testing.T) {
	// A literal "/already/absolute" isn't actually absolute per
	// filepath.IsAbs on Windows (no drive letter) - t.TempDir() gives a
	// real, OS-appropriate absolute path on whatever platform is running.
	abs := filepath.Join(t.TempDir(), "already", "absolute")

	cfg := MediaConfig{ModelDir: abs}
	if err := resolveMediaPaths(&cfg); err != nil {
		t.Fatalf("resolveMediaPaths: %v", err)
	}
	if cfg.ModelDir != abs {
		t.Errorf("ModelDir = %q, want unchanged (%q)", cfg.ModelDir, abs)
	}
}

func TestResolveMediaPaths_RelativeResolvedAgainstExecDir(t *testing.T) {
	dir, err := execDir()
	if err != nil {
		t.Fatalf("execDir: %v", err)
	}

	cfg := MediaConfig{ModelDir: "models/media"}
	if err := resolveMediaPaths(&cfg); err != nil {
		t.Fatalf("resolveMediaPaths: %v", err)
	}

	want := filepath.Join(dir, "models", "media")
	if cfg.ModelDir != want {
		t.Errorf("ModelDir = %q, want %q", cfg.ModelDir, want)
	}
}

func TestLoadForEdit_LeavesDBPathRelative(t *testing.T) {
	sandboxHome(t)

	// Seed the default file first.
	if _, err := Load(); err != nil {
		t.Fatalf("Load (seed): %v", err)
	}

	cfg, err := LoadForEdit()
	if err != nil {
		t.Fatalf("LoadForEdit: %v", err)
	}

	if cfg.DB.Path != "scout.db" {
		t.Errorf("DB.Path = %q, want the raw relative value \"scout.db\" (LoadForEdit must not resolve it)", cfg.DB.Path)
	}
}

func TestSave_RoundTrip(t *testing.T) {
	sandboxHome(t)

	cfg, err := LoadForEdit()
	if err != nil {
		t.Fatalf("LoadForEdit: %v", err)
	}

	if err := Set(cfg, "embedder.batch_size", "64"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if err := Save(cfg); err != nil {
		t.Fatalf("Save: %v", err)
	}

	reloaded, err := LoadForEdit()
	if err != nil {
		t.Fatalf("LoadForEdit (reloaded): %v", err)
	}
	if reloaded.Embedder.BatchSize != 64 {
		t.Errorf("Embedder.BatchSize = %d, want 64 to have persisted", reloaded.Embedder.BatchSize)
	}
}

func TestGetSet_ScalarField(t *testing.T) {
	cfg := &Config{}

	if err := Set(cfg, "search.max_media_results", "3"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if cfg.Search.MaxMediaResults != 3 {
		t.Errorf("Search.MaxMediaResults = %d, want 3", cfg.Search.MaxMediaResults)
	}

	got, err := Get(cfg, "search.max_media_results")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got != "3" {
		t.Errorf("Get = %q, want \"3\"", got)
	}
}

func TestSet_InvalidInteger(t *testing.T) {
	cfg := &Config{}
	if err := Set(cfg, "embedder.batch_size", "not-a-number"); err == nil {
		t.Error("Set with a non-integer value: expected an error, got none")
	}
}

func TestSet_ListFieldRejected(t *testing.T) {
	cfg := &Config{}
	err := Set(cfg, "index.allowed_extensions", ".txt")
	if err == nil {
		t.Fatal("Set on a list field: expected an error, got none")
	}
}

func TestGetSet_UnknownKey(t *testing.T) {
	cfg := &Config{}

	if _, err := Get(cfg, "nonexistent.key"); err == nil {
		t.Error("Get with an unknown key: expected an error, got none")
	}
	if err := Set(cfg, "nonexistent.key", "value"); err == nil {
		t.Error("Set with an unknown key: expected an error, got none")
	}
}

// The shipped default config has to decode its per-type size overrides
// without disturbing the keys around them. A TOML sub-table placed before
// the end of its parent section silently captures every key that follows,
// which is exactly how this table can go wrong.
func TestLoad_SizeOverridesDecodeWithoutSwallowingSiblings(t *testing.T) {
	sandboxHome(t)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if cfg.Index.MaxFileSizeMB != 2 {
		t.Errorf("MaxFileSizeMB = %d, want the 2MB fallback", cfg.Index.MaxFileSizeMB)
	}

	for ext, want := range map[string]int{"pdf": 50, "docx": 25, "heic": 50} {
		if got := cfg.Index.MaxFileSizeMBByType[ext]; got != want {
			t.Errorf("MaxFileSizeMBByType[%q] = %d, want %d", ext, got, want)
		}
	}

	// The keys that follow the table in the file must still be their own.
	if len(cfg.Index.AllowedExtensions) == 0 {
		t.Error("AllowedExtensions is empty - the sub-table swallowed it")
	}
	if len(cfg.Index.IgnoreDirs) == 0 {
		t.Error("IgnoreDirs is empty - the sub-table swallowed it")
	}
	if len(cfg.Index.IgnorePatterns) == 0 {
		t.Error("IgnorePatterns is empty - the sub-table swallowed it")
	}
	if len(cfg.Index.IgnoreDirMarkers) == 0 {
		t.Error("IgnoreDirMarkers is empty - the sub-table swallowed it")
	}
}

// Every key in the shipped config should be readable through `scout config
// get`. A setting that exists in the file but not in the field registry is
// invisible from the command line, which is easy to miss when adding one.
func TestFields_CoverEveryShippedKey(t *testing.T) {
	sandboxHome(t)

	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	for _, key := range []string{
		"embedder.model_path", "embedder.tokenizer_path", "embedder.ort_library_path",
		"embedder.batch_size",
		"media.model_dir", "media.allowed_extensions",
		"db.path",
		"index.max_file_size_mb", "index.max_file_size_mb_by_type", "index.allowed_extensions",
		"index.ignore_dirs", "index.ignore_dir_markers", "index.ignore_patterns",
		"index.index_hidden_dirs",
		"search.max_results", "search.max_media_results", "search.max_results_per_file",
	} {
		if _, err := Get(cfg, key); err != nil {
			t.Errorf("config get %s: %v", key, err)
		}
	}
}
