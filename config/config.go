package config

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/BurntSushi/toml"
)

type Config struct {
	Embedder EmbedderConfig `toml:"embedder"`
	Media    MediaConfig    `toml:"media"`
	DB       DBConfig       `toml:"db"`
	Index    IndexConfig    `toml:"index"`
	Search   SearchConfig   `toml:"search"`
}

type EmbedderConfig struct {
	ModelPath      string `toml:"model_path"`
	TokenizerPath  string `toml:"tokenizer_path"`
	OrtLibraryPath string `toml:"ort_library_path"`
	BatchSize      int    `toml:"batch_size"`
}

// MediaConfig configures the media (image/video) embedding worker.
// ModelDir names a directory rather than individual files: the worker
// resolves the CLIP towers and tokenizer it needs within it. They ship in
// scout's release archive, and a directory missing any of them is an
// error, never a download.
type MediaConfig struct {
	ModelDir string `toml:"model_dir"`

	// AllowedExtensions is a closed list, same convention as
	// IndexConfig.AllowedExtensions: a file is only routed to the media
	// embedder if its extension is here.
	AllowedExtensions []string `toml:"allowed_extensions"`
}

type DBConfig struct {
	Path string `toml:"path"`
}

// SearchConfig configures scout find's default result counts.
// MaxMediaResults is independent of MaxResults - the two are separate
// result sets (matching files vs. matching media), never merged or
// compared against each other.
type SearchConfig struct {
	MaxResults      int `toml:"max_results"`
	MaxMediaResults int `toml:"max_media_results"`
}

type IndexConfig struct {
	// MaxFileSizeMB is the fallback ceiling for any extension not listed
	// in MaxFileSizeMBByType.
	MaxFileSizeMB int `toml:"max_file_size_mb"`

	// MaxFileSizeMBByType overrides that per file extension, keyed without
	// a leading dot and matched case-insensitively. A separate key rather
	// than making MaxFileSizeMB a table, which would fail to decode every
	// config file that already has it as a number.
	MaxFileSizeMBByType map[string]int `toml:"max_file_size_mb_by_type"`

	AllowedExtensions []string `toml:"allowed_extensions"`
	IgnoreDirs        []string `toml:"ignore_dirs"`
	IgnorePatterns    []string `toml:"ignore_patterns"`

	// IndexHiddenDirs walks dot-prefixed directories, skipped by default.
	// Opt-in, not opt-out: configs are never backfilled, so the zero value
	// has to be the behavior worth defaulting to.
	IndexHiddenDirs bool `toml:"index_hidden_dirs"`

	// IgnoreDirMarkers names files whose presence inside a directory means
	// that whole directory should be skipped, regardless of what the
	// directory itself is named. This is for the case IgnoreDirs' exact
	// name matching can't handle - e.g. a Python virtualenv is always
	// identifiable by a pyvenv.cfg file at its root, but the directory
	// holding it can be named anything ("venv", ".venv", "rl-venv", ...).
	IgnoreDirMarkers []string `toml:"ignore_dir_markers"`
}

// scoutDir returns scout's per-user directory, following each OS's own
// convention for where per-user config lives (e.g. ~/.config on Linux,
// ~/Library/Application Support on macOS, %AppData% on Windows). The
// config file, database, and log file all live here.
func scoutDir() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", fmt.Errorf("resolving user config directory: %w", err)
	}
	return filepath.Join(dir, "scout"), nil
}

// Path returns the resolved location of scout's global config file.
func Path() (string, error) {
	dir, err := scoutDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "scoutconfig.toml"), nil
}

// LogPath returns the resolved location of scout's log file.
func LogPath() (string, error) {
	dir, err := scoutDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "scout.log"), nil
}

// OpenLog opens scout's log file for appending, creating scout's directory
// and the file itself if they don't exist yet. The caller is responsible
// for closing it.
func OpenLog() (*os.File, error) {
	path, err := LogPath()
	if err != nil {
		return nil, err
	}

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, fmt.Errorf("creating scout directory: %w", err)
	}

	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("opening log file: %w", err)
	}

	return f, nil
}

// Load reads scout's global config for normal runtime use, writing the
// embedded default to disk first if no config file exists yet, and
// resolving DB.Path and the embedder's asset paths to absolute paths.
func Load() (*Config, error) {
	cfg, path, err := loadRaw()
	if err != nil {
		return nil, err
	}

	if !filepath.IsAbs(cfg.DB.Path) {
		cfg.DB.Path = filepath.Join(filepath.Dir(path), cfg.DB.Path)
	}

	if err := resolveEmbedderPaths(&cfg.Embedder); err != nil {
		return nil, err
	}

	if err := resolveMediaPaths(&cfg.Media); err != nil {
		return nil, err
	}

	return cfg, nil
}

// resolveEmbedderPaths makes the embedder's asset paths absolute,
// resolving relative ones against the running binary's own directory -
// not the working directory, and not the config directory used for
// DB.Path. The model, tokenizer, and onnxruntime library ship alongside
// the binary itself (see scout's release archive layout), so that's what
// a relative path here is relative to.
func resolveEmbedderPaths(cfg *EmbedderConfig) error {
	if filepath.IsAbs(cfg.ModelPath) && filepath.IsAbs(cfg.TokenizerPath) && filepath.IsAbs(cfg.OrtLibraryPath) {
		return nil
	}

	dir, err := execDir()
	if err != nil {
		return fmt.Errorf("resolving embedder asset paths: %w", err)
	}

	if !filepath.IsAbs(cfg.ModelPath) {
		cfg.ModelPath = filepath.Join(dir, cfg.ModelPath)
	}
	if !filepath.IsAbs(cfg.TokenizerPath) {
		cfg.TokenizerPath = filepath.Join(dir, cfg.TokenizerPath)
	}
	if !filepath.IsAbs(cfg.OrtLibraryPath) {
		cfg.OrtLibraryPath = filepath.Join(dir, cfg.OrtLibraryPath)
	}

	return nil
}

// resolveMediaPaths makes the media model directory absolute, resolving a
// relative path against the running binary's own directory - same
// convention as resolveEmbedderPaths, and for the same reason: it's where
// scout's bundled assets live relative to the binary, not the working
// directory.
//
// An empty ModelDir is left untouched rather than resolved to execDir
// itself: existing config files predating this field decode it as "",
// since loadRaw only ever writes the embedded default for a config file
// that doesn't exist yet - it never backfills new fields into one that
// already exists. Empty is treated as "not configured" so callers can
// detect and handle that explicitly, instead of silently pointing the
// media worker at the binary's own directory.
func resolveMediaPaths(cfg *MediaConfig) error {
	if cfg.ModelDir == "" || filepath.IsAbs(cfg.ModelDir) {
		return nil
	}

	dir, err := execDir()
	if err != nil {
		return fmt.Errorf("resolving media model directory: %w", err)
	}

	cfg.ModelDir = filepath.Join(dir, cfg.ModelDir)

	return nil
}

// execDir returns the directory containing the running scout binary.
// Symlinks are resolved first because package managers (e.g. Homebrew)
// install the real binary under a versioned path and only symlink it
// onto PATH - without this, execDir would return the symlink's directory
// instead of the one the bundled assets actually live in.
func execDir() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("resolving executable path: %w", err)
	}

	resolved, err := filepath.EvalSymlinks(exe)
	if err != nil {
		return "", fmt.Errorf("resolving executable symlinks: %w", err)
	}

	return filepath.Dir(resolved), nil
}

// LoadForEdit reads the config file exactly as stored on disk, without
// resolving DB.Path to an absolute path. Use this (with Save) when the
// config is about to be modified and written back - Load's absolute
// DB.Path is only meant for runtime use and would otherwise get baked
// permanently into the file the first time any value is changed.
func LoadForEdit() (*Config, error) {
	cfg, _, err := loadRaw()
	return cfg, err
}

// Save writes cfg back to the config file as TOML. This re-serializes the
// whole file from cfg's current values, so any comments in the existing
// file are lost - hand-editing the file directly is the way to change
// values without disturbing its comments.
func Save(cfg *Config) error {
	path, err := Path()
	if err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("opening config for write: %w", err)
	}
	defer f.Close()

	if err := toml.NewEncoder(f).Encode(cfg); err != nil {
		return fmt.Errorf("encoding config: %w", err)
	}

	return nil
}

func loadRaw() (*Config, string, error) {
	path, err := Path()
	if err != nil {
		return nil, "", err
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, "", fmt.Errorf("creating config directory: %w", err)
		}
		if err := os.WriteFile(path, defaultConfig, 0o644); err != nil {
			return nil, "", fmt.Errorf("writing default config: %w", err)
		}
	} else if err != nil {
		return nil, "", fmt.Errorf("checking config file: %w", err)
	}

	var cfg Config
	if _, err := toml.DecodeFile(path, &cfg); err != nil {
		return nil, "", fmt.Errorf("parsing config %s: %w", path, err)
	}

	return &cfg, path, nil
}
