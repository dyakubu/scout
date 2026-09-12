package config

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// field describes how to get and set one config value by its dotted key
// (e.g. "embedder.batch_size"), for the "scout config get/set" CLI
// commands. List fields (allowed_extensions, ignore_dirs, ignore_patterns)
// are readable but leave set nil - a multi-line list is far more readable
// edited directly in the file than passed as a single CLI argument, so Set
// rejects them with a message pointing at the file instead of silently
// doing something awkward.
type field struct {
	get func(cfg *Config) string
	set func(cfg *Config, value string) error
}

var fields = map[string]field{
	"embedder.model_path": {
		get: func(cfg *Config) string { return cfg.Embedder.ModelPath },
		set: func(cfg *Config, value string) error { cfg.Embedder.ModelPath = value; return nil },
	},
	"embedder.tokenizer_path": {
		get: func(cfg *Config) string { return cfg.Embedder.TokenizerPath },
		set: func(cfg *Config, value string) error { cfg.Embedder.TokenizerPath = value; return nil },
	},
	"embedder.ort_library_path": {
		get: func(cfg *Config) string { return cfg.Embedder.OrtLibraryPath },
		set: func(cfg *Config, value string) error { cfg.Embedder.OrtLibraryPath = value; return nil },
	},
	"embedder.batch_size": {
		get: func(cfg *Config) string { return strconv.Itoa(cfg.Embedder.BatchSize) },
		set: func(cfg *Config, value string) error {
			n, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("expected an integer, got %q", value)
			}
			cfg.Embedder.BatchSize = n
			return nil
		},
	},
	"media.model_dir": {
		get: func(cfg *Config) string { return cfg.Media.ModelDir },
		set: func(cfg *Config, value string) error { cfg.Media.ModelDir = value; return nil },
	},
	"media.allowed_extensions": {
		get: func(cfg *Config) string { return strings.Join(cfg.Media.AllowedExtensions, ", ") },
	},
	"db.path": {
		get: func(cfg *Config) string { return cfg.DB.Path },
		set: func(cfg *Config, value string) error { cfg.DB.Path = value; return nil },
	},
	"index.max_file_size_mb": {
		get: func(cfg *Config) string { return strconv.Itoa(cfg.Index.MaxFileSizeMB) },
		set: func(cfg *Config, value string) error {
			n, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("expected an integer, got %q", value)
			}
			cfg.Index.MaxFileSizeMB = n
			return nil
		},
	},
	"index.allowed_extensions": {
		get: func(cfg *Config) string { return strings.Join(cfg.Index.AllowedExtensions, ", ") },
	},
	// Read-only, like the list fields below: a map can't be expressed as a
	// single command-line value, so it's edited in the file.
	"index.max_file_size_mb_by_type": {
		get: func(cfg *Config) string {
			exts := make([]string, 0, len(cfg.Index.MaxFileSizeMBByType))
			for ext := range cfg.Index.MaxFileSizeMBByType {
				exts = append(exts, ext)
			}
			sort.Strings(exts)

			pairs := make([]string, 0, len(exts))
			for _, ext := range exts {
				pairs = append(pairs, fmt.Sprintf("%s = %d", ext, cfg.Index.MaxFileSizeMBByType[ext]))
			}
			return strings.Join(pairs, ", ")
		},
	},
	"index.index_hidden_dirs": {
		get: func(cfg *Config) string { return strconv.FormatBool(cfg.Index.IndexHiddenDirs) },
		set: func(cfg *Config, value string) error {
			b, err := strconv.ParseBool(value)
			if err != nil {
				return fmt.Errorf("expected true or false, got %q", value)
			}
			cfg.Index.IndexHiddenDirs = b
			return nil
		},
	},
	"index.ignore_dir_markers": {
		get: func(cfg *Config) string { return strings.Join(cfg.Index.IgnoreDirMarkers, ", ") },
	},
	"index.ignore_dirs": {
		get: func(cfg *Config) string { return strings.Join(cfg.Index.IgnoreDirs, ", ") },
	},
	"index.ignore_patterns": {
		get: func(cfg *Config) string { return strings.Join(cfg.Index.IgnorePatterns, ", ") },
	},
	"search.max_results": {
		get: func(cfg *Config) string { return strconv.Itoa(cfg.Search.MaxResults) },
		set: func(cfg *Config, value string) error {
			n, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("expected an integer, got %q", value)
			}
			cfg.Search.MaxResults = n
			return nil
		},
	},
	"search.max_results_per_file": {
		get: func(cfg *Config) string { return strconv.Itoa(cfg.Search.MaxResultsPerFile) },
		set: func(cfg *Config, value string) error {
			n, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("expected an integer, got %q", value)
			}
			cfg.Search.MaxResultsPerFile = n
			return nil
		},
	},
	"search.max_media_results": {
		get: func(cfg *Config) string { return strconv.Itoa(cfg.Search.MaxMediaResults) },
		set: func(cfg *Config, value string) error {
			n, err := strconv.Atoi(value)
			if err != nil {
				return fmt.Errorf("expected an integer, got %q", value)
			}
			cfg.Search.MaxMediaResults = n
			return nil
		},
	},
}

// Get returns the current value of a scalar config key (e.g.
// "embedder.batch_size").
func Get(cfg *Config, key string) (string, error) {
	f, ok := fields[key]
	if !ok {
		return "", fmt.Errorf("unknown config key %q", key)
	}
	return f.get(cfg), nil
}

// Set updates a scalar config key on cfg. It does not save cfg to disk;
// call Save separately. List fields (e.g. "index.allowed_extensions") are
// rejected with a message pointing at the config file, since they aren't
// settable through a single CLI argument.
func Set(cfg *Config, key, value string) error {
	f, ok := fields[key]
	if !ok {
		return fmt.Errorf("unknown config key %q", key)
	}

	if f.set == nil {
		path, err := Path()
		if err != nil {
			return err
		}
		return fmt.Errorf("%q is a list field and can't be set from the CLI - edit it directly in %s", key, path)
	}

	return f.set(cfg, value)
}
