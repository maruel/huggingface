// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/lmittmann/tint"
	"github.com/maruel/huggingface"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func model(ctx context.Context, hfToken, hfRepo, out string) error {
	parts := strings.Split(hfRepo, "/")
	if len(parts) != 2 {
		return fmt.Errorf("%q is not a valid huggingface repo", hfRepo)
	}
	c, err := huggingface.New(hfToken)
	if err != nil {
		return err
	}
	m := huggingface.Model{ModelRef: huggingface.ModelRef{
		Author: parts[0],
		Repo:   parts[1],
	},
	}
	if err = c.GetModelInfo(ctx, &m, "main"); err != nil {
		return err
	}
	b, err := json.MarshalIndent(m, "  ", "  ")
	if err != nil {
		return err
	}
	fmt.Printf("%s\n", b)
	return nil
}

func mainImpl(args []string) error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()
	programLevel := &slog.LevelVar{}
	programLevel.Set(slog.LevelError)
	logger := slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      programLevel,
		TimeFormat: "15:04:05.000", // Like time.TimeOnly plus milliseconds.
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			switch t := a.Value.Any().(type) {
			case string:
				if t == "" {
					return slog.Attr{}
				}
			case bool:
				if !t {
					return slog.Attr{}
				}
			case uint64:
				if t == 0 {
					return slog.Attr{}
				}
			case int64:
				if t == 0 {
					return slog.Attr{}
				}
			case float64:
				if t == 0 {
					return slog.Attr{}
				}
			case time.Time:
				if t.IsZero() {
					return slog.Attr{}
				}
			case time.Duration:
				if t == 0 {
					return slog.Attr{}
				}
			}
			return a
		},
	}))
	slog.SetDefault(logger)
	go func() {
		<-ctx.Done()
		slog.Info("main", "message", "quitting")
	}()

	fs := flag.NewFlagSet("huggingface", flag.ContinueOnError)
	verbose := fs.Bool("v", false, "Enable verbose logging")
	if len(args) == 0 {
		fs.Usage()
		return context.Canceled
	}
	switch args[0] {
	case "model":
		hfToken := fs.String("hf-token", "", "HuggingFace token")
		hfRepo := fs.String("hf-repo", "", "HuggingFace repository, e.g. \"meta-llama/Llama-3.2-1B\"")
		out := fs.String("json", "", "Save stats as a JSON file")
		if fs.Parse(args[1:]) != nil {
			return context.Canceled
		}
		if len(fs.Args()) != 0 {
			return errors.New("unexpected argument")
		}
		if *verbose {
			programLevel.Set(slog.LevelDebug)
		}
		if *hfRepo == "" {
			return errors.New("-hf-repo is required")
		}
		return model(ctx, *hfToken, *hfRepo, *out)
	default:
		fs.Usage()
		return context.Canceled
	}
}

func main() {
	if err := mainImpl(os.Args[1:]); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "huggingface: %s\n", err)
		}
		os.Exit(1)
	}
}
