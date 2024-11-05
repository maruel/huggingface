// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package huggingface is the best library to fetch files from an huggingface
// repository.
package huggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/schollz/progressbar/v3"
)

// PackedFileRef is a packed reference to a file in an hugging face repository.
//
// The form is "hf:<author>/<repo>/HEAD/<file>"
//
// HEAD is the git commit reference. HEAD means the default branch. It can be
// replaced with a branch name or a commit hash. The default branch used by
// huggingface_hub official python library is "main".
//
// DEFAULT_REVISION in
// https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py
type PackedFileRef string

// RepoID returns the canonical "<author>/<repo>" for this repository.
func (p PackedFileRef) RepoID() string {
	if i := strings.LastIndexByte(string(p), '/'); i != -1 {
		if i = strings.LastIndexByte(string(p[:i]), '/'); i != -1 {
			return strings.TrimPrefix(string(p)[:i], "hf:")
		}
	}
	return ""
}

// Author returns the <author> part of the packed reference.
func (p PackedFileRef) Author() string {
	if i := strings.IndexByte(string(p), '/'); i != -1 {
		return strings.TrimPrefix(string(p)[:i], "hf:")
	}
	return ""
}

// Repo returns the <repo> part of the packed reference.
func (p PackedFileRef) Repo() string {
	if i := strings.IndexByte(string(p), '/'); i != -1 {
		if j := strings.IndexByte(string(p)[i+1:], '/'); j != -1 {
			return string(p)[i+1 : i+1+j]
		}
	}
	return ""
}

// Commish returns the HEAD part of the packed reference.
func (p PackedFileRef) Commish() string {
	if i := strings.IndexByte(string(p), '/'); i != -1 {
		if j := strings.IndexByte(string(p)[i+1:], '/'); j != -1 {
			if k := strings.IndexByte(string(p)[j+1:], '/'); k != -1 {
				return string(p)[i+j+1 : i+1+j+k]
			}
		}
	}
	return ""
}

// ModelRef returns the ModelRef reference to the repo containing this file.
func (p PackedFileRef) ModelRef() ModelRef {
	return ModelRef{Author: p.Author(), Repo: p.Repo()}
}

// Basename returns the basename part of this reference.
func (p PackedFileRef) Basename() string {
	if i := strings.LastIndexByte(string(p), '/'); i != -1 {
		return string(p)[i+1:]
	}
	return ""
}

// RepoURL returns the canonical URL for this repository.
func (p PackedFileRef) RepoURL() string {
	return "https://huggingface.co/" + p.RepoID()
}

// Validate checks for obvious errors in the string.
func (p PackedFileRef) Validate() error {
	if !strings.HasPrefix(string(p), "hf:") {
		return fmt.Errorf("invalid file ref %q", p)
	}
	parts := strings.Split(string(p)[4:], "/")
	if len(parts) != 4 {
		return fmt.Errorf("invalid file ref %q", p)
	}
	if len(parts[2]) == 0 {
		return fmt.Errorf("invalid file ref %q", p)
	}
	for _, p := range parts {
		if len(p) < 3 {
			return fmt.Errorf("invalid file ref %q", p)
		}
	}
	return nil
}

// PackedRepoRef is a packed reference to an hugging face repository.
//
// The form is "hf:<author>/<repo>"
type PackedRepoRef string

// RepoID returns the canonical "<author>/<repo>" for this repository.
func (p PackedRepoRef) RepoID() string {
	if !strings.HasPrefix(string(p), "hf:") {
		return ""
	}
	return string(p[3:])
}

// ModelRef converts to a ModelRef reference.
func (p PackedRepoRef) ModelRef() ModelRef {
	out := ModelRef{}
	if parts := strings.SplitN(p.RepoID(), "/", 2); len(parts) == 2 {
		out.Author = parts[0]
		out.Repo = parts[1]
	}
	return out
}

// RepoURL returns the canonical URL for this repository.
func (p PackedRepoRef) RepoURL() string {
	return "https://huggingface.co/" + strings.TrimPrefix(string(p), "hf:")
}

// Validate checks for obvious errors in the string.
func (p PackedRepoRef) Validate() error {
	if strings.Count(string(p), "/") != 1 {
		return fmt.Errorf("invalid repo %q", p)
	}
	if !strings.HasPrefix(string(p), "hf:") {
		return fmt.Errorf("invalid repo %q", p)
	}
	parts := strings.Split(string(p)[3:], "/")
	if len(parts) != 2 {
		return fmt.Errorf("invalid repo %q", p)
	}
	for _, p := range parts {
		if len(p) < 3 {
			return fmt.Errorf("invalid repo %q", p)
		}
	}
	return nil
}

// ModelRef is a reference to a model stored on https://huggingface.co
type ModelRef struct {
	// Author is the owner, either a person or an organization.
	Author string
	// Repo is the name of the repository owned by the Author.
	Repo string

	_ struct{}
}

// RepoID is a shorthand to return .m.Author + "/" + m.Repo
func (m *ModelRef) RepoID() string {
	return m.Author + "/" + m.Repo
}

// URL returns the Model's canonical URL.
func (m *ModelRef) URL() string {
	return "https://huggingface.co/" + m.RepoID()
}

// Model is a model stored on https://huggingface.co
type Model struct {
	ModelRef
	// Upstream is the upstream repo when the model is based on another one.
	Upstream ModelRef

	// Information filled by GetModel():

	// Tensor is the native quantization of the weight. Frequently "BF16" for
	// "bfloat16" type. This is found in config.json in Upstream.
	TensorType string
	// Number of weights. Has direct impact on performance and memory usage.
	NumWeights int64
	// ContentLength is the number of tokens that the LLM can take as context
	// when relevant. Has impact on performance and memory usage. Not relevant
	// for image generators.
	ContextLength int
	// License is the license of the weights, for whatever that means. Use the
	// name for well known licences (e.g. "Apache v2.0" or "MIT") or an URL for
	// custom licenses.
	License string
	// LicenseURL is the URL to the license file.
	LicenseURL string
	// Files is the list of files in the repository.
	Files []string
	// Created is the time the repository was created. It can be at the earliest
	// 2022-03-02 as documented at
	// https://huggingface.co/docs/hub/api#repo-listing-api.
	Created time.Time
	// Modified is the last time the repository was modified.
	Modified time.Time

	_ struct{}
}

// Client is the client for https://huggingface.co/.
type Client struct {
	// serverBase is mocked in test.
	serverBase string
	token      string
	hubHomeDir string
	// Structure is described at https://huggingface.co/docs/huggingface_hub/guides/manage-cache
	// - .locks/ (not implemented)
	// - models--*/
	//   - blobs/
	//     - (sha256 files, not SHA1!)
	//   - refs/
	//     - <git ref>: contains hex encoding git commit hash in snapshots/.
	//   - snapshots/
	//     - <git commit hash>/
	//       - (symlinks to blobs)
	hubCacheDir string
}

// New returns a new *Client client to download files and list repositories.
//
// It uses the endpoints as described at https://huggingface.co/docs/hub/api.
//
// Respects the following environment variables described at
// https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables:
// HF_HOME, HF_HUB_CACHE, HF_TOKEN_PATh and HF_TOKEN.
func New(token string) (*Client, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}
	hubHomeDir := filepath.Join(home, ".cache", "huggingface")
	if e := os.Getenv("HF_HOME"); e != "" {
		hubHomeDir = e
	}
	hubCacheDir := filepath.Join(hubHomeDir, "hub")
	if e := os.Getenv("HF_HUB_CACHE"); e != "" {
		hubCacheDir = e
	}
	if err := os.MkdirAll(hubCacheDir, 0o777); err != nil {
		return nil, err
	}
	tokenFile := filepath.Join(hubHomeDir, "token")
	if e := os.Getenv("HF_TOKEN_PATH"); e != "" {
		tokenFile = e
	}

	if token == "" {
		if token = os.Getenv("HF_TOKEN"); token == "" {
			if t, err := os.ReadFile(tokenFile); err == nil {
				token = strings.TrimSpace(string(t))
				slog.Info("hf", "message", "found token from cache", "file", tokenFile)
			}
		}
	} else if _, err := os.Stat(tokenFile); os.IsNotExist(err) {
		if err = os.WriteFile(tokenFile, []byte(token), 0o644); err != nil {
			return nil, err
		}
		slog.Info("hf", "message", "saved token to cache", "file", tokenFile)
	}
	if token != "" && !strings.HasPrefix(token, "hf_") {
		return nil, errors.New("token is invalid, it must have prefix 'hf_'")
	}
	return &Client{
		serverBase:  "https://huggingface.co",
		token:       token,
		hubHomeDir:  hubHomeDir,
		hubCacheDir: hubCacheDir,
	}, nil
}

// https://huggingface.co/docs/hub/api#get-apimodelsrepoid-or-apimodelsrepoidrevisionrevision
type modelInfoResponse struct {
	LastModified time.Time `json:"lastModified"`
	Siblings     []struct {
		Filename string `json:"rfilename"`
	}
	CardData struct {
		BaseModel  string `json:"base_model"`
		License    string
		LicenseURL string `json:"license_link"`
	} `json:"cardData"`
	CreatedAt   time.Time `json:"createdAt"`
	SafeTensors struct {
		Parameters map[string]int64
		Total      int64
	} `json:"safetensors"`
}

// GetModelInfo fills the supplied Model with information from the HuggingFace Hub.
//
// Use "main" as ref unless you need a specific commit.
func (c *Client) GetModelInfo(ctx context.Context, m *Model, ref string) error {
	slog.Info("hf", "model", m.RepoID())
	url := c.serverBase + "/api/models/" + m.RepoID() + "/revision/" + ref
	resp, err := authGet(ctx, "GET", url, c.token, nil)
	if err != nil {
		return fmt.Errorf("failed to list repoID %s: %w", m.RepoID(), err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to list repoID %s: %w", m.RepoID(), err)
	}
	r := modelInfoResponse{}
	if err := json.Unmarshal(b, &r); err != nil {
		return fmt.Errorf("failed to parse list repoID %s response: %w", m.RepoID(), err)
	}
	m.Files = make([]string, len(r.Siblings))
	m.Created = r.CreatedAt
	m.Modified = r.LastModified
	parts := strings.Split(r.CardData.BaseModel, "/")
	if len(parts) == 2 {
		m.Upstream.Author = parts[0]
		m.Upstream.Repo = parts[1]
	}
	m.License = r.CardData.License
	m.LicenseURL = r.CardData.LicenseURL
	for i := range r.Siblings {
		m.Files[i] = r.Siblings[i].Filename
	}
	for k, s := range r.SafeTensors.Parameters {
		if s > m.NumWeights {
			m.TensorType = k
			m.NumWeights = s
		}
	}
	if m.NumWeights == 0 {
		m.NumWeights = r.SafeTensors.Total
	}
	return nil
}

// for sha1 only.
var reCommit = regexp.MustCompile("^[a-fA-F0-9]{40}$")

// EnsureFile ensures the file is available, downloads it otherwise.
//
// Similar to https://huggingface.co/docs/huggingface_hub/package_reference/file_download
func (c *Client) EnsureFile(ctx context.Context, ref PackedFileRef) (string, error) {
	// TODO: Currently hard-coded for models.
	mdlDir, err := c.prepareModelCache(ref.RepoID())
	if err != nil {
		return "", err
	}
	// Look if there's a ref already there.
	commitish := ""
	etag := ""
	cmtPath := filepath.Join(mdlDir, "refs", ref.Commish())
	if b, err := ioutil.ReadFile(cmtPath); err == nil {
		commitish = string(bytes.TrimSpace(b))
		if !reCommit.MatchString(commitish) {
			return "", fmt.Errorf("%s contains %q which is not a commit hash", cmtPath, commitish)
		}
	} else {
		if commitish, etag, _, err = c.GetFileInfo(ctx, ref); err != nil {
			return "", err
		}
		if !reCommit.MatchString(commitish) {
			return "", fmt.Errorf("%q is not a commit hash", commitish)
		}
	}
	snapshotDir := filepath.Join(mdlDir, "snapshots", commitish)
	if err = os.MkdirAll(snapshotDir, 0o777); err != nil {
		return "", err
	}
	// Symlink is present?
	ln := filepath.Join(snapshotDir, ref.Basename())
	if _, err := os.Stat(ln); err == nil {
		return ln, err
	}
	// We have to download it.
	blob := filepath.Join(mdlDir, "blobs", etag)
	url := c.serverBase + "/" + ref.RepoID() + "/resolve/" + ref.Commish() + "/" + ref.Basename() + "?download=true"
	if err = DownloadFile(ctx, url, blob, c.token); err != nil {
		return "", err
	}
	rel, err := filepath.Rel(filepath.Dir(ln), blob)
	if err != nil {
		return "", err
	}
	if err = os.Symlink(rel, ln); err != nil {
		return "", err
	}
	return ln, nil
}

// GetFileInfo retrieves the information about the file.
//
// Returns the commitish, etag, size.
func (c *Client) GetFileInfo(ctx context.Context, ref PackedFileRef) (string, string, int64, error) {
	hdr := map[string]string{"Accept-Encoding": "identity"}
	url := c.serverBase + "/" + ref.RepoID() + "/resolve/" + ref.Commish() + "/" + ref.Basename() + "?download=true"
	resp, err := authGet(ctx, "GET", url, c.token, hdr)
	if err != nil {
		return "", "", 0, err
	}
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	commitIsh := resp.Header.Get("X-Repo-Commit")
	if commitIsh == "" {
		return "", "", 0, errors.New("missing header X-Repo-Commit")
	}

	etag := resp.Header.Get("X-Linked-Etag")
	if etag == "" {
		etag = resp.Header.Get("Etag")
	}

	sizeStr := resp.Header.Get("X-Linked-Size")
	if sizeStr == "" {
		sizeStr = resp.Header.Get("Content-Length")
	}
	if sizeStr == "" {
		return "", "", 0, errors.New("missing header X-Linked-Size")
	}
	size, err := strconv.ParseInt(sizeStr, 10, 64)
	if err != nil {
		return "", "", 0, fmt.Errorf("invalid header X-Linked-Size %q", sizeStr)
	}
	//resp.Header.Get("Location") or url
	slog.Info("hf", "file_info", "ref", ref, "commit", commitIsh, "etag", etag, "size", size)
	return commitIsh, etag, size, nil
}

// prepareModelCache returns the absolute path to store the model's cache.
//
// Makes sure blobs/, refs/ and snapshots/ exist.
func (c *Client) prepareModelCache(repoID string) (string, error) {
	name := "models--" + strings.ReplaceAll(repoID, "/", "-")
	mdlDir := filepath.Join(c.hubCacheDir, name)
	for _, n := range []string{"blobs", "refs", "snapshots"} {
		if err := os.MkdirAll(filepath.Join(mdlDir, n), 0o777); err != nil {
			return "", err
		}
	}
	return mdlDir, nil
}

//

// DownloadFile downloads a file optionally with a bearer token.
//
// This is a generic utility function. It retries 429 and 5xx automatically.
//
// It prints a progress bar if the file is at least 100kiB.
func DownloadFile(ctx context.Context, url, dst string, token string) error {
	slog.Info("hf", "downloading", url)
	resp, err := authGet(ctx, "GET", url, token, nil)
	if err != nil {
		return fmt.Errorf("failed to download %q: %w", dst, err)
	}
	defer resp.Body.Close()
	// Only then create the file.
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o666)
	if err != nil {
		return fmt.Errorf("failed to download %q: %w", dst, err)
	}
	defer f.Close()

	// Check if resp.ContentLength is small and skip output in this case.
	if resp.ContentLength >= 100*1024 {
		bar := progressbar.DefaultBytes(resp.ContentLength, "downloading")
		_, err = io.Copy(io.MultiWriter(f, bar), resp.Body)
	} else {
		_, err = io.Copy(f, resp.Body)
	}
	return err
}

// authGet does an authenticated HTTP request with a Bearer token.
//
// Method must be HEAD or GET.
func authGet(ctx context.Context, method, url, token string, hdr map[string]string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, nil)
	if err != nil {
		// Unlikely.
		return nil, err
	}
	if token != "" {
		req.Header.Add("Authorization", "Bearer "+token)
	}
	for k, v := range hdr {
		req.Header.Add(k, v)
	}
	for i := 0; i < 10; i++ {
		resp, err := http.DefaultClient.Do(req)
		if resp.StatusCode != 200 {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == 401 {
				if token != "" {
					return nil, fmt.Errorf("double check if your token is valid: %s", resp.Status)
				}
				return nil, fmt.Errorf("a valid token is likely required: %s", resp.Status)
			}
			if resp.StatusCode == 429 || (resp.StatusCode >= 500 && resp.StatusCode < 600) {
				// Sleep and retry.
				time.Sleep(time.Duration(i+1) * time.Second)
				continue
			}
			return nil, fmt.Errorf("request status: %s", resp.Status)
		}
		return resp, err
	}
	return nil, errors.New("failed retrying on 429")
}
