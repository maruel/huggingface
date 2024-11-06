// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/lmittmann/tint"
	"github.com/mattn/go-colorable"
	"github.com/mattn/go-isatty"
)

func TestGetModelInfo(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/microsoft/Phi-3-mini-4k-instruct/revision/main" {
			t.Errorf("unexpected path, got: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(apiRepoPhi3Data))
	}))
	defer server.Close()
	os.Setenv("HF_HOME", t.TempDir())
	c, err := New("")
	if err != nil {
		t.Fatal(err)
	}
	c.serverBase = server.URL

	got := Model{
		ModelRef: ModelRef{
			Author: "microsoft",
			Repo:   "Phi-3-mini-4k-instruct",
		},
	}
	if err := c.GetModelInfo(context.Background(), &got, "main"); err != nil {
		t.Fatal(err)
	}
	want := Model{
		ModelRef: ModelRef{
			Author: "microsoft",
			Repo:   "Phi-3-mini-4k-instruct",
		},
		Files: []string{
			".gitattributes",
			"CODE_OF_CONDUCT.md",
			"LICENSE",
			"NOTICE.md",
			"README.md",
			"SECURITY.md",
			"added_tokens.json",
			"config.json",
			"configuration_phi3.py",
			"generation_config.json",
			"model-00001-of-00002.safetensors",
			"model-00002-of-00002.safetensors",
			"model.safetensors.index.json",
			"modeling_phi3.py",
			"sample_finetune.py",
			"special_tokens_map.json",
			"tokenizer.json",
			"tokenizer.model",
			"tokenizer_config.json",
		},
		Created:    time.Date(2024, 04, 22, 16, 18, 17, 0, time.UTC),
		Modified:   time.Date(2024, 07, 01, 21, 16, 50, 0000, time.UTC),
		TensorType: "BF16",
		NumWeights: 3821079552,
		License:    "mit",
		LicenseURL: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE",
	}
	if diff := cmp.Diff(want, got, cmpopts.IgnoreUnexported(want)); diff != "" {
		t.Fatal(diff)
	}
}

var apiRepoPhi3Data = `
{
		"lastModified": "2024-07-01T21:16:50.000Z",
    "cardData": {
        "license": "mit",
        "license_link": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/LICENSE",
        "language": [
            "en"
        ],
        "inference": {
            "parameters": {
                "temperature": 0
            }
        }
    },
    "siblings": [
        {
            "rfilename": ".gitattributes"
        },
        {
            "rfilename": "CODE_OF_CONDUCT.md"
        },
        {
            "rfilename": "LICENSE"
        },
        {
            "rfilename": "NOTICE.md"
        },
        {
            "rfilename": "README.md"
        },
        {
            "rfilename": "SECURITY.md"
        },
        {
            "rfilename": "added_tokens.json"
        },
        {
            "rfilename": "config.json"
        },
        {
            "rfilename": "configuration_phi3.py"
        },
        {
            "rfilename": "generation_config.json"
        },
        {
            "rfilename": "model-00001-of-00002.safetensors"
        },
        {
            "rfilename": "model-00002-of-00002.safetensors"
        },
        {
            "rfilename": "model.safetensors.index.json"
        },
        {
            "rfilename": "modeling_phi3.py"
        },
        {
            "rfilename": "sample_finetune.py"
        },
        {
            "rfilename": "special_tokens_map.json"
        },
        {
            "rfilename": "tokenizer.json"
        },
        {
            "rfilename": "tokenizer.model"
        },
        {
            "rfilename": "tokenizer_config.json"
        }
    ],
    "createdAt": "2024-04-22T16:18:17.000Z",
    "safetensors": {
        "parameters": {
            "BF16": 3821079552
        },
        "total": 3821079552
    }
}
`

func TestGetModelInfo_Llama(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/meta-llama/Llama-3.2-3B/revision/main" {
			t.Errorf("unexpected path, got: %s", r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(apiRepoLlama3_2Data))
	}))
	defer server.Close()
	os.Setenv("HF_HOME", t.TempDir())
	c, err := New("")
	if err != nil {
		t.Fatal(err)
	}
	c.serverBase = server.URL

	got := Model{
		ModelRef: ModelRef{
			Author: "meta-llama",
			Repo:   "Llama-3.2-3B",
		},
	}
	if err := c.GetModelInfo(context.Background(), &got, "main"); err != nil {
		t.Fatal(err)
	}
	// TODO: verify.
}

var apiRepoLlama3_2Data = `
{
    "_id": "66eaf084b3b3239188f66fa7",
    "author": "meta-llama",
    "cardData": {
      "extra_gated_button_content": "Submit",
      "extra_gated_description": "The information you provide will be collected, stored, processed and shared in accordance with the [Meta Privacy Policy](https://www.facebook.com/privacy/policy/).",
      "extra_gated_fields": {
        "Affiliation": "text",
        "By clicking Submit below I accept the terms of the license and acknowledge that the information I provide will be collected stored processed and shared in accordance with the Meta Privacy Policy": "checkbox",
        "Country": "country",
        "Date of birth": "date_picker",
        "First Name": "text",
        "Job title": {
          "options": [
            "Student",
            "Research Graduate",
            "AI researcher",
            "AI developer/engineer",
            "Reporter",
            "Other"
          ],
          "type": "select"
        },
        "Last Name": "text",
        "geo": "ip_location"
      },
      "extra_gated_prompt": "### LLAMA 3.2 COMMUNITY LICENSE AGREEMENT\n\n blah blah",
      "language": [
        "en",
        "de",
        "fr",
        "it",
        "pt",
        "hi",
        "es",
        "th"
      ],
      "library_name": "transformers",
      "license": "llama3.2",
      "pipeline_tag": "text-generation",
      "tags": [
        "facebook",
        "meta",
        "pytorch",
        "llama",
        "llama-3"
      ]
    },
    "config": {
      "architectures": [
        "LlamaForCausalLM"
      ],
      "model_type": "llama",
      "tokenizer_config": {
        "bos_token": "\u003c|begin_of_text|\u003e",
        "eos_token": "\u003c|end_of_text|\u003e"
      }
    },
    "createdAt": "2024-09-18T15:23:48.000Z",
    "disabled": false,
    "downloads": 235716,
    "gated": "manual",
    "id": "meta-llama/Llama-3.2-3B",
    "lastModified": "2024-10-24T15:07:40.000Z",
    "library_name": "transformers",
    "likes": 286,
    "model-index": null,
    "modelId": "meta-llama/Llama-3.2-3B",
    "pipeline_tag": "text-generation",
    "private": false,
    "safetensors": {
      "parameters": {
        "BF16": 3212749824
      },
      "total": 3212749824
    },
    "sha": "13afe5124825b4f3751f836b40dafda64c1ed062",
    "siblings": [
      {
        "rfilename": ".gitattributes"
      },
      {
        "rfilename": "LICENSE.txt"
      },
      {
        "rfilename": "README.md"
      },
      {
        "rfilename": "USE_POLICY.md"
      },
      {
        "rfilename": "config.json"
      },
      {
        "rfilename": "generation_config.json"
      },
      {
        "rfilename": "model-00001-of-00002.safetensors"
      },
      {
        "rfilename": "model-00002-of-00002.safetensors"
      },
      {
        "rfilename": "model.safetensors.index.json"
      },
      {
        "rfilename": "original/consolidated.00.pth"
      },
      {
        "rfilename": "original/params.json"
      },
      {
        "rfilename": "original/tokenizer.model"
      },
      {
        "rfilename": "special_tokens_map.json"
      },
      {
        "rfilename": "tokenizer.json"
      },
      {
        "rfilename": "tokenizer_config.json"
      }
    ],
    "spaces": [
      "allknowingroger/meta-llama-Llama-3.2-3B",
      "Illia56/kotaemon-pro-Llama-3.2-3B",
      "sakuexe/thesizer",
      "IamVicky111/Psychiatrist_Bot",
      "realmachas/meta-llama-Llama-3.2-3B",
      "mtcporto/meta-llama-Llama-3.2-3B",
      "Inaruss/meta-llama-Llama-3.2-3B",
      "realaer/src",
      "Depsa/meta-llama-Llama-3.2-3B",
      "varun324242/meta-llama-Llama-3.2-3B",
      "BICORP/meta-llama-Llama-3.2-3B",
      "Woziii/LLMnBiasV2",
      "TechXplorer/meta-llama-Llama-3.2-3B",
      "Insikoorit/thesizer",
      "skimike02/meta-llama-Llama-3.2-3B",
      "DjDister/meta-llama-Llama-3.2-3B",
      "IamVicky111/CalmCompass",
      "jesuswithclinton/meta-llama-Llama-3.2-3B",
      "laohan/mini_LAU",
      "erstrik/meta-llama-Llama-3.2-3B",
      "Joaovsales/astro",
      "sunitbana/meta"
    ],
    "tags": [
      "transformers",
      "safetensors",
      "llama",
      "text-generation",
      "facebook",
      "meta",
      "pytorch",
      "llama-3",
      "en",
      "de",
      "fr",
      "it",
      "pt",
      "hi",
      "es",
      "th",
      "arxiv:2204.05149",
      "arxiv:2405.16406",
      "license:llama3.2",
      "autotrain_compatible",
      "text-generation-inference",
      "endpoints_compatible",
      "region:us"
    ],
    "transformersInfo": {
      "auto_model": "AutoModelForCausalLM",
      "pipeline_tag": "text-generation",
      "processor": "AutoTokenizer"
    },
    "widgetData": [
      {
        "text": "My name is Julien and I like to"
      },
      {
        "text": "My name is Thomas and my main"
      },
      {
        "text": "My name is Mariama, my favorite"
      },
      {
        "text": "My name is Clara and I am"
      },
      {
        "text": "My name is Lewis and I like to"
      },
      {
        "text": "My name is Merve and my favorite"
      },
      {
        "text": "My name is Teven and I am"
      },
      {
        "text": "Once upon a time,"
      }
    ]
  }
`

// TestMain sets up the verbose logging.
func TestMain(m *testing.M) {
	flag.Parse()
	l := slog.LevelWarn
	if testing.Verbose() {
		l = slog.LevelDebug
	}
	logger := slog.New(tint.NewHandler(colorable.NewColorable(os.Stderr), &tint.Options{
		Level:      l,
		TimeFormat: time.TimeOnly,
		NoColor:    !isatty.IsTerminal(os.Stderr.Fd()),
	}))
	slog.SetDefault(logger)
	os.Unsetenv("HF_HOME")
	os.Unsetenv("HF_HUB_CACHE")
	os.Unsetenv("HF_TOKEN")
	os.Unsetenv("HF_TOKEN_PATH")
	os.Exit(m.Run())
}
