package extractors

import (
	"cmp"
	"encoding/json"
	"maps"
	"slices"

	"github.com/inngest/inngest/pkg/enums"
	"github.com/inngest/inngest/pkg/tracing/metadata"
)

//tygo:generate
const (
	// KindInngestAISummary is the run-scoped rollup of all AI usage within a
	// run, including usage folded in from invoked child runs. It is
	// synthesized every time a run's span tree is read and is never
	// persisted, so it is always authoritative and can never double-count
	// itself. It must never be added to the allowedInngestKinds allowlist.
	KindInngestAISummary metadata.Kind = "inngest.ai.summary"
)

//tygo:generate
type AISummaryMetadata struct {
	InputTokens   int64    `json:"input_tokens"`
	OutputTokens  int64    `json:"output_tokens"`
	TotalTokens   int64    `json:"total_tokens"`
	EstimatedCost *float64 `json:"estimated_cost,omitempty"`
	Models        []string `json:"models,omitempty"`
	CallCount     int64    `json:"call_count"`
	// Partial marks the summary as known-incomplete: an invoked child run is
	// still running, unreachable, or beyond the depth-1 aggregation.
	Partial bool `json:"partial"`
}

func (ms AISummaryMetadata) Kind() metadata.Kind {
	return KindInngestAISummary
}

func (ms AISummaryMetadata) Scope() metadata.Scope {
	return enums.MetadataScopeRun
}

func (ms AISummaryMetadata) Serialize() (metadata.Values, error) {
	var rawMetadata metadata.Values
	err := rawMetadata.FromStruct(ms)
	if err != nil {
		return nil, err
	}

	return rawMetadata, nil
}

func AISummaryFromValues(values metadata.Values) (AISummaryMetadata, error) {
	var ms AISummaryMetadata
	raw, err := json.Marshal(values)
	if err != nil {
		return ms, err
	}
	err = json.Unmarshal(raw, &ms)
	return ms, err
}

// AIUsageScopeCounted reports whether an inngest.ai metadata entry at the
// given scope counts toward the run-level AI summary. Step and legacy
// step-attempt entries are executor-reported usage; run-scoped entries are
// how users report out-of-step usage. Extended-trace entries are excluded
// because the same LLM call can be reported at both the step and extended
// trace levels.
func AIUsageScopeCounted(scope metadata.Scope) bool {
	switch scope {
	case enums.MetadataScopeStep, enums.MetadataScopeStepAttempt, enums.MetadataScopeRun:
		return true
	default:
		return false
	}
}

// AISummaryBuilder accumulates inngest.ai metadata entries and child-run
// summaries into a single AISummaryMetadata. Every counted entry is summed,
// including entries from retried attempts: spend on a retried step is real
// spend.
type AISummaryBuilder struct {
	sum     AISummaryMetadata
	cost    float64
	hasCost bool
	models  map[string]struct{}
}

func NewAISummaryBuilder() *AISummaryBuilder {
	return &AISummaryBuilder{models: map[string]struct{}{}}
}

// aiUsageValues is the minimal projection of an inngest.ai entry needed to
// aggregate usage. It deliberately does not reuse AIMetadata: that struct
// types fields like latency_ms as *int64, but producers emit them as floats
// (e.g. 2165.798), so unmarshalling the full struct fails and would drop the
// entry's tokens entirely. Numerics are float64 here so both integer and
// fractional encodings parse.
type aiUsageValues struct {
	InputTokens   float64  `json:"input_tokens"`
	OutputTokens  float64  `json:"output_tokens"`
	TotalTokens   *float64 `json:"total_tokens"`
	EstimatedCost *float64 `json:"estimated_cost"`
	RequestModel  string   `json:"request_model"`
	ResponseModel string   `json:"response_model"`
}

// AddCall folds one inngest.ai metadata entry's values into the summary.
func (b *AISummaryBuilder) AddCall(values metadata.Values) error {
	raw, err := json.Marshal(values)
	if err != nil {
		return err
	}
	var m aiUsageValues
	if err := json.Unmarshal(raw, &m); err != nil {
		return err
	}

	b.sum.InputTokens += int64(m.InputTokens)
	b.sum.OutputTokens += int64(m.OutputTokens)
	if m.TotalTokens != nil {
		b.sum.TotalTokens += int64(*m.TotalTokens)
	} else {
		b.sum.TotalTokens += int64(m.InputTokens) + int64(m.OutputTokens)
	}
	if m.EstimatedCost != nil {
		b.cost += *m.EstimatedCost
		b.hasCost = true
	}
	// Mirrors COALESCE(response_model, request_model), the label Cloud's AI
	// dashboards group every model-scoped metric by. Recording both would
	// surface request aliases that are never a dashboard category.
	if model := cmp.Or(m.ResponseModel, m.RequestModel); model != "" {
		b.models[model] = struct{}{}
	}
	b.sum.CallCount++

	return nil
}

// AddSummary folds another summary (e.g. an invoked child run's) into this
// one. A partial input makes the result partial.
func (b *AISummaryBuilder) AddSummary(s AISummaryMetadata) {
	b.sum.InputTokens += s.InputTokens
	b.sum.OutputTokens += s.OutputTokens
	b.sum.TotalTokens += s.TotalTokens
	if s.EstimatedCost != nil {
		b.cost += *s.EstimatedCost
		b.hasCost = true
	}
	for _, m := range s.Models {
		b.models[m] = struct{}{}
	}
	b.sum.CallCount += s.CallCount
	if s.Partial {
		b.sum.Partial = true
	}
}

func (b *AISummaryBuilder) MarkPartial() {
	b.sum.Partial = true
}

// Empty reports whether nothing has been accumulated. The partial flag is
// deliberately ignored: an empty-but-partial summary still carries no usage.
func (b *AISummaryBuilder) Empty() bool {
	return b.sum.CallCount == 0 && !b.hasCost
}

func (b *AISummaryBuilder) Summary() AISummaryMetadata {
	out := b.sum
	if b.hasCost {
		cost := b.cost
		out.EstimatedCost = &cost
	}
	if len(b.models) > 0 {
		out.Models = slices.Sorted(maps.Keys(b.models))
	}
	return out
}
