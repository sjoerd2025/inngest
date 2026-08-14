package manager

import (
	"crypto/rand"
	"fmt"
	"strconv"
	"testing"

	"github.com/inngest/inngest/pkg/cqrs"
	"github.com/inngest/inngest/pkg/enums"
	"github.com/inngest/inngest/pkg/tracing/meta"
	"github.com/inngest/inngest/pkg/tracing/metadata/extractors"
	"github.com/oklog/ulid/v2"
	"github.com/stretchr/testify/require"
)

func metadataSpanAttrs(kind, scope, values string) []byte {
	return fmt.Appendf(nil,
		`{"_inngest.metadata.kind":%q,"_inngest.metadata.scope":%q,"_inngest.metadata.op":"merge","_inngest.metadata.values":%s}`,
		kind, scope, strconv.Quote(values),
	)
}

func findAISummary(t *testing.T, span *cqrs.OtelSpan) []*cqrs.SpanMetadata {
	t.Helper()
	var out []*cqrs.SpanMetadata
	for _, md := range span.Metadata {
		if md.Kind == extractors.KindInngestAISummary {
			out = append(out, md)
		}
	}
	return out
}

func TestCQRSAISummaryMetadata(t *testing.T) {
	runAttr := []byte(`{"_inngest.dynamic.status":"Running"}`)
	stepAttrs := func(id string, attempt int) []byte {
		return fmt.Appendf(nil, `{"_inngest.step.id":%q,"_inngest.step.attempt":%d}`, id, attempt)
	}

	t.Run("sums counted scopes, excludes extended_trace, strips spoofed summaries", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("a", 0)},
			{DynamicSpanID: "step2", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("b", 0)},
			// Executor-reported usage at legacy step_attempt scope.
			{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step_attempt",
				`{"input_tokens":100,"output_tokens":20,"request_model":"gpt-4o","response_model":"gpt-4o-mini","estimated_cost":0.05}`,
			)},
			// Step-scoped usage with an explicit total that beats input+output.
			{DynamicSpanID: "md2", ParentSpanID: "step2", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step",
				`{"input_tokens":10,"output_tokens":5,"total_tokens":40,"request_model":"claude-3-5"}`,
			)},
			// Extended-trace entries can duplicate step-level reporting and must
			// not be counted.
			{DynamicSpanID: "md3", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "extended_trace",
				`{"input_tokens":1000,"output_tokens":1000}`,
			)},
			// User-written run-scoped usage is additive.
			{DynamicSpanID: "md4", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "run",
				`{"input_tokens":1,"output_tokens":2}`,
			)},
			// A stored summary must be stripped and recomputed, never trusted.
			{DynamicSpanID: "md5", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai.summary", "run",
				`{"input_tokens":99999,"call_count":50}`,
			)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)
		require.NotNil(t, root)

		summaries := findAISummary(t, root)
		require.Len(t, summaries, 1)
		require.Equal(t, enums.MetadataScopeRun, summaries[0].Scope)

		sum, err := extractors.AISummaryFromValues(summaries[0].Values)
		require.NoError(t, err)
		require.Equal(t, int64(111), sum.InputTokens)
		require.Equal(t, int64(27), sum.OutputTokens)
		require.Equal(t, int64(163), sum.TotalTokens)
		require.NotNil(t, sum.EstimatedCost)
		require.InDelta(t, 0.05, *sum.EstimatedCost, 1e-9)
		// md1 reported both models, so the response model wins; md2 reported
		// only a request model, so that is used as the fallback.
		require.Equal(t, []string{"claude-3-5", "gpt-4o-mini"}, sum.Models)
		require.Equal(t, int64(3), sum.CallCount)
		require.False(t, sum.Partial)

		// The user's own run-scoped entry stays visible alongside the summary.
		userEntries := 0
		for _, md := range root.Metadata {
			if md.Kind == extractors.KindInngestAI {
				userEntries++
			}
		}
		require.Equal(t, 1, userEntries)
	})

	// Producers emit latency_ms (and other optional fields) as floats, which
	// the step-level AIMetadata struct types as *int64. Parsing must not be so
	// strict that such an entry's tokens are dropped from the summary.
	t.Run("counts entries whose optional fields are fractional", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("a", 0)},
			{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step",
				`{"input_tokens":37,"output_tokens":6,"total_tokens":43,"latency_ms":2165.79833984375,"request_model":"gpt-5.4-mini","estimated_cost":0.000055}`,
			)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)

		summaries := findAISummary(t, root)
		require.Len(t, summaries, 1)
		sum, err := extractors.AISummaryFromValues(summaries[0].Values)
		require.NoError(t, err)
		require.Equal(t, int64(1), sum.CallCount)
		require.Equal(t, int64(37), sum.InputTokens)
		require.Equal(t, int64(43), sum.TotalTokens)
	})

	t.Run("partial when the run invokes a child run", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		childRunID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		invokeAttrs := fmt.Appendf(nil,
			`{"_inngest.step.id":"inv","_inngest.step.attempt":0,"_inngest.step.invoke.run.id":%q}`,
			childRunID,
		)

		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: invokeAttrs},
			{DynamicSpanID: "md1", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "run", `{"input_tokens":5,"output_tokens":5}`,
			)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)

		summaries := findAISummary(t, root)
		require.Len(t, summaries, 1)
		sum, err := extractors.AISummaryFromValues(summaries[0].Values)
		require.NoError(t, err)
		require.True(t, sum.Partial)
		require.Equal(t, int64(10), sum.TotalTokens)
	})

	t.Run("no summary without AI usage or invokes", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("a", 0)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)
		require.Empty(t, findAISummary(t, root))
	})
}

func TestCQRSGetRunsAIUsage(t *testing.T) {
	cm, cleanup := initCQRS(t)
	defer cleanup()

	completedAttr := []byte(`{"_inngest.dynamic.status":"Completed"}`)
	runningAttr := []byte(`{"_inngest.dynamic.status":"Running"}`)

	completedRun := ulid.MustNew(ulid.Now(), rand.Reader)
	runningRun := ulid.MustNew(ulid.Now(), rand.Reader)
	invokingRun := ulid.MustNew(ulid.Now(), rand.Reader)
	skippedRun := ulid.MustNew(ulid.Now(), rand.Reader)
	missingRun := ulid.MustNew(ulid.Now(), rand.Reader)
	grandchildRun := ulid.MustNew(ulid.Now(), rand.Reader)

	fixtures := map[ulid.ULID][]testSpanFields{
		completedRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: completedAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: []byte(`{"_inngest.step.id":"a","_inngest.step.attempt":0}`)},
			{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step_attempt",
				`{"input_tokens":30,"output_tokens":12,"request_model":"gpt-4o","estimated_cost":0.01}`,
			)},
			// Extended-trace usage must be excluded here too.
			{DynamicSpanID: "md2", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "extended_trace", `{"input_tokens":500,"output_tokens":500}`,
			)},
		},
		runningRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runningAttr},
			{DynamicSpanID: "md1", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "run", `{"input_tokens":7,"output_tokens":3}`,
			)},
		},
		invokingRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: completedAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: fmt.Appendf(nil,
				`{"_inngest.step.id":"inv","_inngest.step.attempt":0,"_inngest.step.invoke.run.id":%q}`, grandchildRun.String())},
		},
		skippedRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: []byte(`{"_inngest.dynamic.status":"Skipped"}`)},
		},
	}
	for runID, spans := range fixtures {
		for _, s := range spans {
			s.RunID = runID.String()
			insertTestSpan(t, cm, s)
		}
	}

	usage, err := cm.GetRunsAIUsage(t.Context(), []ulid.ULID{completedRun, runningRun, invokingRun, skippedRun, missingRun})
	require.NoError(t, err)

	completed, ok := usage[completedRun]
	require.True(t, ok)
	require.Equal(t, int64(30), completed.InputTokens)
	require.Equal(t, int64(12), completed.OutputTokens)
	require.Equal(t, int64(42), completed.TotalTokens)
	require.Equal(t, int64(1), completed.CallCount)
	require.Equal(t, []string{"gpt-4o"}, completed.Models)
	require.False(t, completed.Partial)

	running, ok := usage[runningRun]
	require.True(t, ok)
	require.Equal(t, int64(10), running.TotalTokens)
	require.True(t, running.Partial, "a still-running child's usage is incomplete")

	invoking, ok := usage[invokingRun]
	require.True(t, ok)
	require.True(t, invoking.Partial, "grandchild usage is beyond the depth-1 aggregation")

	skipped, ok := usage[skippedRun]
	require.True(t, ok)
	require.False(t, skipped.Partial, "a skipped child run is terminal; no further usage can arrive")

	_, ok = usage[missingRun]
	require.False(t, ok, "runs with no spans are omitted so callers can treat them as unreachable")
}
