package loader

import (
	"context"
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/inngest/inngest/pkg/cqrs"
	"github.com/inngest/inngest/pkg/enums"
	"github.com/inngest/inngest/pkg/tracing/meta"
	"github.com/inngest/inngest/pkg/tracing/metadata/extractors"
	"github.com/oklog/ulid/v2"
	"github.com/stretchr/testify/require"
)

func summarySpan(t *testing.T, sum extractors.AISummaryMetadata, children ...*cqrs.OtelSpan) *cqrs.OtelSpan {
	t.Helper()
	values, err := sum.Serialize()
	require.NoError(t, err)
	return &cqrs.OtelSpan{
		Metadata: []*cqrs.SpanMetadata{{
			Scope:  enums.MetadataScopeRun,
			Kind:   extractors.KindInngestAISummary,
			Values: values,
		}},
		Children: children,
	}
}

func invokeSpan(childRunID ulid.ULID) *cqrs.OtelSpan {
	return &cqrs.OtelSpan{
		Attributes: &meta.ExtractedValues{StepInvokeRunID: &childRunID},
	}
}

func parseSummary(t *testing.T, root *cqrs.OtelSpan) extractors.AISummaryMetadata {
	t.Helper()
	for _, md := range root.Metadata {
		if md.Kind == extractors.KindInngestAISummary {
			sum, err := extractors.AISummaryFromValues(md.Values)
			require.NoError(t, err)
			return sum
		}
	}
	t.Fatal("no inngest.ai.summary entry on root span")
	return extractors.AISummaryMetadata{}
}

func TestFoldChildRunAIUsage(t *testing.T) {
	ctx := context.Background()
	childID := ulid.MustNew(ulid.Now(), rand.Reader)
	cost := 0.02

	inTree := extractors.AISummaryMetadata{
		InputTokens:  10,
		OutputTokens: 5,
		TotalTokens:  15,
		Models:       []string{"claude-3-5"},
		CallCount:    1,
		Partial:      true,
	}

	t.Run("folds a resolved child and clears partial", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			require.Equal(t, []ulid.ULID{childID}, ids)
			return map[ulid.ULID]extractors.AISummaryMetadata{
				childID: {
					InputTokens:   90,
					OutputTokens:  45,
					TotalTokens:   135,
					EstimatedCost: &cost,
					Models:        []string{"gpt-4o"},
					CallCount:     2,
				},
			}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(100), sum.InputTokens)
		require.Equal(t, int64(50), sum.OutputTokens)
		require.Equal(t, int64(150), sum.TotalTokens)
		require.NotNil(t, sum.EstimatedCost)
		require.InDelta(t, 0.02, *sum.EstimatedCost, 1e-9)
		require.Equal(t, []string{"claude-3-5", "gpt-4o"}, sum.Models)
		require.Equal(t, int64(3), sum.CallCount)
		require.False(t, sum.Partial)
	})

	t.Run("stays partial when a child is unreachable", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return map[ulid.ULID]extractors.AISummaryMetadata{}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(15), sum.TotalTokens)
		require.True(t, sum.Partial)
	})

	t.Run("stays partial when a child reports partial usage", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return map[ulid.ULID]extractors.AISummaryMetadata{
				childID: {TotalTokens: 100, CallCount: 1, Partial: true},
			}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(115), sum.TotalTokens)
		require.True(t, sum.Partial)
	})

	t.Run("keeps the in-tree summary when the fetch fails", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return nil, fmt.Errorf("boom")
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(15), sum.TotalTokens)
		require.True(t, sum.Partial)
	})

	t.Run("drops an empty placeholder once all children resolve with no usage", func(t *testing.T) {
		root := summarySpan(t, extractors.AISummaryMetadata{Partial: true}, invokeSpan(childID))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return map[ulid.ULID]extractors.AISummaryMetadata{childID: {}}, nil
		})

		for _, md := range root.Metadata {
			require.NotEqual(t, extractors.KindInngestAISummary, md.Kind)
		}
	})

	t.Run("no-op without invoked children", func(t *testing.T) {
		root := summarySpan(t, extractors.AISummaryMetadata{TotalTokens: 15, CallCount: 1})

		called := false
		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			called = true
			return nil, nil
		})
		require.False(t, called)

		sum := parseSummary(t, root)
		require.Equal(t, int64(15), sum.TotalTokens)
	})

	t.Run("pending invoke without a stamped run ID keeps partial via in-tree flag", func(t *testing.T) {
		op := enums.OpcodeInvokeFunction
		pendingInvoke := &cqrs.OtelSpan{Attributes: &meta.ExtractedValues{StepOp: &op}}
		root := summarySpan(t, inTree, invokeSpan(childID), pendingInvoke)

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return map[ulid.ULID]extractors.AISummaryMetadata{
				childID: {TotalTokens: 100, CallCount: 1},
			}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(115), sum.TotalTokens)
		require.True(t, sum.Partial, "an invoke step whose child run isn't stamped yet keeps the summary partial")
	})
}
