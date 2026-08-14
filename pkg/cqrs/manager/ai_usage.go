package manager

import (
	"context"
	"fmt"

	"github.com/inngest/inngest/pkg/cqrs"
	"github.com/inngest/inngest/pkg/enums"
	"github.com/inngest/inngest/pkg/logger"
	"github.com/inngest/inngest/pkg/tracing/meta"
	"github.com/inngest/inngest/pkg/tracing/metadata/extractors"
	"github.com/oklog/ulid/v2"
	"golang.org/x/sync/errgroup"
)

// GetRunsAIUsage sums each run's counted inngest.ai metadata into a run-level
// AI usage summary, reading only metadata spans rather than assembling the
// full span tree. It exists so the GraphQL loader layer can fold step.invoke
// child run usage into a parent run's inngest.ai.summary; GetSpansByRunID
// must never call it, since that primitive also serves the rerun path.
//
// A run's summary is marked partial when the run is still executing or when
// it invokes runs of its own — grandchild usage is beyond the depth-1
// aggregation. Runs with no spans at all are omitted from the result so
// callers can treat them as unreachable.
func (w wrapper) GetRunsAIUsage(ctx context.Context, runIDs []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
	if len(runIDs) == 0 {
		return map[ulid.ULID]extractors.AISummaryMetadata{}, nil
	}

	var metadataSpans, runSpans, stepSpans map[ulid.ULID][]*cqrs.OtelSpan
	eg, egctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		var err error
		metadataSpans, err = w.GetSpansByRunIDsAndName(egctx, runIDs, meta.SpanNameMetadata)
		return err
	})
	eg.Go(func() error {
		var err error
		runSpans, err = w.GetSpansByRunIDsAndName(egctx, runIDs, meta.SpanNameRun)
		return err
	})
	eg.Go(func() error {
		var err error
		stepSpans, err = w.GetSpansByRunIDsAndName(egctx, runIDs, meta.SpanNameStep)
		return err
	})
	if err := eg.Wait(); err != nil {
		return nil, fmt.Errorf("error loading spans for AI usage: %w", err)
	}

	out := make(map[ulid.ULID]extractors.AISummaryMetadata, len(runIDs))
	for _, runID := range runIDs {
		if len(runSpans[runID]) == 0 {
			continue
		}

		builder := extractors.NewAISummaryBuilder()
		for _, span := range metadataSpans[runID] {
			for _, md := range span.Metadata {
				if md.Kind != extractors.KindInngestAI || !extractors.AIUsageScopeCounted(md.Scope) {
					continue
				}
				if err := builder.AddCall(md.Values); err != nil {
					logger.StdlibLogger(ctx).Warn(
						"skipping malformed inngest.ai metadata entry",
						"run_id", runID.String(),
						"error", err,
					)
				}
			}
		}

		ended := false
		for _, span := range runSpans[runID] {
			// Skipped is terminal but not covered by IsEnded.
			if span.GetIsRoot() && (span.Status.IsEnded() || span.Status == enums.StepStatusSkipped) {
				ended = true
				break
			}
		}

		hasInvokes := false
		for _, span := range stepSpans[runID] {
			if span.IsInvokeStep() {
				hasInvokes = true
				break
			}
		}

		if !ended || hasInvokes {
			builder.MarkPartial()
		}

		out[runID] = builder.Summary()
	}

	return out, nil
}
