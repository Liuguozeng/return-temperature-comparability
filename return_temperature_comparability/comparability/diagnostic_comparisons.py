"""Empirical distributions, reference strata and exact paired diagnostic values."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter

BLUE, PINK, INK = '#2879AD', '#C97E9C', '#26333D'
DURATIONS = [7, 14, 30, 60, 90]
ENDPOINTS = ['median', 'heat_weighted']

CAPTIONS = {
    'S1': ('Supplementary Figure S1. Comparison error across campaign lengths. '
           'The three pages show PreDist empirical distributions, PreDist reference-contrast strata, '
           'and exact XAI4HEAT pair-season comparisons. Rows show 7, 14, 30, 60 and 90 days; '
           'columns separate median and heat-weighted endpoints. Blue denotes synchronous and '
           'pink asynchronous sampling. Each ECDF gives equal weight to its pair-season values. '
           'Stratum boxes show the median, interquartile range and full observed range; positive '
           'reference-contrast quartiles are fixed across durations and designs, with zero contrasts '
           'separate. XAI4HEAT connectors join the two values of the same pair-season at its exact '
           'reference contrast. The 41,110 PreDist and 140 XAI4HEAT source rows, including 160 '
           'PreDist zero-reference rows, are retained. Reused systems, pairs and seasons form '
           'dependent descriptive observations. Source: comparability/results/PAIR_SEASON_SUMMARY.csv.'),
    'S2': ('Supplementary Figure S2. Strict ordering reversals across campaign lengths. '
           'The page order, endpoints and design colors follow Figure S1. Values are pair-season '
           'strict-reversal percentages. ECDFs describe their empirical distribution; boxes show '
           'the median, interquartile range and full range within fixed reference-contrast strata, '
           'not confidence intervals. Each PreDist median panel excludes 32 zero-reference '
           'pair-seasons whose ordering is undefined. Campaign ties remain in the defined-order '
           'denominator; strict reversals have opposite campaign and nonzero reference signs. '
           'All defined XAI4HEAT values are shown at their exact reference contrast. '
           'Source: comparability/results/PAIR_SEASON_SUMMARY.csv.')}


def reference_strata(pairs, provider, endpoint):
    rows = pairs.loc[pairs.provider.eq(provider) & pairs.endpoint.eq(endpoint)]
    unique = rows[['pair_season_id', 'reference_contrast_c']].drop_duplicates()
    if unique.pair_season_id.duplicated().any():
        raise ValueError('Reference contrast changed across durations')
    positive = unique.reference_contrast_c.abs()
    positive = positive[positive > 0].to_numpy()
    boundaries = np.unique(np.quantile(positive, [.25, .5, .75]))
    def assign(values):
        values = np.abs(values)
        return np.where(values == 0, 0, np.searchsorted(boundaries, values, side='left') + 1)
    labels = ['0', *[f'Q{i+1}' for i in range(len(boundaries)+1)]]
    return boundaries, labels, assign


def validate_source(pairs):
    required = {'provider', 'endpoint', 'duration_days', 'pair_season_id', 'reference_contrast_c',
                'sync_mae_c', 'async_mae_c', 'sync_reversal_rate', 'async_reversal_rate', 'zero_reference'}
    if not required.issubset(pairs.columns):
        raise ValueError('Missing comparison field')
    if pairs.duplicated(['provider', 'endpoint', 'duration_days', 'pair_season_id']).any():
        raise ValueError('Repeated pair-season identity')
    expected = {'PreDist': 4111, 'XAI4HEAT': 14}
    for (provider, endpoint, days), rows in pairs.groupby(['provider', 'endpoint', 'duration_days']):
        if provider not in expected or endpoint not in ENDPOINTS or days not in DURATIONS:
            raise ValueError('Unknown comparison branch')
        if len(rows) != expected[provider]:
            raise ValueError('Missing comparison object')
        for design in ['sync', 'async']:
            if not np.isfinite(rows[design+'_mae_c']).all():
                raise ValueError('Nonfinite contrast MAE')
            if not np.array_equal(rows[design+'_reversal_rate'].isna(), rows.zero_reference.astype(bool)):
                raise ValueError('Undefined ordering changed')
    if len(pairs) != 41250:
        raise ValueError('Incomplete paired diagnostic source')


def layout(plt, provider, metric, view):
    fig, axes = plt.subplots(5, 2, figsize=(178/25.4, 185/25.4))
    fig.subplots_adjust(left=.12, right=.98, bottom=.12, top=.85, hspace=.42, wspace=.25)
    name = 'Comparison error' if metric == 'error' else 'Strict ordering reversals'
    subtitle = {'ecdf': 'Empirical distributions', 'strata': 'Reference-contrast strata',
                'paired': 'Exact pair-season comparisons'}[view]
    fig.text(.12, .97, provider+' | '+name, size=11, weight='bold', va='top')
    fig.text(.12, .93, subtitle, size=8.5, va='top')
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=1.6, label='Synchronous'),
               Line2D([], [], color=PINK, lw=1.6, ls='--', label='Asynchronous')]
    if view == 'paired':
        handles = [Line2D([], [], color=BLUE, marker='o', ls='', label='Synchronous'),
                   Line2D([], [], color=PINK, marker='s', markerfacecolor='none', ls='', label='Asynchronous')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.995, .96), frameon=False, fontsize=8)
    for col, endpoint in enumerate(ENDPOINTS):
        p = axes[0, col].get_position()
        fig.text((p.x0+p.x1)/2, .872, 'Median' if col == 0 else 'Heat weighted',
                 ha='center', size=9, weight='bold')
    for row, days in enumerate(DURATIONS):
        p = axes[row, 0].get_position()
        fig.text(.045, (p.y0+p.y1)/2, str(days)+' d', size=9, weight='bold', va='center')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E5EBEF', lw=.45)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8, length=2, pad=2)
    return fig, axes


def build(pairs, output, plt, save_figure):
    from diagnostic_facets import derive_maps, validate_maps
    validate_source(pairs)
    output = Path(output)
    mapping, grid, facets = derive_maps(pairs)
    validate_maps(pairs, mapping, grid, facets)
    mapping.to_csv(output/'DIAGNOSTIC_SOURCE_MAP.csv', index=False, float_format='%.17g')
    grid.to_csv(output/'DIAGNOSTIC_FREQUENCY_MAP.csv', index=False, float_format='%.17g')
    facets.to_csv(output/'DIAGNOSTIC_FACETS.csv', index=False, float_format='%.17g')
    page_records, box_records, strata_records = [], [], []
    for metric, figure in [('error', 'S1'), ('reversal', 'S2')]:
        writer = PdfWriter()
        for provider, view in [('PreDist', 'ecdf'), ('PreDist', 'strata'), ('XAI4HEAT', 'paired')]:
            local = pairs.loc[pairs.provider.eq(provider)]
            suffix = '_mae_c' if metric == 'error' else '_reversal_rate'
            factor = 1 if metric == 'error' else 100
            maximum = float(np.nanmax(local[['sync'+suffix, 'async'+suffix]].to_numpy())*factor)
            limit = np.ceil(maximum) if metric == 'error' else 100
            reference_limit = float(np.ceil(local.reference_contrast_c.abs().max()*1.05))
            fig, axes = layout(plt, provider, metric, view)
            described = 0
            for col, endpoint in enumerate(ENDPOINTS):
                boundaries, labels, assign = reference_strata(pairs, provider, endpoint)
                if view == 'strata':
                    strata_records.append(dict(figure=figure, provider=provider, endpoint=endpoint,
                        positive_quartile_boundaries_c=boundaries.tolist(), labels=labels,
                        basis='unique pair-seasons, absolute reference contrast; numpy linear quantiles'))
                for row, days in enumerate(DURATIONS):
                    ax = axes[row, col]
                    rows = local.loc[local.endpoint.eq(endpoint) & local.duration_days.eq(days)]
                    groups = assign(rows.reference_contrast_c.to_numpy())
                    for design, color, linestyle in [('sync', BLUE, '-'), ('async', PINK, '--')]:
                        raw = rows[design+suffix].to_numpy()*factor
                        valid = np.isfinite(raw)
                        described += int(valid.sum())
                        values = raw[valid]
                        if view == 'ecdf':
                            ordered = np.sort(values)
                            cdf = np.arange(1, len(ordered)+1)/len(ordered)
                            ax.step(np.r_[0., ordered, limit], np.r_[0., cdf, 1.], where='post',
                                    color=color, ls=linestyle, lw=1.25)
                            ax.set(xlim=(0, limit), ylim=(0, 1.04), yticks=[0, .5, 1])
                        elif view == 'strata':
                            summaries, positions = [], []
                            for group_id, label in enumerate(labels):
                                v = raw[(groups == group_id) & valid]
                                if not len(v):
                                    continue
                                q1, med, q3 = np.quantile(v, [.25, .5, .75])
                                stat = dict(label=label, med=float(med), q1=float(q1), q3=float(q3),
                                            whislo=float(v.min()), whishi=float(v.max()), fliers=[])
                                summaries.append(stat)
                                positions.append(group_id+(-.18 if design == 'sync' else .18))
                                box_records.append(dict(figure=figure, endpoint=endpoint, duration_days=days,
                                    design=design, stratum=label, count=len(v), **{k:stat[k] for k in ['med','q1','q3','whislo','whishi']}))
                            ax.bxp(summaries, positions=positions, widths=.29, showfliers=False,
                                   patch_artist=True, manage_ticks=False,
                                   boxprops=dict(facecolor=color+'33', edgecolor=color, lw=.8),
                                   medianprops=dict(color=color, lw=1.3),
                                   whiskerprops=dict(color=color, lw=.65), capprops=dict(color=color, lw=.65))
                            display_labels = ['0*', *labels[1:]] if metric == 'reversal' else labels
                            ax.set(xlim=(-.6, len(labels)-.4), ylim=(0, limit*1.04),
                                   xticks=range(len(labels)), xticklabels=display_labels)
                        else:
                            ref = rows.reference_contrast_c.abs().to_numpy()
                            ax.scatter(raw[valid], ref[valid], s=13, facecolor=color if design == 'sync' else 'none',
                                       marker='o' if design == 'sync' else 's', zorder=3,
                                       linewidths=.2 if design == 'sync' else .8,
                                       edgecolor='white' if design == 'sync' else color)
                            if design == 'sync':
                                other = rows['async'+suffix].to_numpy()*factor
                                for a, b, y in zip(raw[valid], other[valid], ref[valid]):
                                    ax.plot([a,b],[y,y], color='#C4CDD3', lw=.7, zorder=1)
                            ax.set(xlim=(-.02*limit, limit*1.03), ylim=(0, reference_limit))
                    if row != 4:
                        ax.tick_params(labelbottom=False)
            ylabel = 'Empirical cumulative fraction' if view == 'ecdf' else (
                ('Contrast MAE (°C)' if metric == 'error' else 'Strict reversals (%)') if view == 'strata'
                else 'Absolute reference contrast (°C)')
            xlabel = 'Reference-contrast stratum' if view == 'strata' else (
                'Pair-season contrast MAE (°C)' if metric == 'error' else 'Pair-season strict reversals (%)')
            fig.text(.01, .49, ylabel, rotation=90, size=8.5, va='center', ha='center')
            fig.text(.56, .067, xlabel, size=8.5, ha='center')
            footer = ('Q1-Q4: positive reference contrasts; 0: zero reference contrast' if view == 'strata'
                      else '4,111 pair-seasons per endpoint and duration' if provider == 'PreDist'
                      else '14 pair-seasons per endpoint and duration')
            if metric == 'reversal' and provider == 'PreDist':
                footer = 'Median: 4,079 defined pair-seasons; heat weighted: 4,111'
                if view == 'strata':
                    footer = '0*: ordering undefined; median n=4,079; heat weighted n=4,111'
            fig.text(.12, .017, footer, size=8)
            for ax in axes.flat:
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.set_xticks([x for x in ax.get_xticks() if xmin <= x <= xmax])
                ax.set_yticks([y for y in ax.get_yticks() if ymin <= y <= ymax])
            stem = f'Figure_{figure}_{provider}_{view}'
            qa = save_figure(fig, output, stem, 178, 185)
            plt.close(fig)
            page_records.append(dict(**qa, figure=figure, provider=provider, view=view,
                                     described_design_rows=described, panels=10))
            writer.add_page(PdfReader(output/(stem+'.pdf')).pages[0])
        with (output/f'Figure_{figure}.pdf').open('wb') as stream:
            writer.write(stream)
    pd.DataFrame(box_records).to_csv(output/'DIAGNOSTIC_STRATUM_SUMMARIES.csv', index=False, float_format='%.17g')
    (output/'DIAGNOSTIC_REFERENCE_STRATA.json').write_text(json.dumps(strata_records, indent=2)+'\n')
    return dict(pages=page_records, source_rows=len(pairs), source_mapping_rows=len(mapping),
        frequency_cells=len(grid), facet_rows=len(facets), zero_reference_error_rows=160,
        undefined_ordering_excluded_design_rows=320, ties_unchanged=True,
        frozen_source_basename='PAIR_SEASON_SUMMARY.csv', scientific_aggregation_changed=False,
        captions=CAPTIONS)
