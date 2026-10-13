'Portable presentation-only return_temperature_comparability renderer with explicit inputs and display numbering.'

import argparse

import copy

import hashlib

import importlib.util

import json

import os

import re

import shutil

import subprocess

import sys

import unittest

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel_geometry import panel_label, write_panel_geometry

BLUE = '#2879AD'

PINK = '#C97E9C'

GRAY = '#607F92'

INK = '#26333D'

PALETTE = {'sync': BLUE, 'async': PINK, 'effect': GRAY}

DURATIONS = [7, 14, 30, 60, 90]

PROVIDERS = ['PreDist', 'XAI4HEAT']

ENDPOINTS = ['median', 'heat_weighted']

SYSTEMS = ['L4', 'L8', 'L12', 'L17', 'L22']

FROZEN_HASHES = {
    'PROVIDER_SUMMARY.csv': 'ce507f9ea630be9713e1b26ca4f4334990a3921b7c99dc8ea188bbb022542c79',
    'FOCAL_SYSTEM_SUMMARY.csv': '5d8f978f6a36e86387fea8c616743e03ceb89e4afc59f373b7a973c6071c0e6b',
    'XAI_LEAVE_ONE_SYSTEM.csv': '69a73504998ba513274fe7c6f3c1a2c081746664460ac5b5fe0e5e64932a117a',
    'PAIR_SEASON_SUMMARY.csv': '1f501031a24212448e8238012a538adfc3e3a5936f674db36a691e015c23f867',
}

FIG5_CAPTION = (
    'Figure 2. Absolute comparison error and timing effects at fixed marginal sampling. '
    'Panels (a-d) show all 20 provider-endpoint-duration groups: blue synchronous and pink asynchronous contrast MAEs. '
    'The Cartesian asynchronous design includes the synchronous diagonal and preserves both marginal start distributions. '
    'Solid whiskers in PreDist (a,b) give pointwise 95% physical-node bootstrap CIs (5,000 draws); '
    'dashed whiskers in XAI4HEAT (c,d) span all five leave-one-system (LOS) estimates, a descriptive range rather than a CI. '
    'Paired 7- and 90-day timing effects beneath each row are synchronous minus asynchronous MAE (°C), '
    'with their own CI or LOS range, not differences of marginal bounds; negative values favor synchronization. '
    'Hatching marks 90 days. Panel (e) retains 68 PreDist focal systems in each endpoint-duration row, '
    'with offsets separating overlapping circles rather than representing density or uncertainty. '
    'Panel (f) retains all five XAI4HEAT systems: circles/squares denote median/heat-weighted endpoints, '
    "and filled/open symbols denote 7/90 days. Blue-gray focal effects average each system's eligible partners "
    'under pair-specific shared-date references; they are signed differences, not absolute errors or intrinsic system properties. '
    'Complete pair-season errors and ordering diagnostics are in Supplementary Figures S1-S2 and Section S13. '
    'Systems reused across pairs, seasons and durations are dependent observations.'
)

from diagnostic_facets import S1_CAPTION, S2_CAPTION

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def require(condition, message):
    if not bool(condition):
        raise ValueError(message)

def check_palette(palette):
    require(palette == {'sync': BLUE, 'async': PINK, 'effect': GRAY}, 'Palette role mismatch')

def load_tables(results):
    import pandas as pd
    results = Path(results)
    require(results.is_dir(), 'Explicit results directory is missing')
    tables = {}
    for name, digest in FROZEN_HASHES.items():
        path = results / name
        require(path.is_file(), 'Required input missing: ' + name)
        tables[name] = pd.read_csv(path)
        require(sha(path) == digest, 'Frozen input identity/numbers changed: ' + name)
    validate_tables(tables)
    return tables

def validate_tables(tables):
    import numpy as np
    specs = {
        'PROVIDER_SUMMARY.csv': ['provider', 'estimand', 'endpoint', 'duration_days', 'metric', 'estimate', 'ci95_lower', 'ci95_upper', 'physical_systems', 'uncertainty'],
        'FOCAL_SYSTEM_SUMMARY.csv': ['provider', 'estimand', 'endpoint', 'duration_days', 'metric', 'system', 'eligible_counterparts', 'estimate'],
        'XAI_LEAVE_ONE_SYSTEM.csv': ['provider', 'estimand', 'endpoint', 'duration_days', 'metric', 'omitted_system', 'full_estimate', 'leave_one_system_estimate', 'retained_systems'],
        'PAIR_SEASON_SUMMARY.csv': ['pair_season_id', 'provider', 'system_i', 'system_j', 'heating_year', 'estimand', 'endpoint', 'duration_days', 'reference_i_c', 'reference_j_c', 'reference_contrast_c', 'sync_mae_c', 'async_mae_c', 'sync_minus_async_mae_c', 'sync_reversal_rate', 'async_reversal_rate', 'zero_reference'],
    }
    keys = [specs['PROVIDER_SUMMARY.csv'][:5],
            ['provider', 'estimand', 'endpoint', 'duration_days', 'system', 'metric'],
            ['provider', 'estimand', 'endpoint', 'duration_days', 'metric', 'omitted_system'],
            ['pair_season_id', 'endpoint', 'duration_days']]
    for (name, columns), key in zip(specs.items(), keys):
        require(name in tables, 'Required table missing: ' + name)
        table = tables[name]
        require(set(columns).issubset(table.columns), 'Required columns missing: ' + name)
        require(not table.duplicated(key).any(), 'Duplicate identities: ' + name)
        require(set(table.provider).issubset(PROVIDERS), 'Wrong provider identity: ' + name)
        require(set(table.duration_days) == set(DURATIONS), 'Wrong duration identities: ' + name)
    summary, focal, los, pairs = [tables[n] for n in specs]
    primary = summary.loc[summary.estimand.eq('same_season')]
    require(set(primary.endpoint) == set(ENDPOINTS), 'Wrong endpoint identities')
    for provider in PROVIDERS:
        physical = 68 if provider == 'PreDist' else 5
        for endpoint in ENDPOINTS:
            for days in DURATIONS:
                rows = primary.loc[primary.provider.eq(provider) & primary.endpoint.eq(endpoint) & primary.duration_days.eq(days)]
                require(rows.physical_systems.eq(physical).all(), 'Wrong physical-system count')
                values = rows.set_index('metric').estimate
                require(np.isclose(values['sync_mae_c'] - values['async_mae_c'], values['sync_minus_async_mae_c'], atol=1e-12), 'Wrong timing contrast numbers')
                for metric in ['sync_mae_c', 'async_mae_c', 'sync_minus_async_mae_c']:
                    p = rows.loc[rows.metric.eq(metric)].iloc[0]
                    f = focal.loc[focal.provider.eq(provider) & focal.estimand.eq('same_season') & focal.endpoint.eq(endpoint) & focal.duration_days.eq(days) & focal.metric.eq(metric)]
                    require(len(f) == physical and f.system.nunique() == physical, 'Incomplete focal identities')
                    require(np.isfinite(f.estimate).all(), 'Nonfinite focal values')
                    require(np.isclose(f.estimate.mean(), p.estimate, atol=1e-12), 'Focal/provider number mismatch')
                    if provider == 'PreDist':
                        require(p.ci95_lower <= p.estimate <= p.ci95_upper, 'Invalid node CI')
                        require(p.uncertainty == 'node_bootstrap_percentile_95', 'Wrong CI identity')
                    else:
                        require(set(f.system) == set(SYSTEMS), 'Wrong XAI physical identities')
                        require(np.isnan(p.ci95_lower) and np.isnan(p.ci95_upper), 'XAI population CI prohibited')
                        l = los.loc[los.estimand.eq('same_season') & los.endpoint.eq(endpoint) & los.duration_days.eq(days) & los.metric.eq(metric)]
                        require(set(l.omitted_system) == set(SYSTEMS) and len(l) == 5, 'Incomplete XAI omission identities')
                        require(l.retained_systems.eq(4).all(), 'Wrong XAI omission system count')
                        require(np.allclose(l.full_estimate, p.estimate, atol=1e-12), 'LOS full estimate mismatch')
    require(len(pairs) == 41250, 'Incomplete full-object pair-season table')
    require(set(pairs.estimand) == {'same_season'}, 'Wrong pair-season estimand')
    for provider, count in [('PreDist', 4111), ('XAI4HEAT', 14)]:
        part = pairs.loc[pairs.provider.eq(provider)]
        require(part.pair_season_id.nunique() == count and len(part) == count * 10, 'Wrong pair-season identities/count')
        require(part.groupby('pair_season_id').size().eq(10).all(), 'Missing pair-season protocol objects')
    require(np.allclose(pairs.reference_i_c - pairs.reference_j_c, pairs.reference_contrast_c, atol=1e-12), 'Wrong reference contrast numbers')
    require(np.allclose(pairs.sync_mae_c - pairs.async_mae_c, pairs.sync_minus_async_mae_c, atol=1e-12), 'Wrong pair-season timing numbers')
    require((pairs.system_i != pairs.system_j).all(), 'Self-pair identity')
    for design in ['sync', 'async']:
        require(pairs[design + '_mae_c'].ge(0).all(), 'Negative absolute error')
        rate = pairs[design + '_reversal_rate']
        require(rate.dropna().between(0, 1).all(), 'Wrong reversal fraction')
        require(rate.loc[pairs.zero_reference.eq(1)].isna().all(), 'Zero reference ordering must be undefined')
    require(los.provider.eq('XAI4HEAT').all(), 'Wrong LOS provider')
    check_palette(PALETTE)

def setup_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': GRAY, 'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.linewidth': 0.6, 'hatch.linewidth': 0.5})
    return plt

def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E5EBEF', linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=0.6, pad=3)

def save_figure(fig, output, stem, width, height):
    import matplotlib
    from pypdf import PdfReader
    write_panel_geometry(fig, output / stem)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    require(min(t.get_fontsize() for t in texts) >= 8, 'Figure label below 8 pt')
    for t in texts:
        box = t.get_window_extent(renderer)
        require(box.x0 >= -0.1 and box.y0 >= -0.1 and box.x1 <= fig.bbox.x1 + .1 and box.y1 <= fig.bbox.y1 + .1, 'Text outside figure: ' + t.get_text())
    collisions = []
    for i, left in enumerate(texts):
        a = left.get_window_extent(renderer)
        for right in texts[i + 1:]:
            b = right.get_window_extent(renderer)
            dx = min(a.x1, b.x1) - max(a.x0, b.x0)
            dy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if dx > 0.5 and dy > 0.5:
                collisions.append([left.get_text(), right.get_text()])
    require(not collisions, 'Overlapping figure labels: ' + repr(collisions))
    for suffix, dpi in [('pdf', 600), ('png', 300), ('svg', 600)]:
        fig.savefig(output / (stem + '.' + suffix), dpi=dpi)
    page = PdfReader(output / (stem + '.pdf')).pages[0]
    font_names = [str(v.get_object()['/BaseFont']) for v in page['/Resources']['/Font'].values()]
    require(all('TimesNewRoman' in f for f in font_names), 'Non-TNR exported font')
    require(abs(float(page.mediabox.width) * 25.4 / 72 - width) < 1e-5, 'Wrong export width')
    require(abs(float(page.mediabox.height) * 25.4 / 72 - height) < 1e-5, 'Wrong export height')
    return {'width_mm': width, 'height_mm': height, 'minimum_font_pt': 8, 'fonts': font_names,
            'text_bounds_pass': True, 'text_overlap_pass': True, 'palette': PALETTE}

def paired_annotation(primary, los, provider, endpoint, days):
    'Read the actual paired estimand; marginal absolute-MAE intervals are not interchangeable.'
    import numpy as np
    metric = 'sync_minus_async_mae_c'
    row = primary.loc[primary.provider.eq(provider) & primary.endpoint.eq(endpoint)
                      & primary.duration_days.eq(days) & primary.metric.eq(metric)]
    require(len(row) == 1, 'Non-unique paired annotation identity')
    point = row.iloc[0]
    if provider == 'PreDist':
        lower, upper = float(point.ci95_lower), float(point.ci95_upper)
        uncertainty = 'pointwise_node_bootstrap_percentile_95'
        source = 'PROVIDER_SUMMARY.csv'
    else:
        omitted = los.loc[los.estimand.eq('same_season') & los.endpoint.eq(endpoint)
                          & los.duration_days.eq(days) & los.metric.eq(metric)]
        require(len(omitted) == 5 and set(omitted.omitted_system) == set(SYSTEMS),
                'Incomplete paired leave-one-system range')
        require(np.allclose(omitted.full_estimate, point.estimate, atol=1e-12),
                'Paired annotation full-estimate drift')
        lower = float(omitted.leave_one_system_estimate.min())
        upper = float(omitted.leave_one_system_estimate.max())
        uncertainty = 'descriptive_leave_one_system_range_not_CI'
        source = 'XAI_LEAVE_ONE_SYSTEM.csv'
    label = f"{days} d: {point.estimate:.3f} [{lower:.3f}, {upper:.3f}]"
    return label, {'provider': provider, 'endpoint': endpoint, 'duration_days': days,
                   'metric': metric, 'estimate': float(point.estimate), 'lower': lower,
                   'upper': upper, 'uncertainty': uncertainty, 'source': source}

def build_figures(results, output):
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), 'Use a new empty figure output directory')
    os.environ['MPLCONFIGDIR'] = str(output / '.mplconfig')
    tables = load_tables(results)
    plt = setup_plot()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    summary, focal, los, pairs = [tables[n] for n in FROZEN_HASHES]
    primary = summary.loc[summary.estimand.eq('same_season')]
    f = focal.loc[focal.estimand.eq('same_season') & focal.metric.eq('sync_minus_async_mae_c') & focal.duration_days.isin([7, 90])]
    fig = plt.figure(figsize=(178 / 25.4, 185 / 25.4))
    fig.text(.095, .975, 'Absolute comparison error', fontsize=10, weight='bold')
    fig.legend(handles=[Patch(facecolor=BLUE, label='Synchronous'), Patch(facecolor=PINK, label='Asynchronous'),
        Patch(facecolor='white', edgecolor=GRAY, hatch='//', label='90-day protocol')],
        loc='upper right', bbox_to_anchor=(.99, .971), ncol=3, frameon=False, columnspacing=1)
    upper = []
    annotation_records = []
    for idx, (provider, endpoint, box) in enumerate([
        ('PreDist', 'median', [.095, .780, .40, .125]),
        ('PreDist', 'heat_weighted', [.59, .780, .40, .125]),
        ('XAI4HEAT', 'median', [.095, .490, .40, .15]),
        ('XAI4HEAT', 'heat_weighted', [.59, .490, .40, .15])]):
        ax = fig.add_axes(box)
        part = primary.loc[primary.provider.eq(provider) & primary.endpoint.eq(endpoint)]
        for design, offset in [('sync', -.18), ('async', .18)]:
            rows = part.loc[part.metric.eq(design + '_mae_c')].set_index('duration_days').loc[DURATIONS]
            x = np.arange(5) + offset
            bars = ax.bar(x, rows.estimate, width=.33, color=PALETTE[design], linewidth=.4, edgecolor='white', zorder=2)
            bars[-1].set_hatch('//')
            bars[-1].set_edgecolor('#FFFFFF')
            if provider == 'PreDist':
                low, high = rows.ci95_lower.to_numpy(), rows.ci95_upper.to_numpy()
                label = '95% node CI'
                linestyle = '-'
            else:
                local = los.loc[los.estimand.eq('same_season') & los.endpoint.eq(endpoint) & los.metric.eq(design + '_mae_c')]
                low = np.array([local.loc[local.duration_days.eq(d), 'leave_one_system_estimate'].min() for d in DURATIONS])
                high = np.array([local.loc[local.duration_days.eq(d), 'leave_one_system_estimate'].max() for d in DURATIONS])
                label = 'LOS range (not CI)'
                linestyle = '--'
            ax.vlines(x, low, high, colors=INK, linewidth=.7, linestyle=linestyle, zorder=3)
            ax.hlines(low, x - .07, x + .07, colors=INK, linewidth=.7, zorder=3)
            ax.hlines(high, x - .07, x + .07, colors=INK, linewidth=.7, zorder=3)
        ax.set_ylim(0, 4.25)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_xticks(np.arange(5), DURATIONS)
        ax.set_xlabel('Campaign duration (days)')
        ax.set_ylabel('Contrast MAE (°C)')
        panel_label(ax, chr(97 + idx), provider + ' | ' + ('Hourly median' if endpoint == 'median' else 'Heat weighted'))
        ax.text(.98, .95, label, transform=ax.transAxes, ha='right', va='top', fontsize=8)
        style_axis(ax)
        upper.append(ax)
        if idx in [0, 2]:
            fig.text(.095, box[1] - .064,
                     'Paired MAE difference: sync - async (°C)',
                     fontsize=8.5, weight='bold', color=INK)
        for row, days in enumerate([7, 90]):
            label, record = paired_annotation(primary, los, provider, endpoint, days)
            fig.text(box[0], box[1] - .084 - .021 * row, label,
                     fontsize=8.5, weight='bold' if days == 7 else 'normal', color=INK)
            annotation_records.append(record)
    fig.text(.095, .350, 'Variation in physical-system timing effects', fontsize=10, weight='bold')
    ax = fig.add_axes([.095, .11, .40, .185])
    groups = [('median', 7), ('median', 90), ('heat_weighted', 7), ('heat_weighted', 90)]
    for y, (endpoint, days) in enumerate(groups):
        rows = f.loc[f.provider.eq('PreDist') & f.endpoint.eq(endpoint) & f.duration_days.eq(days)].sort_values('system')
        if days == 90:
            ax.axhspan(y - .37, y + .37, facecolor='#EDF2F5', linewidth=0, zorder=0)
        offsets = (np.arange(len(rows)) % 7 - 3) * .055
        ax.scatter(rows.estimate, y + offsets, s=8, facecolors=GRAY if days == 7 else 'white', edgecolors=GRAY, linewidths=.5, zorder=3)
    ax.axvline(0, color=INK, linewidth=.7)
    ax.set_xlim(-1, .5)
    ax.set_xticks([-1, -.5, 0, .5])
    ax.set_yticks(range(4), ['Median\n7 d', 'Median\n90 d', 'Heat\n7 d', 'Heat\n90 d'])
    ax.set_ylim(3.6, -.7)
    ax.set_xlabel('Sync - async MAE (°C)')
    panel_label(ax, 'e', 'PreDist | 68 focal systems per row')
    style_axis(ax)
    ax.legend(handles=[Line2D([], [], color=GRAY, marker='o', linestyle='', label='7 d filled'),
        Line2D([], [], color=GRAY, marker='o', markerfacecolor='white', linestyle='', label='90 d open')],
        loc='upper center', bbox_to_anchor=(.47, -.30), ncol=2, frameon=False,
        columnspacing=.8, handletextpad=.3)
    ax = fig.add_axes([.59, .11, .40, .185])
    for endpoint, marker, offset in [('median', 'o', -.13), ('heat_weighted', 's', .13)]:
        for days, shift in [(7, -.035), (90, .035)]:
            rows = f.loc[f.provider.eq('XAI4HEAT') & f.endpoint.eq(endpoint) & f.duration_days.eq(days)].set_index('system').loc[SYSTEMS]
            ax.scatter(np.arange(5) + offset + shift, rows.estimate, marker=marker, s=19,
                facecolors=GRAY if days == 7 else 'white', edgecolors=GRAY, linewidths=.7, zorder=3)
    ax.axhline(0, color=INK, linewidth=.7)
    ax.set_ylim(-1.1, .2)
    ax.set_yticks([-1, -.5, 0])
    ax.set_xticks(range(5), SYSTEMS)
    ax.set_xlabel('Physical system')
    ax.set_ylabel('Sync - async MAE (°C)')
    panel_label(ax, 'f', 'XAI4HEAT | all five focal systems')
    style_axis(ax)
    ax.legend(handles=[Line2D([], [], color=GRAY, marker='o', linestyle='', label='Median'),
        Line2D([], [], color=GRAY, marker='s', linestyle='', label='Heat weighted'),
        Line2D([], [], color=GRAY, marker='o', linestyle='', label='7 d filled'),
        Line2D([], [], color=GRAY, marker='o', markerfacecolor='white', linestyle='', label='90 d open')],
        loc='upper center', bbox_to_anchor=(.50, -.30), ncol=2, frameon=False,
        columnspacing=.8, handletextpad=.3)
    main_qa = save_figure(fig, output, 'Figure_2', 178, 185)
    plt.close(fig)
    main_qa.update({'provider_endpoint_duration_groups': 20, 'paired_bars': 40,
        'PreDist_focal_points': 272, 'XAI_focal_points': 20,
        'paired_annotations': annotation_records,
        'paired_annotation_intervals_from_marginal_error_bounds': False,
        'CI_to_density_conversion': False, 'signed_effects_zero_truncation': False,
        'new_scientific_analysis': False})
    from diagnostic_facets import build_diagnostics, S1_CAPTION, S2_CAPTION
    si_qa = build_diagnostics(pairs, output, plt, save_figure)
    for name in FROZEN_HASHES:
        shutil.copyfile(Path(results) / name, output / name)
    write_json(output / 'FIGURE_METADATA.json', {'inputs': [{'path': n, 'sha256': h} for n, h in FROZEN_HASHES.items()],
        'main': main_qa, 'supplement': si_qa, 'source_code': Path(__file__).name,
        'title': 'Temporal alignment shapes return-temperature comparability between heating systems',
        'main_caption': FIG5_CAPTION, 'supplement_caption': S1_CAPTION,
        'supplement_captions': {'S1': S1_CAPTION, 'S2': S2_CAPTION},
        'public_package_binding': 'return_temperature_comparability'} )
    (output / 'FIGURE_CAPTIONS.md').write_text(FIG5_CAPTION + '\n\n' + S1_CAPTION + '\n\n' + S2_CAPTION + '\n')
    print(json.dumps({'figures': 3, 'supplement_pages': 6, 'main_groups': 20, 'full_objects': len(pairs), 'status': 'technical_pass'}))

def run_tests(results, output):
    tables = load_tables(results)
    class Contracts(unittest.TestCase):
        def test_valid_frozen_tables(self):
            validate_tables(tables)
        def test_missing_directory(self):
            with self.assertRaisesRegex(ValueError, 'directory is missing'):
                load_tables(Path(results) / 'not_present')
        def test_missing_input(self):
            altered = dict(tables)
            altered.pop('PROVIDER_SUMMARY.csv')
            with self.assertRaises(ValueError):
                validate_tables(altered)
        def test_missing_column(self):
            altered = {n: t.copy() for n, t in tables.items()}
            altered['FOCAL_SYSTEM_SUMMARY.csv'].drop(columns='system', inplace=True)
            with self.assertRaisesRegex(ValueError, 'columns missing'):
                validate_tables(altered)
        def test_wrong_provider(self):
            altered = {n: t.copy() for n, t in tables.items()}
            altered['PROVIDER_SUMMARY.csv'].loc[0, 'provider'] = 'Other'
            with self.assertRaisesRegex(ValueError, 'provider identity'):
                validate_tables(altered)
        def test_wrong_numbers(self):
            altered = {n: t.copy() for n, t in tables.items()}
            t = altered['PROVIDER_SUMMARY.csv']
            index = t.index[t.estimand.eq('same_season') & t.metric.eq('sync_mae_c')][0]
            t.loc[index, 'estimate'] += 1
            with self.assertRaisesRegex(ValueError, 'contrast numbers'):
                validate_tables(altered)
        def test_wrong_system_identity(self):
            altered = {n: t.copy() for n, t in tables.items()}
            altered['FOCAL_SYSTEM_SUMMARY.csv'].loc[altered['FOCAL_SYSTEM_SUMMARY.csv'].system.eq('L4'), 'system'] = 'WRONG'
            with self.assertRaisesRegex(ValueError, 'physical identities'):
                validate_tables(altered)
        def test_duplicate(self):
            import pandas as pd
            altered = {n: t.copy() for n, t in tables.items()}
            altered['PAIR_SEASON_SUMMARY.csv'] = pd.concat([altered['PAIR_SEASON_SUMMARY.csv'], altered['PAIR_SEASON_SUMMARY.csv'].iloc[:1]])
            with self.assertRaisesRegex(ValueError, 'Duplicate identities'):
                validate_tables(altered)
        def test_palette_roles(self):
            check_palette(PALETTE)
            with self.assertRaisesRegex(ValueError, 'Palette role'):
                check_palette({'sync': PINK, 'async': BLUE, 'effect': GRAY})
        def test_no_xai_population_ci(self):
            altered = {n: t.copy() for n, t in tables.items()}
            t = altered['PROVIDER_SUMMARY.csv']
            t.loc[t.provider.eq('XAI4HEAT') & t.estimand.eq('same_season'), 'ci95_lower'] = 0
            with self.assertRaisesRegex(ValueError, 'population CI'):
                validate_tables(altered)
        def test_paired_annotation_source_identity(self):
            primary = tables['PROVIDER_SUMMARY.csv'].loc[tables['PROVIDER_SUMMARY.csv'].estimand.eq('same_season')]
            los = tables['XAI_LEAVE_ONE_SYSTEM.csv']
            for provider in PROVIDERS:
                for endpoint in ENDPOINTS:
                    for days in [7, 90]:
                        label, record = paired_annotation(primary, los, provider, endpoint, days)
                        row = primary.loc[primary.provider.eq(provider) & primary.endpoint.eq(endpoint)
                                          & primary.duration_days.eq(days) & primary.metric.eq('sync_minus_async_mae_c')].iloc[0]
                        self.assertEqual(record['estimate'], float(row.estimate))
                        self.assertIn(f"{days} d:", label)
                        if provider == 'PreDist':
                            self.assertEqual((record['lower'], record['upper']), (float(row.ci95_lower), float(row.ci95_upper)))
                        else:
                            omitted = los.loc[los.estimand.eq('same_season') & los.endpoint.eq(endpoint)
                                              & los.duration_days.eq(days) & los.metric.eq('sync_minus_async_mae_c')]
                            self.assertEqual((record['lower'], record['upper']),
                                             (float(omitted.leave_one_system_estimate.min()), float(omitted.leave_one_system_estimate.max())))
                            self.assertIn('not_CI', record['uncertainty'])
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    proc = subprocess.run([sys.executable, str(Path(__file__)), 'figures'], capture_output=True, text=True)
    require(proc.returncode != 0 and '--results' in proc.stderr and '--output' in proc.stderr, 'Explicit CLI missing-input test')
    write_json(output, {'tests_run': result.testsRun + 1, 'passed': result.wasSuccessful(),
        'failures': len(result.failures), 'errors': len(result.errors), 'explicit_cli_required_paths': True,
        'frozen_input_hashes': FROZEN_HASHES, 'palette_roles': PALETTE})
    require(result.wasSuccessful(), 'Unit/negative tests failed')

def main():
    parser = argparse.ArgumentParser(description='Presentation-only renderer; original Figure 5 displays as Figure 2.')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['figures', 'test']:
        command = commands.add_parser(name)
        command.add_argument('--results', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'figures':
        build_figures(args.results, args.output)
    else:
        run_tests(args.results, args.output)

if __name__ == '__main__':
    main()
