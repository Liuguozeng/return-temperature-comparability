'Deterministic presentation of frozen values; no model fitting or CI estimation.'
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
BLUE = '#2879AD'
MID = '#C97E9C'
RED = '#C97E9C'
GRAY = '#7C8792'
INK = '#26333D'
PALE = '#E5EBEF'
CATS = {'undercoverage': RED, 'nominal_band': GRAY, 'overcoverage': BLUE}
PAPERS = {'return_temperature_comparability': '.'}
SUPS = {'return_temperature_comparability': '.'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def configure_style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9, 'axes.labelsize': 9, 'axes.labelcolor': INK, 'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'axes.edgecolor': GRAY, 'axes.linewidth': 0.6, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})

def clean(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    if grid:
        ax.grid(axis=grid, color='#E5EBEF', linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)

def validate_categories(frame):
    unknown = set(frame.coverage_direction.dropna()) - set(CATS)
    if unknown:
        raise ValueError(f'Unknown coverage categories: {unknown}')
    expected = np.where(frame.coverage_90 < 0.85, 'undercoverage', np.where(frame.coverage_90 > 0.95, 'overcoverage', 'nominal_band'))
    if not np.array_equal(expected, frame.coverage_direction):
        raise ValueError('Coverage category disagrees with frozen thresholds')

def bound(value):
    return float(value) if np.isfinite(value) else None

def uncertainty(provider):
    return 'point_only' if provider == 'AAU' else 'descriptive_range' if provider == 'Empa/NEST' else 'cluster_CI'

def effect(ax, y, value, lo, hi, color, provider, height=0.18):
    'Magnitude bar plus a co-located interval band, not a duplicate estimate.'
    bar = ax.barh(y, value, height=height * 0.5, color=color, alpha=0.9, zorder=3)[0]
    if np.isfinite(lo) and np.isfinite(hi):
        ax.add_patch(Rectangle((lo, y - height / 2), hi - lo, height, facecolor=color, alpha=0.12, edgecolor='none', zorder=2))
        ax.add_patch(Rectangle((lo, y - height / 2), hi - lo, height, fill=False, edgecolor=color, lw=0.5, linestyle='--' if provider == 'Empa/NEST' else '-', zorder=4))
    ax.plot([value, value], [y - height * 0.63, y + height * 0.63], color=color, lw=1.25, zorder=5)
    return float(bar.get_width())

def proportion_bands(ax, model, rows, panel):
    'Paired zero-based proportion tracks; only the operator estimate has a CI.'
    ax.clear()
    ax.axis('off')
    ax.set(xlim=(-0.17, 1.39), ylim=(3, 0))
    ax.set_title(f'{panel}  {model} | 120 common facilities', loc='left', fontsize=9.2, weight='bold', pad=25)
    for x, label, color in [(0, 'Operator equal', BLUE), (0.72, 'Facility equal', MID)]:
        ax.add_patch(Rectangle((x, -0.33), 0.06, 0.12, facecolor=color, clip_on=False))
        ax.text(x + 0.085, -0.27, label, va='center', fontsize=7.8)
    for y, r in enumerate(rows.sort_values('year').itertuples()):
        ax.text(-0.04, y + 0.39, str(int(r.year)), ha='right', va='center', fontsize=8.1)
        for offset, value, color, height in [(0.16, float(r.operator_equal_supported_and_covered), BLUE, 0.2), (0.55, float(r.facility_equal_supported_and_covered), MID, 0.16)]:
            ax.add_patch(Rectangle((0, y + offset - height / 2), 1, height, facecolor=PALE, edgecolor='none'))
            ax.add_patch(Rectangle((0, y + offset - height / 2), value, height, facecolor=color, edgecolor='none'))
            ax.text(1.045, y + offset, f'{value:.1%}', va='center', fontsize=8.5, color=color, weight='bold')
        lo, hi = (float(r.operator_cluster_ci95_lower), float(r.operator_cluster_ci95_upper))
        ax.add_patch(Rectangle((lo, y + 0.025), hi - lo, 0.27, facecolor='white', alpha=0.55, edgecolor=INK, lw=0.6))
        ax.text(0.02, y + 0.86, f'95% CI [{lo:.1%}, {hi:.1%}]', va='center', fontsize=7.2, color=GRAY)
    ax.text(0, 3.1, '0%', ha='center', va='top', fontsize=7.2, color=GRAY)
    ax.text(1, 3.1, '100%', ha='center', va='top', fontsize=7.2, color=GRAY)

def supported_coverage_tracks(ax, model, rows, panel):
    'Two full-scale tracks, with numerical CI beside its operator estimate.'
    ax.clear()
    ax.axis('off')
    ax.set(xlim=(-0.17, 1.55), ylim=(3, 0))
    ax.set_title(f'{panel}  {model} | Supported-and-covered', loc='left', fontsize=9.2, weight='bold', pad=24)
    ax.text(0, -0.66, '120 common facilities', fontsize=8.1, color=GRAY, clip_on=False)
    for x, label, color in [(0, 'Operator equal', BLUE), (0.82, 'Facility equal', MID)]:
        ax.add_patch(Rectangle((x, -0.35), 0.055, 0.12, facecolor=color, clip_on=False))
        ax.text(x + 0.08, -0.29, label, va='center', fontsize=7.8)
    for y, r in enumerate(rows.sort_values('year').itertuples()):
        year = int(r.year)
        ax.text(-0.05, y + 0.43, str(year), ha='right', va='center', fontsize=8.1)
        for offset, value, color, weight in [(0.2, float(r.operator_equal_supported_and_covered), BLUE, 'operator'), (0.67, float(r.facility_equal_supported_and_covered), MID, 'facility')]:
            height = 0.38
            ax.add_patch(Rectangle((0, y + offset - height / 2), 1, height, facecolor=PALE, edgecolor='none'))
            bar = Rectangle((0, y + offset - height / 2), value, height, facecolor=color, edgecolor='none')
            bar.set_gid(f'proportion:{model}:{year}:{weight}')
            ax.add_patch(bar)
            ax.text(0.035, y + offset, f'{value:.1%}', va='center', fontsize=8.0, color='white' if weight == 'operator' else INK, weight='bold')
        lo, hi = (float(r.operator_cluster_ci95_lower), float(r.operator_cluster_ci95_upper))
        band = Rectangle((lo, y + 0.01), hi - lo, 0.38, facecolor='white', alpha=0.7, edgecolor=INK, lw=0.6)
        band.set_gid(f'operator_ci:{model}:{year}')
        ax.add_patch(band)
        value = float(r.operator_equal_supported_and_covered)
        ax.plot([value, value], [y + 0.01, y + 0.39], color=INK, lw=0.85)
        ax.text(1.04, y + 0.2, f'[{lo:.1%}, {hi:.1%}]', va='center', fontsize=7.8, color=GRAY)
    for x, label in [(0, '0%'), (0.5, '50%'), (1, '100%')]:
        ax.text(x, 3.08, label, ha='center', va='top', fontsize=7.5, color=GRAY)

class Builder:

    def __init__(self, candidate, output, selected=None, fig4_variant='A'):
        self.candidate, self.output = (candidate, output)
        self.selected = set(selected or [])
        self.records = []
        self.fig4_variant = fig4_variant

    def data(self, paper, name):
        return pd.read_csv(self.candidate / PAPERS[paper] / SUPS[paper] / 'results' / name)

    def wanted(self, paper, number):
        return not self.selected or f'{paper}:{number}' in self.selected

    def export(self, paper, index, fig, artists, supporting=None):
        import json
        parent = self.candidate / 'figures' / ('Figure_' + str(index))
        out = self.output / ('Figure_' + str(index))
        if out.resolve().is_relative_to(self.candidate.resolve()):
            raise ValueError('Output must be outside the read-only release')
        out.mkdir(parents=True, exist_ok=True)
        for suffix in ['pdf', 'svg', 'png']:
            fig.savefig(out / ('Figure_' + str(index) + '.' + suffix), dpi=300)
        plt.close(fig)
        (out / 'ARTIST_DATA.json').write_text(json.dumps(artists, indent=2) + '\n')
        (out / 'PRESENTATION_DATA.json').write_text(json.dumps(supporting or {}, indent=2) + '\n')

    def return_temperature_2(self):
        df = self.data('return_temperature_comparability', 'CROSS_ECOSYSTEM_CONTRASTS.csv')
        assert len(df) == 18 and (not df.duplicated(['provider', 'endpoint_family', 'comparison']).any())
        providers = ['PreDist', 'XAI4HEAT', 'AAU', 'Empa/NEST']
        fig, axes = plt.subplots(1, 2, figsize=(178 / 25.4, 4.65), sharex=True)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.83, wspace=0.18)
        artists = []
        for i, (ax, ep) in enumerate(zip(axes, ['median', 'heat_weighted'])):
            ps = providers + (['Aalborg'] if i else [])
            for j, provider in enumerate(ps):
                y = 4 - j
                if j % 2 == 0:
                    ax.axhspan(y - 0.43, y + 0.43, color=PALE, zorder=-2)
                for k, (co, color) in enumerate(zip(['7_minus_90_days', '14_minus_90_days'], [BLUE, MID])):
                    r = df[df.provider.eq(provider) & df.endpoint_family.eq(ep) & df.comparison.eq(co)].iloc[0]
                    yy = y + (0.17 if k == 0 else -0.17)
                    v = effect(ax, yy, r.estimate, r.ci_low, r.ci_high, color, provider, 0.24)
                    ax.text(2.23, yy, f'{v:.2f}', ha='right', va='center', fontsize=8.5, color=INK)
                    artists.append(dict(provider=provider, endpoint=ep, comparison=co, estimate=v, lower=bound(r.ci_low), upper=bound(r.ci_high), uncertainty=uncertainty(provider)))
            ticks = [4, 3, 2, 1] if i == 0 else [0]
            ax.set(xlim=(0, 2.3), ylim=(-0.48, 4.52), xticks=[0, 0.5, 1, 1.5, 2], yticks=ticks, yticklabels=providers if i == 0 else ['Aalborg'])
            ax.set_title(f'{chr(97 + i)}  ' + ('Median' if i == 0 else 'Heat-weighted'), loc='left', weight='bold', pad=12)
            clean(ax, 'x')
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
        fig.legend(handles=[Patch(facecolor=BLUE, label='7−90 days'), Patch(facecolor=MID, label='14−90 days')], loc='upper center', bbox_to_anchor=(0.57, 0.98), ncol=2, frameon=False)
        fig.text(0.57, 0.085, 'Short minus 90-day absolute error (°C)', ha='center', fontsize=9)
        fig.text(0.14, 0.028, 'Bands: 95% CI; dashed: two-system range. AAU: estimate only.', fontsize=8.2, color=GRAY)
        self.export('return_temperature_comparability', 2, fig, artists)

    def return_temperature_3(self):
        df = self.data('return_temperature_comparability', 'CROSS_ECOSYSTEM_DURATION_SUMMARY.csv')
        df = df[df.endpoint_family.eq('median')]
        fig, axes = plt.subplots(4, 1, figsize=(178 / 25.4, 5.85), sharex=True)
        fig.subplots_adjust(left=0.15, right=0.975, bottom=0.14, top=0.94, hspace=0.34)
        artists = []
        for i, (ax, provider) in enumerate(zip(axes, ['PreDist', 'XAI4HEAT', 'AAU', 'Empa/NEST'])):
            rr = df[df.provider.eq(provider)].sort_values('duration_days')
            upper = max(rr.estimate.max(), rr.ci_high.max() if rr.ci_high.notna().any() else 0)
            ymax = np.ceil(upper * 5) / 5 + 0.16
            for j, r in enumerate(rr.itertuples()):
                color = RED if r.duration_days == 90 else BLUE
                if np.isfinite(r.ci_low):
                    ax.add_patch(Rectangle((j - 0.27, r.ci_low), 0.54, r.ci_high - r.ci_low, facecolor=color, alpha=0.12, edgecolor='none', zorder=2))
                    ax.add_patch(Rectangle((j - 0.27, r.ci_low), 0.54, r.ci_high - r.ci_low, fill=False, edgecolor=color, lw=0.55, ls='--' if provider == 'Empa/NEST' else '-', zorder=4))
                bar = ax.bar(j, r.estimate, width=0.2, color=color, alpha=0.9, zorder=3)[0]
                ax.plot([j - 0.29, j + 0.29], [r.estimate] * 2, color=color, lw=1.1, zorder=5)
                ax.text(j + 0.35, r.estimate, f'{r.estimate:.2f}', va='center', fontsize=8.8, color=INK)
                artists.append(dict(provider=provider, duration=int(r.duration_days), estimate=float(bar.get_height()), lower=bound(r.ci_low), upper=bound(r.ci_high), uncertainty=uncertainty(provider)))
            ax.set(ylim=(0, ymax), xlim=(-0.62, 4.72), yticks=[0, round(ymax - 0.16, 1)], xticks=range(5), xticklabels=['7', '14', '30', '60', '90'])
            clean(ax, 'y')
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#C9D2D9')
            ax.tick_params(axis='y', length=0)
            ax.text(-0.17, 0.65, f'{chr(97 + i)}\n{provider}', transform=ax.transAxes, va='center', fontsize=9, weight='bold')
        axes[-1].set_xlabel('Campaign duration (days)')
        fig.text(0.15, 0.975, 'Mean absolute error for the median endpoint (°C)', fontsize=10, weight='bold')
        fig.text(0.15, 0.043, 'Within-provider scales; bands: 95% CI; dashed: two-system range; AAU: estimates.', fontsize=8.1, color=GRAY)
        fig.text(0.15, 0.016, 'Pink marks the 90-day comparator.', fontsize=8.1, color=GRAY)
        self.export('return_temperature_comparability', 3, fig, artists, {'axis_scales': 'provider-specific, zero based', 'durations': [7, 14, 30, 60, 90]})

    def return_temperature_4(self):
        equal = self.data('return_temperature_comparability', 'EQUAL_DAY_CONTRASTS.csv')
        disjoint = self.data('return_temperature_comparability', 'NONOVERLAP_CONTRASTS.csv')
        fig = plt.figure(figsize=(178 / 25.4, 5.35))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 2], left=0.18, right=0.975, bottom=0.19, top=0.88, hspace=0.52)
        artists = []
        for i, (df, providers, title) in enumerate([(equal, ['PreDist', 'XAI4HEAT', 'AAU', 'Empa/NEST'], 'Equal-day aggregation'), (disjoint, ['PreDist', 'XAI4HEAT'], 'Date-disjoint reference')]):
            ax = fig.add_subplot(gs[i])
            n = len(providers)
            for j, provider in enumerate(providers):
                y = n - 1 - j
                if j % 2 == 0:
                    ax.axhspan(y - 0.43, y + 0.43, color=PALE, zorder=-2)
                for offset, short, color in [(0.17, 7, BLUE), (-0.17, 14, MID)]:
                    r = df[df.provider.eq(provider) & df.short_days.eq(short)].iloc[0]
                    v = effect(ax, y + offset, r.contrast_c, r.ci95_lower, r.ci95_upper, color, provider, 0.23)
                    ax.text(1.67, y + offset, f'{v:+.2f}', ha='right', va='center', color=INK, fontsize=8.5)
                    artists.append(dict(analysis='equal_day' if i == 0 else 'date_disjoint', provider=provider, short_days=short, estimate=v, lower=bound(r.ci95_lower), upper=bound(r.ci95_upper), uncertainty=uncertainty(provider)))
            ax.axvline(0, color=GRAY, lw=0.75, zorder=1)
            ax.set(xlim=(-0.58, 1.73), ylim=(-0.48, n - 0.48), yticks=range(n), yticklabels=providers[::-1], xticks=[-0.5, 0, 0.5, 1, 1.5])
            clean(ax, 'x')
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='y', length=0)
            ax.set_title(f'{chr(97 + i)}  {title}', loc='left', weight='bold', pad=8)
        ax.set_xlabel('Median-endpoint error contrast (°C)')
        fig.legend(handles=[Patch(facecolor=BLUE, label='7−90 days'), Patch(facecolor=MID, label='14−90 days')], loc='upper right', bbox_to_anchor=(0.985, 0.985), ncol=2, frameon=False)
        fig.text(0.18, 0.085, 'Bands: 95% CI; dashed: two-system range; AAU: estimate only.', fontsize=8.2, color=GRAY)
        fig.text(0.18, 0.04, 'Date-disjoint eligibility: ≥5 systems and ≥30 common-start instances;', fontsize=8.2, color=GRAY)
        fig.text(0.18, 0.014, 'AAU and Empa/NEST do not meet this requirement.', fontsize=8.2, color=GRAY)
        self.export('return_temperature_comparability', 4, fig, artists)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    args=parser.parse_args()
    configure_style()
    builder=Builder(Path(__file__).resolve().parents[1], args.output_root)
    for number in [2, 3, 4]:
        getattr(builder, 'return_temperature_'+str(number))()
