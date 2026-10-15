"""Plot registered strict-reversal rates and their existing uncertainty summaries."""
import argparse
import csv
import json
from pathlib import Path

from render_comparability import (BLUE, PINK, INK, DURATIONS, PROVIDERS,
                                  ENDPOINTS, load_tables, setup_plot,
                                  save_figure, style_axis, require)
from panel_geometry import panel_label

CAPTION = (
    'Figure 3. Strict reversals of seasonal temperature ordering. '
    'The same-start and Cartesian designs use the same marginal campaign distributions. '
    'Panels show PreDist (a,b) and XAI4HEAT (c,d), with median and heat-weighted endpoints in columns. '
    'Blue circles denote synchronous pairing and pink squares denote asynchronous pairing. '
    'Solid whiskers are pointwise 95% physical-system bootstrap intervals in PreDist; '
    'dashed whiskers span the five leave-one-system estimates in XAI4HEAT. '
    'Zero-reference contrasts have no defined ordering; campaign ties follow the definition in Section 2. '
    'Each point is the registered provider-level rate, with shared physical systems across pairs.'
)


def ordering_records(tables):
    primary = tables['PROVIDER_SUMMARY.csv']
    primary = primary.loc[primary.estimand.eq('same_season')]
    los = tables['XAI_LEAVE_ONE_SYSTEM.csv']
    records = []
    for provider in PROVIDERS:
        for endpoint in ENDPOINTS:
            for duration in DURATIONS:
                for design in ['sync', 'async']:
                    metric = design + '_reversal_rate'
                    rows = primary.loc[primary.provider.eq(provider) & primary.endpoint.eq(endpoint)
                                       & primary.duration_days.eq(duration) & primary.metric.eq(metric)]
                    require(len(rows) == 1, 'Missing or duplicate strict-reversal estimate')
                    r = rows.iloc[0]
                    if provider == 'PreDist':
                        low, high = float(r.ci95_lower), float(r.ci95_upper)
                        interval = 'pointwise_node_bootstrap_95'
                        source = 'PROVIDER_SUMMARY.csv'
                    else:
                        bounds = los.loc[los.estimand.eq('same_season') & los.endpoint.eq(endpoint)
                                         & los.duration_days.eq(duration) & los.metric.eq(metric)]
                        require(len(bounds) == 5 and bounds.omitted_system.nunique() == 5,
                                'Incomplete strict-reversal leave-one-system range')
                        require(bounds.full_estimate.sub(r.estimate).abs().max() < 1e-12,
                                'Leave-one-system full estimate disagrees')
                        low = float(bounds.leave_one_system_estimate.min())
                        high = float(bounds.leave_one_system_estimate.max())
                        interval = 'leave_one_system_range'
                        source = 'XAI_LEAVE_ONE_SYSTEM.csv'
                    records.append(dict(provider=provider, endpoint=endpoint,
                                        duration_days=duration, design=design,
                                        estimate=float(r.estimate), lower=low, upper=high,
                                        interval=interval, interval_source=source,
                                        estimate_source='PROVIDER_SUMMARY.csv'))
    validate_records(records)
    return records


def validate_records(records):
    import math
    expected = {(p, e, d, s) for p in PROVIDERS for e in ENDPOINTS
                for d in DURATIONS for s in ['sync', 'async']}
    keys = [(r['provider'], r['endpoint'], r['duration_days'], r['design']) for r in records]
    require(len(keys) == len(set(keys)) == 40 and set(keys) == expected,
            'Incomplete strict-reversal display identities')
    for r in records:
        require(all(math.isfinite(r[k]) and 0 <= r[k] <= 1 for k in ['estimate', 'lower', 'upper']),
                'Invalid strict-reversal fraction')
        require(r['lower'] <= r['upper'], 'Reversed strict-reversal bounds')
        expected_interval = ('pointwise_node_bootstrap_95' if r['provider'] == 'PreDist'
                             else 'leave_one_system_range')
        require(r['interval'] == expected_interval, 'Wrong strict-reversal uncertainty identity')


def build(results, output):
    import numpy as np
    rows = ordering_records(load_tables(results))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt = setup_plot()
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(178 / 25.4, 126 / 25.4), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.15, top=.85, hspace=.52, wspace=.27)
    maximum = max(r['upper'] for r in rows)
    ymax = np.ceil(maximum * 100 / 5) * 5
    for i, (provider, endpoint) in enumerate((p, e) for p in PROVIDERS for e in ENDPOINTS):
        ax = axes.flat[i]
        for design, color, marker, shift in [('sync', BLUE, 'o', -.065), ('async', PINK, 's', .065)]:
            part = [r for r in rows if r['provider'] == provider and r['endpoint'] == endpoint
                    and r['design'] == design]
            x = np.arange(5) + shift
            points = [100 * r['estimate'] for r in part]
            low, high = [[100 * r[k] for r in part] for k in ['lower', 'upper']]
            ax.plot(x, points, color=color, marker=marker, ms=4.2, lw=1.1,
                    ls='-' if design == 'sync' else '--', zorder=3)
            ax.vlines(x, low, high, color=color, lw=.8,
                      ls='-' if provider == 'PreDist' else '--', zorder=2)
            ax.hlines(low, x - .04, x + .04, color=color, lw=.7)
            ax.hlines(high, x - .04, x + .04, color=color, lw=.7)
        ax.set(xlim=(-.35, 4.35), ylim=(0, ymax), xticks=np.arange(5),
               xticklabels=DURATIONS, xlabel='Campaign duration (days)')
        if i % 2 == 0:
            ax.set_ylabel('Strict reversal rate (%)')
        name = 'Median' if endpoint == 'median' else 'Heat weighted'
        panel_label(ax, chr(97 + i), provider + ' | ' + name)
        style_axis(ax)
    fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', lw=1.1, label='Synchronous'),
                        Line2D([], [], color=PINK, marker='s', lw=1.1, ls='--', label='Asynchronous')],
               loc='upper center', bbox_to_anchor=(.54, .985), ncol=2, frameon=False)
    fig.text(.10, .027, 'Whiskers: PreDist 95% node intervals; XAI4HEAT leave-one-system ranges.',
             fontsize=8, color=INK)
    qa = save_figure(fig, output, 'Figure_3', 178, 126)
    plt.close(fig)
    with (output / 'Figure_3_values.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'Figure_3_caption.txt').write_text(CAPTION + '\n')
    qa.update(points=len(rows), y_limits_percent=[0, float(ymax)], source_values_changed=False)
    (output / 'Figure_3_contract.json').write_text(json.dumps(qa, indent=2) + '\n')
    return qa


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.results, args.output), indent=2))
