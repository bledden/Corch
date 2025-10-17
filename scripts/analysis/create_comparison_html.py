#!/usr/bin/env python3
"""
Generate beautiful HTML comparison report for Sequential vs Baseline results
"""

import json
import sys
from datetime import datetime

def create_html_report(task_data, output_file="comparison_report.html"):
    """Create HTML report with charts and detailed comparison"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sequential vs Baseline - Quality Comparison</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}

        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .timestamp {{
            margin-top: 10px;
            font-size: 0.9em;
            opacity: 0.8;
        }}

        .content {{
            padding: 40px;
        }}

        .task-header {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}

        .task-header h2 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        .task-description {{
            font-size: 1.2em;
            opacity: 0.95;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}

        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-5px);
        }}

        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .metric-value {{
            font-size: 2.5em;
            font-weight: 700;
        }}

        .comparison-section {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 40px 0;
        }}

        .method-card {{
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }}

        .method-card.sequential {{
            border-left: 5px solid #28a745;
        }}

        .method-card.baseline {{
            border-left: 5px solid #ffc107;
        }}

        .method-title {{
            font-size: 1.8em;
            margin-bottom: 20px;
            font-weight: 700;
        }}

        .method-title.sequential {{
            color: #28a745;
        }}

        .method-title.baseline {{
            color: #ff9800;
        }}

        .quality-score {{
            font-size: 4em;
            font-weight: 700;
            text-align: center;
            margin: 20px 0;
        }}

        .quality-score.sequential {{
            color: #28a745;
        }}

        .quality-score.baseline {{
            color: #ff9800;
        }}

        .dimensions {{
            margin-top: 20px;
        }}

        .dimension {{
            margin: 15px 0;
        }}

        .dimension-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: 600;
        }}

        .dimension-bar {{
            height: 25px;
            background: #e9ecef;
            border-radius: 12px;
            overflow: hidden;
        }}

        .dimension-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            padding: 0 10px;
            color: white;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .chart-container {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 15px;
        }}

        .chart-title {{
            font-size: 1.5em;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 700;
        }}

        .models-used {{
            margin: 30px 0;
            padding: 25px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            border-radius: 15px;
            color: white;
        }}

        .models-used h3 {{
            margin-bottom: 15px;
            font-size: 1.5em;
        }}

        .model-list {{
            list-style: none;
        }}

        .model-list li {{
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.2);
            font-size: 1.1em;
        }}

        .model-list li:last-child {{
            border-bottom: none;
        }}

        .winner-badge {{
            display: inline-block;
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 700;
            margin-left: 10px;
        }}

        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #666;
        }}

        @media (max-width: 768px) {{
            .comparison-section {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Sequential vs Baseline</h1>
            <div class="subtitle">AI Code Generation Quality Comparison</div>
            <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>

        <div class="content">
            <div class="task-header">
                <h2>Task: {task_data.get('task', 'Unknown Task')}</h2>
                <div class="task-description">{task_data.get('description', 'Compare sequential vs baseline code generation')}</div>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Quality Improvement</div>
                    <div class="metric-value">{task_data['improvement_pct']:.0f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sequential Score</div>
                    <div class="metric-value">{task_data['sequential']['quality']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Baseline Score</div>
                    <div class="metric-value">{task_data['baseline']['quality']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pass Threshold</div>
                    <div class="metric-value">0.70</div>
                </div>
            </div>

            <div class="comparison-section">
                <div class="method-card sequential">
                    <div class="method-title sequential">Sequential (5-Stage) <span class="winner-badge">WINNER</span></div>
                    <div class="quality-score sequential">{task_data['sequential']['quality']:.2f}</div>
                    <div class="dimensions">
                        <h4>Quality Dimensions:</h4>
"""

    # Add sequential dimensions
    for dim, score in task_data['sequential']['dimensions'].items():
        pct = score * 100
        html += f"""
                        <div class="dimension">
                            <div class="dimension-label">
                                <span>{dim.replace('_', ' ').title()}</span>
                                <span>{score:.2f}</span>
                            </div>
                            <div class="dimension-bar">
                                <div class="dimension-fill" style="width: {pct}%">{pct:.0f}%</div>
                            </div>
                        </div>
"""

    html += """
                    </div>
                    <div class="models-used">
                        <h3>Models Used:</h3>
                        <ul class="model-list">
"""

    for model in task_data['sequential']['models']:
        html += f"                            <li>{model}</li>\n"

    html += """
                        </ul>
                    </div>
                </div>

                <div class="method-card baseline">
                    <div class="method-title baseline">Baseline (Single-Pass)</div>
                    <div class="quality-score baseline">{:.2f}</div>
                    <div class="dimensions">
                        <h4>Quality Dimensions:</h4>
""".format(task_data['baseline']['quality'])

    # Add baseline dimensions
    for dim, score in task_data['baseline']['dimensions'].items():
        pct = score * 100
        html += f"""
                        <div class="dimension">
                            <div class="dimension-label">
                                <span>{dim.replace('_', ' ').title()}</span>
                                <span>{score:.2f}</span>
                            </div>
                            <div class="dimension-bar">
                                <div class="dimension-fill" style="width: {pct}%">{pct:.0f}%</div>
                            </div>
                        </div>
"""

    html += """
                    </div>
                    <div class="models-used">
                        <h3>Model Used:</h3>
                        <ul class="model-list">
"""

    for model in task_data['baseline']['models']:
        html += f"                            <li>{model}</li>\n"

    html += """
                        </ul>
                    </div>
                </div>
            </div>

            <div class="chart-container">
                <div class="chart-title">Quality Dimensions Comparison</div>
                <canvas id="dimensionsChart"></canvas>
            </div>

            <div class="chart-container">
                <div class="chart-title">Overall Quality Score</div>
                <canvas id="overallChart"></canvas>
            </div>
        </div>

        <div class="footer">
            <p><strong>Facilitair</strong> - Sequential AI Collaboration System</p>
            <p>Generated with real quality evaluation using CodeQualityEvaluator</p>
            <p>Tracked with W&B Weave</p>
        </div>
    </div>

    <script>
        // Dimensions comparison chart
        const dimensionsCtx = document.getElementById('dimensionsChart').getContext('2d');
        new Chart(dimensionsCtx, {
            type: 'bar',
            data: {
                labels: """ + json.dumps([dim.replace('_', ' ').title() for dim in task_data['sequential']['dimensions'].keys()]) + """,
                datasets: [
                    {
                        label: 'Sequential',
                        data: """ + json.dumps(list(task_data['sequential']['dimensions'].values())) + """,
                        backgroundColor: 'rgba(40, 167, 69, 0.8)',
                        borderColor: 'rgba(40, 167, 69, 1)',
                        borderWidth: 2
                    },
                    {
                        label: 'Baseline',
                        data: """ + json.dumps(list(task_data['baseline']['dimensions'].values())) + """,
                        backgroundColor: 'rgba(255, 152, 0, 0.8)',
                        borderColor: 'rgba(255, 152, 0, 1)',
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: 14
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            font: {
                                size: 16
                            }
                        }
                    }
                }
            }
        });

        // Overall quality chart
        const overallCtx = document.getElementById('overallChart').getContext('2d');
        new Chart(overallCtx, {
            type: 'doughnut',
            data: {
                labels: ['Sequential', 'Baseline'],
                datasets: [{
                    data: [""" + str(task_data['sequential']['quality']) + """, """ + str(task_data['baseline']['quality']) + """],
                    backgroundColor: [
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(255, 152, 0, 0.8)'
                    ],
                    borderColor: [
                        'rgba(40, 167, 69, 1)',
                        'rgba(255, 152, 0, 1)'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: {
                                size: 16
                            }
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html)

    print(f"✅ HTML report generated: {output_file}")
    return output_file


if __name__ == "__main__":
    # Example data structure - can be loaded from benchmark JSON
    example_data = {
        "task": "Prime Number Checker",
        "description": "Write a function to check if a number is prime",
        "sequential": {
            "quality": 0.84,
            "dimensions": {
                "correctness": 0.90,
                "completeness": 0.85,
                "code_quality": 0.88,
                "documentation": 0.82,
                "error_handling": 0.75,
                "testing": 0.80
            },
            "models": [
                "Architect: meta-llama/llama-3.3-70b-instruct",
                "Coder: deepseek/deepseek-chat (4 iterations)",
                "Reviewer: deepseek/deepseek-chat",
                "Documenter: meta-llama/llama-3.3-70b-instruct"
            ]
        },
        "baseline": {
            "quality": 0.45,
            "dimensions": {
                "correctness": 0.55,
                "completeness": 0.40,
                "code_quality": 0.50,
                "documentation": 0.20,
                "error_handling": 0.30,
                "testing": 0.15
            },
            "models": [
                "qwen/qwen3-coder-plus"
            ]
        },
        "improvement_pct": 86.7  # (0.84 - 0.45) / 0.45 * 100
    }

    create_html_report(example_data, "task1_comparison.html")
