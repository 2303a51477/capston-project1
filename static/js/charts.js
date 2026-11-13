/**
 * charts.js
 * Chart.js visualizations for AI-Based Student Performance Prediction System.
 */

// Global Chart Defaults
if (typeof Chart !== 'undefined') {
  Chart.defaults.font.family = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  Chart.defaults.color = '#64748b';
  Chart.defaults.borderColor = '#e2e8f0';
}

function initDashboardCharts() {
  // 1. Dashboard Attendance Doughnut Chart
  const attCanvas = document.getElementById('dashboardAttendanceGauge');
  if (attCanvas) {
    const attPct = parseFloat(attCanvas.getAttribute('data-pct')) || 0;
    const remaining = Math.max(0, 100 - attPct);

    let color = '#16a34a'; // Green
    if (attPct < 65) color = '#dc2626'; // Red
    else if (attPct < 75) color = '#d97706'; // Yellow

    new Chart(attCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Attended', 'Missed / Remaining'],
        datasets: [{
          data: [attPct, remaining],
          backgroundColor: [color, '#e2e8f0'],
          borderWidth: 0,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '76%',
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return `${ctx.label}: ${ctx.parsed}%`;
              }
            }
          }
        }
      }
    });
  }

  // 2. Dashboard Subject Marks Bar Chart
  const marksCanvas = document.getElementById('dashboardSubjectMarksChart');
  if (marksCanvas) {
    fetch('/api/student-analytics')
      .then(res => res.json())
      .then(data => {
        if (!data.marks.labels.length) return;

        new Chart(marksCanvas, {
          type: 'bar',
          data: {
            labels: data.marks.labels.map(l => l.length > 18 ? l.substring(0, 18) + '...' : l),
            datasets: [
              {
                label: 'IA 1 (25)',
                data: data.marks.ia1,
                backgroundColor: '#38bdf8',
                borderRadius: 4
              },
              {
                label: 'IA 2 (25)',
                data: data.marks.ia2,
                backgroundColor: '#0284c7',
                borderRadius: 4
              },
              {
                label: 'Assignment (25)',
                data: data.marks.assignments,
                backgroundColor: '#0f172a',
                borderRadius: 4
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { stacked: false, grid: { display: false } },
              y: { max: 25, beginAtZero: true, grid: { color: '#f1f5f9' } }
            },
            plugins: {
              legend: { position: 'top', align: 'end' }
            }
          }
        });
      })
      .catch(err => console.error("Could not load dashboard chart data:", err));
  }
}

function initAttendanceCharts() {
  const canvas = document.getElementById('attendanceSubjectBarChart');
  if (!canvas) return;

  fetch('/api/student-analytics')
    .then(res => res.json())
    .then(data => {
      if (!data.attendance.labels.length) return;

      const pcts = data.attendance.percentages;
      const barColors = pcts.map(p => p >= 75 ? '#16a34a' : (p >= 65 ? '#d97706' : '#dc2626'));

      new Chart(canvas, {
        type: 'bar',
        data: {
          labels: data.attendance.labels,
          datasets: [{
            label: 'Attendance %',
            data: pcts,
            backgroundColor: barColors,
            borderRadius: 6,
            barThickness: 28
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              max: 100,
              beginAtZero: true,
              ticks: { callback: v => v + '%' }
            },
            x: { grid: { display: false } }
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: function (ctx) {
                  const idx = ctx[0].dataIndex;
                  return `${data.attendance.labels[idx]}: ${data.attendance.names[idx]}`;
                },
                label: function (ctx) {
                  return `Attendance: ${ctx.parsed.y}%`;
                }
              }
            }
          }
        }
      });
    });
}

function initReportsCharts() {
  const historyCanvas = document.getElementById('predictionHistoryChart');
  const subjectRadarCanvas = document.getElementById('subjectPerformanceRadar');

  fetch('/api/student-analytics')
    .then(res => res.json())
    .then(data => {
      // 1. Prediction History Line Chart
      if (historyCanvas && data.predictions.timestamps.length) {
        new Chart(historyCanvas, {
          type: 'line',
          data: {
            labels: data.predictions.timestamps,
            datasets: [{
              label: 'AI Predicted Final Marks',
              data: data.predictions.scores,
              borderColor: '#0284c7',
              backgroundColor: 'rgba(2, 132, 199, 0.1)',
              fill: true,
              tension: 0.35,
              borderWidth: 3,
              pointBackgroundColor: '#0284c7',
              pointRadius: 6,
              pointHoverRadius: 8
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              y: {
                min: 30,
                max: 100,
                ticks: { callback: v => v + ' / 100' }
              },
              x: { grid: { display: false } }
            },
            plugins: {
              tooltip: {
                callbacks: {
                  afterLabel: function (ctx) {
                    return `Tier: ${data.predictions.categories[ctx.dataIndex]}`;
                  }
                }
              }
            }
          }
        });
      }

      // 2. Subject Performance Radar
      if (subjectRadarCanvas && data.marks.labels.length) {
        const scaledTotals = data.marks.totals.map(t => Math.round((t / 75) * 100));

        new Chart(subjectRadarCanvas, {
          type: 'radar',
          data: {
            labels: data.marks.labels.map(l => l.length > 15 ? l.substring(0, 15) + '..' : l),
            datasets: [{
              label: 'Internal Score (Scaled %)',
              data: scaledTotals,
              backgroundColor: 'rgba(56, 189, 248, 0.25)',
              borderColor: '#0284c7',
              pointBackgroundColor: '#0284c7',
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              r: {
                min: 0,
                max: 100,
                ticks: { stepSize: 25 },
                pointLabels: { font: { size: 11, weight: 'bold' } }
              }
            }
          }
        });
      }
    });
}

document.addEventListener('DOMContentLoaded', function () {
  initDashboardCharts();
  initAttendanceCharts();
  initReportsCharts();
});

