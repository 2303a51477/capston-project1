/**
 * main.js
 * Frontend UI interactions, mobile sidebar toggling, and helpers.
 */

document.addEventListener('DOMContentLoaded', function () {
  // 1. Mobile Sidebar Toggle
  const sidebar = document.querySelector('.sidebar');
  const sidebarToggle = document.getElementById('sidebarToggle');
  const sidebarClose = document.getElementById('sidebarClose');

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', function () {
      sidebar.classList.toggle('show');
    });
  }

  if (sidebarClose && sidebar) {
    sidebarClose.addEventListener('click', function () {
      sidebar.classList.remove('show');
    });
  }

  // Close sidebar when clicking outside on mobile
  document.addEventListener('click', function (event) {
    if (sidebar && sidebar.classList.contains('show')) {
      if (!sidebar.contains(event.target) && sidebarToggle && !sidebarToggle.contains(event.target)) {
        sidebar.classList.remove('show');
      }
    }
  });

  // 2. Auto-dismiss Bootstrap Alerts after 5 seconds
  const autoAlerts = document.querySelectorAll('.alert-dismissible');
  autoAlerts.forEach(function (alert) {
    setTimeout(function () {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
      if (bsAlert) {
        bsAlert.close();
      }
    }, 5000);
  });

  // 3. Prediction Form Autofill Helper
  const autofillBtn = document.getElementById('btnAutofillAcademicData');
  if (autofillBtn) {
    autofillBtn.addEventListener('click', function () {
      const curAtt = autofillBtn.getAttribute('data-att');
      const curInternal = autofillBtn.getAttribute('data-internal');

      const attInput = document.getElementById('inputAttendance');
      const internalInput = document.getElementById('inputInternalMarks');
      const assignmentInput = document.getElementById('inputAssignmentMarks');
      const studyHoursInput = document.getElementById('inputStudyHours');
      const prevSemesterInput = document.getElementById('inputPrevSemester');

      if (attInput && curAtt) attInput.value = parseFloat(curAtt).toFixed(1);
      if (internalInput && curInternal) internalInput.value = parseFloat(curInternal).toFixed(1);
      if (assignmentInput && curInternal) assignmentInput.value = Math.min(100, (parseFloat(curInternal) * 1.05)).toFixed(1);
      if (studyHoursInput && (!studyHoursInput.value || studyHoursInput.value === '0')) studyHoursInput.value = '3.5';
      if (prevSemesterInput && (!prevSemesterInput.value || prevSemesterInput.value === '0')) {
        prevSemesterInput.value = curInternal ? parseFloat(curInternal).toFixed(1) : '72.0';
      }

      // Visual feedback
      autofillBtn.innerHTML = '<i class="bi bi-check-lg me-1"></i> Data Filled!';
      autofillBtn.classList.remove('btn-outline-primary');
      autofillBtn.classList.add('btn-success');
      setTimeout(function () {
        autofillBtn.innerHTML = '<i class="bi bi-magic me-1"></i> Autofill from My Records';
        autofillBtn.classList.remove('btn-success');
        autofillBtn.classList.add('btn-outline-primary');
      }, 2500);
    });
  }

  // 4. Quick Attendance Calculator Helper (Inline simulator)
  const calcBtn = document.getElementById('btnCalculateTargetAttendance');
  if (calcBtn) {
    calcBtn.addEventListener('click', function () {
      const attended = parseInt(document.getElementById('simAttended').value) || 0;
      const total = parseInt(document.getElementById('simTotal').value) || 0;
      const target = parseFloat(document.getElementById('simTarget').value) || 75.0;
      const resultBox = document.getElementById('simResult');

      if (total <= 0) {
        resultBox.innerHTML = '<div class="alert alert-warning py-2 mb-0">Total classes must be greater than 0.</div>';
        return;
      }

      const curPct = (attended / total) * 100;
      if (curPct >= target) {
        // Can miss how many?
        // (attended) / (total + y) >= target / 100 => total + y <= attended / (target/100)
        const maxTotal = Math.floor(attended / (target / 100));
        const canMiss = Math.max(0, maxTotal - total);
        resultBox.innerHTML = `
          <div class="alert alert-success py-2 mb-0">
            <strong>Great news!</strong> Current attendance is <strong>${curPct.toFixed(1)}%</strong>. You can safely miss up to <strong>${canMiss}</strong> upcoming classes while staying at or above ${target}%.
          </div>`;
      } else {
        // Need to attend how many consecutively?
        // (attended + x) / (total + x) >= target/100
        const factor = target / 100;
        const x = Math.ceil(((factor * total) - attended) / (1 - factor));
        resultBox.innerHTML = `
          <div class="alert alert-danger py-2 mb-0">
            <strong>Action Needed:</strong> Current attendance is <strong>${curPct.toFixed(1)}%</strong>. You need to attend the next <strong>${x}</strong> consecutive classes without absence to reach ${target}%.
          </div>`;
      }
    });
  }

  // 5. Real-Time "What-If" AI Scenario Simulator
  const simSliderHours = document.getElementById('simSliderHours');
  const simSliderAtt = document.getElementById('simSliderAtt');
  const simSliderInternal = document.getElementById('simSliderInternal');
  const simSliderAssign = document.getElementById('simSliderAssign');
  const simSliderPrev = document.getElementById('simSliderPrev');

  if (simSliderHours && simSliderAtt) {
    let debounceTimer = null;

    function runRealtimeSimulation() {
      const hours = parseFloat(simSliderHours.value) || 4.0;
      const att = parseFloat(simSliderAtt.value) || 80.0;
      const internal = parseFloat(simSliderInternal.value) || 75.0;
      const assign = parseFloat(simSliderAssign.value) || 80.0;
      const prev = parseFloat(simSliderPrev.value) || 72.0;

      // Update slider badge labels
      document.getElementById('sliderHoursVal').textContent = hours.toFixed(1) + ' hrs/day';
      document.getElementById('sliderAttVal').textContent = att.toFixed(1) + '%';
      document.getElementById('sliderInternalVal').textContent = internal.toFixed(1) + '%';
      document.getElementById('sliderAssignVal').textContent = assign.toFixed(1) + '%';
      document.getElementById('sliderPrevVal').textContent = prev.toFixed(1) + '%';

      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            attendance: att,
            study_hours: hours,
            internal_marks: internal,
            assignment_marks: assign,
            previous_semester_marks: prev
          })
        })
          .then(res => res.json())
          .then(data => {
            if (data.success) {
              const scoreElem = document.getElementById('simLiveScore');
              const badgeElem = document.getElementById('simLiveBadge');
              const barElem = document.getElementById('simLiveBar');
              const commentElem = document.getElementById('simLiveComment');

              if (scoreElem) scoreElem.innerHTML = `${data.predicted_marks} <span class="fs-5 text-muted fw-normal">/ 100</span>`;
              if (barElem) {
                barElem.style.width = `${data.predicted_marks}%`;
                barElem.className = `progress-bar bg-${data.badge_color}`;
              }
              if (badgeElem) {
                badgeElem.innerHTML = `<span class="badge badge-status badge-cat-${data.category.toLowerCase().replace(' ', '-')} fs-6 px-3 py-2">${data.category}</span>`;
              }
              if (commentElem) {
                commentElem.textContent = `At ${hours} hrs/day & ${att}% attendance, the AI model forecasts a ${data.category} final result.`;
              }
            }
          })
          .catch(err => console.error("Sim error:", err));
      }, 150);
    }

    [simSliderHours, simSliderAtt, simSliderInternal, simSliderAssign, simSliderPrev].forEach(slider => {
      slider.addEventListener('input', runRealtimeSimulation);
    });

    // Run once on load to initialize
    runRealtimeSimulation();
  }
});

