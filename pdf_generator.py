"""
pdf_generator.py
Generates a professional, branded PDF Academic Performance & AI Prediction Report using ReportLab.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_student_pdf_report(student, attendance_records, internals_records, latest_prediction, recommendations):
    """
    Builds a high quality PDF report in-memory and returns the byte stream.
    """
    # Normalize student and records to dicts if sqlite3.Row
    student = dict(student) if student is not None else {}
    latest_prediction = dict(latest_prediction) if latest_prediction is not None else None
    attendance_records = [dict(r) for r in attendance_records]
    internals_records = [dict(r) for r in internals_records]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom palette matching our web app theme (light blue / slate navy)
    primary_color = colors.HexColor("#0284c7")     # Light blue accent
    primary_dark = colors.HexColor("#0f172a")      # Dark slate
    slate_gray = colors.HexColor("#475569")        # Muted slate
    table_header_bg = colors.HexColor("#e0f2fe")   # Very light blue
    accent_green = colors.HexColor("#16a34a")
    accent_yellow = colors.HexColor("#ca8a04")
    accent_red = colors.HexColor("#dc2626")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=primary_dark,
        alignment=TA_CENTER
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=slate_gray,
        alignment=TA_CENTER
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceAfter=6,
        spaceBefore=10
    )

    cell_style = ParagraphStyle(
        'CellText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=primary_dark
    )

    cell_style_bold = ParagraphStyle(
        'CellTextBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=primary_dark
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("AI-BASED STUDENT PERFORMANCE PREDICTION SYSTEM", title_style))
    elements.append(Paragraph("Official Academic Progress & Predictive AI Assessment Report", subtitle_style))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=4, spaceAfter=12))

    # 2. Student Info Card Table
    gen_time = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    student_info_data = [
        [
            Paragraph(f"<b>Student Name:</b> {student['name']}", cell_style),
            Paragraph(f"<b>Student ID:</b> {student['student_id']}", cell_style)
        ],
        [
            Paragraph(f"<b>Email Address:</b> {student['email']}", cell_style),
            Paragraph(f"<b>Department:</b> {student.get('department', 'CSE')}", cell_style)
        ],
        [
            Paragraph(f"<b>Current Semester:</b> Semester {student.get('semester', 6)}", cell_style),
            Paragraph(f"<b>Report Generated:</b> {gen_time}", cell_style)
        ]
    ]

    info_table = Table(student_info_data, colWidths=[270, 270])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 12))

    # 3. Key Summary Metrics
    # Calculate summary metrics
    total_att = sum(r['attended_classes'] for r in attendance_records) if attendance_records else 0
    total_classes = sum(r['total_classes'] for r in attendance_records) if attendance_records else 0
    overall_att_pct = round((total_att / total_classes * 100), 1) if total_classes > 0 else 0.0

    avg_internal = 0.0
    if internals_records:
        avg_internal = round(sum(
            ((r['ia1'] + r['ia2'] + r['assignment_marks']) / 75.0 * 100) for r in internals_records
        ) / len(internals_records), 1)

    predicted_val = latest_prediction['predicted_marks'] if latest_prediction else "N/A"
    category_val = latest_prediction['category'] if latest_prediction else "Not Evaluated"

    cat_color = "#16a34a" if category_val == "Excellent" else ("#0284c7" if category_val == "Good" else ("#ca8a04" if category_val == "Average" else "#dc2626"))

    metrics_data = [
        [
            Paragraph("<b>Overall Attendance</b>", cell_style_bold),
            Paragraph("<b>Avg Internal Marks</b>", cell_style_bold),
            Paragraph("<b>AI Predicted Final Marks</b>", cell_style_bold),
            Paragraph("<b>Performance Category</b>", cell_style_bold)
        ],
        [
            Paragraph(f"<font size=12 color='#0f172a'><b>{overall_att_pct}%</b></font><br/>{total_att}/{total_classes} Classes", cell_style),
            Paragraph(f"<font size=12 color='#0f172a'><b>{avg_internal}%</b></font><br/>Scaled out of 100", cell_style),
            Paragraph(f"<font size=12 color='#0284c7'><b>{predicted_val} / 100</b></font><br/>Random Forest ML", cell_style),
            Paragraph(f"<font size=12 color='{cat_color}'><b>{category_val}</b></font><br/>Status Tier", cell_style)
        ]
    ]

    metrics_table = Table(metrics_data, colWidths=[135, 135, 135, 135])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), table_header_bg),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#f0f9ff")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#bae6fd")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bae6fd")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(metrics_table)
    elements.append(Spacer(1, 12))

    # 4. Subject-wise Attendance Tracking
    elements.append(Paragraph("1. Subject-Wise Attendance Record", section_heading))
    att_table_data = [
        [
            Paragraph("<b>Code</b>", cell_style_bold),
            Paragraph("<b>Subject Name</b>", cell_style_bold),
            Paragraph("<b>Attended / Total</b>", cell_style_bold),
            Paragraph("<b>Percentage</b>", cell_style_bold),
            Paragraph("<b>Status</b>", cell_style_bold)
        ]
    ]

    for record in attendance_records:
        t = record['total_classes']
        a = record['attended_classes']
        pct = round((a / t * 100), 1) if t > 0 else 0
        if pct >= 75:
            status_text = "<font color='#16a34a'><b>Normal (>= 75%)</b></font>"
        elif pct >= 65:
            status_text = "<font color='#ca8a04'><b>Warning (65-74%)</b></font>"
        else:
            status_text = "<font color='#dc2626'><b>Critical (< 65%)</b></font>"

        att_table_data.append([
            Paragraph(record['subject_code'], cell_style),
            Paragraph(record['subject_name'], cell_style),
            Paragraph(f"{a} / {t}", cell_style),
            Paragraph(f"{pct}%", cell_style_bold),
            Paragraph(status_text, cell_style)
        ])

    if not attendance_records:
        att_table_data.append([Paragraph("No attendance records logged.", cell_style), "", "", "", ""])

    att_table = Table(att_table_data, colWidths=[65, 220, 95, 80, 80])
    att_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(att_table)
    elements.append(Spacer(1, 10))

    # 5. Internal Assessment Marks
    elements.append(Paragraph("2. Internal Assessment & Coursework Marks", section_heading))
    marks_table_data = [
        [
            Paragraph("<b>Subject Name</b>", cell_style_bold),
            Paragraph("<b>IA-1 (25)</b>", cell_style_bold),
            Paragraph("<b>IA-2 (25)</b>", cell_style_bold),
            Paragraph("<b>Assign (25)</b>", cell_style_bold),
            Paragraph("<b>Total (75)</b>", cell_style_bold),
            Paragraph("<b>Scaled %</b>", cell_style_bold)
        ]
    ]

    for m in internals_records:
        total_internal = m['ia1'] + m['ia2'] + m['assignment_marks']
        scaled_pct = round((total_internal / 75.0) * 100, 1)
        marks_table_data.append([
            Paragraph(m['subject_name'], cell_style),
            Paragraph(f"{m['ia1']:.1f}", cell_style),
            Paragraph(f"{m['ia2']:.1f}", cell_style),
            Paragraph(f"{m['assignment_marks']:.1f}", cell_style),
            Paragraph(f"{total_internal:.1f}", cell_style_bold),
            Paragraph(f"{scaled_pct}%", cell_style)
        ])

    if not internals_records:
        marks_table_data.append([Paragraph("No internal marks recorded.", cell_style), "", "", "", "", ""])

    marks_table = Table(marks_table_data, colWidths=[230, 60, 60, 65, 65, 60])
    marks_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(marks_table)
    elements.append(Spacer(1, 10))

    # 6. Personalized AI Recommendations
    elements.append(Paragraph("3. Actionable Improvement Recommendations", section_heading))
    if recommendations:
        rec_data = []
        for idx, rec in enumerate(recommendations[:4], 1):
            rec_text = f"<b>{idx}. {rec['title']}</b> ({rec['category']})<br/>{rec['advice']}"
            rec_data.append([Paragraph(rec_text, cell_style)])
        rec_table = Table(rec_data, colWidths=[540])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(rec_table)
    else:
        elements.append(Paragraph("Ensure continuous study and maintain current high performance.", cell_style))

    elements.append(Spacer(1, 14))

    # 7. Signature Footer
    elements.append(KeepTogether([
        Table([
            [
                Paragraph("<b>Student Signature:</b> ___________________", cell_style),
                Paragraph("<b>Academic Advisor Signature:</b> ___________________", cell_style)
            ]
        ], colWidths=[270, 270], style=[
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]),
        Spacer(1, 6),
        Paragraph("<font size=7 color='#94a3b8'>Generated automatically by AI-Based Student Performance Prediction System. This document reflects certified institutional prediction analytics.</font>", subtitle_style)
    ]))

    doc.build(elements)
    buffer.seek(0)
    return buffer
