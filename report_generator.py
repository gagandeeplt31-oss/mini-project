"""
report_generator.py - Professional Farm Advisory PDF Report Generator.

Generates a formatted, printable PDF document containing:
- Farm soil and weather input conditions
- Recommended crop with live/historical market valuation
- Agronomic NPK and pH baseline gap analysis
- Exact, actionable fertilizer and soil amendment treatment plan
"""

import io
import re
import time
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)


def sanitize_text(text):
    """
    Sanitizes raw text for ReportLab Flowables:
    - Replaces Rupee symbol with 'Rs. ' to prevent latin-1 encoding errors.
    - Converts markdown bold (**text**) to ReportLab XML tags (<b>text</b>).
    - Converts markdown italics (*text*) to <i>text</i>.
    - Strips unsupported unicode emojis.
    - Escapes bare ampersands and XML special characters.
    """
    if not text:
        return ""
    t = str(text)
    
    # Currency symbols
    t = t.replace('₹', 'Rs. ')
    
    # Strip emojis
    t = re.sub(r'[\U00010000-\U0010ffff]', '', t)
    t = re.sub(r'[\u2600-\u27bf]', '', t)
    
    # XML entity escaping (escape ampersand first if not part of an existing entity)
    t = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', t)
    
    # Convert markdown to ReportLab HTML tags
    t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
    t = re.sub(r'\*(.*?)\*', r'<i>\1</i>', t)
    
    return t.strip()


def generate_farm_report_pdf(
    user_inputs,
    feature_names,
    recommended_crop,
    market_price,
    price_badge,
    treatment_plan,
    top_candidates=None,
    farmer_name="Valued Farmer",
    location="Field Location",
    farm_economics=None,
    **kwargs
):
    """
    Builds a professional multi-section Farm Advisory PDF report in memory.
    
    Returns:
        bytes: Raw PDF file bytes ready for download or disk write.
    """
    if farm_economics is None:
        farm_economics = kwargs.get('farm_economics', kwargs.get('econ_data', kwargs.get('economics', None)))
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Brand Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1b5e20'),
        alignment=0,
        spaceAfter=2
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#555555'),
        alignment=0,
        spaceAfter=6
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#777777'),
        alignment=2
    )
    
    section_h2 = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2e7d32'),
        spaceBefore=10,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#222222')
    )
    
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#222222'),
        spaceAfter=4
    )
    
    rec_title_style = ParagraphStyle(
        'RecTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#2e7d32'),
        alignment=1
    )
    
    rec_price_style = ParagraphStyle(
        'RecPrice',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1b5e20'),
        alignment=1
    )
    
    elements = []
    
    # -------------------------------------------------------------
    # HEADER SECTION
    # -------------------------------------------------------------
    gen_time_str = time.strftime('%d-%b-%Y %H:%M')
    header_data = [
        [
            Paragraph("AgriSmart: Farm Decision &amp; Treatment Report", title_style),
            Paragraph(f"<b>Generated:</b> {gen_time_str}<br/><b>Advisory ID:</b> AGY-{int(time.time())%1000000}", meta_style)
        ],
        [
            Paragraph("AI-Powered Crop Recommendation, Market Valuation &amp; Soil Amendment Plan", subtitle_style),
            Paragraph("Status: Official Advisory", meta_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[380, 160])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 4))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#2e7d32'), spaceAfter=8))
    
    # -------------------------------------------------------------
    # 1. FIELD CONDITIONS & SOIL PROFILE
    # -------------------------------------------------------------
    elements.append(Paragraph("1. Field Soil &amp; Environmental Profile", section_h2))
    
    input_dict = dict(zip(feature_names, user_inputs)) if isinstance(user_inputs, (list, tuple)) else user_inputs
    
    field_rows = [
        [
            Paragraph("<b>Nitrogen (N)</b>", body_style),
            Paragraph(f"{input_dict.get('N', 0):.1f} kg/ha", body_style),
            Paragraph("<b>Temperature</b>", body_style),
            Paragraph(f"{input_dict.get('temperature', 0):.1f} °C", body_style),
        ],
        [
            Paragraph("<b>Phosphorus (P)</b>", body_style),
            Paragraph(f"{input_dict.get('P', 0):.1f} kg/ha", body_style),
            Paragraph("<b>Humidity</b>", body_style),
            Paragraph(f"{input_dict.get('humidity', 0):.1f} %", body_style),
        ],
        [
            Paragraph("<b>Potassium (K)</b>", body_style),
            Paragraph(f"{input_dict.get('K', 0):.1f} kg/ha", body_style),
            Paragraph("<b>Soil pH</b>", body_style),
            Paragraph(f"{input_dict.get('ph', 0):.2f}", body_style),
        ],
        [
            Paragraph("<b>Rainfall</b>", body_style),
            Paragraph(f"{input_dict.get('rainfall', 0):.1f} mm", body_style),
            Paragraph("<b>Soil Category</b>", body_style),
            Paragraph("Loam / Agricultural Baseline", body_style),
        ]
    ]
    
    field_table = Table(field_rows, colWidths=[120, 150, 120, 150])
    field_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f8e9')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f1f8e9')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dcedc8')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(field_table)
    elements.append(Spacer(1, 8))
    
    # -------------------------------------------------------------
    # 2. RECOMMENDED CROP & ECONOMIC VALUATION
    # -------------------------------------------------------------
    elements.append(Paragraph("2. Optimized Crop Recommendation &amp; Market Valuation", section_h2))
    
    clean_badge = sanitize_text(price_badge).replace('[', '').replace(']', '').strip()
    price_val = float(market_price) if str(market_price).replace('.', '', 1).isdigit() else market_price
    
    rec_card_data = [
        [Paragraph(f"RECOMMENDED CROP: {recommended_crop.upper()}", rec_title_style)],
        [Paragraph(f"Market Price: Rs. {price_val}/kg  |  Rate Source: [{clean_badge}]", rec_price_style)],
        [Paragraph("Biologically viable and economically optimized for highest farmer return on investment.", subtitle_style)]
    ]
    
    rec_table = Table(rec_card_data, colWidths=[540])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#e8f5e9')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#2e7d32')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(rec_table)
    elements.append(Spacer(1, 8))
    
    # -------------------------------------------------------------
    # 3. SOIL NUTRIENT GAP ANALYSIS (TABLE)
    # -------------------------------------------------------------
    elements.append(Paragraph("3. Nutrient Gap Analysis vs. Ideal Agronomic Baseline", section_h2))
    
    nutrients = treatment_plan.get('nutrients', {})
    ph_info = treatment_plan.get('ph', {})
    
    th_style = ParagraphStyle('TH', parent=body_style, fontName='Helvetica-Bold', textColor=colors.white)
    
    gap_table_data = [
        [
            Paragraph("Parameter", th_style),
            Paragraph("Measured", th_style),
            Paragraph("Ideal Baseline", th_style),
            Paragraph("Gap (Delta)", th_style),
            Paragraph("Agronomic Status", th_style),
        ]
    ]
    
    for key, label in [('N', 'Nitrogen (N)'), ('P', 'Phosphorus (P)'), ('K', 'Potassium (K)')]:
        info = nutrients.get(key, {})
        u_val = info.get('user', 0.0)
        i_val = info.get('ideal', 0.0)
        d_val = info.get('delta', 0.0)
        st_val = info.get('status', 'Optimal')
        
        status_color = "#c62828" if st_val == "Deficit" else ("#2e7d32" if st_val == "Optimal" else "#1565c0")
        st_para = Paragraph(f"<b><font color='{status_color}'>{st_val}</font></b>", body_style)
        
        gap_table_data.append([
            Paragraph(label, body_style),
            Paragraph(f"{u_val:.1f} kg/ha", body_style),
            Paragraph(f"{i_val:.1f} kg/ha", body_style),
            Paragraph(f"{d_val:+.1f} kg/ha", body_style),
            st_para
        ])
        
    # pH row
    ph_u = ph_info.get('user', 6.5)
    ph_i = ph_info.get('ideal', 6.5)
    ph_d = ph_info.get('delta', 0.0)
    ph_st = ph_info.get('status', 'Optimal')
    ph_color = "#2e7d32" if ph_st == "Optimal" else "#e65100"
    
    gap_table_data.append([
        Paragraph("Soil pH", body_style),
        Paragraph(f"{ph_u:.2f}", body_style),
        Paragraph(f"{ph_i:.2f}", body_style),
        Paragraph(f"{ph_d:+.2f}", body_style),
        Paragraph(f"<b><font color='{ph_color}'>{ph_st}</font></b>", body_style)
    ])
    
    gap_table = Table(gap_table_data, colWidths=[130, 95, 105, 100, 110])
    gap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fbe7')]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(gap_table)
    elements.append(Spacer(1, 8))
    
    # -------------------------------------------------------------
    # 4. ACTIONABLE AMENDMENT ADVICE (BULLETS)
    # -------------------------------------------------------------
    elements.append(Paragraph("4. Soil Treatment &amp; Fertilizer Application Plan", section_h2))
    
    for advice in treatment_plan.get('actionable_advice', []):
        clean_advice = sanitize_text(advice)
        elements.append(Paragraph(f"&bull;&nbsp;&nbsp;{clean_advice}", bullet_style))
        
    elements.append(Spacer(1, 8))
    
    # -------------------------------------------------------------
    # 5. FARM ECONOMICS & ROI PROJECTIONS
    # -------------------------------------------------------------
    if farm_economics and isinstance(farm_economics, dict):
        elements.append(Paragraph("5. Farm Economics &amp; ROI Projection", section_h2))
        
        econ_table_data = [
            [
                Paragraph("<b>Metric</b>", th_style),
                Paragraph("<b>Value</b>", th_style),
                Paragraph("<b>Basis / Note</b>", th_style)
            ],
            [
                Paragraph("Cultivated Farm Size", body_style),
                Paragraph(f"{farm_economics.get('farm_size_ha', 1.0)} hectare(s)", body_style),
                Paragraph("User-specified operational acreage", body_style)
            ],
            [
                Paragraph("Projected Harvest Yield", body_style),
                Paragraph(f"{farm_economics.get('total_yield_kg', 0.0):,.0f} kg", body_style),
                Paragraph(f"Avg baseline: {farm_economics.get('yield_per_ha_kg', 0.0):,.0f} kg/ha", body_style)
            ],
            [
                Paragraph("Projected Gross Revenue", body_style),
                Paragraph(f"<b>Rs. {farm_economics.get('gross_revenue', 0.0):,.2f}</b>", body_style),
                Paragraph(f"Spot price: Rs. {farm_economics.get('market_price_per_kg', 0.0):.2f}/kg", body_style)
            ],
            [
                Paragraph("Total Fertilizer Input Cost", body_style),
                Paragraph(f"Rs. {farm_economics.get('fertilizer_cost', 0.0):,.2f}", body_style),
                Paragraph("Estimated Urea, DAP, MOP amendment cost", body_style)
            ],
            [
                Paragraph("Estimated Net Margin", body_style),
                Paragraph(f"<b><font color='#2e7d32'>Rs. {farm_economics.get('net_profit', 0.0):,.2f}</font></b>", body_style),
                Paragraph(f"Projected margin: {farm_economics.get('profit_margin_pct', 0.0):.1f}%", body_style)
            ]
        ]
        
        econ_table = Table(econ_table_data, colWidths=[160, 150, 230])
        econ_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2e7d32')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fbe7')]),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(econ_table)
        elements.append(Spacer(1, 6))

        # Itemized Fertilizer Breakdown
        breakdown = farm_economics.get('fertilizer_breakdown', {})
        if breakdown and isinstance(breakdown, dict):
            elements.append(Paragraph("<b>Itemized Fertilizer Amendment Expenditure:</b>", body_style))
            elements.append(Spacer(1, 2))
            
            f_table_data = [
                [
                    Paragraph("<b>Fertilizer</b>", th_style),
                    Paragraph("<b>Target Nutrient</b>", th_style),
                    Paragraph("<b>Deficit</b>", th_style),
                    Paragraph("<b>Quantity (kg)</b>", th_style),
                    Paragraph("<b>Rate (Rs./kg)</b>", th_style),
                    Paragraph("<b>Cost (Rs.)</b>", th_style)
                ]
            ]
            for f_name, f_info in breakdown.items():
                f_table_data.append([
                    Paragraph(str(f_name), body_style),
                    Paragraph(str(f_info.get('nutrient', '')), body_style),
                    Paragraph(f"{float(f_info.get('deficit_per_ha', 0.0)):.1f} kg/ha", body_style),
                    Paragraph(f"{float(f_info.get('total_qty_kg', 0.0)):.1f} kg", body_style),
                    Paragraph(f"Rs. {float(f_info.get('rate_per_kg', 0.0)):.2f}", body_style),
                    Paragraph(f"<b>Rs. {float(f_info.get('cost', 0.0)):,.2f}</b>", body_style)
                ])
            
            f_table = Table(f_table_data, colWidths=[80, 110, 85, 85, 80, 100])
            f_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#388e3c')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fbe7')]),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(f_table)
            elements.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # FOOTER NOTICE
    # -------------------------------------------------------------
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc'), spaceBefore=6, spaceAfter=6))
    footer_text = (
        "<b>Notice:</b> This recommendation is generated by the AgriSmart Decision Support System based on soil science, "
        "historical agronomic data, and commodity market pricing. Conduct periodic soil tests for precision agriculture."
    )
    elements.append(Paragraph(footer_text, ParagraphStyle('Footer', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=7.5, leading=10, textColor=colors.HexColor('#777777'))))
    
    # Build Document
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


if __name__ == "__main__":
    print("--- Testing report_generator.py ---")
    dummy_inputs = [90, 42, 43, 20.8, 82.0, 6.5, 202.9]
    dummy_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    from fertilizer_calc import calculate_soil_amendments
    dummy_plan = calculate_soil_amendments(dummy_inputs, 'rice')
    
    pdf = generate_farm_report_pdf(
        user_inputs=dummy_inputs,
        feature_names=dummy_names,
        recommended_crop='rice',
        market_price=48.69,
        price_badge='🟢 Live Mandi Rate',
        treatment_plan=dummy_plan
    )
    print("PDF generated successfully! File size:", len(pdf), "bytes")
    with open("test_output_report.pdf", "wb") as f:
        f.write(pdf)
    print("Saved test_output_report.pdf")
