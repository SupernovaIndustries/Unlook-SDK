"""
Certification reporting for ISO/ASTM 52902 compliance.

This module generates comprehensive certification reports documenting
scanner compliance with ISO/ASTM 52902 standard requirements.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

# For PDF generation (optional dependency)
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available. PDF reports will not be generated.")

logger = logging.getLogger(__name__)


@dataclass
class CertificationReport:
    """Container for certification report data."""
    report_id: str
    generation_date: datetime
    scanner_info: Dict[str, Any]
    test_results: List[Dict[str, Any]]
    uncertainty_data: Dict[str, Any]
    calibration_status: Dict[str, Any]
    compliance_summary: Dict[str, bool]
    overall_compliance: bool
    recommendations: List[str]


class CertificationReporter:
    """
    Generates certification reports for ISO/ASTM 52902 compliance.
    
    Creates comprehensive documentation of scanner performance,
    uncertainty measurements, and compliance status.
    """
    
    # ISO/ASTM 52902 requirements
    STANDARD_REQUIREMENTS = {
        'length_measurement': {
            'max_uncertainty': 0.1,  # mm
            'description': 'Maximum permissible measurement uncertainty for length'
        },
        'angle_measurement': {
            'max_uncertainty': 0.5,  # degrees
            'description': 'Maximum permissible measurement uncertainty for angles'
        },
        'form_measurement': {
            'max_uncertainty': 0.15,  # mm
            'description': 'Maximum permissible measurement uncertainty for form features'
        },
        'repeatability': {
            'max_std_dev': 0.05,  # mm
            'description': 'Maximum standard deviation for repeated measurements'
        },
        'calibration_validity': {
            'max_days': 365,  # days
            'description': 'Maximum time between calibration validations'
        }
    }
    
    def __init__(self, 
                 scanner_info: Dict[str, Any],
                 output_directory: str = "./certification_reports"):
        """
        Initialize certification reporter.
        
        Args:
            scanner_info: Scanner specifications and identification
            output_directory: Directory for saving reports
        """
        self.scanner_info = scanner_info
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
    def generate_report(self,
                       calibration_results: List[Dict[str, Any]],
                       uncertainty_measurements: Dict[str, Any],
                       pattern_test_results: Dict[str, Any],
                       save_pdf: bool = True) -> CertificationReport:
        """
        Generate a comprehensive certification report.
        
        Args:
            calibration_results: Results from calibration validation tests
            uncertainty_measurements: Uncertainty data for each pattern type
            pattern_test_results: Test results for different pattern types
            save_pdf: Whether to generate PDF report
            
        Returns:
            CertificationReport object
        """
        logger.info("Generating ISO/ASTM 52902 certification report")
        
        # Generate report ID
        report_id = f"CERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze compliance for each requirement
        compliance_summary = self._analyze_compliance(
            calibration_results,
            uncertainty_measurements,
            pattern_test_results
        )
        
        # Determine overall compliance
        overall_compliance = all(compliance_summary.values())
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            compliance_summary,
            calibration_results,
            uncertainty_measurements
        )
        
        # Calculate calibration status
        calibration_status = self._assess_calibration_status(calibration_results)
        
        # Compile test results
        test_results = self._compile_test_results(
            calibration_results,
            pattern_test_results
        )
        
        # Create report object
        report = CertificationReport(
            report_id=report_id,
            generation_date=datetime.now(),
            scanner_info=self.scanner_info,
            test_results=test_results,
            uncertainty_data=uncertainty_measurements,
            calibration_status=calibration_status,
            compliance_summary=compliance_summary,
            overall_compliance=overall_compliance,
            recommendations=recommendations
        )
        
        # Save report data as JSON
        self._save_json_report(report)
        
        # Generate PDF if requested and available
        if save_pdf and REPORTLAB_AVAILABLE:
            self._generate_pdf_report(report)
        
        # Generate summary text report
        self._generate_text_report(report)
        
        return report
    
    def _analyze_compliance(self,
                          calibration_results: List[Dict[str, Any]],
                          uncertainty_measurements: Dict[str, Any],
                          pattern_test_results: Dict[str, Any]) -> Dict[str, bool]:
        """Analyze compliance with each requirement."""
        compliance = {}
        
        # Length measurement compliance
        max_length_uncertainty = self._get_max_uncertainty(
            uncertainty_measurements, 'length'
        )
        compliance['length_measurement'] = (
            max_length_uncertainty <= self.STANDARD_REQUIREMENTS['length_measurement']['max_uncertainty']
        )
        
        # Angle measurement compliance
        max_angle_uncertainty = self._get_max_uncertainty(
            uncertainty_measurements, 'angle'
        )
        compliance['angle_measurement'] = (
            max_angle_uncertainty <= self.STANDARD_REQUIREMENTS['angle_measurement']['max_uncertainty']
        )
        
        # Form measurement compliance
        max_form_uncertainty = self._get_max_uncertainty(
            uncertainty_measurements, 'form'
        )
        compliance['form_measurement'] = (
            max_form_uncertainty <= self.STANDARD_REQUIREMENTS['form_measurement']['max_uncertainty']
        )
        
        # Repeatability compliance
        repeatability_std = self._calculate_repeatability(pattern_test_results)
        compliance['repeatability'] = (
            repeatability_std <= self.STANDARD_REQUIREMENTS['repeatability']['max_std_dev']
        )
        
        # Calibration validity compliance
        days_since_calibration = self._days_since_last_calibration(calibration_results)
        compliance['calibration_validity'] = (
            days_since_calibration <= self.STANDARD_REQUIREMENTS['calibration_validity']['max_days']
        )
        
        return compliance
    
    def _get_max_uncertainty(self,
                           uncertainty_measurements: Dict[str, Any],
                           measurement_type: str) -> float:
        """Get maximum uncertainty for a measurement type."""
        max_uncertainty = 0.0
        
        for pattern_type, data in uncertainty_measurements.items():
            if measurement_type == 'length':
                # For length, use mean uncertainty
                max_uncertainty = max(max_uncertainty, data.get('mean_uncertainty', 0))
            elif measurement_type == 'angle':
                # For angles, check specific angle uncertainties if available
                angle_uncertainty = data.get('statistics', {}).get('angle_uncertainty', 0)
                max_uncertainty = max(max_uncertainty, angle_uncertainty)
            elif measurement_type == 'form':
                # For form features, use maximum uncertainty
                max_uncertainty = max(max_uncertainty, data.get('max_uncertainty', 0))
        
        return max_uncertainty
    
    def _calculate_repeatability(self,
                               pattern_test_results: Dict[str, Any]) -> float:
        """Calculate repeatability from test results."""
        repeatability_values = []
        
        for pattern_type, results in pattern_test_results.items():
            if 'repeatability_tests' in results:
                for test in results['repeatability_tests']:
                    repeatability_values.append(test.get('std_dev', 0))
        
        if repeatability_values:
            return np.mean(repeatability_values)
        
        return 0.0
    
    def _days_since_last_calibration(self,
                                   calibration_results: List[Dict[str, Any]]) -> int:
        """Calculate days since last successful calibration."""
        if not calibration_results:
            return 9999  # No calibration data
        
        # Find most recent successful calibration
        latest_date = None
        for result in calibration_results:
            if result.get('passed', False):
                test_date = result.get('test_date')
                if isinstance(test_date, str):
                    test_date = datetime.fromisoformat(test_date)
                
                if latest_date is None or test_date > latest_date:
                    latest_date = test_date
        
        if latest_date is None:
            return 9999  # No successful calibration
        
        return (datetime.now() - latest_date).days
    
    def _assess_calibration_status(self,
                                 calibration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess current calibration status."""
        if not calibration_results:
            return {
                'status': 'unknown',
                'last_calibration': None,
                'days_since_calibration': None,
                'drift_detected': False
            }
        
        # Get most recent result
        latest_result = calibration_results[-1]
        
        return {
            'status': 'valid' if latest_result.get('passed', False) else 'invalid',
            'last_calibration': latest_result.get('test_date'),
            'days_since_calibration': self._days_since_last_calibration(calibration_results),
            'drift_detected': latest_result.get('drift_detected', False),
            'test_object_used': latest_result.get('test_object'),
            'errors': latest_result.get('errors', {})
        }
    
    def _compile_test_results(self,
                            calibration_results: List[Dict[str, Any]],
                            pattern_test_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compile all test results into a structured format."""
        compiled_results = []
        
        # Add calibration test results
        for cal_result in calibration_results:
            compiled_results.append({
                'test_type': 'calibration_validation',
                'test_date': cal_result.get('test_date'),
                'test_object': cal_result.get('test_object'),
                'passed': cal_result.get('passed'),
                'measurements': cal_result.get('measurements'),
                'errors': cal_result.get('errors')
            })
        
        # Add pattern test results
        for pattern_type, results in pattern_test_results.items():
            compiled_results.append({
                'test_type': f'pattern_test_{pattern_type}',
                'test_date': results.get('test_date'),
                'pattern_type': pattern_type,
                'num_correspondences': results.get('num_correspondences'),
                'coverage_percent': results.get('coverage_percent'),
                'mean_uncertainty': results.get('mean_uncertainty')
            })
        
        return compiled_results
    
    def _generate_recommendations(self,
                                compliance_summary: Dict[str, bool],
                                calibration_results: List[Dict[str, Any]],
                                uncertainty_measurements: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on compliance analysis."""
        recommendations = []
        
        # Check each compliance area
        if not compliance_summary.get('length_measurement', True):
            recommendations.append(
                "Length measurement uncertainty exceeds limits. "
                "Consider improving pattern resolution or correspondence algorithms."
            )
        
        if not compliance_summary.get('angle_measurement', True):
            recommendations.append(
                "Angle measurement uncertainty exceeds limits. "
                "Verify projector-camera calibration and consider using hybrid ArUco patterns."
            )
        
        if not compliance_summary.get('form_measurement', True):
            recommendations.append(
                "Form measurement uncertainty exceeds limits. "
                "Increase point cloud density or improve surface reconstruction algorithms."
            )
        
        if not compliance_summary.get('repeatability', True):
            recommendations.append(
                "Repeatability does not meet requirements. "
                "Check for mechanical stability and environmental conditions."
            )
        
        if not compliance_summary.get('calibration_validity', True):
            recommendations.append(
                "Calibration validation is overdue. "
                "Perform calibration validation with certified test objects immediately."
            )
        
        # Check for pattern-specific issues
        for pattern_type, data in uncertainty_measurements.items():
            if data.get('mean_uncertainty', 0) > 0.08:  # Getting close to limit
                recommendations.append(
                    f"{pattern_type} pattern showing elevated uncertainty. "
                    f"Consider optimizing {pattern_type} parameters."
                )
        
        # Add positive feedback if fully compliant
        if all(compliance_summary.values()):
            recommendations.append(
                "Scanner fully compliant with ISO/ASTM 52902 requirements. "
                "Continue regular calibration validation schedule."
            )
        
        return recommendations
    
    def _save_json_report(self, report: CertificationReport):
        """Save report as JSON file."""
        json_path = self.output_directory / f"{report.report_id}.json"
        
        # Convert report to dictionary
        report_dict = {
            'report_id': report.report_id,
            'generation_date': report.generation_date.isoformat(),
            'scanner_info': report.scanner_info,
            'test_results': report.test_results,
            'uncertainty_data': report.uncertainty_data,
            'calibration_status': report.calibration_status,
            'compliance_summary': report.compliance_summary,
            'overall_compliance': report.overall_compliance,
            'recommendations': report.recommendations
        }
        
        with open(json_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Saved JSON report to {json_path}")
    
    def _generate_text_report(self, report: CertificationReport):
        """Generate human-readable text report."""
        text_path = self.output_directory / f"{report.report_id}.txt"
        
        with open(text_path, 'w') as f:
            f.write("ISO/ASTM 52902 CERTIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Report ID: {report.report_id}\n")
            f.write(f"Generated: {report.generation_date.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Scanner: {report.scanner_info.get('name', 'Unknown')}\n")
            f.write(f"Model: {report.scanner_info.get('model', 'Unknown')}\n\n")
            
            f.write("COMPLIANCE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Compliance: {'PASS' if report.overall_compliance else 'FAIL'}\n\n")
            
            for requirement, compliant in report.compliance_summary.items():
                status = "PASS" if compliant else "FAIL"
                f.write(f"{requirement}: {status}\n")
            
            f.write("\nCALIBRATION STATUS\n")
            f.write("-" * 20 + "\n")
            cal_status = report.calibration_status
            f.write(f"Status: {cal_status.get('status', 'Unknown')}\n")
            f.write(f"Days Since Calibration: {cal_status.get('days_since_calibration', 'N/A')}\n")
            f.write(f"Drift Detected: {cal_status.get('drift_detected', False)}\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
            f.write("\nTEST SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Tests Performed: {len(report.test_results)}\n")
            passed_tests = sum(1 for t in report.test_results if t.get('passed', False))
            f.write(f"Tests Passed: {passed_tests}/{len(report.test_results)}\n")
            
            # Pattern-specific uncertainty summary
            f.write("\nUNCERTAINTY SUMMARY\n")
            f.write("-" * 20 + "\n")
            for pattern_type, data in report.uncertainty_data.items():
                f.write(f"\n{pattern_type.upper()} Pattern:\n")
                f.write(f"  Mean Uncertainty: {data.get('mean_uncertainty', 0):.3f} mm\n")
                f.write(f"  Max Uncertainty: {data.get('max_uncertainty', 0):.3f} mm\n")
                stats = data.get('statistics', {})
                f.write(f"  Coverage: {stats.get('coverage_percent', 0):.1f}%\n")
                f.write(f"  Correspondences: {stats.get('num_correspondences', 0)}\n")
        
        logger.info(f"Saved text report to {text_path}")
    
    def _generate_pdf_report(self, report: CertificationReport):
        """Generate PDF report using ReportLab."""
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. Skipping PDF generation.")
            return
        
        pdf_path = self.output_directory / f"{report.report_id}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        # Title
        story.append(Paragraph("ISO/ASTM 52902 Certification Report", title_style))
        story.append(Spacer(1, 20))
        
        # Scanner Information
        scanner_data = [
            ['Scanner Information', ''],
            ['Report ID:', report.report_id],
            ['Generated:', report.generation_date.strftime('%Y-%m-%d %H:%M:%S')],
            ['Scanner Name:', report.scanner_info.get('name', 'Unknown')],
            ['Model:', report.scanner_info.get('model', 'Unknown')],
            ['Serial Number:', report.scanner_info.get('serial', 'Unknown')]
        ]
        
        scanner_table = Table(scanner_data, colWidths=[150, 350])
        scanner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(scanner_table)
        story.append(Spacer(1, 20))
        
        # Compliance Summary
        compliance_data = [['Requirement', 'Status', 'Details']]
        
        for req, compliant in report.compliance_summary.items():
            status = 'PASS' if compliant else 'FAIL'
            color = colors.green if compliant else colors.red
            details = self.STANDARD_REQUIREMENTS[req]['description']
            compliance_data.append([req.replace('_', ' ').title(), status, details])
        
        compliance_table = Table(compliance_data, colWidths=[150, 50, 300])
        compliance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        # Color code status column
        for i, row in enumerate(compliance_data[1:], 1):
            if row[1] == 'PASS':
                compliance_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (1, i), (1, i), colors.green)
                ]))
            else:
                compliance_table.setStyle(TableStyle([
                    ('TEXTCOLOR', (1, i), (1, i), colors.red)
                ]))
        
        story.append(Paragraph("Compliance Summary", styles['Heading2']))
        story.append(compliance_table)
        story.append(Spacer(1, 30))
        
        # Overall Result
        overall_style = ParagraphStyle(
            'OverallResult',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.green if report.overall_compliance else colors.red,
            alignment=TA_CENTER
        )
        
        overall_text = f"Overall Compliance: {'PASS' if report.overall_compliance else 'FAIL'}"
        story.append(Paragraph(overall_text, overall_style))
        story.append(PageBreak())
        
        # Detailed Results
        story.append(Paragraph("Detailed Test Results", styles['Heading2']))
        story.append(Spacer(1, 20))
        
        # Add test results summary
        for i, test_result in enumerate(report.test_results[-5:]):  # Last 5 results
            test_type = test_result.get('test_type', 'Unknown')
            passed = test_result.get('passed', False)
            test_date = test_result.get('test_date', 'Unknown')
            
            test_text = f"{i+1}. {test_type} - {test_date} - {'PASS' if passed else 'FAIL'}"
            story.append(Paragraph(test_text, styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Recommendations", styles['Heading2']))
        story.append(Spacer(1, 10))
        
        for rec in report.recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
            story.append(Spacer(1, 5))
        
        # Build PDF
        doc.build(story)
        logger.info(f"Saved PDF report to {pdf_path}")
    
    def generate_summary_report(self,
                              reports: List[CertificationReport],
                              filename: str = "compliance_summary.json"):
        """Generate summary report from multiple certification reports."""
        summary = {
            'total_reports': len(reports),
            'date_range': {
                'start': min(r.generation_date for r in reports).isoformat(),
                'end': max(r.generation_date for r in reports).isoformat()
            },
            'compliance_trend': [],
            'common_issues': {},
            'improvement_areas': []
        }
        
        # Analyze compliance trend
        for report in sorted(reports, key=lambda r: r.generation_date):
            summary['compliance_trend'].append({
                'date': report.generation_date.isoformat(),
                'compliant': report.overall_compliance,
                'compliance_areas': report.compliance_summary
            })
        
        # Identify common issues
        issue_counts = {}
        for report in reports:
            for req, compliant in report.compliance_summary.items():
                if not compliant:
                    issue_counts[req] = issue_counts.get(req, 0) + 1
        
        summary['common_issues'] = issue_counts
        
        # Identify improvement areas
        for issue, count in issue_counts.items():
            if count > len(reports) / 2:  # Issue in more than half the reports
                summary['improvement_areas'].append({
                    'area': issue,
                    'frequency': count / len(reports),
                    'description': self.STANDARD_REQUIREMENTS[issue]['description']
                })
        
        # Save summary
        summary_path = self.output_directory / filename
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved compliance summary to {summary_path}")
        
        return summary