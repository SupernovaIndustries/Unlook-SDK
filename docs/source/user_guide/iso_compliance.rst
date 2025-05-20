ISO/ASTM 52902 Compliance
=========================

The UnLook SDK provides support for ISO/ASTM 52902 compliance, which is the standard for "Additive manufacturing — Test artifacts — Geometric capability assessment of additive manufacturing systems".

Overview
--------

The ISO/ASTM 52902 standard defines test methods to assess the performance of additive manufacturing systems. For 3D scanning systems like UnLook, this means providing quantitative measurements of:

* Measurement uncertainty
* Calibration validation
* Geometric accuracy
* Certification reporting

These features enable the UnLook scanner to be used in industrial and manufacturing environments where traceability and certification are required.

Uncertainty Measurement
---------------------

The SDK provides built-in functionality to calculate measurement uncertainty for different pattern types:

.. code-block:: python

   from unlook.client.scanning.compliance import (
       MazeUncertaintyMeasurement,
       VoronoiUncertaintyMeasurement,
       HybridArUcoUncertaintyMeasurement
   )

   # Create uncertainty measurement for a specific pattern type
   uncertainty_measure = MazeUncertaintyMeasurement((1280, 720))

   # Compute uncertainty
   uncertainty_data = uncertainty_measure.compute_uncertainty(
       correspondences,  # From pattern matching
       {'pixel_to_mm': 0.1}
   )

   # Get results
   print(f"Mean uncertainty: {uncertainty_data.mean_uncertainty:.3f}mm")
   print(f"Max uncertainty: {uncertainty_data.max_uncertainty:.3f}mm")
   print(f"Uncertainty distribution: {uncertainty_data.histogram}")

Calibration Validation
--------------------

Validate calibration using standardized test objects:

.. code-block:: python

   from unlook.client.scanning.compliance import CalibrationValidator

   # Create validator with scanner specifications
   validator = CalibrationValidator({
       'name': 'UnLook Scanner',
       'model': 'UL-1000',
       'accuracy': 0.1,  # mm
       'resolution': (1280, 720)
   })

   # Validate with test object
   validation_result = validator.validate_with_test_object(
       point_cloud,  # Scanned test object
       'sphere_25mm'  # Standard test object ID
   )

   # Check results
   if validation_result.passed:
       print("Calibration validation passed!")
       print(f"Mean error: {validation_result.mean_error:.3f}mm")
       print(f"Max error: {validation_result.max_error:.3f}mm")
   else:
       print("Calibration validation failed")
       print(f"Failures: {validation_result.failures}")

Certification Reporting
---------------------

Generate certification reports that demonstrate compliance with the standard:

.. code-block:: python

   from unlook.client.scanning.compliance import CertificationReporter

   # Create reporter
   reporter = CertificationReporter({
       'name': 'UnLook Scanner',
       'model': 'UL-1000',
       'accuracy': 0.1,  # mm
       'resolution': (1280, 720)
   })

   # Generate report
   certification = reporter.generate_report(
       calibration_results,
       uncertainty_measurements,
       pattern_test_results,
       save_pdf=True
   )

   # Print summary
   print(f"Compliance status: {'COMPLIANT' if certification.overall_compliance else 'NON-COMPLIANT'}")
   print(f"Report saved to: {certification.report_path}")

Integration with Scanning
----------------------

You can enable ISO compliance features in the scanning process:

.. code-block:: python

   from unlook import UnlookClient
   from unlook.client.scanning import StaticScanner, StaticScanConfig
   from unlook.client.scan_config import PatternType

   # Create config with ISO compliance enabled
   config = StaticScanConfig(
       quality="high",
       pattern_type=PatternType.MAZE,
       enable_uncertainty_calculation=True,
       iso_compliance_report=True
   )

   # Create scanner
   scanner = StaticScanner(client=client, config=config)

   # Perform scan with compliance features
   result = scanner.perform_scan()

   # Get uncertainty data
   uncertainty = result.get('uncertainty_data')
   print(f"Mean uncertainty: {uncertainty.mean_uncertainty:.3f}mm")

   # Save compliance report
   scanner.save_compliance_report("scan_certification.pdf")

Related Documentation
------------------

For more information about the ISO/ASTM 52902 standard and its implementation in the UnLook SDK, refer to:

* `ISO/ASTM 52902 Standard <https://www.iso.org/standard/74509.html>`_
* :doc:`../api_reference/compliance`