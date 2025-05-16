# Claude Session Notes - Investor Demo Priority (2025-05-17)

## CRITICAL CONTEXT
- **Investor demo approaching** - Need reliable box scan + depth map
- **Test object**: E-leather box (non-light-absorbing, better than bunny)
- **Current status**: Enhanced processor failing (black masks), standard processor previously worked (57k correspondences)

## PROJECT CONTEXT (from README.md)
UnLook SDK controls modular 3D scanning hardware including:
- Structured light modules with DLPC342X projectors
- Stereo cameras for triangulation
- Auto-discovery of scanners on network
- Real-time and static scanning modes
- Simple Codes for easy integration in workflows
- Open-source, extensible, and modular design
- Supports various scanning modules like structured light, time-of-flight, and laser triangulation
## INVESTOR DEMO PRIORITIES (Most Doable First)

### 1. DISABLE ENHANCED PROCESSOR (Immediate Fix)
```python
# In static_scanner.py, force use of standard processing:
use_enhanced = False  # Hard-code this temporarily
```
- Standard processor found 57k correspondences before
- Enhanced processor is causing complete failure
- This is the quickest fix for demo

### 2. OPTIMIZE FOR E-LEATHER BOX
The e-leather box is ideal because:
- Doesn't absorb light (unlike bunny)
- Clean, flat surfaces
- Good for pattern visibility

Suggested scan settings:
```python
scanner_config = {
    "pattern": "multi_scale",  # or "gray_code"
    "num_bits": 4,  # Start simple (16 levels)
    "exposure": -2,  # Lower exposure
    "gain": 1,  # Minimal gain
    "pattern_hold_time": 0.5  # Give time for capture
}
```

### 3. VERIFY AUTO-OPTIMIZER
Check if it's actually running:
```python
# Add debug logging in camera_auto_optimizer.py:
logger.info(f"Auto-optimizer chose: exposure={exposure}, gain={gain}")
```

### 4. EMERGENCY FALLBACK DEMO MODE
Create a simple demo script that:
1. Uses known-good settings
2. Captures with standard processing only
3. Saves intermediate results
4. Shows partial results even if incomplete

```python
# demo_scan.py
def investor_demo_scan():
    # Hard-coded settings that work
    scanner = StaticScanner()
    scanner.use_enhanced = False
    scanner.exposure = -2
    scanner.pattern = "multi_scale"
    
    # Capture and show whatever we get
    try:
        result = scanner.scan()
        show_depth_map(result)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        show_partial_results()
```

### 5. PATTERN VISIBILITY QUICK FIXES

#### A. Increase Projector Brightness
```python
# In projector control
projector.set_brightness(255)  # Maximum
projector.set_contrast(255)   # Maximum
```

#### B. Adjust Timing
```python
# Ensure capture happens during projection
time.sleep(0.1)  # After pattern display
camera.capture()
time.sleep(0.1)  # Before next pattern
```

#### C. Use Simpler Patterns
- Start with 3-bit Gray code (8 levels)
- Use coarser patterns for initial test
- Once working, increase complexity

### 6. DEBUG DEPTH MAP GENERATION
Fix the depth_map_diagnostic.py errors:
1. Handle correspondence format variations
2. Fix divide-by-zero warnings
3. Add debug output for each step

### 7. DEMO PREPARATION CHECKLIST

**Before Demo:**
- [ ] Test in actual demo room lighting
- [ ] Use e-leather box (not bunny)
- [ ] Disable enhanced processor
- [ ] Set projector to maximum brightness
- [ ] Use simple patterns (3-4 bits)
- [ ] Have fallback screenshots ready

**During Demo:**
- Show live pattern projection first
- Explain structured light concept
- Show captured patterns
- Display depth map (even if partial)
- Emphasize modular/opensource nature

**Backup Plan:**
- Have previous successful scans ready
- Show depth map from earlier 57k correspondence run
- Explain temporary lighting challenges
- Focus on SDK architecture and potential

## CODE CHANGES PRIORITY ORDER

1. **FIX depth_map_diagnostic.py IndexError (CRITICAL)**
   ```python
   # Fix correspondence visualization error:
   # Current error: IndexError at line 435
   # Issue: correspondences format inconsistent
   
   # Replace correspondence visualization loop with:
   if correspondences and len(correspondences) > 0:
       # Detect format dynamically
       first_corr = correspondences[0]
       if isinstance(first_corr, dict):
           # Dictionary format
           for corr in correspondences[:1000]:
               x_l = corr['left_x']
               y = corr['y']
               cv2.circle(left_viz, (x_l, y), 3, (0, 255, 0), -1)
       elif isinstance(first_corr, (tuple, list)):
           # Tuple/list format
           for corr in correspondences[:1000]:
               x_l = corr[0]
               y = corr[2]
               cv2.circle(left_viz, (x_l, y), 3, (0, 255, 0), -1)
       elif isinstance(first_corr, np.ndarray):
           # Numpy array format
           for corr in correspondences[:1000]:
               x_l = int(corr[0])
               y = int(corr[2])
               cv2.circle(left_viz, (x_l, y), 3, (0, 255, 0), -1)
   ```

2. **static_scanning_example.py**
   ```python
   # Force standard processing
   if args.use_enhanced:
       args.use_enhanced = False  # Override for now
   ```

3. **camera_auto_optimizer.py**
   ```python
   # Add verbose logging
   def optimize_settings(self):
       logger.info("Starting auto-optimization...")
       # Log each decision
   ```

4. **enhanced_pattern_processor.py**
   ```python
   # Add try/except with fallback
   def process_stereo_patterns(self):
       try:
           # Current code
       except:
           logger.warning("Enhanced processing failed, returning None")
           return None
   ```

## EXPECTED OUTCOMES
With these changes, you should achieve:
- Basic depth map of e-leather box
- Reliable pattern capture
- Consistent results for demo
- Clear understanding of what's failing

## POST-DEMO IMPROVEMENTS
After successful demo, focus on:
1. Fixing enhanced processor
2. Implementing hardware upgrades
3. Improving ambient light handling
4. Adding real-time scanning
5. Enhancing pattern visibility
6. Testing with various objects
7. Improving documentation and examples
8. Adding neural network support for pattern recognition
9. Addign more advanced features like texture mapping
10. Implementing user-friendly GUI for scanning
11. Improving error handling and logging
12. Adding more detailed documentation for developers
13. Creating a community forum for user feedback
14. Implementing a bug tracking system
15. Creating a roadmap for future features
16. Adding more test cases for edge scenarios
17. Implementing a CI/CD pipeline for automated testing
18. Creating a user manual for end-users
19. Adding a troubleshooting guide
20. Creating a FAQ section for common issues
21. Implementing a feedback loop for user suggestions
22. Creating a marketing strategy for the SDK
23. Implementing a user onboarding process
24. Creating a demo video for the SDK
25. Creating a press kit for media outreach
26. Implementing a social media strategy
27. Reddit community engagement
28. Kickstarter campaign for funding, goal 150k
29. 

Remember: For investor demo, **reliability > features**. Better to show basic working scan than advanced features that fail.
## ADDITIONAL NOTES
After demo, consider:
- Gathering feedback from investors
- Identifying areas for improvement
- Hard testing in various environments 
- Hard testing with different objects based on ISO/ASTM 52902:2019(E) standards