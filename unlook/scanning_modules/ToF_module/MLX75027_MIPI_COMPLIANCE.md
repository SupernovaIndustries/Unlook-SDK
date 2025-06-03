# MLX75027 MIPI 2-Lane Configuration - Compliance Check

## ✅ Datasheet Requirements vs Implementation

### 1. **I2C Address Configuration**
**Requirement**: "Impostare il pin SLASEL a GND per usare l'indirizzo I2C 0x57"
**Implementation**: ✅ Hardware configuration (external to software)

### 2. **Power Sequence**
**Requirement**: "RESETB alto solo dopo che tutte le tensioni e l'8 MHz sono stabili"
**Implementation**: ✅ Hardware/driver responsibility

### 3. **Software Standby Entry**
**Requirement**: "Entrare in software standby scrivendo 0x01 in 0x1000"
**Implementation**: ✅ 
```python
def _enter_standby(self):
    SOFTWARE_STANDBY_REG = 0x1000
    self._set_tof_register(SOFTWARE_STANDBY_REG, 0x01, 0x01)
```

### 4. **MIPI Lane Configuration**
**Requirement**: "DATA_LANE_CONFIG = 0 (registro 0x1010) → MIPI 2 lanes"
**Implementation**: ✅ 
```python
DATA_LANE_CONFIG_REG = 0x1010
lane_value = 0x00 if self.config.mipi_lanes == 2 else 0x01
self._set_tof_register(DATA_LANE_CONFIG_REG, lane_value, 0x01)
```

### 5. **Speed Configuration**
**Requirement**: "Impostare velocità nei registri: 0x100C, 0x100D, ..., 0x1071"
**Implementation**: ⚠️ Partial - Example values only
```python
# Simplified implementation - specific values needed from datasheet
self._set_tof_register(0x100C, 0x04, 0x01)  # Example
self._set_tof_register(0x100D, 0x00, 0x01)  # Example
```

### 6. **Sensor Configuration**
**Requirement**: Configure integration time, modulation frequency, output mode
**Implementation**: ✅ Already implemented in main configuration

### 7. **Exit Standby**
**Requirement**: "Uscire da standby: scrivere 0x00 in 0x1000"
**Implementation**: ✅ 
```python
def _exit_standby(self):
    SOFTWARE_STANDBY_REG = 0x1000
    self._set_tof_register(SOFTWARE_STANDBY_REG, 0x00, 0x01)
```

### 8. **Start Streaming**
**Requirement**: "Avviare streaming: scrivere 0x01 in 0x1001"
**Implementation**: ✅ 
```python
def _start_streaming(self):
    STREAMING_REG = 0x1001
    self._set_tof_register(STREAMING_REG, 0x01, 0x01)
```

## Correct Configuration Sequence

The implementation follows the exact sequence:
1. ✅ Enter standby (0x01 → 0x1000)
2. ✅ Configure MIPI lanes (0x00 → 0x1010 for 2-lane)
3. ⚠️ Configure speed registers (needs exact values)
4. ✅ Configure ToF parameters
5. ✅ Exit standby (0x00 → 0x1000)
6. ✅ Start streaming (0x01 → 0x1001)

## Usage Example

```python
# Configure for 2-lane MIPI as per datasheet
config = ToFConfig(
    mipi_lanes=2,
    fps=12
)

with MLX7502x(config=config) as sensor:
    # Automatically:
    # 1. Enters standby
    # 2. Sets DATA_LANE_CONFIG = 0
    # 3. Configures sensor
    # 4. Exits standby
    # 5. Starts streaming
    sensor.capture_continuous()
```

## Note
For complete compliance, the exact speed register values (0x100C-0x1071) need to be extracted from the datasheet tables and implemented.