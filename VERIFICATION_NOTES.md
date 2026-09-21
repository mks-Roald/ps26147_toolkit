Verification that MODULATION_BAUD_RANGES implementation is correct:

1. Constant defined correctly:
   - MODULATION_BAUD_RANGES contains all modulation types with min, max, step
   - Values match the requested ranges: 
     * BPSK: [1000,100000] step 1000
     * QPSK: [1000,200000] step 1000
     * 8PSK/16QAM/64QAM: higher ranges with step 1000
     * FSK2/FSK4: lower ranges [1000,50000] and [1000,100000] with step 500

2. Slider uses dynamic values:
   - min={MODULATION_BAUD_RANGES[controls.modulation]?.min ?? 1000}
   - max={MODULATION_BAUD_RANGES[controls.modulation]?.max ?? 500000}
   - step={MODULATION_BAUD_RANGES[controls.modulation]?.step ?? 1000}

3. Clamping logic updated:
   - Uses MODULATION_BAUD_RANGES[controls.modulation] for range
   - Preserves existing functionality to clamp baud_rate when modulation changes

4. Fallback values provided:
   - ?? 1000 for min, ?? 500000 for max, ?? 1000 for step
   - Ensures slider still works if modulation not found in map

All requirements from the user's request have been met.