import React from 'react';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Box from '@mui/material/Box'; // Importing Box for additional styling

function UIToggleButtons(props) {
  const handleAlignment = (event, newAlignment) => {
    // Avoid setting null value when the selected button is clicked again
    if (newAlignment !== null) {
      props.onChange({ target: { name: props.id, value: newAlignment } });
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2, maxHeight: '20vh', overflowY: 'auto' }}>
      <label htmlFor={props.id}>{props.label}</label>
      <ToggleButtonGroup
        multiple
        value={props.value}
        onChange={handleAlignment}
        aria-label={props.label}
        sx={{ flexWrap: 'wrap',  }} // This enables the wrapping of toggle buttons
      >
        {props.options.map((option) => (
          <ToggleButton
            key={option.value}
            value={option.value}
            aria-label={option.label}
            sx={{ textTransform: 'none',
                  fontSize: '11px',
                  '&.Mui-selected, &.Mui-selected:hover': { // Styles when the button is selected
                    bgcolor: 'success.light', // Background color when selected
                    color: 'white', // Text color when selected
                  },
                  '&:hover': { // Styles on hover
                    bgcolor: 'primary.light', // Lighter background on hover when not selected
                  },
                  padding: '8px',
             }} // This prevents text from being transformed to uppercase
            >
            {option.label}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
}

export { UIToggleButtons };