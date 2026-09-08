import React from 'react';
import ToggleButton from '@mui/material/ToggleButton';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';
import Box from '@mui/material/Box'; 

function UIToggleButtonsExclusive(props) {
  const handleAlignment = (event, newAlignment) => {
    // Avoid setting null value when the selected button is clicked again
    if (newAlignment !== null) {
      props.onChange({ target: { name: props.id, value: newAlignment } });
    }
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 2, maxHeight: '20vh', overflowY: 'auto', justifyContent: 'space-between', width: props.width ? props.width : 'auto' }}>
      <label htmlFor={props.id}>{props.label}</label>
      <ToggleButtonGroup
        size = {props.size ? props.size : "medium"}
        exclusive
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
                  fontSize: props.fontSize ? props.fontSize : '14px',
                  '&.Mui-selected, &.Mui-selected:hover': { // Styles when the button is selected
                    bgcolor: props.selectedColor ? props.selectedColor : 'success.light', // Background color when selected
                    color: 'white', // Text color when selected
                  },
                  '&:hover': { // Styles on hover
                    bgcolor: 'primary.light', // Lighter background on hover when not selected
                  },
             }} // This prevents text from being transformed to uppercase
            disabled={option.disabled ? option.disabled : false}
            >
            {option.label}
            {option.icon ? option.icon : ""}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
    </Box>
  );
}

export { UIToggleButtonsExclusive };