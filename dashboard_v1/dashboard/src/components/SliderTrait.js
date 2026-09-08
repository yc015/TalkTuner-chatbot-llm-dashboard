import React from "react";
import { Slider } from "@mui/material";

function SliderTrait(props) {
  return (
    <div className="trait">
      <div className="trait-icon">{props.icon}</div>
      <span className="trait-text">
        <p className="trait-label">{props.answer}:</p>
        <p className="trait-answer">{`${Math.round(props.confidence)}%`}</p>
      </span>
      <SliderBar
        confidence={props.confidence}
        onSliderChange={props.onSliderChange}
      />
    </div>
  );
}

function SliderBar(props) {
  // Convert confidence (0 to 1) to slider value (0 to 100)
  const sliderValue = Math.round(props.confidence);

  const handleSliderChange = (event, newValue) => {
    if (props.onSliderChange) {
      props.onSliderChange(newValue);
    }
  };

  return (
    // <div className="progress-bar">
    <Slider
      value={isNaN(sliderValue) ? 0 : sliderValue}
      defaultValue={isNaN(sliderValue) ? 0 : sliderValue}
      onChangeCommitted={handleSliderChange}
      valueLabelDisplay="on"
      sx={{
        width: "auto", // Set the width of the slider
        flex: "1 1 auto", // Optionally control the flex property (e.g., 'none', 1, 'auto')
        // Add other style properties as needed
        "& .MuiSlider-thumb": {
          width: "14px",
          height: "14px",
          color: "#384958",
        },
        "& .MuiSlider-track": {
          color: "#536a80",
          // backgroundImage: "linear-gradient(.25turn, #8fabc0, #4d6276)"
        },
        "& .MuiSlider-rail": {
          color: "#658196",
        },
        "& .MuiSlider-active": {
          color: "#3e6080",
        },
      }}
      // color='primary'
      aria-label="Small"
    />
    // </div>
  );
}

export { SliderTrait, SliderBar };
