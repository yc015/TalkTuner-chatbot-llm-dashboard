import React from 'react';
import { Slider } from '@mui/material';

import { Button } from '../components/Button';
import { ImCross } from "react-icons/im";

function Trait(props) {
    return (
        <div className="trait">
            <div className="trait-icon">
                {props.icon}
            </div>
            <span className="trait-text">
                <p className="trait-label">{props.answer}:</p>
                <p className="trait-answer">{`${Math.round(props.confidence * 100)}%`}</p>
            </span>
            <ProgressBar width={props.confidence}/>
        </div>
    );
}

function ProgressBar(props) {
    const getProgressBarColor = (width) => {
        // Define two colors for the gradient
        const color1 = [255, 255, 255]; // RGB values for the start color
        const color2 = [38, 61, 87];   // RGB values for the end color

        // Calculate the interpolated color based on the width
        const interpolatedColor = interpolateColor(color1, color2, width);

        // Convert the interpolated color to an rgba string
        const progressBarColor = `rgba(${interpolatedColor.join(',')}, 1)`;

        return progressBarColor;
    };
    
    // Linear interpolation between two colors
    const interpolateColor = (color1, color2, factor) => {
        return color1.map((channel, index) => {
            const interpolatedValue = Math.round(channel + (color2[index] - channel) * factor);
            return Math.min(255, Math.max(0, interpolatedValue));
        });
    };

    const progressBarColor = getProgressBarColor(props.width);

    const handleSliderChange = (event, newValue) => {
        if (props.onSliderChange) {
            console.log(newValue)
            props.onSliderChange(newValue);
        }
    };

    function LabelFormat(x) { 
        return Math.round(x); 
    }

    return (
        <div className="progress-bar">
            <div className='progress-bar wrapper'>
                <div className="progress-bar inner" 
                    style={{
                        width: `${props.width * 100}%`,
                        backgroundColor: progressBarColor
                    }}>
                </div>
                <div className='progress-bar-slider'>
                    {(props.traitName !== "Unknown" && props.traitName !== "Uncertainty"  && props.traitName != "Sycophancy") && 
                        <Slider 
                            value={isNaN(props.controlConfidence) ? 0 : props.controlConfidence}
                            defaultValue={isNaN(props.controlConfidence) ? 0 : props.controlConfidence} 
                            onChangeCommitted={handleSliderChange}
                            valueLabelDisplay="auto" 
                            valueLabelFormat={LabelFormat}
                            sx={{
                                width: `auto`, // Set the width of the slider
                                flex: '1 1 auto', // Optionally control the flex property (e.g., 'none', 1, 'auto')
                                // Add other style properties as needed
                                "& .MuiSlider-thumb": {
                                    width: '10px',
                                    height: '16px',
                                    // color: `${Math.round(Math.abs(props.controlConfidence - props.width * 100)) > 1 ? "#23C712" : '#115DAD'}`,
                                    color: `${props.beingControlled ? "#23C712" : '#bababa'}`,
                                    borderRadius: '0px',
                                    border: 'solid black',
                                    borderWidth: '1px',
                                },
                                '& .MuiSlider-track': {
                                    color: "rgba(255, 255, 255, 0)"
                                    // backgroundImage: "linear-gradient(.25turn, #8fabc0, #4d6276)"
                                },
                                '& .MuiSlider-rail': {
                                    color: "rgba(255, 255, 255, 0)"
                                },
                            }}
                    // color='primary'
                    aria-label="Small"
                    />}
                    <Button
                        // data-id={props.mid}
                        onClick={props.onClearButtonClick}
                        // className={`reset-button ${Math.round(Math.abs(props.controlConfidence - props.width * 100)) > 1 ? "active" : ""}`}
                        className={`reset-button ${props.beingControlled ? "active" : ""}`}      
                    >
                        <ImCross/>
                    </Button>
                </div>
            </div>
        </div>
    );
}

function UnknownBar(props) {
    return (
        <div className="progress-bar-unknown">
            <div className='progress-bar-stripes-left'></div>
            <div className='progress-bar-label'>
                Unknown 
            </div>
            <div className='progress-bar-stripes-right'></div>
        </div>
    );
}

export { Trait, ProgressBar, UnknownBar }