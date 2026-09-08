import React, { useContext } from "react";
import { Slider } from "@mui/material";

import { Button } from "../components/Button";
import { ImCross } from "react-icons/im";
import {
  FiAlertCircle,
  FiArrowLeft,
  FiArrowRight,
  FiChevronsDown,
  FiChevronsUp,
} from "react-icons/fi";
import { ConfigContext } from "../context/ConfigContext";

// So I guess this just isn't being used now???
function Trait(props) {
  return (
    <div className="trait">
      <div className="trait-icon">{props.icon}</div>
      <span className="trait-text">
        <p className="trait-label">{props.answer}:</p>
        <p className="trait-answer">{`${Math.round(
          props.confidence * 100
        )}%`}</p>
      </span>
      <ProgressBar width={props.confidence} />
    </div>
  );
}

function ProgressBar(props) {
  const { config } = useContext(ConfigContext);
  
  const getProgressBarColor = (width) => {
    // Define two colors for the gradient
    // const color1 = [255, 255, 255]; // RGB values for the start color
    // const color2 = [38, 61, 87];   // RGB values for the end color
    //var light-green-shadow-light
    const color2 = [9, 184, 79];
    // const color2 = [44,50,114]
    //var dark-teal
    const color1 = [100, 189, 192];
    // const color1 = [161,203,144];

    // Calculate the interpolated color based on the width
    const interpolatedColor = interpolateColor(color1, color2, width);

    // Convert the interpolated color to an rgba string
    const progressBarColor = `rgba(${interpolatedColor.join(",")}, 1)`;

    return progressBarColor;
  };

  // Linear interpolation between two colors
  const interpolateColor = (color1, color2, factor) => {
    return color1.map((channel, index) => {
      const interpolatedValue = Math.round(
        channel + (color2[index] - channel) * factor
      );
      return Math.min(255, Math.max(0, interpolatedValue));
    });
  };

    const progressBarColor = getProgressBarColor(props.width);

  return (
    <div className="progress-bar">
      <div className="progress-bar wrapper">
        {/* Solid gray for the part below 50% */}
        <div
          className="progress-bar inner"
          style={{
            flex: `${Math.min(props.width, 0.5)}`, // Proportional width for below 50%
            // backgroundImage:
            //   "repeating-linear-gradient(45deg, #b5b5b5, #b5b5b5 9px, white 9px, white 18px)", // Stripe effect
            background: "var(--main-bg-color-shadow-ligher)",
            transition: "flex 0.7s ease",
            borderRadius: props.width < 0.5 ? "20px":"20px 0 0 20px"
          }}
        ></div>

        {/* Gradient for the part above 50% */}
        {props.width > 0.5 && (
          <div
            className="progress-bar inner"
            style={{
              flex: `${props.width - 0.5}`, // Proportional width for above 50%
              background: props.beingControlled && !props.posControlled
                ? ""
                : `linear-gradient(to right, rgb(100, 189, 192), ${progressBarColor})`, // Gradient effect
              transition: "flex 0.7s ease",
              borderRadius: props.beingControlled && props.posControlled ? "0 0 0 0" : "0 20px 20px 0"
            }}
          ></div>
        )}

        {/* <div
          className={`progress-bar remaining ${
            props.beingControlled && props.posControlled ? "active" : ""
          }`}
          style={{
            width: `${(1 - props.width) * 100}%`,
            transition: "width 0.7s ease",
            right: "0%"
          }}
        ></div> */}

        < div
          className={`progress-bar remaining ${
            props.beingControlled && props.posControlled && config.probeType !== 'prompt_based' ? "active" : ""
          }`}
          style={{
            width: `${(1 - props.width) * 100}%`,
            position: "absolute",
            left: `${props.width * 100}%`, // Starts where the filled section ends
            transition: "width 0.7s ease",
            background: `${progressBarColor}`
          }}
        ></div>
        {/* <div className='progress-bar-slider'>
                    <Button
                        onClick={props.onClearButtonClick}
                        className={`reset-button ${props.beingControlled ? "active" : ""}`}      
                    >
                        <ImCross/>
                    </Button>
                </div> */}
      </div>
      {props.traitName !== "Unknown" &&
        props.traitName !== "Uncertainty" && 
        props.controlEnabled && 
        config.probeType !== 'prompt_based' && (
          <Button
            onClick={props.onNegButtonClick}
            className={`intervention-option left ${
              props.beingControlled && !props.posControlled ? "active" : ""
            }`}
            disabled={false}
          >
            {" "}
            <FiArrowLeft
              style={{ height: "19px", width: "19px", color: "#64BDC0" }}
            />{" "}
          </Button>
        )}
      {props.traitName !== "Unknown" &&
        props.traitName !== "Uncertainty" && 
        props.controlEnabled && 
        config.probeType !== 'prompt_based' && (
          <Button
            onClick={props.onPosButtonClick}
            className={`intervention-option right ${
              props.beingControlled && props.posControlled ? "active" : ""
            }`}
            disabled={false}
          >
            {" "}
            <FiArrowRight
              style={{ height: "19px", width: "19px", color: "#09b84f" }}
            />{" "}
          </Button>
        )}
    </div>
  );
}

function UnknownBar(props) {
  return (
    <div className="progress-bar-unknown">
      <div className="progress-bar-stripes-left"></div>
      {/* <div className='progress-bar-label'>
                Unknown 
            </div> */}
      {/* <div className='progress-bar-stripes-right'></div> */}
    </div>
  );
}

function TraitAlert(props) {
  return (
    <div
      className={`trait-alert ${props.answerChange ? "new" : ""} ${
        Math.abs(props.change) > 0.2 || props.answerChanged ? "active" : ""
      }`}
    >
      <FiAlertCircle style={{color: "#606060"}}/>
      {props.answerChange ? (
        <span>Answer Changed</span>
      ) : (
        <>
          <span>Confidence</span>
          {props.change > 0 ? <FiChevronsUp style={{color: "#606060"}}/> : <FiChevronsDown style={{color: "#606060"}}/>}
        </>
      )}
    </div>
  );
}

function ControlAlert(props) {
    console.log(props.posControlled)
    return (
      <div
        // className={`trait-alert ${props.posControlled ? "pos active": "neg active"}`}
        className={`trait-alert controlled active`}
      >
        {/* <FiAlertCircle /> */}
        <span>Pinned</span>
        {/* {props.posControlled ? <FiChevronsUp /> : <FiChevronsDown />} */}
      </div>
    );
  }

export { Trait, ProgressBar, UnknownBar, TraitAlert, ControlAlert };