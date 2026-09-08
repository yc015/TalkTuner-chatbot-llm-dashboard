import React, { useState, useRef, useContext, useEffect } from "react";
import { Button } from "../components/Button";
import {
  Trait,
  TraitAlert,
  ControlAlert,
  ProgressBar,
  UnknownBar,
} from "../components/Trait";
import { FiChevronDown, FiChevronUp } from "react-icons/fi";
import { AttributionContext } from "../context/AttributionContext";
import { ChatContext } from "../context/ChatContext";
import * as d3 from "d3";
import { ConfigContext } from "../context/ConfigContext";
import { API_PORT, API_IP, BACKEND_ADDR } from '../helpers/constants.js';

function TraitFamily(props) {
  const [optionsVisible, setOptionsVisible] = useState(false);
  const [change, setChange] = useState(0);
  const [answerChanged, setAnswerChanged] = useState(false);
  const { setAttribution } = useContext(AttributionContext);
  const ctx = useContext(ChatContext).chatInfo;
  const config = useContext(ConfigContext);
  // var traits = [...props.traits].sort((a, b) => b.confidence - a.confidence);
  var traits = props.traits;
  if (config.config.sort === "on") {
    traits = [...props.traits].sort((a, b) => b.confidence - a.confidence);
  }

  var controlled = false
  for (let i=0; i < traits.length; i++) {
    // console.log(i)
    if (ctx.controlYouModelStatus[props.category][traits[i].indexTrait]) {
        controlled = true
    }
  }

  // console.log(controlled)

  const toggleDropdown = () => {
    setOptionsVisible(!optionsVisible);
  };

  const setLastUserMsg = (msg) => {
    let lastMsgInChat = d3
      .selectAll("div.message-container div.message > span")
      .nodes()
      .at(-2);
    lastMsgInChat.innerHTML = msg;
  };

  const setLastBotMsg = (msg) => {
    let lastMsgInChat = d3
      .selectAll("div.message-container div.message > span")
      .nodes()
      .at(-1);
    lastMsgInChat.innerHTML = msg;
  };

  const setLastChatMsg = (msg, i) => {
    let lastMsgInChat = d3
      .selectAll("div.message-container div.message > span")
      .nodes()
      .at(i);
    lastMsgInChat.innerHTML = msg;
  };

  useEffect(() => {
    // This will trigger every time ctx.history.length changes
    if (
      ctx.history.length &&
      ctx.history.length > 2 &&
      ctx.history.length % 2 === 1
    ) {
      for (let i = 0; i < ctx.history.length; i++) {
        setLastChatMsg(ctx.history.at(i).msg, i);
      }
      props.setHighlightedTrait(null);
    }
  }, [ctx.history.length]);

    useEffect(() => {
        if (!props.blockUpdate && !props.loadingTraits && props.historyIndex > 0 && ctx.historyYouModel.length > 1 && props.historyIndex < ctx.historyYouModel.length) {
            // console.log("updating trait family");
            // console.log(props.historyIndex)
            var lastState = ctx.historyYouModel[props.historyIndex - 1][props.category];
            // Check if lastState exists (it may not for newly added probes)
            if (!lastState || typeof lastState !== 'object' || Object.keys(lastState).length === 0) {
                // Skip history comparison for newly added attributes that don't have history
                setChange(0);
                setAnswerChanged(false);
                return;
            }
            const lastTopTrait = Object.entries(lastState).reduce((maxKey, [key, value]) => {
                return value > lastState[maxKey] ? key : maxKey;
            }, Object.keys(lastState)[0]);
            const confidenceDifference = traits[0].confidence - lastState[traits[0].indexTrait];
            // console.log(`Did the value change: ${lastTopTrait !== traits[0].indexTrait}`);
            // console.log(`The confidence went ${confidenceDifference > 0 ? "up" : "down"}`);
            setChange(confidenceDifference);
            setAnswerChanged(lastTopTrait !== traits[0].indexTrait);
        }
    }
        , [props.historyIndex, props.blockUpdate, props.loadingTraits, traits]);

  const handleIconClick = (trait) => {
    if (ctx.history.length < 1) {
      return;
    }
    if (props.highlightedTrait === trait) {
      props.setHighlightedTrait(null);
      for (let i = 0; i < ctx.history.length; i++) {
        setLastChatMsg(ctx.history.at(i).msg, i);
      }
      return;
    }
    let xhr = new XMLHttpRequest();
    props.setLoadingTraits(true);
    if (config.config.subject === "input" && trait != "Uncertainty") {
      xhr.open(
        "POST",
        `${BACKEND_ADDR}/post_chat_attribution`,
        true
      );
      xhr.setRequestHeader("Content-Type", "application/json");

      xhr.onload = () => {
        props.setLoadingTraits(false);
        if (xhr.status === 201) {
          console.log(ctx.history);
          // Assuming the API response's structure is { message: '...' }
          const data = JSON.parse(xhr.responseText);
          console.log(data);
          setAttribution((prevMessages) => ({
            ...prevMessages,
            [trait]: data.message,
          }));
          if (data.msg_ids) {
            for (let i = 0; i < data.msg_ids.length; i++) {
              setLastChatMsg(data["message"][i], data.msg_ids[i]);
            }
          } else {
            setLastUserMsg(data["message"]);
          }
        } else {
          // Handle non-200 status codes appropriately
          console.error("Error fetching attribution:", xhr.statusText);
        }
      };

      xhr.onerror = () => {
        props.setLoadingTraits(false);
        // Handle network error
        console.error("Network error occurred while fetching attribution");
      };

      // Send the trait as part of the request payload
      xhr.send(
        JSON.stringify({
          id: ctx.id,
          subattribute: trait,
          msg_ids: ctx.attrMsg,
          model: config.model,
        })
      );
    } else {
      xhr.open(
        "POST",
        `${BACKEND_ADDR}/post_chat_response_attribution`,
        true
      );
      xhr.setRequestHeader("Content-Type", "application/json");

      xhr.onload = () => {
        props.setLoadingTraits(false);
        if (xhr.status === 201) {
          console.log(ctx.history);
          // Assuming the API response's structure is { message: '...' }
          const data = JSON.parse(xhr.responseText);
          console.log(data);
          setAttribution((prevMessages) => ({
            ...prevMessages,
            [trait]: data.message,
          }));
          setLastBotMsg(data["message"]);
        } else {
          props.setLoadingTraits(false);
          // Handle non-200 status codes appropriately
          console.error("Error fetching attribution:", xhr.statusText);
        }
      };

      xhr.onerror = () => {
        props.setLoadingTraits(false);
        // Handle network error
        console.error("Network error occurred while fetching attribution");
      };

      // Send the trait as part of the request payload
      xhr.send(
        JSON.stringify({ id: ctx.id, subattribute: trait, model: config.model })
      );
    }
    props.setHighlightedTrait(trait);
    console.log(trait);
    console.log(props.highlightedTrait);
  };
  

    return (
        <div className={"trait-family " + (props.blockEvents ? "block-events " : "") + ((ctx.history.length > 0 && "loading" in ctx.history[ctx.history.length - 1] && ctx.history[ctx.history.length - 1].loading) ? "loading" : "")}>
            <div className="trait">
                {/* <div className={`trait-icon${traits[0].trait === "Unknown" ? '-unknown' : ''} ${traits[0].trait === props.highlightedTrait ? 'highlighted' : ''}`} >
                    {traits[0].icon}
                </div> */}
                <div className="inner-content">
                    <div className="text-details">
                        {props.displayBar && traits[0].trait !== "Unknown" && <span className="trait-text">
                            <p className="trait-label">{props.label}</p>
                            <p className="trait-answer">{`${traits[0].trait} | ${Math.round(traits[0].confidence * 100)}%`}</p>
                        </span>}
                        {props.displayBar && traits[0].trait === "Unknown" && <span className="trait-text">
                            <p className="trait-label">{props.label}</p>
                            <p className="trait-answer">{`${traits[0].trait}`}</p>
                        </span>}
                        <div style={{justifyContent: "flex-end", flexDirection: "row", display: "flex", gap: "8px"}}>
                          {props.historyIndex > 0 && <TraitAlert answerChange={answerChanged} change={change} />}
                          {controlled && <ControlAlert posControlled={false} />}
                        </div>
                    </div>
                    {props.displayBar && traits[0].trait !== "Unknown" && <ProgressBar width={traits[0].confidence} controlConfidence={traits[0].controlConfidence} onPosButtonClick={() => props.onPositiveIntervButtonClick(props.category, traits[0].indexTrait)} onNegButtonClick={() => props.onNegativeIntervButtonClick(props.category, traits[0].indexTrait)} beingControlled={ctx.controlYouModelStatus[props.category][traits[0].indexTrait]} posControlled={ctx.controlYouModel[props.category][traits[0].indexTrait] > 50} traitName={traits[0].trait} onClearButtonClick={() => props.onClearButtonClick(props.category, traits[0].indexTrait)} controlEnabled={props.enableControl} />}
                    {props.displayBar && traits[0].trait === "Unknown" && <UnknownBar />}
                </div>
                {(props.displayBar && traits.length > 1) && <Button className="question" onClick={toggleDropdown}>
                    {optionsVisible ? <FiChevronUp /> : <FiChevronDown />}
                </Button>}
                {traits.length === 1 && (props.displayBar) && <span style={{ width: "20px", marginLeft: "10px" }}></span>}
            </div>
            {optionsVisible && traits.slice(1).filter((obj, i) => (obj.trait !== "Unknown")).map((obj, i) => (
                <div className="trait trait-dropdown-child" key={i}>
                    {/* <Trait 
                        icon={obj.icon}
                        answer={obj.trait}
                        confidence={obj.confidence}
                    /> */}
              {/* <div
                className={`trait-icon-child${
                  obj.trait === "Unknown" ? "-unknown" : ""
                } ${obj.trait === props.highlightedTrait ? "highlighted" : ""}`}
              >
                {obj.icon}
              </div> */}
              <span className="trait-text-child">
                <p className="trait-answer">{`${obj.trait} | ${Math.round(
                  obj.confidence * 100
                )}%`}</p>
              </span>
              {props.displayBar && (
                <ProgressBar
                  width={obj.confidence}
                  controlConfidence={obj.controlConfidence}
                  beingControlled={
                    ctx.controlYouModelStatus[props.category][obj.indexTrait]
                  }
                  posControlled={
                    ctx.controlYouModel[props.category][obj.indexTrait] > 50
                  }
                  onPosButtonClick={() =>
                    props.onPositiveIntervButtonClick(
                      props.category,
                      obj.indexTrait
                    )
                  }
                  onNegButtonClick={() =>
                    props.onNegativeIntervButtonClick(
                      props.category,
                      obj.indexTrait
                    )
                  }
                  traitName={obj.trait}
                  onClearButtonClick={() =>
                    props.onClearButtonClick(props.category, obj.indexTrait)
                  }
                controlEnabled={props.enableControl} />
              )}
              <div className="questionspacing"></div>
            </div>
          ))}
    </div>
  );
}

export { TraitFamily };
