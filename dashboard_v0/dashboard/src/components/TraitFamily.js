import React, { useState, useRef, useContext, useEffect } from 'react';
import { Trait } from '../components/Trait';
import { Button } from '../components/Button';
import { ProgressBar, UnknownBar } from '../components/Trait';
import { FiChevronDown, FiChevronUp } from 'react-icons/fi';
import { AttributionContext } from '../context/AttributionContext';
import { ChatContext } from '../context/ChatContext';
import * as d3 from "d3";
import { ConfigContext } from '../context/ConfigContext';
import { API_PORT, API_IP } from '../helpers/constants.js';

function TraitFamily(props) {
    const [optionsVisible, setOptionsVisible] = useState(false);
    const { setAttribution } = useContext(AttributionContext);
    const ctx = useContext(ChatContext).chatInfo;
    const config = useContext(ConfigContext);
    // var traits = [...props.traits].sort((a, b) => b.confidence - a.confidence);
    if (config.config.sort === "off"){
        var traits = props.traits
    } else {
        var traits = [...props.traits].sort((a, b) => b.confidence - a.confidence);
    }
    
    const toggleDropdown = () => {
        setOptionsVisible(!optionsVisible);
    };

    const setLastUserMsg = (msg) => {
        let lastMsgInChat = d3.selectAll('div.message-container div.message > span').nodes().at(-2);
        lastMsgInChat.innerHTML = msg;
    }

    const setLastBotMsg = (msg) => {
        let lastMsgInChat = d3.selectAll('div.message-container div.message > span').nodes().at(-1);
        lastMsgInChat.innerHTML =  msg;
    }

    const setLastChatMsg = (msg, i) => {
        let lastMsgInChat = d3.selectAll('div.message-container div.message > span').nodes().at(i);
        lastMsgInChat.innerHTML = msg;
    }

    useEffect(() => {
        // This will trigger every time ctx.history.length changes
        if (ctx.history.length && ctx.history.length > 2 && ctx.history.length % 2 === 1) {
            setLastChatMsg(ctx.history.at(-3).msg, -3);
            setLastChatMsg(ctx.history.at(-2).msg, -2);
            props.setHighlightedTrait(null);
        }
      }, [ctx.history.length]);

    const handleIconClick = (trait) => {
        if (ctx.history.length < 1) { return }
        if (trait === "Sycophancy") { return }
        if (props.highlightedTrait === trait) {
            props.setHighlightedTrait(null); 
            for (let i=0; i < ctx.history.length;i++) {
                setLastChatMsg(ctx.history.at(i).msg, i);
            }
            return;
        }
        let xhr = new XMLHttpRequest();
        props.setLoadingTraits(true);
        if (config.config.subject === "input" && trait != "Uncertainty") {
            xhr.open('POST', `http://localhost:8505/post_chat_attribution`, true);
            xhr.setRequestHeader('Content-Type', 'application/json');

            xhr.onload = () => {
                props.setLoadingTraits(false);
                if (xhr.status === 201) {
                    console.log(ctx.history)
                    // Assuming the API response's structure is { message: '...' }
                    const data = JSON.parse(xhr.responseText);
                    console.log(data)
                    setAttribution(prevMessages => ({
                        ...prevMessages,
                        [trait]: data.message
                    }));
                    if (data.msg_ids) {
                        for (let i=0; i < data.msg_ids.length; i++) {
                            setLastChatMsg(data['message'][i], data.msg_ids[i])
                        }
                    } else {
                        setLastUserMsg(data['message'])
                    }
                    
                } else {
                    // Handle non-200 status codes appropriately
                    console.error('Error fetching attribution:', xhr.statusText);
                }
            };

            xhr.onerror = () => {
                props.setLoadingTraits(false);
                // Handle network error
                console.error('Network error occurred while fetching attribution');
            };

            // Send the trait as part of the request payload
            xhr.send(JSON.stringify({ id: ctx.id,
                                      subattribute: trait,
                                      msg_ids: ctx.attrMsg,
                                      model: config.config.model
                                    }));}
        else {
            xhr.open('POST', `http://localhost:8505/post_chat_response_attribution`, true);
            xhr.setRequestHeader('Content-Type', 'application/json');

            xhr.onload = () => {
                props.setLoadingTraits(false);
                if (xhr.status === 201) {
                    console.log(ctx.history)
                    // Assuming the API response's structure is { message: '...' }
                    const data = JSON.parse(xhr.responseText);
                    console.log(data)
                    setAttribution(prevMessages => ({
                        ...prevMessages,
                        [trait]: data.message
                    }));
                    setLastBotMsg(data['message'])
                } else {
                    props.setLoadingTraits(false);
                    // Handle non-200 status codes appropriately
                    console.error('Error fetching attribution:', xhr.statusText);
                }
            };

            xhr.onerror = () => {
                props.setLoadingTraits(false);
                // Handle network error
                console.error('Network error occurred while fetching attribution');
            };

            // Send the trait as part of the request payload
            xhr.send(JSON.stringify({ id: ctx.id,
                                      subattribute: trait,
                                      model: config.config.model
                                     }));
        }
        props.setHighlightedTrait(trait);
    };

    return (
        <div className="trait-family">
            <div className="trait">
                <div className={`trait-icon ${traits[0].trait === props.highlightedTrait ? 'highlighted' : ''}`} onClick={traits[0].trait === "Unknown" ? null : () => handleIconClick(traits[0].trait)}>
                    {traits[0].icon}
                </div>
                {props.displayBar && traits[0].trait !== "Unknown" && <span className="trait-text">
                    <p className="trait-label">{props.label}</p>
                    <p className="trait-answer">{`${traits[0].trait} | ${Math.round(traits[0].confidence * 100)}%`}</p>
                </span>}
                {props.displayBar && traits[0].trait === "Unknown" && <span className="trait-text">
                    <p className="trait-label">{props.label}</p>
                    <p className="trait-answer">{`${traits[0].trait}`}</p>
                </span>}
                {props.displayBar && traits[0].trait !== "Unknown" && <ProgressBar width={traits[0].confidence} controlConfidence={traits[0].controlConfidence} onSliderChange={(newValue) => props.onSliderChange(props.category, traits[0].indexTrait, newValue)} beingControlled={ctx.controlYouModelStatus[props.category][traits[0].indexTrait]} traitName={traits[0].trait} onClearButtonClick={() => props.onClearButtonClick(props.category, traits[0].indexTrait)}/>}
                {props.displayBar && traits[0].trait === "Unknown" && <UnknownBar/>}
                {(props.displayBar && traits.length > 1) && <Button className="question" onClick={toggleDropdown}>
                    {optionsVisible ? <FiChevronUp /> : <FiChevronDown />}
                </Button>}
                {traits.length === 1 && (props.displayBar) && <span style={{width: "20px", marginLeft: "10px"}}></span>}
            </div>
            {optionsVisible && traits.slice(1).filter((obj, i) => (obj.trait !== "Unknown")).map((obj, i) => (
                <div className="trait trait-dropdown-child" key={i}>
                    {/* <Trait 
                        icon={obj.icon}
                        answer={obj.trait}
                        confidence={obj.confidence}
                    /> */}
                    <div className={`trait-icon-child ${obj.trait === props.highlightedTrait ? 'highlighted' : ''}`} onClick={obj.trait === "Unknown" ? null :() => handleIconClick(obj.trait)}>
                        {obj.icon}
                    </div>
                    <span className="trait-text-child">
                        <p className="trait-answer">{`${obj.trait} | ${Math.round(obj.confidence * 100)}%`}</p>
                    </span>
                    {props.displayBar && <ProgressBar width={obj.confidence} controlConfidence={obj.controlConfidence} beingControlled={ctx.controlYouModelStatus[props.category][obj.indexTrait]} onSliderChange={(newValue) => props.onSliderChange(props.category, obj.indexTrait, newValue)} traitName={obj.trait} onClearButtonClick={() => props.onClearButtonClick(props.category, obj.indexTrait)}/>}
                    <div className="questionspacing"></div>
                </div>
            ))}
        </div>
    );
}

export { TraitFamily };