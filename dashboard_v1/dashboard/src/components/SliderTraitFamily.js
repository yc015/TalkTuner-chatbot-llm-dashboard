import React, { useState, useRef, useContext, useEffect } from 'react';
import { SliderTrait } from '../components/SliderTrait';
import { Button } from '../components/Button';
import { SliderBar } from '../components/SliderTrait';
import { FiChevronDown, FiChevronUp } from 'react-icons/fi';
import { AttributionContext } from '../context/AttributionContext';
import { ChatContext } from '../context/ChatContext';
import * as d3 from "d3";
import { ConfigContext } from '../context/ConfigContext';
import { API_PORT, API_IP, BACKEND_ADDR } from '../helpers/constants.js';

function SliderTraitFamily(props) {
    const [optionsVisible, setOptionsVisible] = useState(false);
    const { setAttribution } = useContext(AttributionContext);
    const ctx = useContext(ChatContext).chatInfo;
    const config = useContext(ConfigContext);
    // var traits = [...props.traits].sort((a, b) => b.confidence - a.confidence);
    var traits = props.traits;

    const toggleDropdown = () => {
        setOptionsVisible(!optionsVisible);
    };

    const setLastUserMsg = (msg) => {
        let lastMsgInChat = d3.selectAll('div.message-container div.message > span').nodes().at(-2);
        lastMsgInChat.innerHTML = msg;
    }

    const setLastBotMsg = (msg) => {
        let lastMsgInChat = d3.selectAll('div.message-container div.message > span').nodes().at(-1);
        lastMsgInChat.innerHTML = msg;
    }

    const setLastChatMsg = (msg, i) => {
        console.log(d3.selectAll('div.message-container div.message > span'))
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
        if (props.highlightedTrait === trait) {
            props.setHighlightedTrait(null); 
            setLastChatMsg(ctx.history.at(-2).msg, -2);
            setLastChatMsg(ctx.history.at(-1).msg, -1); 
            return;
        }
        let xhr = new XMLHttpRequest();
        props.setLoadingTraits(true);
        if (config.config.subject === "input") {
            xhr.open('POST', `${BACKEND_ADDR}/post_chat_attribution`, true);
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
                    setLastUserMsg(data['message'])
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
                                    subattribute: trait }));}
        else {
            xhr.open('POST', `${BACKEND_ADDR}/post_chat_response_attribution`, true);
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
                                      subattribute: trait }));
        }
        props.setHighlightedTrait(trait);
    };

    return (
        <div className="trait-family">
            <div className="trait">
                <div className={`trait-icon ${traits[0].trait === props.highlightedTrait ? 'highlighted' : ''}`} onClick={() => handleIconClick(traits[0].trait)}>
                    {traits[0].icon}
                </div>
                {props.displayBar && 
                    <span className="trait-text">
                        <p className="trait-label">{props.label}</p>
                        <p className="trait-answer">{`${traits[0].trait} | ${Math.round(traits[0].confidence)}`}</p>
                    </span>
                }
                {props.displayBar && <SliderBar confidence={traits[0].confidence} onSliderChange={(newValue) => props.onSliderChange(props.category, traits[0].indexTrait, newValue)}/>}
                {props.displayBar && 
                    <Button className="question" onClick={toggleDropdown}>
                        {optionsVisible ? <FiChevronUp /> : <FiChevronDown />}
                    </Button>
                }
            </div>
            {optionsVisible && traits.slice(1).map((obj, i) => (
                <div className="trait trait-dropdown-child" key={i}>
                    {/* <Trait 
                        icon={obj.icon}
                        answer={obj.trait}
                        confidence={obj.confidence}
                    /> */}
                    <div className={`trait-icon-child ${obj.trait === props.highlightedTrait ? 'highlighted' : ''}`} onClick={() => handleIconClick(obj.trait)}>
                        {obj.icon}
                    </div>
                    <span className="trait-text-child">
                        <p className="trait-answer">{`${obj.trait} | ${Math.round(obj.confidence)}`}</p>
                    </span>
                    <SliderBar confidence={obj.confidence} onSliderChange={(newValue) => props.onSliderChange(props.category, obj.indexTrait, newValue)}/>
                    <div className="questionspacing"></div>
                </div>
            ))}
        </div>
    );
}

export { SliderTraitFamily };