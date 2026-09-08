import React, { useState, useEffect, useContext, useCallback } from 'react';
import { useMountEffect } from '../helpers/useMountEffect.js';
import { Button } from './Button.js';
import { FiSend } from 'react-icons/fi';
import { RxReset } from "react-icons/rx";

import { ModalContext } from '../context/ModalContext.js';
import { ChatContext } from '../context/ChatContext.js';
import { ConfigContext } from '../context/ConfigContext.js';
import { LiaHandPointLeftSolid, LiaHandPointRightSolid, LiaHandPointUpSolid } from "react-icons/lia";
import { UIToggleButtonsExclusive } from '../components/UIToggleButtonsExclusive.js';
import { defaultControlYouModelStatus } from '../context/ChatContext.js';
import { FiSave } from "react-icons/fi";
import { TextInput } from './TextInput.js';
import logoImage from '../imgs/favicon3.png'
import mistralAILogoImage from "../imgs/mistralAI.png"

import llamaBoxer from '../imgs/llama_boxer.png'
import mistralBoxer from "../imgs/mistral_boxer.png"
// import { LiaDharmachakraSolid, LiaCrossSolid } from 'react-icons/lia'

import { CircularProgressbarWithChildren, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

function Settings(props) {
    const ctx = useContext(ChatContext).chatInfo;
    const { config, setConfig } = useContext(ConfigContext);
    const [systemPromptInput, setSystemPromptInput] = useState('');
    const [isBusy, setIsBusy] = useState(false);

    useMountEffect(() => {
        setSystemPromptInput('');
    });

    const isPromptEmpty = () => { return !/(.|\s)*\S(.|\s)*/.test(systemPromptInput); }

    function onSend() {
        if (isPromptEmpty()) return;
        setIsBusy(true)
        setConfig(prevConfig => {
            return { ...prevConfig, systemPrompt: systemPromptInput };
        });
        setSystemPromptInput("");
        setIsBusy(false)
    }

    function onReset() {
        setIsBusy(true)
        setConfig(prevConfig => {
            return { ...prevConfig, systemPrompt: prevConfig.defaultSystemPrompt };
        });
        setIsBusy(false)
    }

    return (
        <div className="dashboard container">
            <h3>Current System Prompt:</h3>
            <div className='system-prompt-container'>
                {config.model === "llama" ? (config.systemPrompt !== null ? config.systemPrompt : config.defaultSystemPrompt) : "Mistral-7B V2 does not support system message."}
                {isBusy && 
                <div>
                    <span className="loader" />
                </div>
            }
            </div>
            <div>
                <TextInput
                    name="chatInput"
                    id="chat-message-input"
                    className="system-prompt-input"
                    value={systemPromptInput}
                    onChange={e => setSystemPromptInput(e.target.value)}
                    onKeyDown={e => { if (e.keyCode === 13) onSend();}}
                    placeholder={"Input your system prompt here"}
                    area={true}
                />
                <div style={{alignItems: "center", justifyContent: "center", display: "flex"}}>
                    <Button className="system-prompt-submit" onClick={onSend} disabled={isPromptEmpty() || config.model === "mistral"}>
                        Save System Prompt
                        <FiSave />
                    </Button>
                    <Button className="system-prompt-submit" onClick={onReset} disabled={isPromptEmpty() || config.model === "mistral"}>
                        Reset
                        <RxReset/>
                    </Button>
                </div>
            </div>
            <h3>Assigned Chat ID: {ctx.id}</h3> 
            <h3>Current Length of Conversation: {ctx.history.length}</h3> 
            <div style={{alignItems: "center", display: "flex"}}><img src={config.model === "llama" ? llamaBoxer : mistralBoxer} alt="Placeholder" height="60vw" style={{transform: config.model === "mistral" ? "scaleX(-1)": ""}} />: 
                <div style={{border: "1px solid", borderRadius: "7px 7px 7px 0px", padding: "7px", marginLeft: "8px"}}>
                    {ctx.history.length < 4 && "I am your assistant. Try to talk more with me!"}{ctx.history.length >= 4 && ctx.history.length < 6 && "I want to know more about you!"}{ctx.history.length >6 && "Wow, we have talked for so long. I think I have learned a lot about you!"} 
                </div>
            </div>
            
        </div>
    );
}

export { Settings }