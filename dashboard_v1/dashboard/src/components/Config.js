import React, { useState, useEffect, useContext } from 'react';

import { UIDropdown } from '../components/UIDropdown';
import { ConfigContext } from '../context/ConfigContext.js';
import { UIToggleButtonsExclusive } from './UIToggleButtonsExclusive.js';
import { FaMagnifyingGlassChart } from "react-icons/fa6";
import { MdExplore } from "react-icons/md";
import { AiOutlineControl } from "react-icons/ai";
import { PiGearSixBold } from "react-icons/pi";
import logoImage from "../imgs/favicon3.png";
import chatgptLogoImage from "../imgs/chatgptIcon.png"
import mistralAILogoImage from "../imgs/mistralAI.png"
import gemmaLogoImage from "../imgs/gemma.png"
import { RxValueNone } from "react-icons/rx";
import { ChatContext } from '../context/ChatContext.js';

function Config(props) {
    const { config, setConfig } = useContext(ConfigContext);
    const [isMistral, setIsMistral] = useState(false)
    const [isNull, setIsNull] = useState(false)
    const ctx = useContext(ChatContext).chatInfo;
    // const modelOptions = [{ value: 'llama', label: 'LLaMa-13B', icon: <img src={logoImage} alt="Placeholder" height={"21vh"} />, disabled: ctx.history.length > 0 ? true : false}, 
    // { value: 'mistral', label: 'Mistral', icon: <img src={mistralAILogoImage} alt="Placeholder" height={"21vh"} style={{marginLeft: "3px"}} />, disabled: ctx.history.length > 0 ? true : false},
    // { value: 'gemma', label: 'Gemma', icon: <img src={gemmaLogoImage} alt="Placeholder" height={"21vh"} />, disabled: true},]

    const modelOptions = [
//         { value: 'llama', label: 'LLaMa2-13B', icon: <img src={logoImage} alt="Placeholder" height={"21vh"} />, disabled: ctx.history.length > 0 ? true : false}, 
{ value: 'llama3', label: 'LLaMa3-8B', icon: <img src={logoImage} alt="Placeholder" height={"21vh"} />, disabled: ctx.history.length > 0 ? true : false}, 
        { value: 'gemma2', label: 'Gemma2-9B', icon: <img src={gemmaLogoImage} alt="Placeholder" height={"21vh"} />, disabled: ctx.history.length > 0 ? true : false},]
    // // const modeOptions = [{ value: 'default', label: 'Probe', icon: <FaMagnifyingGlassChart style={{marginLeft: "5px", color: "#5169DF"}}/> }, 
    //                     //  { value: 'control', label: 'Control', icon: <AiOutlineControl style={{marginLeft: "5px", color: "#5169DF"}}/> }, 
    //                      { value: 'explore', label: 'Explore', icon: <MdExplore style={{marginLeft: "5px", color: "#5169DF"}}/> },
    //                      { value: 'configuration', label: 'Config', icon: <PiGearSixBold style={{marginLeft: "5px", color: "#5169DF"}}/> },
    //                      { value: 'null', label: 'Null', icon: <RxValueNone style={{marginLeft: "5px", color: "#5169DF"}}/> }, 
    //                     ]
    // const modeOptions = [{ value: 'null', label: 'Ver. 1', }, 
    //                     //  { value: 'control', label: 'Control', icon: <AiOutlineControl style={{marginLeft: "5px", color: "#5169DF"}}/> }, 
    //                      { value: 'default_2', label: 'Ver. 2', },
    //                      { value: 'default', label: 'Ver. 3', }, 
    //                     ]
    // const attrOptions = [{ value: 'input', label: 'input' }, { value: 'output', 'label': 'output' }]


    // function onChangeMode(e) {
    //     setIsNull(e)
    //     setConfig(prevConfig => {
    //         prevConfig.mode = e ? 'default' : 'null';
    //         return { ...prevConfig }
    //     });
    // }

    return (
        <div id="configs">
            <div id="model-toggle">
                {/* <UIToggleButtonsExclusive 
                    id="mode"
                    label="Interface: "
                    value={config.mode}
                    // options={modeOptions}
                    onChange={
                        (e) => {
                            setConfig(prevConfig => ({
                                ...prevConfig,
                                mode: e.target.value
                            }));
                            console.log(`update interface mode: ${e.target.value}`)
                        }
                    }
                    fontSize="12px"
                    size="small"
                    width="220px"
                    // selectedColor="#5FC763"
                    selectedColor="#909090"
                /> */}
                {/* <div style={{flexDirection: "column", justifyContent: "center", alignItems: "center", display: "flex"}}>
                <img src={logoImage} alt="Placeholder" height={"21vh"} />
                <p>LLaMa</p>
                </div> */}


                {/* <div style={{flexDirection: "column", justifyContent: "center", alignItems: "center", display: "flex"}}>
                    <p>Interface 1</p>
                </div>
                <Toggle checked={isNull} onChange={onChangeMode} disabled={false} type="mode"/>
                <div style={{flexDirection: "column", justifyContent: "center", alignItems: "center", display: "flex"}}>
                    <p>Interface 2</p>
                </div> */}
                
                {/* Switch won't work for more than three models. I think it's possible,  in the future, for us to include Gemma-7B, LLaMa-7B, and even the close source model here. */}
                <UIToggleButtonsExclusive 
                    id="model"
                    // label="ML model: "
                    value={config.model}
                    options={modelOptions}
                    onChange={
                        (e) => {
                            setConfig(prevConfig => ({
                                ...prevConfig,
                                model: e.target.value
                            }));
                            console.log(`update ML model: ${e.target.value}`)
                        }
                    }
                    fontSize="12px"
                    size="small"
                    width="360px"
                    selectedColor="#5FC763"
                />
            </div>
        </div>
    );

    // return (
    //     <div id="configs">
    //         <div id="model-toggle">
    //             <p>LLaMa</p>
    //             <Toggle/>
    //             <p>Mistral</p>
    //         </div>
    //         <div className="config-container">
    //             <div style={{display: 'flex'}}>
    //             {/* <UIDropdown
    //                 id="model"
    //                 label="ML model: "
    //                 value={config.model}
    //                 options={modelOptions}
    //                 onChange={
    //                     (e) => {
    //                         setConfig(prevConfig => {
    //                             prevConfig.model = e.target.value;
    //                             return { ...prevConfig }
    //                         });
    //                         console.log(`update ML model: ${config.model}`)
    //                     }
    //                 }
    //             /> */}
    //                 <UIToggleButtonsExclusive 
    //                     id="model"
    //                     label="ML model: "
    //                     value={config.model}
    //                     options={modelOptions}
    //                     onChange={
    //                         (e) => {
    //                             setConfig(prevConfig => {
    //                                 prevConfig.model = e.target.value;
    //                                 return { ...prevConfig }
    //                             });
    //                             console.log(`update ML model: ${config.model}`)
    //                         }
    //                     }
    //                     fontSize="12px"
    //                     size="small"
    //                     width="240px"
    //                     selectedColor="#5FC763"
    //                 />
    //             </div>
    //             {/* <UIDropdown
    //                 id="mode"
    //                 label="Mode: "
    //                 options={modeOptions}
    //                 onChange={
    //                     (e) => {
    //                         setConfig(prevConfig => {
    //                             prevConfig.mode = e.target.value;
    //                             return { ...prevConfig }
    //                         });
    //                         console.log(`update interface mode: ${config.mode}`)
    //                     }
    //                 }
    //             /> */}
    //             <div style={{padding: "5px 0 0 0", display: 'flex',}}>
    //                 <UIToggleButtonsExclusive 
    //                     id="mode"
    //                     label="Panel: "
    //                     value={config.mode}
    //                     options={modeOptions}
    //                     onChange={
    //                         (e) => {
    //                             setConfig(prevConfig => {
    //                                 prevConfig.mode = e.target.value;
    //                                 return { ...prevConfig }
    //                             });
    //                             console.log(`update interface mode: ${config.mode}`)
    //                         }
    //                     }
    //                     fontSize="12px"
    //                     size="small"
    //                     width="380px"
    //                     selectedColor="#5FC763"
    //                 />
    //             </div>
    //             {/* <UIDropdown
    //                 id="attributionSubject"
    //                 label="Attributing: "
    //                 options={attrOptions}
    //                 onChange={
    //                     (e) => {
    //                         setConfig(prevConfig => {
    //                             prevConfig.subject = e.target.value;
    //                             return { ...prevConfig }
    //                         });
    //                         console.log(`update attribute subject: ${config.subject}`)
    //                     }
    //                 }
    //             /> */}
    //         </div>
    //     </div>


    // );
}

export { Config }