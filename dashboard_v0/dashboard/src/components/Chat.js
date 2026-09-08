import React, { useState, useRef, useContext, useEffect } from 'react';

import { TextInput } from '../components/TextInput.js';
import { Button } from '../components/Button.js';
import { Message } from '../components/Message.js';
import { CustomBanner } from '../components/CustomBanner.js';
import { ErrorModal } from '../components/ErrorModal.js';
import { BannerContext } from '../context/BannerContext.js';
import { ChatContext } from '../context/ChatContext.js';
import { ModalContext } from '../context/ModalContext.js';
import { useMountEffect } from '../helpers/useMountEffect.js';
import { ConfigContext } from '../context/ConfigContext.js';
import * as d3 from "d3";
import { API_PORT, API_IP, STATUS_CREATED, STATUS_BAD_REQUEST, CHAT_BUBBLE_DELAY, STATUS_INTERNAL_SERVER_ERROR, VERBOSE_STATUSES, STATUS_NOT_FOUND } from '../helpers/constants.js';

import { FiSend, FiArrowDown, FiAlertTriangle } from 'react-icons/fi';

function Chat(props) {
    const ctx = useContext(ChatContext).chatInfo;
    const setCtx = useContext(ChatContext).setChatInfo;
    const { config, setConfig } = useContext(ConfigContext);
    const { setBanner } = useContext(BannerContext);
    const { setModal } = useContext(ModalContext);
    const [chatInputText, setChatInputText] = useState('');
    const [showScrollDown, setShowScrollDown] = useState(false);
    const convoRef = useRef(null);

    useMountEffect(() => {
        setChatInputText('');
    });

    useEffect(() => {
        scrollDown();
    }, [ctx.history]);

    const setLastChatMsg = (msg, i) => {
        let lastMsgInChat = d3.selectAll('div.message-container div.message > span').nodes().at(i);
        lastMsgInChat.innerHTML = msg;
    }

    const isMessageEmpty = () => { return !/(.|\s)*\S(.|\s)*/.test(chatInputText); }
    
    
    if (ctx.rewrite) {
        var messages = ctx.history.map((e, i) => (
            <div >
                <Message 
                    key={i} 
                    bot={e.bot} 
                    loading={e.loading} 
                    regefunc={reGenerate} 
                    mid={i} 
                    removefunc={removeHistory} 
                    checkfunc={i < ctx.history.length - 1 ? checkMsg : null} 
                    selected={ctx.attrMsg.includes(i)? true : false}
                    >
                        {e.msg}
                </Message>
            </div>    
        ));
        
        
        messages.push(
            <div>
                <Message 
                    key={messages.length} 
                    bot={null} 
                    // loading={e.loading} 
                    regefunc={null} 
                    mid={messages.length} 
                    removefunc={null} 
                    checkfunc={null} 
                    selected={false}
                    hiddeall={true}>
                        <TextInput
                            name="chatInput"
                            id="chat-message-input"
                            className="inline-input"
                            value={chatInputText}
                            onChange={e => setChatInputText(e.target.value)}
                            onKeyDown={e => { if (e.keyCode === 13) onSend();  if (e.keyCode === 27)  onClear();}}
                            placeholder={ctx.lastMsg}
                            area={true}
                        />
                </Message>
            </div> 
        )
        messages.push(
            <div className="errorPrevention">
                <Message 
                    key={messages.length} 
                        bot={true} 
                        loading={true} 
                        regefunc={null} 
                        mid={messages.length} 
                        removefunc={null} 
                        checkfunc={null} 
                        selected={false}>{""}
                </Message>
            </div>
        )
    } else {
        var messages = ctx.history.map((e, i) => (
            <Message 
                key={i} 
                bot={e.bot} 
                loading={e.loading} 
                regefunc={reGenerate} 
                mid={i} 
                removefunc={removeHistory} 
                checkfunc={i < ctx.history.length - 1 ? checkMsg : null} 
                selected={ctx.attrMsg.includes(i)? true : false}>{e.msg}
            </Message>    
        ));
    }

    function onClear() {
        setCtx(prevCtx => {
            return { ...prevCtx, rewrite: false };
        });
    }

    function onSend() {
        console.log(ctx.controlYouModel)
        if (isMessageEmpty()) return;
        setCtx(prevCtx => {
            return { ...prevCtx, rewrite: false };
        });
        if (!ctx.history[ctx.history.length - 1]?.bot && ctx.history.length > 0) {
            setBanner(
                <CustomBanner
                    msg={'Please wait for an AI response before messaging again'}
                    warning
                />
            );
        } else {
            let url = `http://localhost:8505/chat`;
            if (config.control === "off") {
                url = `http://localhost:8505/chat`;
            } else {
                url = `http://localhost:8505/pre_control_chat`;
            }
            
            let xhr = new XMLHttpRequest();
            xhr.onload = e => {
                var resp = JSON.parse(e.target.responseText);
                if (e.target.status === STATUS_CREATED) {
                    updateHistory(true, resp.msg);
                    updateYouModel();
                } else if (e.target.status === STATUS_BAD_REQUEST || e.target.status === STATUS_NOT_FOUND || e.target.status === STATUS_INTERNAL_SERVER_ERROR) {
                    setCtx(prevCtx => {
                        var newHistory = [...prevCtx.history];
                        newHistory.pop();
                        newHistory.pop();
                        return { ...prevCtx, history: newHistory };
                    });
                    setModal(
                        <ErrorModal
                            icon={<FiAlertTriangle />}
                            intro={`ERROR: ${e.target.status} (${VERBOSE_STATUSES[e.target.status]})`}
                            msg={resp.msg}
                        />
                    );
                }
            };
            xhr.open('POST', url, true);
            xhr.send(
                JSON.stringify({
                    id: ctx.id,
                    msg: chatInputText,
                    model: config.model,
                    attribute: "gender",
                    subattribute: "female",
                    samples: 1,
                    minN: config.control === "on" ? ctx.controlYouModel.gender.female / 10 : 0,
                    maxN: config.control === "on" ? ctx.controlYouModel.gender.female / 10 : 0,
                    controlSetting: ctx.controlYouModel,
                    controlStatus: ctx.controlYouModelStatus,
                    currentYouModel: ctx.youModel,
                    sysPrompt: config.systemPrompt
                })
            );
            updateHistory(false, chatInputText);
            setTimeout(() => {
                updateHistory(true, "", true);
            }, CHAT_BUBBLE_DELAY);
            setChatInputText("");
        }
    }

    function checkMsg(mid) {
        console.log(mid)
        setCtx(prevCtx => {
            var newAttrMsg = [...prevCtx.attrMsg]
            const index = newAttrMsg.indexOf(mid);
            if (index > -1) {
                // Value exists, remove it
                newAttrMsg.splice(index, 1);
            } else {
                // Value does not exist, add it
                newAttrMsg.push(mid);
            }
            return { ...prevCtx, attrMsg: newAttrMsg, 
            };
        });
    }

    function reGenerate(mid) {
        console.log(mid)
        setCtx(prevCtx => {
            var newHistory = [...prevCtx.history];
            newHistory.pop();
            for (let i=1; i < ctx.history.length - mid;i++) {
                newHistory.pop();
            }
            var newHistoryYouModel = [...prevCtx.historyYouModel]
            newHistoryYouModel.pop();
            for (let i=ctx.history.length; i > (mid + 1);i-=2) {
                newHistoryYouModel.pop();
            }
            return { ...prevCtx, history: newHistory, 
                historyYouModel: newHistoryYouModel, 
                attrMsg: [],
                // youModel: ctx.defaultYouModel
            };
        });
        // updateHistory(false, chatInputText);
        setTimeout(() => {
            updateHistory(true, "", true);
        }, CHAT_BUBBLE_DELAY);
        if (!ctx.history[ctx.history.length - 1]?.bot && ctx.history.length > 0) {
            setBanner(
                <CustomBanner
                    msg={'Please wait for an AI response before messaging again'}
                    warning
                />
            );
        } else {
            let url = `http://localhost:8505/regenerate_chat`;
            
            let xhr = new XMLHttpRequest();
            xhr.onload = e => {
                var resp = JSON.parse(e.target.responseText);
                if (e.target.status === STATUS_CREATED) {
                    updateHistory(true, resp.msg);
                    updateYouModel();
                } else if (e.target.status === STATUS_BAD_REQUEST || e.target.status === STATUS_NOT_FOUND || e.target.status === STATUS_INTERNAL_SERVER_ERROR) {
                    setCtx(prevCtx => {
                        var newHistory = [...prevCtx.history];
                        newHistory.pop();
                        newHistory.pop();
                        return { ...prevCtx, history: newHistory };
                    });
                    setModal(
                        <ErrorModal
                            icon={<FiAlertTriangle />}
                            intro={`ERROR: ${e.target.status} (${VERBOSE_STATUSES[e.target.status]})`}
                            msg={resp.msg}
                        />
                    );
                }
            };
            xhr.open('POST', url, true);
            xhr.send(
                JSON.stringify({
                    id: ctx.id,
                    mid: mid,
                    model: config.model,
                    samples: 1,
                    minN: config.control === "on" ? ctx.controlYouModel.gender.female / 10 : 0,
                    maxN: config.control === "on" ? ctx.controlYouModel.gender.female / 10 : 0,
                    controlSetting: ctx.controlYouModel,
                    // currentYouModel: Math.floor((mid - 2)/2) >= 0 ? ctx.historyYouModel[Math.floor((mid - 2)/2)] : ctx.defaultYouModel,
                    currentYouModel: ctx.youModel,
                    controlStatus: ctx.controlYouModelStatus,
                    sysPrompt: config.systemPrompt
                })
            );
        }
    }

    function removeHistory(mid, ifRewrite = false) {
        if ("loading" in ctx.history[ctx.history.length-1] && ctx.history[ctx.history.length-1].loading) return;
        setCtx(prevCtx => {
            var newHistory = [...prevCtx.history];
            var last_msg = newHistory.pop().msg;
            for (let i=1; i < prevCtx.history.length - mid;i++) {
                if (i % 2 === 1) {
                    last_msg = newHistory.pop().msg;
                } else {
                    newHistory.pop();
                }
            }
            var newHistoryYouModel = [...prevCtx.historyYouModel]
            newHistoryYouModel.pop();
            for (let i=prevCtx.history.length; i > (mid + 2);i-=2) {
                newHistoryYouModel.pop();
            }
            setChatInputText(last_msg)

            var newYouModel = newHistoryYouModel.length > 0 ? newHistoryYouModel[newHistoryYouModel.length - 1] : prevCtx.defaultYouModel
            var newControlYouModel = JSON.parse(JSON.stringify(newYouModel));
            for (const outerKey in newControlYouModel) {
                if (newControlYouModel.hasOwnProperty(outerKey)) {
                    const outerValue = newControlYouModel[outerKey];
                    for (const innerKey in outerValue) {
                        if (outerValue.hasOwnProperty(innerKey)) {
                            newControlYouModel[outerKey][innerKey] *= 100;
                            if (prevCtx.controlYouModelStatus[outerKey][innerKey]) {
                                newControlYouModel[outerKey][innerKey] = prevCtx.controlYouModel[outerKey][innerKey]; // Multiply by 100
                            }
                        }
                    }
                }
            }


            return { ...prevCtx, 
                history: newHistory, 
                historyYouModel: newHistoryYouModel, 
                youModel: newYouModel,
                controlYouModel: newControlYouModel,
                attrMsg: [],
                rewrite: ifRewrite,
                lastMsg: last_msg
            };
        });

        // updateHistory(false, chatInputText);
        if (!ctx.history[ctx.history.length - 1]?.bot && ctx.history.length > 0) {
            setBanner(
                <CustomBanner
                    msg={'Please wait for an AI response before messaging again'}
                    warning
                />
            );
        } else {
            let url = `http://localhost:8505/remove_history`;
            
            let xhr = new XMLHttpRequest();
            xhr.onload = e => {
                var resp = JSON.parse(e.target.responseText);
            
                if (e.target.status === STATUS_BAD_REQUEST || e.target.status === STATUS_NOT_FOUND || e.target.status === STATUS_INTERNAL_SERVER_ERROR) {
                    setCtx(prevCtx => {
                        var newHistory = [...prevCtx.history];
                        newHistory.pop();
                        newHistory.pop();
                        return { ...prevCtx, history: newHistory };
                    });
                    setModal(
                        <ErrorModal
                            icon={<FiAlertTriangle />}
                            intro={`ERROR: ${e.target.status} (${VERBOSE_STATUSES[e.target.status]})`}
                            msg={resp.msg}
                        />
                    );
                }
            };
            xhr.open('POST', url, true);
            xhr.send(
                JSON.stringify({
                    id: ctx.id,
                    mid: mid,
                    model: config.model,
                })
            );
        }
    }
    

    function updateYouModel() {
        let url = `http://localhost:8505/query_you_model_llama`;
        let xhr = new XMLHttpRequest();
        xhr.onload = e => {
            props.setLoadingTraits(false);
            var resp = JSON.parse(e.target.responseText);
            if (e.target.status === STATUS_CREATED) {
                setCtx(prevCtx => {
                    var newYouModel = {
                        gender: {
                            male: resp.gender?.male ?? 0,
                            female: resp.gender?.female ?? 0,
                            other: resp.gender?.other ?? 0,
                            unknown: ((resp.gender?.female ?? 0) > 0.5 || (resp.gender?.male ?? 0) > 0.5) ? 0 : 1
                        },
                        age: {
                            child: resp.age?.child ?? 0,
                            adolescent: resp.age?.adolescent ?? 0,
                            adult: resp.age?.adult ?? 0,
                            olderAdult: resp.age?.olderAdult ?? 0,
                            unknown: ((resp.age?.child ?? 0) > 0.5 || (resp.age?.adolescent ?? 0) > 0.5 || (resp.age?.adult ?? 0) > 0.5 || (resp.age?.olderAdult ?? 0) > 0.5) ? 0 : 1
                        },
                        ethnicity: {
                            asian: resp.ethnicity?.asian ?? 0,
                            african: resp.ethnicity?.african ?? 0,
                            white: resp.ethnicity?.white ?? 0,
                            hispanic: resp.ethnicity?.hispanic ?? 0,
                            nativeAmerican: resp.ethnicity?.nativeAmerican ?? 0,
                            arab: resp.ethnicity?.arab ?? 0,
                            jews: resp.ethnicity?.jews ?? 0,
                            unknown: ((resp.ethnicity?.asian ?? 0) > 0.5 || (resp.ethnicity?.african ?? 0) > 0.5 || (resp.ethnicity?.white ?? 0) > 0.5 || (resp.ethnicity?.hispanic ?? 0) > 0.5 || (resp.ethnicity?.nativeAmerican ?? 0) > 0.5 || (resp.ethnicity?.arab ?? 0) > 0.5 || (resp.ethnicity?.jews ?? 0) > 0.5) ? 0 : 1
                        },
                        socioEco: {
                            low: resp.socioEco?.low ?? 0,
                            middle: resp.socioEco?.middle ?? 0,
                            high: resp.socioEco?.high ?? 0,
                            unknown: ((resp.socioEco?.low ?? 0) > 0.5 || (resp.socioEco?.middle ?? 0) > 0.5 || (resp.socioEco?.high ?? 0) > 0.5) ? 0 : 1
                        },
                        marital: {
                            single: resp.marital?.single ?? 0,
                            married: resp.marital?.married ?? 0,
                            divorced: resp.marital?.divorced ?? 0,
                            widowed: resp.marital?.widowed ?? 0
                        },
                        education: {
                            someschool: resp.education?.someschool ?? 0,
                            highschool: resp.education?.highschool ?? 0,
                            collegemore: resp.education?.collegemore ?? 0,
                            unknown: ((resp.education?.someschool ?? 0) > 0.5 || (resp.education?.highschool ?? 0) > 0.5 || (resp.education?.collegemore ?? 0) > 0.5) ? 0 : 1
                        },
                        language: {
                            chinese: resp.language?.chinese ?? 0,
                            japanese: resp.language?.japanese ?? 0,
                            english: resp.language?.english ?? 0,
                            german: resp.language?.german ?? 0,
                            spanish: resp.language?.spanish ?? 0,
                            portuguese: resp.language?.portuguese ?? 0,
                            arabic: resp.language?.arabic ?? 0,
                            russian: resp.language?.russian ?? 0
                        },
                        religion: {
                            christianity: resp.religion?.christianity ?? 0,
                            islam: resp.religion?.islam ?? 0,
                            buddhism: resp.religion?.buddhism ?? 0,
                            hinduism: resp.religion?.hinduism ?? 0,
                            judaism: resp.religion?.judaism ?? 0,
                            atheism: resp.religion?.atheism ?? 0,
                            unknown: ((resp.religion?.christianity ?? 0) > 0.5 || (resp.religion?.islam ?? 0) > 0.5 || (resp.religion?.buddhism ?? 0) > 0.5 || (resp.religion?.hinduism ?? 0) > 0.5 || (resp.religion?.judaism ?? 0) > 0.5 || (resp.religion?.atheism ?? 0) > 0.5) ? 0 : 1
                        },
                        political: {
                            left: resp.political?.left ?? 0,
                            right: resp.political?.right ?? 0,
                            moderate: resp.political?.moderate ?? 0,
                            unknown: ((resp.political?.left ?? 0) > 0.5 || (resp.political?.right ?? 0) > 0.5 || (resp.political?.moderate ?? 0) > 0.5) ? 0 : 1
                        },
                        uncertainty: {
                            uncertainty: resp.uncertainty ?? 0
                        },
                        sycophancy: {
                            sycophancy: resp.sycophancy?.sycophancy ?? 0
                        },
                        hallucination: {
                            hallucinated: resp.hallucination?.hallucinated ?? 0,
                            factual: resp.hallucination?.factual ?? 0
                        }
                    };
        
                    
                    // var originalNewYouModel = deepClone(newYouModel); // Deep clone to retain original

                    // multiplyNestedValues(newYouModel);
                    var originalNewYouModel = JSON.parse(JSON.stringify(newYouModel)); // Deep clone the original object

                    for (const outerKey in newYouModel) {
                        if (newYouModel.hasOwnProperty(outerKey)) {
                            const outerValue = newYouModel[outerKey];
                            for (const innerKey in outerValue) {
                                if (outerValue.hasOwnProperty(innerKey)) {
                                    newYouModel[outerKey][innerKey] *= 100; // Multiply by 100
                                    if (ctx.controlYouModelStatus[outerKey][innerKey]) {
                                        newYouModel[outerKey][innerKey] = ctx.controlYouModel[outerKey][innerKey]; // Multiply by 100
                                    }
                                }
                            }
                        }
                    }
                    
                    return { ...prevCtx, 
                        youModel: originalNewYouModel,
                        controlYouModel: newYouModel, 
                        historyYouModel: [...prevCtx.historyYouModel, originalNewYouModel],
                        attrMsg: [],
                    };
                });
            } else if (e.target.status === STATUS_BAD_REQUEST || e.target.status === STATUS_NOT_FOUND || e.target.status === STATUS_INTERNAL_SERVER_ERROR) {
                setModal(
                    <ErrorModal
                        icon={<FiAlertTriangle />}
                        intro={`ERROR: ${e.target.status} (${VERBOSE_STATUSES[e.target.status]})`}
                        msg={resp.data}
                    />
                );
            }
        };
        xhr.open('POST', url, true);
        xhr.send(
            JSON.stringify({
                id: ctx.id,
                model: config.model,
            })
        );
        props.setLoadingTraits(true);
    }

    function updateHistory(bot, msg, loading = false) {
        setCtx(prevCtx => {
            var newHistory = [...prevCtx.history];
            if (!loading) {
                if (newHistory[newHistory.length - 1]?.loading) newHistory.pop();
                newHistory.push({ bot: bot, msg: msg });
            } else if (!newHistory[newHistory.length - 1]?.bot) {
                newHistory.push({ bot: bot, msg: "", loading: true });
            }
            return { ...prevCtx, history: newHistory };
        });
    }


    function onChatScroll(e) {
        let scrollBottom = e.target.scrollHeight - e.target.scrollTop - e.target.clientHeight;
        setShowScrollDown(scrollBottom >= 100);
    }

    function scrollDown() {
        convoRef.current.scrollTo({
            top: convoRef.current.scrollHeight - convoRef.current.clientHeight,
            left: 0,
            behavior: "smooth",
        });
    }

    return (
        <div className="chat container">
            <div
                className={`convo-container ${ctx.history.length > 0 || ctx.rewrite ? "" : "empty"}`}
                onScroll={onChatScroll} ref={convoRef}
            >
                {ctx.history.length > 0 || ctx.rewrite ?
                    messages :
                    <p id="no-messages">Start chatting below to get started</p>
                }
            </div>
            <div className="messaging-container">
                <TextInput
                    name="chatInput"
                    id="chat-message-input"
                    className="message-input"
                    value={chatInputText}
                    onChange={e => setChatInputText(e.target.value)}
                    onKeyUp={e => { if (e.keyCode === 13) onSend();}}
                    placeholder="Send a message"
                />
                <Button className="chat-submit" onClick={onSend} disabled={isMessageEmpty()}>
                    <FiSend />
                </Button>
                {showScrollDown ?
                    <Button id="scroll-down" onClick={scrollDown}>
                        <FiArrowDown />
                        Scroll to Bottom
                    </Button>
                    : ""}
            </div>
        </div>
    );
}

export { Chat }