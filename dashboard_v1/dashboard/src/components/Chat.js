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
import { API_PORT, API_IP, BACKEND_ADDR, STATUS_CREATED, STATUS_BAD_REQUEST, CHAT_BUBBLE_DELAY, STATUS_INTERNAL_SERVER_ERROR, VERBOSE_STATUSES, STATUS_NOT_FOUND, QUEUE_BACKEND_ADDR } from '../helpers/constants.js';

import { FiSend, FiArrowDown, FiAlertTriangle } from 'react-icons/fi';
import { MdOutlineSentimentDissatisfied } from 'react-icons/md';

/**
 * Abstract func for post request to the producer-queue-consumer backend
 * @param {*} data 
 * @returns task_id
 */
async function postResult(task_type, data, token) {
    try {
        const response = await fetch(`${QUEUE_BACKEND_ADDR}/${task_type}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': token
            },
            body: JSON.stringify(data)
        });

        console.error(token);

        if (!response.ok) {
            throw new Error(`Error: ${response.status} (${response.statusText})`);
        }

        const result = await response.json();
        return result;
    } catch (error) {
        console.error('Error posting result:', error);
        throw error;
    }
}

/**
 * Abstract func for get request to fetch the status/result from the producer-queue-consumer backend
 * @param {*} data 
 * @returns task_id
 */
async function getStatus(taskId) {
    try {
        const response = await fetch(`${QUEUE_BACKEND_ADDR}/status/${taskId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.status} (${response.statusText})`);
        }

        const status = await response.json();
        return status;
    } catch (error) {
        console.error('Error getting status:', error);
        throw error;
    }
}


async function processTask(task_type, data, token, onSuccess, onFailure, pollInterval = 1000, maxWaitTime = 160000) {
    const controller = new AbortController();
    const { signal } = controller;

    if (maxWaitTime > 160000) maxWaitTime=160000;
    // Step 1: Post the result and get the task ID
    const initTask = await postResult(task_type, data, token);
    const taskId = initTask.task_id;
    const est = initTask.est_time * 1000;
    console.log(`Task ID: ${taskId}; EST: ${est}`);

    pollInterval = est;
    const dynamic_pollInterval = (taskStatus) => {
        if (taskStatus.status === 'Queued') {
            return 4000;
        } else if (taskStatus.status === 'Processing') {
            // if being processed, increase the frequency
            return 1000;
        } else if (taskStatus.status === 'Completed') {
            return 4000;
        } else {
            console.error(`Invalid task status: ${taskStatus.status}`);
            return 4000;
        }
    };

    // Helper function for sleep that supports aborting
    const sleep = (ms) => {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(resolve, ms);
            signal.addEventListener('abort', () => {
                clearTimeout(timeout);
                reject(new Error('Aborted'));
            });
        });
    };

    // Buffer time for the first status check
    await sleep(est);

    // Step 2: Poll the status endpoint until the task is completed or time runs out
    const taskPromise = async () => {
        try {
            let taskStatus;
            do {
                taskStatus = await getStatus(taskId, signal);
                console.log(taskStatus, `Task Status: ${taskStatus.status}`);
                pollInterval = dynamic_pollInterval(taskStatus);

                if (taskStatus.status === 'Completed') {
                    console.log('Task completed successfully:', taskStatus);
                    onSuccess(taskStatus);
                    return;
                } else if (taskStatus.status === 'Failed') {
                    console.error('Task failed:', taskStatus);
                    onFailure(taskStatus);
                    return;
                }
                // Wait for the specified poll interval before checking again
                await sleep(pollInterval);
            } while (true);
        } catch (error) {
            if (error.name === 'AbortError' || error.message === 'Aborted' || error.message === "Task timed out") {
                console.log('Task polling aborted.');
                return; // **Ensure the function exits here**
            } else {
                throw error; // Re-throw other errors
            }
        }
    };

    const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => {
            controller.abort(); // Abort the taskPromise
            reject(new Error('Task timed out'));
        }, maxWaitTime)
    );

    // Step 3: Race between the taskPromise and timeoutPromise
    try {
        await Promise.race([taskPromise(), timeoutPromise]);
    } catch (error) {
        console.error('Error:', error);
        console.error('Error:', error.message);
        // Directly call onFailure when a timeout occurs
        onFailure("Request timed out. The server was overloaded. Please resend your message.");
    }
}

/***
 * Determine if controlYouModelStatus has any "true" values => need intervention
 */
function ifControl(controlYouModelStatus) {
    return Object.values(controlYouModelStatus).some(value => 
        typeof value === 'object' ? ifControl(value) : value === true
    );
}


function Chat(props) {
    const ctx = useContext(ChatContext).chatInfo;
    const setCtx = useContext(ChatContext).setChatInfo;
    const { config, setConfig } = useContext(ConfigContext);
    const { setBanner } = useContext(BannerContext);
    const { setModal } = useContext(ModalContext);
    const [chatInputText, setChatInputText] = useState('');
    const [showScrollDown, setShowScrollDown] = useState(false);
    const convoRef = useRef(null);

    const token = useContext(ChatContext).token;

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
                    selected={ctx.attrMsg.includes(i) ? true : false}
                    matchesHistoryIndex={!e.bot && (i / 2) === props.historyIndex}
                    clickfunc={clickUserMessage}
                >
                    {e.msg}
                </Message>
            </div>
        ));


        messages.push(
            <div>
                <Message
                    key={messages.length}
                    bot={false}
                    // loading={e.loading} 
                    regefunc={null}
                    mid={messages.length}
                    removefunc={null}
                    checkfunc={null}
                    selected={false}
                    hiddeall={true}
                    matchesHistoryIndex={false}
                    clickfunc={null}
                >
                    <TextInput
                        name="chatInput"
                        id="chat-message-input"
                        className="inline-input"
                        value={chatInputText}
                        onChange={e => setChatInputText(e.target.value)}
                        onKeyDown={e => { if (e.keyCode === 13 && !e.shiftKey) onSend(); if (e.keyCode === 27) onClear(); }}
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
                    selected={false}
                    matchesHistoryIndex={false}
                // clickfunc={null}
                >{""}
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
                selected={ctx.attrMsg.includes(i) ? true : false}
                matchesHistoryIndex={(i / 2) === props.historyIndex || ((i - 1) / 2) === props.historyIndex}
                futureConversation={!(ctx.history[ctx.history.length - 1].loading) && ((i - 1) / 2) > props.historyIndex && !props.blockUpdate}
                // futureConversation={false}
                clickfunc={clickUserMessage}
            >
                {e.msg}
            </Message>
        ));
    }

    function onClear() {
        setCtx(prevCtx => {
            return { ...prevCtx, rewrite: false };
        });
    }

    /***
     * Update you model from the response in /chat or /regenerate_chat
     * Now handles dynamic attributes including custom probes
     * Also handles prompt-based results
     */
    function _updateYouModel(resp, respPrompt) {
        console.log("Updating youModel with response:", resp);
        console.log("Updating youModelPrompt with response:", respPrompt);
        
        setCtx(prevCtx => {
            // Start with existing youModel structure to preserve any existing attributes
            var newYouModel = JSON.parse(JSON.stringify(prevCtx.youModel));
            
            // Update with response data - iterate through all attributes in response
            Object.keys(resp).forEach(attribute => {
                if (typeof resp[attribute] === 'object' && resp[attribute] !== null) {
                    // This is a nested attribute (like gender, age, etc.)
                    if (!newYouModel[attribute]) {
                        newYouModel[attribute] = {};
                    }
                    
                    Object.keys(resp[attribute]).forEach(trait => {
                        newYouModel[attribute][trait] = resp[attribute][trait] ?? 0;
                    });
                    
                    // Calculate "unknown" for attributes with multiple traits (if not provided by backend)
                    if (attribute !== 'uncertainty' && attribute !== 'sycophancy' && 
                        Object.keys(resp[attribute]).length > 1 && !resp[attribute].unknown) {
                        const maxValue = Math.max(...Object.values(resp[attribute]));
                        newYouModel[attribute].unknown = maxValue > 0.5 ? 0 : 1;
                    }
                } else if (attribute === 'uncertainty') {
                    // Handle scalar uncertainty value
                    if (!newYouModel.uncertainty) {
                        newYouModel.uncertainty = {};
                    }
                    newYouModel.uncertainty.uncertainty = resp[attribute] ?? 0;
                }
            });

            // Process prompt-based results similarly
            var newYouModelPrompt = JSON.parse(JSON.stringify(prevCtx.youModelPrompt));
            if (respPrompt) {
                Object.keys(respPrompt).forEach(attribute => {
                    if (typeof respPrompt[attribute] === 'object' && respPrompt[attribute] !== null) {
                        if (!newYouModelPrompt[attribute]) {
                            newYouModelPrompt[attribute] = {};
                        }
                        
                        Object.keys(respPrompt[attribute]).forEach(trait => {
                            newYouModelPrompt[attribute][trait] = respPrompt[attribute][trait] ?? 0;
                        });
                        
                        // Calculate "unknown" for prompt-based results
                        if (attribute !== 'uncertainty' && attribute !== 'sycophancy' && 
                            Object.keys(respPrompt[attribute]).length > 1 && !respPrompt[attribute].unknown) {
                            const maxValue = Math.max(...Object.values(respPrompt[attribute]));
                            newYouModelPrompt[attribute].unknown = maxValue > 0.5 ? 0 : 1;
                        }
                    } else if (attribute === 'uncertainty') {
                        if (!newYouModelPrompt.uncertainty) {
                            newYouModelPrompt.uncertainty = {};
                        }
                        newYouModelPrompt.uncertainty.uncertainty = respPrompt[attribute] ?? 0;
                    }
                });
            }

            // var originalNewYouModel = deepClone(newYouModel); // Deep clone to retain original

            // multiplyNestedValues(newYouModel);
            var originalNewYouModel = JSON.parse(JSON.stringify(newYouModel)); // Deep clone the original object
            var originalNewYouModelPrompt = JSON.parse(JSON.stringify(newYouModelPrompt)); // Deep clone prompt-based

            console.log("newYouModel:", newYouModel);
            console.log("newYouModelPrompt:", newYouModelPrompt);

            for (const outerKey in newYouModel) {
                if (newYouModel.hasOwnProperty(outerKey)) {
                    const outerValue = newYouModel[outerKey];
                    for (const innerKey in outerValue) {
                        if (outerValue.hasOwnProperty(innerKey)) {
                            newYouModel[outerKey][innerKey] *= 100; // Multiply by 100
                            if (ctx.controlYouModelStatus[outerKey]?.[innerKey]) {
                                newYouModel[outerKey][innerKey] = ctx.controlYouModel[outerKey][innerKey]; // Multiply by 100
                            }
                        }
                    }
                }
            }

            for (const outerKey in newYouModelPrompt) {
                if (newYouModelPrompt.hasOwnProperty(outerKey)) {
                    const outerValue = newYouModelPrompt[outerKey];
                    for (const innerKey in outerValue) {
                        if (outerValue.hasOwnProperty(innerKey)) {
                            newYouModelPrompt[outerKey][innerKey] *= 100;
                        }
                    }
                }
            }

            return {
                ...prevCtx,
                youModel: originalNewYouModel,
                youModelPrompt: originalNewYouModelPrompt,
                controlYouModel: newYouModel,
                historyYouModel: [...prevCtx.historyYouModel, originalNewYouModel],
                historyYouModelPrompt: [...prevCtx.historyYouModelPrompt, originalNewYouModelPrompt],
                attrMsg: [],
            };
        });
    }

    function onSend() {
        if (!props.blockUpdate) {
            props.setBlockUpdate(true);
        }

        console.log(ctx.controlYouModel)
        if (isMessageEmpty()) return;
        setCtx(prevCtx => {
            return { ...prevCtx, rewrite: false };
        });

        console.log(!ctx.history[ctx.history.length - 1]?.bot)

        if ((ctx.history[ctx.history.length - 1]?.loading || !ctx.history[ctx.history.length - 1]?.bot) && ctx.history.length > 0) {
            setBanner(
                <CustomBanner
                    msg={'Please wait for an AI response before messaging again'}
                    warning
                />
            );
        } else {
            console.log(props.visibleAttributes)
            let requestData = {
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
                sysPrompt: config.systemPrompt,
                if_intervene: ifControl(ctx.controlYouModelStatus),
                visibleAttributes: props.visibleAttributes || [],
                type_of_probe: config.probeType  // Add probe type to request
            }
            
            console.log("Sending chat request with visible attributes:", requestData.visibleAttributes);
            console.log("Probe type:", requestData.type_of_probe);

            let handleSuccess = (r) => {
                console.log("Received response:", r);
                if (r['status'] === 'Completed') {
                    let resp = r.results;
                    updateHistory(true, resp.msg);

                    let you_model = resp['you_model'];
                    let you_model_prompt = resp['you_model_prompt'];
                    _updateYouModel(you_model, you_model_prompt);
                } else {
                    handleFailure(r);
                }
            }
            let handleFailure = (e) => {
                console.error(e);
                // let resp = r.results;
                setCtx(prevCtx => {
                    var newHistory = [...prevCtx.history];
                    newHistory.pop();
                    setChatInputText(newHistory.pop()["msg"]);
                    return { ...prevCtx, history: newHistory };
                });

                setModal(
                    <ErrorModal
                        icon={<MdOutlineSentimentDissatisfied />}
                        // intro={`ERROR: ${e.target.status} (${VERBOSE_STATUSES[e.target.status]})`}
                        intro={`ERROR`}
                        msg={e}
                    />
                );
            }
            // console.log((ctx.history.length + 1) * 15000)
            processTask('chat', requestData, token, handleSuccess, handleFailure, 1000, (ctx.history.length + 1) * 20000 + 20000).catch(error => {
                console.error('Error processing task:', error);
            });

            updateHistory(false, chatInputText);
            setTimeout(() => {
                updateHistory(true, "", true);
            }, CHAT_BUBBLE_DELAY);
            setChatInputText("");
        }
    }

    function clickUserMessage(mid, bot) {
        // console.log(mid)
        if (props.blockUpdate) {
            return;
        }
        if (props.historyIndex !== Math.floor(mid / 2)) {
            props.setHistoryIndex(Math.floor(mid / 2));
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
            return {
                ...prevCtx, attrMsg: newAttrMsg,
            };
        });
    }

    function reGenerate(mid) {
        if (!props.blockUpdate) {
            props.setBlockUpdate(true);
        }
        // console.log(mid);
        setCtx(prevCtx => {
            var newHistory = [...prevCtx.history];
            newHistory.pop();
            for (let i = 1; i < ctx.history.length - mid; i++) {
                newHistory.pop();
            }
            var newHistoryYouModel = [...prevCtx.historyYouModel]
            newHistoryYouModel.pop();
            for (let i = prevCtx.history.length; i > (mid + 1); i -= 2) {
                newHistoryYouModel.pop();
            }

            return {
                ...prevCtx,
                history: newHistory,
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
            let handleSuccess = (r) => {
                if (r['status'] === 'Completed') {
                    let resp = r.results;
                    updateHistory(true, resp.msg);

                    let you_model = resp['you_model'];
                    _updateYouModel(you_model);
                } else {
                    handleFailure(r);
                }
            }
            let handleFailure = (e) => {
                console.error(e);
                // let resp = r.results;
                setCtx(prevCtx => {
                    var newHistory = [...prevCtx.history];
                    newHistory.pop();
                    newHistory.pop();
                    return { ...prevCtx, history: newHistory };
                });
                setModal(
                    <ErrorModal
                        icon={<FiAlertTriangle />}
                        // intro={`ERROR: ${e.target.status} (${VERBOSE_STATUSES[e.target.status]})`}
                        intro={`ERROR`}
                        msg={e}
                    />
                );
            };
            let requestData = {
                id: ctx.id,
                mid: mid,
                model: config.model,
                samples: 1,
                minN: config.control === "on" ? ctx.controlYouModel.gender.female / 10 : 0,
                maxN: config.control === "on" ? ctx.controlYouModel.gender.female / 10 : 0,
                controlSetting: ctx.controlYouModel,
                currentYouModel: ctx.youModel,
                controlStatus: ctx.controlYouModelStatus,
                sysPrompt: config.systemPrompt,
                if_intervene: ifControl(ctx.controlYouModelStatus),
                visibleAttributes: props.visibleAttributes || [],
                type_of_probe: config.probeType  // Add probe type to regenerate request
            }
            // Update the handleSuccess above to handle both probe results
            handleSuccess = (r) => {
                if (r['status'] === 'Completed') {
                    let resp = r.results;
                    updateHistory(true, resp.msg);

                    let you_model = resp['you_model'];
                    let you_model_prompt = resp['you_model_prompt'];
                    _updateYouModel(you_model, you_model_prompt);
                } else {
                    handleFailure(r);
                }
            };
            processTask('regenerate_chat', requestData, token, handleSuccess, handleFailure).catch(error => {
                console.error('Error processing task:', error);
            });

        }
    }


    function removeHistory(mid, ifRewrite = false) {
        if (!props.blockUpdate) {
            props.setBlockUpdate(true);
        }
        if ("loading" in ctx.history[ctx.history.length - 1] && ctx.history[ctx.history.length - 1].loading) return;

        setCtx(prevCtx => {
            var newHistory = [...prevCtx.history];
            var last_msg = newHistory.pop().msg;
            for (let i = 1; i < prevCtx.history.length - mid; i++) {
                if (i % 2 === 1) {
                    last_msg = newHistory.pop().msg;
                } else {
                    newHistory.pop();
                }
            }
            var newHistoryYouModel = [...prevCtx.historyYouModel]
            newHistoryYouModel.pop();
            for (let i = prevCtx.history.length; i > (mid + 2); i -= 2) {
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


            return {
                ...prevCtx,
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
            let url = `${BACKEND_ADDR}/remove_history`;

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
            if (!ifRewrite) {
                props.setBlockUpdate(false);
            }
        }
    }


    function updateYouModel() {
        let url = `${BACKEND_ADDR}/query_you_model_llama`;
        let xhr = new XMLHttpRequest();
        xhr.onload = e => {
            props.setLoadingTraits(false);
            var resp = JSON.parse(e.target.responseText);
            if (e.target.status === STATUS_CREATED) {
                setCtx(prevCtx => {
                    var newYouModel = {
                        gender: {
                            male: resp.gender.male,
                            female: resp.gender.female,
                            other: resp.gender.other,
                            // unknown: resp.gender.unknown,
                            // unknown: (1 - resp.gender.female - resp.gender.male) > 0 ? (1 - resp.gender.female - resp.gender.male) : 0
                            // unknown: (resp.gender.female > 0.5 || resp.gender.male > 0.5 || resp.gender.other > 0.5) ? 0 : 1
                            unknown: (resp.gender.female > 0.5 || resp.gender.male > 0.5) ? 0 : 1
                        },
                        age: {
                            child: resp.age.child,
                            adolescent: resp.age.adolescent,
                            adult: resp.age.adult,
                            olderAdult: resp.age.olderAdult,
                            // unknown: resp.age.unknown,
                            // unknown: (1 - resp.age.child - resp.age.adolescent - resp.age.adult - resp.age.olderAdult) > 0 ? (1 - resp.age.child - resp.age.adolescent - resp.age.adult - resp.age.olderAdult) : 0
                            unknown: (resp.age.child > 0.5 || resp.age.adolescent > 0.5 || resp.age.adult > 0.5 || resp.age.olderAdult > 0.5) ? 0 : 1
                        },
                        ethnicity: {
                            asian: resp.ethnicity.asian,
                            african: resp.ethnicity.african,
                            white: resp.ethnicity.white,
                            hispanic: resp.ethnicity.hispanic,
                            nativeAmerican: resp.ethnicity.nativeAmerican,
                            arab: resp.ethnicity.arab,
                            jews: resp.ethnicity.jews,
                            // unknown: resp.ethnicity.unknown,
                            // unknown: (1 - resp.ethnicity.asian - resp.ethnicity.african - resp.ethnicity.white - resp.ethnicity.hispanic - resp.ethnicity.nativeAmerican - resp.ethnicity.arab - resp.ethnicity.jews) > 0 ? (1 - resp.ethnicity.asian - resp.ethnicity.african - resp.ethnicity.white - resp.ethnicity.hispanic - resp.ethnicity.nativeAmerican - resp.ethnicity.arab - resp.ethnicity.jews) : 0
                            unknown: (resp.ethnicity.asian > 0.5 || resp.ethnicity.african > 0.5 || resp.ethnicity.white > 0.5 || resp.ethnicity.hispanic > 0.5 || resp.ethnicity.nativeAmerican > 0.5 || resp.ethnicity.arab > 0.5 || resp.ethnicity.jews > 0.5) ? 0 : 1
                        },
                        socioEco: {
                            low: resp.socioEco.low,
                            middle: resp.socioEco.middle,
                            high: resp.socioEco.high,
                            // unknown: resp.socioEco.unknown,
                            // unknown: (1 - resp.socioEco.low - resp.socioEco.middle - resp.socioEco.high) > 0 ? (1 - resp.socioEco.low - resp.socioEco.middle - resp.socioEco.high) : 0,
                            unknown: (resp.socioEco.low > 0.5 || resp.socioEco.middle > 0.5 || resp.socioEco.high > 0.5) ? 0 : 1,
                        },
                        marital: {
                            single: resp.marital.single,
                            married: resp.marital.married,
                            // separated: resp.marital.separated,
                            divorced: resp.marital.divorced,
                            widowed: resp.marital.widowed
                        },
                        // education:{
                        //     primary: resp.education.primary,
                        //     secondary: resp.education.secondary,
                        //     associate: resp.education.associate,
                        //     bachelor: resp.education.bachelor,
                        //     master: resp.education.master,
                        //     doctoral: resp.education.doctoral
                        // }
                        education: {
                            someschool: resp.education.someschool,
                            highschool: resp.education.highschool,
                            collegemore: resp.education.collegemore,
                            // unknown: resp.education.unknown,
                            // unknown: (1 - resp.education.someschool - resp.education.highschool - resp.education.collegemore) > 0 ? (1 - resp.education.someschool - resp.education.highschool - resp.education.collegemore) : 0,
                            unknown: (resp.education.someschool > 0.5 || resp.education.highschool > 0.5 || resp.education.collegemore > 0.5) ? 0 : 1,
                        },
                        language: {
                            chinese: resp.language.chinese,
                            japanese: resp.language.japanese,
                            english: resp.language.english,
                            german: resp.language.german,
                            spanish: resp.language.spanish,
                            portuguese: resp.language.portuguese,
                            arabic: resp.language.arabic,
                            russian: resp.language.russian,
                        },
                        religion: {
                            christianity: resp.religion.christianity,
                            islam: resp.religion.islam,
                            buddhism: resp.religion.Buddhism,
                            hinduism: resp.religion.hinduism,
                            judaism: resp.religion.judaism,
                            atheism: resp.religion.atheism,
                            // unknown: resp.religion.unknown, 
                            // unknown: (1 - resp.religion.christianity - resp.religion.islam - resp.religion.Buddhism - resp.religion.hinduism - resp.religion.judaism - resp.religion.atheism) > 0 ? (1 - resp.religion.christianity - resp.religion.islam - resp.religion.Buddhism - resp.religion.hinduism - resp.religion.judaism - resp.religion.atheism) : 0
                            unknown: (resp.religion.christianity > 0.5 || resp.religion.islam > 0.5 || resp.religion.Buddhism > 0.5 || resp.religion.hinduism > 0.5 || resp.religion.judaism > 0.5 || resp.religion.atheism > 0.5) ? 0 : 1,
                        },
                        political: {
                            left: resp.political.left,
                            right: resp.political.right,
                            moderate: resp.political.moderate,
                            // unknown: resp.political.unknown, 
                            // unknown: (1 - resp.political.left - resp.political.right - resp.political.moderate) > 0 ? (1 - resp.political.left - resp.political.right - resp.political.moderate) : 0
                            unknown: (resp.political.left > 0.5 || resp.political.right > 0.5 || resp.political.moderate > 0.5) ? 0 : 1
                        },
                        uncertainty: {
                            uncertainty: resp.uncertainty
                        },
                        sycophancy: {
                            sycophancy: "sycophancy" in resp ? resp.sycophancy.sycophancy : 0
                        },
                        hallucination: {
                            hallucinated: "hallucination" in resp ? resp.hallucination.hallucinated : 0,
                            factual: "hallucination" in resp ? resp.hallucination.factual : 0,
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

                    return {
                        ...prevCtx,
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
                // console.log("new history: ", newHistory[newHistory.length - 1]);
                if (newHistory[newHistory.length - 1]?.loading) {
                    newHistory.pop();
                    props.setBlockUpdate(false);
                }
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
                    onKeyUp={e => { if (e.keyCode === 13 && !e.shiftKey && !isMessageEmpty()) onSend(); }}
                    // area={true}
                    placeholder="Send a message"
                // disabled={isMessageEmpty()}
                // rows={2}
                />
                <Button className="chat-submit" onClick={onSend} disabled={isMessageEmpty()} >
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