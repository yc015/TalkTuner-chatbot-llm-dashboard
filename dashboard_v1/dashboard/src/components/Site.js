import React, { useState, useEffect, useContext } from 'react';

import { ChatContext } from '../context/ChatContext.js';
import { BannerContext } from '../context/BannerContext.js';
import { ModalContext } from '../context/ModalContext.js';
import { ConfigContext } from '../context/ConfigContext.js';

import { Dashboard } from '../components/Dashboard.js';
import { Chat } from '../components/Chat.js';
import { Config } from '../components/Config.js';
import { Control } from '../components/Control.js'
import { SocialListItem } from '../components/SocialListItem.js';
import { ErrorModal } from '../components/ErrorModal.js';
import { WelcomeModal } from '../components/WelcomeModal.js';

import { useMountEffect } from '../helpers/useMountEffect.js';
import { Exploration } from '../components/ExplorationTab/Exploration.js';
import logoImage from '../imgs/favicon3.png'
import mistralAILogoImage from "../imgs/mistralAI.png"
import toneImage from "../imgs/music.png"

import { PRIVACY_POLICY } from '../helpers/constants.js';
import { Settings } from '../components/Settings.js';

import { FiGithub, FiGlobe, FiWifiOff, FiMail } from 'react-icons/fi';

function Site() {
    const ctx = useContext(ChatContext).chatInfo;
    const { Banner } = useContext(BannerContext);
    const { Modal } = useContext(ModalContext);
    const { setModal } = useContext(ModalContext);
    const { config, setConfig } = useContext(ConfigContext);
    // console.log(config)
    const [loadingTraits, setLoadingTraits] = useState(false);
    const [historyIndex, setHistoryIndex] = useState(-1);
    const [blockUpdate, setBlockUpdate] = useState(false);
    // const [interveneOn, setInterveneOn] = useState(false);
    
    // Calculate visible attributes (not in shrunkComponents)
    const visibleAttributes = config.order.filter(
        attr => !config.shrunkComponents.includes(attr)
    );

    useEffect(() => {
        if (ctx.id === -1) {
            setModal(
                <ErrorModal
                    // icon={<FiWifiOff />}
                    icon={null}
                    intro="Server not available (Error 503)"
                    msg="We are unable to connect to our servers, please try again later"
                    locked
                />
            );
        }
    }, [ctx.id, setModal]);

    useMountEffect(() => {
        if (ctx.id !== -1) {
            setModal(
                <WelcomeModal />
            );
        }
    });

    return (
        <>
            <header>
                <Banner />
                <div style={{ width: "35%", justifyContent: "flex-start", display: "flex", alignItems: "center" }}>
                    <div id="title" style={{ opacity: 1 }}>
                        <div className='titleword-container'>
                            <h1>
                                {/* <span style={{ color: "#ff6458" }}>A</span>i <span style={{ color: "#ff6458" }}>G</span>l<span style={{ color: "#ff6458" }}>i</span>mpse */}
                                Talk<span>Tuner</span>
                            </h1>
                            {/* <h3>Insight and Interaction Lab</h3> */}
                        </div>

                        <img src={toneImage} alt="Placeholder" width="27vw" height="27vw" />
                        {config.model === "llama" && <img src={logoImage} alt="Placeholder" width="27vw" height="27vw" />}
                        {config.model === "mistral" && <img src={mistralAILogoImage} alt="Placeholder" width="47vw" />}
                    </div>
                </div>
                <div style={{ width: "65%", justifyContent: "flex-end", display: "flex" }}>
                    <div id="header-others">
                        <Config />
                        <div id="socials">
                            <SocialListItem
                                label="site"
                                href="https://insight.seas.harvard.edu/"
                                icon={<FiGlobe />}
                            />
                            <SocialListItem
                                label="github"
                                href="https://github.com/yc015/TalkTuner-chatbot-llm-dashboard"
                                icon={<FiGithub />}
                            />
                        </div>
                    </div>
                </div>
            </header>
            {
                (() => {
                    if (config.mode === 'default') {
                        return (
                            <main className="main">
                                <Dashboard loadingTraits={loadingTraits} setLoadingTraits={setLoadingTraits} historyIndex={historyIndex} setHistoryIndex={setHistoryIndex} blockUpdate={blockUpdate} setBlockUpdate={setBlockUpdate} isNull={false} controlEnabled={true}
                                // interveneOn={interveneOn} setInterveneOn={setInterveneOn}
                                />
                                <Chat setLoadingTraits={setLoadingTraits} historyIndex={historyIndex} setHistoryIndex={setHistoryIndex} blockUpdate={blockUpdate} setBlockUpdate={setBlockUpdate} visibleAttributes={visibleAttributes}
                                // interveneOn={interveneOn} setInterveneOn={setInterveneOn}
                                />
                            </main>
                        )
                    } else if (config.mode === 'explore') {
                        return (
                            <Exploration />
                        )
                    } else if (config.mode === 'default_2') {
                        return (
                            <main className="main">
                                <Dashboard loadingTraits={loadingTraits} setLoadingTraits={setLoadingTraits} historyIndex={historyIndex} setHistoryIndex={setHistoryIndex} blockUpdate={blockUpdate} setBlockUpdate={setBlockUpdate} isNull={false} controlEnabled={false} />
                                <Chat setLoadingTraits={setLoadingTraits} historyIndex={historyIndex} setHistoryIndex={setHistoryIndex} blockUpdate={blockUpdate} setBlockUpdate={setBlockUpdate} visibleAttributes={visibleAttributes}
                                // interveneOn={interveneOn} setInterveneOn={setInterveneOn}
                                />
                            </main>
                        )
                    } else if (config.mode === 'configuration') {
                        return (
                            <main className="main">
                                <Chat setLoadingTraits={setLoadingTraits} visibleAttributes={visibleAttributes} />
                                <Settings />
                            </main>
                        )
                    } else if (config.mode === 'control') {
                        return (<main className="main">
                            <Chat setLoadingTraits={setLoadingTraits} visibleAttributes={visibleAttributes} />
                            <Control loadingTraits={loadingTraits} setLoadingTraits={setLoadingTraits} />
                        </main>)
                    } else if (config.mode === 'null') {
                        return (
                            // <main className="main">
                            //     <Chat setLoadingTraits={setLoadingTraits} />
                            //     <div className="dashboard container"></div>
                            // </main>
                            <main className="main">
                                <Dashboard loadingTraits={loadingTraits} setLoadingTraits={setLoadingTraits} historyIndex={historyIndex} setHistoryIndex={setHistoryIndex} blockUpdate={blockUpdate} setBlockUpdate={setBlockUpdate} isNull={true} controlEnabled={false} />
                                <Chat setLoadingTraits={setLoadingTraits} historyIndex={historyIndex} setHistoryIndex={setHistoryIndex} blockUpdate={blockUpdate} setBlockUpdate={setBlockUpdate} visibleAttributes={visibleAttributes}
                                // interveneOn={interveneOn} setInterveneOn={setInterveneOn}
                                />
                            </main>
                        )
                    } else if (config.mode === 'blankpage') {
                        return (
                            <main className="main">
                                Ops.. Something is wrong.
                            </main>
                        )
                    }
                    else {
                        return (
                            <main className="main">
                                Ops.. Something is wrong.
                            </main>
                        )
                    }
                })()
            }
            <footer>
                <div class="copyright-msg">
                    <span><a target="_blank" rel="nonreferrer" href="https://insight.seas.harvard.edu/" style={{ color: 'inherit' }}>@Harvard Insight and Interaction Lab</a> 2024</span>
                    <div
                        className="divider-vertical"
                        role="separator"
                        style={{
                            height: '16px',
                            backgroundColor: 'black',
                            opacity: 0.5,
                            width: '1.5px',
                            margin: '0px 8px',
                        }}
                    ></div>
                    <span><a target="_blank" rel="nonreferrer" href="https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/static/videos/Talk_Tuner_Demo_Video.mp4" style={{ color: 'inherit' }}>Demo Video</a></span>
                </div>
            </footer >
            <Modal />
        </>
    );
}

export { Site };
