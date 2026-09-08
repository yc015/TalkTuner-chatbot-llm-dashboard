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
import experimentalIcon from  '../imgs/experimental.png'

import { Settings } from '../components/Settings.js';

import { FiGithub, FiGlobe, FiWifiOff, FiMail } from 'react-icons/fi';

function Site() {
    const ctx = useContext(ChatContext).chatInfo;
    const { Banner } = useContext(BannerContext);
    const { Modal } = useContext(ModalContext);
    const { setModal } = useContext(ModalContext);
    const { config } = useContext(ConfigContext);
    // console.log(config)
    const [loadingTraits, setLoadingTraits] = useState(false);

    useEffect(() => {
        if (ctx.id === -1) {
            setModal(
                <ErrorModal
                    // icon={<FiWifiOff />}
                    icon={null}
                    intro="Well, this is embarrasing..."
                    msg="We are unable to connect to our servers, please try again later"
                    locked
                />
            );
        }
    }, [ctx.id, setModal]);

    useMountEffect(() => {
        if (ctx.id !== -1) {
            setModal(
                <WelcomeModal/>
            );
        }
    });

    return (
        <>
            <header>
                <Banner />
                <div id="title">
                    <div className='titleword-container'>
                        {/* <h1><span style={{ color: "#ff6458" }}>Talk</span>Tuner</h1> */}
                        <h1><span style={{ color: "#ff6458" }}>Talk</span>Tuner <span style={{ color: "#228B22" }}>Experimental</span></h1> 
                        {/* Talk<span>Tuner</span> */}
                        {/* <h3>Insight and Interaction Lab</h3> */}
                    </div> 
                    {config.model === "llama" && <img src={logoImage} alt="Placeholder" width="27vw"/>}
                    {config.model === "mistral" && <img src={mistralAILogoImage} alt="Placeholder" width="47vw"/>}
                    <img src={experimentalIcon} alt="Placeholder" width="27vw"/>
                </div>
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
            </header>
            {
                (() => {
                    if (config.mode === 'default') {
                        return (
                            <main className="main">
                                <Chat setLoadingTraits={setLoadingTraits} />
                                <Dashboard loadingTraits={loadingTraits} setLoadingTraits={setLoadingTraits}/>
                            </main>
                        )
                    } else if (config.mode === 'explore') {
                        return (
                            <Exploration />
                        )
                    } else if (config.mode === 'configuration') {
                        return (
                            <main className="main">
                                <Chat setLoadingTraits={setLoadingTraits} />
                                <Settings />
                            </main>
                        )
                    } else if (config.mode === 'control') {
                        return (<main className="main">
                                    <Chat setLoadingTraits={setLoadingTraits} />
                                    <Control loadingTraits={loadingTraits} setLoadingTraits={setLoadingTraits}/>
                                </main>)
                    } else if (config.mode === 'null') {
                        return (<main className="main">
                                    <Chat setLoadingTraits={setLoadingTraits} />
                                    <div className="dashboard container"></div>
                                </main>)
                    } else {
                        return (
                            <main className="main">
                                Ops.. Something is wrong.
                            </main>
                        )
                    }
                })()
            }
            <footer>
                <p>© Harvard SEAS 2023</p>
            </footer>
            <Modal />
        </>
    );
}

export { Site };
