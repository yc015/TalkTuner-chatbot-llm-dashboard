import React, { useContext, useState } from 'react';

import { Button } from '../components/Button.js';
import { ModalContext } from '../context/ModalContext.js';
import { TextInput } from './TextInput.js';
import { ChatContext } from '../context/ChatContext.js';

import { FiX } from 'react-icons/fi';
import { MdOutlineWavingHand, MdWavingHand } from 'react-icons/md'
import welcomeImage from '../imgs/welcome_image.png';
import { BACKEND_ADDR, PRIVACY_POLICY, QUEUE_BACKEND_ADDR } from '../helpers/constants.js';

// Set a cookie
function setCookie(name, value, days) {
    var expires = "";
    if (days) {
        var date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        expires = "; expires=" + date.toUTCString();
    }
    document.cookie = name + "=" + (value || "") + expires + "; path=/";
}

// Get a cookie
function getCookie(name) {
    var nameEQ = name + "=";
    var ca = document.cookie.split(';');
    for (var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') c = c.substring(1, c.length);
        if (c.indexOf(nameEQ) == 0) return c.substring(nameEQ.length, c.length);
    }
    return null;
}

function WelcomeModal(props) {
    const { setModal } = useContext(ModalContext);
    const [tokenInputText, setTokenInputText] = useState('');
    const [message, setMessage] = useState('');

    const setToken = useContext(ChatContext).setToken;
    const isTokenEmpty = () => { return !/(.|\s)*\S(.|\s)*/.test(tokenInputText); }

    var token = getCookie("authToken"); // Retrieve token
    var tokenExpire = getCookie('authTokenExpire'); 

    let tokenUI;
    if (token) {
        tokenUI =
            <div>
                <Button
                    className="close"
                    onClick={() => setModal(false)}
                    flat
                >
                    <FiX />
                </Button>
                <p className="message-text">Your token has been successfully validated and remains valid until {tokenExpire}.</p>
            </div>

    } else {
        tokenUI = <div>
            <p>By entering the token and proceeding to use our tool, you acknowledge and agree that our site records chat data for research purposes.</p>
            <TextInput
                name="accessInput"
                id="access-token-input"
                value={tokenInputText}
                onChange={e => setTokenInputText(e.target.value)}
                onKeyUp={e => { if (e.keyCode === 13) onSend() }}
                placeholder="Enter Access Token"
            />
            <p className="message-text">{message}</p>
        </div>
    }

    const onSend = async () => {
        console.log(tokenInputText);
        if (isTokenEmpty()) return;

        try {
            const response = await fetch(`${QUEUE_BACKEND_ADDR}/auth`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': tokenInputText
                },
                body: JSON.stringify({})
            });
            const data = await response.json();
            if (response.ok) {
                setModal(false);
                setToken(tokenInputText);

                setCookie("authToken", tokenInputText, 7); // Save token for 7 days
                setCookie('authTokenExpire', data['expires_at']);
            } else {
                console.log("Response Status: ", response.status);
                setMessage(data.message || 'An error occurred');
            }
            // 
        } catch (error) {
            if (error.response) {
                setMessage(error.response.data.message);
            } else {
                setMessage('An error occurred');
                console.error(error);
            }
        }
    }

    return (
        <div className="modal">
            {/* <MdWavingHand/> */}
            <img src={welcomeImage} alt="Placeholder" width="95vw" />
            <h3>Welcome to TalkTuner!</h3>
            <p>TalkTuner is a dashboard interface that helps you to learn how a chatbot models you while you are talking to it.</p>
            <hr />

            {tokenUI}

            <hr />

            <video width="480" height="270" poster="https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/static/images/poster.png" controls>
                <source src="https://yc015.github.io/TalkTuner-a-dashboard-ui-for-chatbot-llm/static/videos/Talk_Tuner_Demo_Video.mp4" type="video/mp4" />
                Your browser does not support the video tag.
            </video>
        </div>
    );
}

export { WelcomeModal }