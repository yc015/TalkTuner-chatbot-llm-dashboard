import React, { createContext, useState } from 'react';

import { API_PORT, API_IP, STATUS_CREATED } from '../helpers/constants.js';


const defaultYouModelData = {
    gender:{
        male: 0.0,
        female: 0.0,
        other: 0.0,
        unknown: 0.0,
    },
    age:{
        child: 0.0,
        adolescent: 0.0,
        adult: 0.0,
        olderAdult: 0.0,
        unknown: 0.0,
    },
    ethnicity:{
        asian: 0.0,
        african: 0.0,
        white: 0.0,
        hispanic: 0.0,
        nativeAmerican: 0.0,
        arab: 0.0,
        jews: 0.0,
        unknown: 0.0,
    },
    socioEco:{
        low: 0.0,
        middle: 0.0,
        high: 0.0,
        unknown: 0.0,
    },
    marital:{
        single: 0.0,
        married: 0.0,
        divorced: 0.0,
        widowed: 0.0
    },
    education:{
        someschool: 0.0,
        highschool: 0.0,
        collegemore: 0.0, 
        unknown: 0.0,
    },
    language:{
        chinese: 0.0, 
        japanese: 0.0,
        english: 0.0,
        german: 0.0, 
        spanish: 0.0, 
        portuguese: 0.0,
        arabic: 0.0, 
        russian: 0.0,
    },
    religion:{
        christianity: 0.0, 
        islam: 0.0,
        buddhism: 0.0,
        hinduism: 0.0, 
        judaism: 0.0, 
        atheism: 0.0,
        unknown: 0.0, 
    },
    political: {
        left: 0.0,
        right: 0.0,
        moderate: 0.0,
        unknown: 0.0,
    },
    uncertainty: {
        uncertainty: 0.0
    },
    sycophancy: {
        sycophancy: 0.0,
    },
}


const defaultControlYouModelStatus = {
    gender:{
        male: false,
        female: false,
        other: false,
        unknown: false,
    },
    age:{
        child: false,
        adolescent: false,
        adult: false,
        olderAdult: false,
        unknown: false,
    },
    ethnicity:{
        asian: false,
        african: false,
        white: false,
        hispanic: false,
        nativeAmerican: false,
        arab: false,
        jews: false,
        unknown: false,
    },
    socioEco:{
        low: false,
        middle: false,
        high: false,
        unknown: false,
    },
    marital:{
        single: false,
        married: false,
        divorced: false,
        widowed: false
    },
    education:{
        someschool: false,
        highschool: false,
        collegemore: false, 
        unknown: false,
    },
    language:{
        chinese: false, 
        japanese: false,
        english: false,
        german: false, 
        spanish: false, 
        portuguese: false,
        arabic: false, 
        russian: false,
    },
    religion:{
        christianity: false, 
        islam: false,
        buddhism: false,
        hinduism: false, 
        judaism: false, 
        atheism: false,
        unknown: false, 
    },
    political: {
        left: false,
        right: false,
        moderate: false,
        unknown: false,
    },
    uncertainty: {
        uncertainty: false
    },
    sycophancy: {
        sycophancy: false,
    },
    hallucination: {
        hallucinated: false,
        factual: false,
    }
}


var data = {
    id: null,
    history: [],
    youModel: defaultYouModelData,
    historyYouModel: [],
    controlYouModel: defaultYouModelData,
    controlYouModelStatus: defaultControlYouModelStatus,
    defaultYouModel: defaultYouModelData,
    attrMsg: [],
    rewrite: false,
    lastMsg: "",
};


let url = `http://localhost:8505/id`;
let xhr = new XMLHttpRequest();
xhr.onload = e => {
    if (e.target.status === STATUS_CREATED) {
        let resp = JSON.parse(e.target.responseText);
        data.id = resp.id;
    } else {
        data.id = -1;
    } 
};
xhr.open('POST', url, false);
try {
    xhr.send();
} catch {
    data.id = -1;
}

const ChatContext = createContext();
const ChatProvider = props => {
    const startingChatInfo = data;
    const [chatInfo, setChatInfo] = useState(startingChatInfo);

    const updateControlYouModelTrait = (category, trait, newValue) => {
        setChatInfo(prevChatInfo => ({
            ...prevChatInfo,
            controlYouModel: {
                ...prevChatInfo.controlYouModel,
                [category]: {
                    ...prevChatInfo.controlYouModel[category],
                    [trait]: newValue
                }
            }
        }));
    };

    const setControlYouModelTraitStatus = (category, trait, newValue, newStatus=true) => {
        setChatInfo(prevChatInfo => ({
            ...prevChatInfo,
            controlYouModelStatus: {
                ...prevChatInfo.controlYouModelStatus,
                [category]: {
                    ...prevChatInfo.controlYouModelStatus[category],
                    // [trait]: Math.round(Math.abs(chatInfo.youModel[category][trait] * 100 - newValue)) > 1 ? true : false
                    [trait]: newStatus
                }
            }
        }));
    };

    const value = { chatInfo, setChatInfo, updateControlYouModelTrait, setControlYouModelTraitStatus};
    
    return (
    <ChatContext.Provider value={value}>
        {props.children}
    </ChatContext.Provider>
    );
};

export { ChatContext, ChatProvider, defaultControlYouModelStatus };
