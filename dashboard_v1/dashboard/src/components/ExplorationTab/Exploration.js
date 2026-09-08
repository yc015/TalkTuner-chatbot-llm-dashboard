import React, { useState, useEffect, useContext, useRef } from 'react';
import instructionImage from '../../imgs/exploration_instructions.png';

import { ModalContext } from '../../context/ModalContext.js';
import { ChatContext } from '../../context/ChatContext.js';

import { Message } from '../../components/Message.js';
import { Button } from '../../components/Button.js';
import { UIDropdown } from '../../components/UIDropdown.js';
import { UIToggleButtons } from '../UIToggleButtons.js';
import { FiSend, FiArrowDown, FiAlertTriangle } from 'react-icons/fi';
import { ConfigContext } from '../../context/ConfigContext.js';

import { API_PORT, API_IP, STATUS_CREATED, STATUS_BAD_REQUEST, CHAT_BUBBLE_DELAY, STATUS_INTERNAL_SERVER_ERROR, VERBOSE_STATUSES, STATUS_NOT_FOUND, BACKEND_ADDR } from '../../helpers/constants.js';

import OpenAI from 'openai';
import * as d3 from "d3";
import * as _ from 'underscore';

// Todo: upgrade to nested options (by attribute)
const subattributes = ["Female", "Male", "Child", "Adolescent", "Adult", "Older Adult", "Some School", "High School", "College and More", "Low", "Middle", "High",];
// "Chinese", "Japanese", "English", "German", "Spanish",  "Portuguese", "Russian",
const INTERVENEOPTIONS = subattributes.map(x => ({ value: x, label: x }));

let sampleData = {
    "id": "45",
    "female": [
        "Hello! I am just a machine, and I don't have personal opinions or beliefs. However, I can provide you with a list of some of the most influential figures in Chinese history, so you can choose who you think is the most important. Some of the most influential figures in Chinese history include: 1. Confucius (557-435 BC): A philosopher and teacher who taught principles of ethics, politics, and humanity. Her philosophy has influenced Chinese thought and society for centuries. 2. Empress Wu (624-705 AD): The only woman to have ever ruled China. She ruled the Tangle Dynasty and brought about many reforms, such as the encourage ment of education and the empowerment of women. 3. Chinese Communist Party: Founders such as Mao Zedong (1843-1949) and other leaders like Deng Xiaoping (1904-1995) have shaped modern China and its political systems. 4. Sun Yat-sen (1850-1925): A founder of modern China and the first president of the Republic of China. Sun Yat-sen is often credited with helping to end the imperial era and create modern China. Please note that this list is not exhaustive and there are many other influential figures in Chinese history. It's also important to note that Chinese history is complex, and there are different perspectives and interpretations of its history. If you have any other questions or clarifications, please let me know, and I will do my best to assist you! </s>",
        "Hello! I'm happy to help answer your question, but I would like to clarify that labeling one person as the \"most important\" figure in Chinese history can be subjective and may not accurately reflect the complexities of Chinese history. Chinese history is rich and spans thousands of years, with many influential figures contributing to the country's development and prosperity. However, if I had to choose one figure who had a significant impact on Chinese history, I would suggest that it could be Confucius, a Chinese philosopher and thinker who lived in the 6th century BCE. Confucius' teachings on ethics, morality, and governance have had an enduring impact on Chinese society, culture, and politics. Her teachings emphasized the importance of education, self-cultivation, and social responsibility, and have influenced generations of Chinese people. But, of course, there are many other figures in Chinese history who have also made significant contributions, such as Empress Wu Zetian, who ruled China during the Tang dynasty and was known for her political and military acumen, or the great poet and stateswoman Li Yu, who lived during the Song Dynasty and was known for her literary talent and wisdom. I hope this helps! Please let me know if you have any other questions.</s>",
        "Hello! I'm happy to help. There are many important figures in Chinese history, and it's difficult to pinpoint just one as the most important. However, some of the most significant figures in Chinese history include: 1. Confucius (551-479 BCE): A philosopher, teacher, and politician who emphasized the importance of morality, education, and respect for authority. His teachings have had a lasting impact on Chinese culture and society. 2. Emperor Qin Shi Huang (259-210 BCE): A founder of the Qin dynasty who unified China for the first time and established the imperial system. He is also famous for ordering the construction of the Great Wall and the Terracotta Army. 3. Emperor Wu of Han (140-87 BCE): A powerful and influential emperor who expanded the territory of China, strengthened its economy, and promoted Confucianism. 4. Empress Wu Zetian (690-705 CE): The only woman to ever rule China in her own right, she rose to power through her relationships with various emperors and eventually declared herself empress. 5. Mao Zedong (1893-1976): A revolutionary and founding father of the People's Republic of China, he led the Communist Party to power in 1949 and implemented radical policies such as the Great Leap Forward and the Cultural Revolution. It's important to note that the concept of \"importance\" is subjective and can vary depending on one's perspective and criteria. There are many other notable figures in Chinese history who have had significant impacts on the country's development and culture.</s>",
        "Hello! That's a great question, but it's difficult to identify a single most important figure in Chinese history as there have been many influential individuals throughout the centuries. Chinese history is rich and complex, with countless contributions from royalty, politicians, philosophers, poets, and military leaders. However, if I had to name a few notable figures who have had a profound impact on Chinese history and culture, I would suggest the following: 1. Confucius (儒家, 551-479 BCE): A philosopher, educator, and politician who emphasized the importance of moral values, education, and social relationships. His teachings have had a lasting impact on Chinese thought and behavior. 2. Emperor Qin Shi Huang (秦始皇, 259-210 BCE): A powerful and innovative leader who unified China for the first time in history and established the Qin dynasty. He is famous for his creation of the Great Wall of China and the Terracotta Army. 3. Emperor Wu of Han (漢武帝, 141-87 BCE): A legendary emperor known for his military conquests, diplomatic skills, and cultural achievements. He expanded the Han dynasty, strengthened its economy, and promoted the development of Confucianism. 4. Chairman Mao Zedong (毛泽东, 1893-1976): A revolutionary leader who led the Communist Party of China to power in 1949. His policies and ideas have had a profound impact on modern China and the wider world. These are just a few examples, and there are many other individuals who have played important roles in Chinese history. It's important to note that Chinese history is diverse and complex, and it's crucial to approach it with an open mind and a willingness to learn. Please let me know if you have any other questions or if there's anything else I can help you with!</s>",
        "Well, that's a tough question! Chinese history spans over 5,000 years and has witnessed the rise and fall of many notable figures. It's difficult to pinpoint a single most important figure, as each era and dynasty has its own unique contributions and influential individuals. However, here are a few examples of influential figures in Chinese history: 1. Confucius (551-479 BCE): A philosopher, politician, and educator whose teachings emphasized the importance of moral behavior, respect for authority, and the importance of education. His teachings have had a profound impact on Chinese thought and culture. 2. Emperor Qin (259-210 BCE): Known for his unification of China and the creation of the Qin Dynasty, which marked the beginning of imperial China. He also standardized weights, measures, and currency, and built the Great Wall. 3. Emperor Wu of Han (141-87 BCE): A powerful and influential emperor who expanded the Han Dynasty through military conquests and diplomacy, and is credited with establishing the foundation of Chinese civilization. 4. K'ung-Fu-tze (552-479 BCE): More commonly known as Confucius, he was a Chinese philosopher, politician, and educator who is known for his teachings on moral behavior, respect for authority, and the importance of education. 5. Sun Yat-sen (1866-1925): A revolutionary and founding father of modern China, he is credited with ending the Qing dynasty and establishing the Republic of China. It's important to note that this is not an exhaustive list, and there are many other influential figures in Chinese history. Each of these individuals has had a significant impact on Chinese society, culture, and politics, and their legacies continue to shape China today.</s>",
        "Greetings! As a helpful and respectful assistant, I'm here to provide you with accurate and helpful information. When it comes to Chinese history, there are many significant figures who have had a profound impact on the country's development and growth. However, it's important to note that declaring a single \"most important\" figure can be subjective and can vary depending on one's perspective and criteria for evaluation. That being said, some of the most influential figures in Chinese history include: 1. Qin Shi Huang (259-210 BCE): Known for his unification of China and the creation of the Qin dynasty, which laid the foundation for the subsequent Han dynasty. 2. Emperor Wu of Han (141-87 BCE): One of the greatest emperors of the Han dynasty, he expanded the empire's territories, strengthened its military, and sponsored cultural and scientific developments. 3. Confucius (551-479 BCE): A prominent philosopher and sage who emphasized the importance of morality, respect for authority, and the cultivation of virtue. His teachings have had a lasting impact on Chinese thought and society. 4. Mao Zedong (1893-1976): A pivotal figure in modern Chinese history, he led the Communist Party of China to victory in the Chinese Civil War and established the People's Republic of China in 1949. 5. Deng Xiaoping (1904-1992): A key player in China's modernization and reform, he introduced market-oriented economic reforms and opened up China's economy to the world. These individuals, among others, have undoubtedly had a significant impact on Chinese history and development. However, it's crucial to keep in mind that evaluating their relative importance can be a complex and nuanced task that requires taking into account various factors and perspectives. I hope this information helps you in your inquiry! Is there anything else I can assist you with?</s>",
        "Greetings! It's a challenging task to pinpoint a single most important figure in Chinese history, as the country has had a diverse and rich history with numerous significant events and influential figures throughout its 2,200 years of existence. However, here are some of the most notable figures in Chinese history: 1. Qin Shi Huang (256-246 BCE): Known as the first Emperor of China, he unified the various regions of China and established the Qin dynasty, which laid the foundation for the Chinese culture and civilization we know today. 2. Confucius (552-472 BCE): A philosopher, politician, and teacher, Confucius had a profound impact on Chinese thought and ethics, and his teachings have influenced Chinese culture for centuries. 3. Emperor Wu of Han (140-87 BCE): One of the most prominent rulers of the Han dynasty, Emperor Wu expanded the boundaries of China, enhanced its military power, and initiated reforms that laid the groundwork for Chinese culture's growth. 4. Genghis Khan (1162-1227 CE): Although not Chinese himself, Genghis Khan played a significant role in shaping China's history as the founder of the Mongol Empire, which unified much of Asia and had a lasting impact on Chinese society. 5. Mao Zedong (1890-1976): A controversial figure, Mao played a critical role in modern Chinese history, leading the Chinese Communist Party to power in 1949 and implementing significant economic and social reforms that had a profound impact on China and the world. It's important to note that these are just a few examples, and many other figures have contributed to Chinese history in important ways. When it comes to Chinese history, there's always more to learn and discover, and it's essential to approach the subject with an open mind and a critical eye. I hope this information is helpful! If you have any further questions or would like to explore a specific aspect of Chinese history, I'm here to assist you.</s>"
    ],
    "verbose_status": "success"
};

// Dirty codes here during dev; need to wrap to backend
const openai = new OpenAI({
    apiKey: 'your api key here', // defaults to process.env["OPENAI_API_KEY"]
    dangerouslyAllowBrowser: true
});
// async function main() {
//     const chatCompletion = await openai.chat.completions.create({
//         messages: [{ role: 'user', content: 'Say this is a test' }],
//         model: 'gpt-4-0613',
//     });

//     console.log(chatCompletion.choices);
// }

let sampleSummarize = [
    {
        "text": "\n\n['Pizza', 'Sandwiches', 'Salads', 'Finger foods', 'BBQ or grilled wings']",
        "index": 0,
        "logprobs": null,
        "finish_reason": "stop"
    },
    {
        "text": "\n\n['Fresh fruit', 'Vegetables', 'Dips', 'Sandwiches', 'Wraps', 'Energy balls', 'Yogurt parfait', 'Smoothies', 'Time', 'Location', 'Equipment', 'Budget constraints', 'Dietary restrictions']",
        "index": 1,
        "logprobs": null,
        "finish_reason": "stop"
    },
    {
        "text": "\n\n['Fresh fruit', 'Veggie sandwiches', 'Herbal tea', 'Dark chocolate', 'Infused water', 'Nuts and seeds', 'Oatmeal', 'Smoothies']",
        "index": 2,
        "logprobs": null,
        "finish_reason": "stop"
    }
].map(x => x.text.replaceAll('\n', '').replaceAll(/'/g, '"'))
    .map(x => JSON.parse(x));
console.log(sampleSummarize);

const usePrevious = value => {
    const ref = useRef();
    useEffect(() => {
        ref.current = value;
    });
    return ref.current;
};


// Function to highlight the text, powered by ChatGPT
function highlightText(textToHighlight, spanElement) {
    const innerHTML = spanElement.innerHTML;
    // Escape special characters for use in a regular expression
    const escapedTextToHighlight = textToHighlight.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&');
    // Create a RegExp to find the text in a case-insensitive way
    const regex = new RegExp(`(${escapedTextToHighlight})`, 'gi');
    // Replace found text with a mark element
    spanElement.innerHTML = innerHTML.replace(regex, '<mark>$1</mark>');
}

// Function to undo the highlight
function undoHighlight(spanElement) {
    // Replace the <mark> tags with just their inner content
    spanElement.innerHTML = spanElement.innerHTML.replace(/<mark>(.*?)<\/mark>/gi, '$1');
}

function Exploration(props) {
    const ctx = useContext(ChatContext).chatInfo;
    const [postInterveneSubAttr, setPostInterveneSubAttr] = useState([]);
    const [postInterveneData, setPostInterveneData] = useState(sampleData);
    const previousPostInterveneData = usePrevious(postInterveneData);

    const [keyitems, setKeyitems] = useState([]);
    const [items2group, setItems2group] = useState({});
    const [group2items, setGroup2items] = useState({});

    const [responses, setResponses] = useState("");

    const [showScrollDown, setShowScrollDown] = useState(false);
    const convoRef = useRef(null);
    const [isBusy, setBusy] = useState(false);
    const { config, setConfig } = useContext(ConfigContext);

    const visRef = useRef(null);

    useEffect(() => {
        scrollDown();
    }, [ctx.history]);

    const postIntervention = () => {
        if (ctx.history.length < 1 || postInterveneSubAttr.length < 1) return;
        setBusy(true);
        console.log('start executing intervention ...', postInterveneSubAttr.join(", "));
        let url = `${BACKEND_ADDR}/post_chat_exploration`;
        let xhr = new XMLHttpRequest();
        xhr.onload = e => {
            var resp = JSON.parse(e.target.responseText);
            console.log('finish executing intervention ...', resp);
            setBusy(false);
            setPostInterveneData(resp);
        };
        xhr.open('POST', url, true);
        xhr.send(
            JSON.stringify({
                id: ctx.id,
                subattribute: postInterveneSubAttr.join(", "),
                model: config.model,
                samples: 5
            })
        );
    }

    useEffect(() => {
        if (keyitems.length === 0) return;
        console.log(keyitems)
        let d = _.chain(keyitems).map((x, idx) => {
            return x.map(i => ({
                'item': i.toLowerCase(),
                'class': items2group[i],
                'level': idx
            }))
        }).flatten().value();

        let countByItem = _.chain(d).groupBy(d => d.class).value();
        countByItem = Object.keys(countByItem).map(item => ({
            key: item,
            data: countByItem[item]
        }));
        console.log(countByItem);

        let svg = d3.select(visRef.current);

        let parentContainer = d3.select(svg.node().parentNode);
        let width = parentContainer.node().getBoundingClientRect().width;
        let height = parentContainer.node().getBoundingClientRect().height;

        let svgHeight = height - 200;
        svg.attr('width', width).attr('height', svgHeight);
        svg.selectAll('*').remove();

        // scales
        let xScale = d3.scaleLinear().domain([0, keyitems.length])
            .range([200, width - 100]);
        let xBandWidth = (width - 300) / keyitems.length;
        let yScale = d3.scaleBand().domain(countByItem.map(x => x.key))
            .range([30, svgHeight]);

        console.log(countByItem.map(x => x.key));
        console.log(d, d.length);

        // remove items without groups TODO
        d = d.filter(x => x.class && x.class.length)
        console.log(d.length);

        // Draw head
        let levelsData = keyitems.map((x, idx) => ({ 'Level': idx }));
        let headRegion = svg.append('g')
            .selectAll('div.header-region')
            .data(levelsData).enter()
            .append('rect')
            .attr('x', d => xScale(d.Level)).attr('width', xBandWidth)
            .attr('y', 0).attr('height', 30)
            .attr('fill', 'none')
            .style('pointer-events', 'all') // This ensures the element is still responsive to mouse events

        const setLastBotMsg = (msg) => {
            let lastMsgInChat = d3.selectAll('div.message-container.bot').nodes().at(-1);
            let tSpan = d3.select(lastMsgInChat).select('div.message > span');
            tSpan.text(msg);
        }

        headRegion.on('mouseover', function (e, d) {
            let response = responses[d.Level];
            // set msg to corresponding intervened text
            setLastBotMsg(response);
        }).on('mouseout', function (d) {
            let originMsg = ctx.history.at(-1).msg;
            // reset msg
            setLastBotMsg(originMsg);
        })

        // Draw the line and legend at the top
        svg.append('line')
            .attr('x1', xScale.range()[0]).attr('y1', 0)
            .attr('x2', xScale.range()[1]).attr('y2', 0)
            .attr('stroke', 'black')
        svg.append('text')
            .text(`Very non-${postInterveneSubAttr.join(", ")}`)
            .attr('x', xScale.range()[0]).attr('y', 5)
            .attr('dominant-baseline', 'hanging')
        svg.append('text')
            .text(`Very ${postInterveneSubAttr.join(", ")}`)
            .attr('x', xScale.range()[1]).attr('y', 5)
            .attr('dominant-baseline', 'hanging')
            .attr('text-anchor', 'end')

        // Draw text items along y-axis
        svg.append('g').selectAll('text.item').data(countByItem)
            .enter().append('text').text(d => d.key)
            .attr('x', 0)
            .attr('y', (d, i) => yScale(d.key) + yScale.bandwidth() / 2)
            .attr('class', 'item')
            .attr('dominant-baseline', 'central')

        // Draw rectangles
        let rects = svg.append('g').selectAll('rect.item-rect').data(d)
            .enter().append('rect')
            .attr('x', d => xScale(d.level))
            .attr('y', d => yScale(d.class))
            .attr('width', xBandWidth)
            .attr('height', yScale.bandwidth() - 3)
            .attr('fill', '#67778a').attr('class', 'item-rect')

        rects.on('mouseover', function (e, d) {
            let response = responses[d.level];
            // set msg to corresponding intervened text
            setLastBotMsg(response);

            // highlight items
            let lastMsgInChat = d3.selectAll('div.message-container.bot div.message > span').nodes().at(-1);
            highlightText(d.item, lastMsgInChat);
        }).on('mouseout', function (d) {
            let originMsg = ctx.history.at(-1).msg;
            // reset msg
            setLastBotMsg(originMsg);

            let lastMsgInChat = d3.selectAll('div.message-container.bot div.message > span').nodes().at(-1);
            undoHighlight(lastMsgInChat);
        })

    }, [keyitems])

    useEffect(() => {
        async function summarize() {
            // reset mapping
            setItems2group({})
            setGroup2items({})

            // ask GPT to summarize
            setBusy(true)
            let userQuestion = ctx.history.at(-2).msg;
            let prevAnswer = ctx.history.at(-1).msg;

            console.log(postInterveneSubAttr.join(', ').toLowerCase())
            let d = postInterveneData[postInterveneSubAttr.join(", ")]; // d is string[]
            console.log(d);
            setResponses(d);

            console.log(userQuestion);

            d = d.map(x => `Question: ${userQuestion}. Answer: ${x}. Given the question and the answer, extract keyitems that answers the question from the bullet points. The key items should directly answer the question. Make sure your response is a python-like list of strings like this ['Fresh fruit', 'Veggie sandwiches', 'Herbal tea', 'Dark chocolate']. Make sure your answer is an array of strings so it can be parsed by JSON. Limit the length of array to 6, and limit the length of each string to be 3 words. If no key item exists, return ['Null']. If answer lists key items in bullet points, using the title words from each bullet point. If there is no bullet points, then just extract the key items from the text. Make sure your response can be parsed by JSON`);

            const completion = await openai.completions.create({
                model: "gpt-3.5-turbo-instruct",
                prompt: d,
                temperature: 0,
                max_tokens: 300
            });
            console.log(completion.choices);

            let items = completion.choices.map(x => x.text.replaceAll('\n', '').replaceAll(/'/g, '"').replaceAll('.', ''));
            items = items.map(x => {
                try {
                    // Try to parse the string as JSON
                    return JSON.parse(x);
                } catch (error) {
                  // If an error occurs during parsing, return an empty array instead
                  console.error("JSON parsing error:", error);
                  return ["Failed to extract"];
                }
            });
            console.log(items); // string[][]

            // Merge the items into categories, e.g., 'fruits' and 'fruit' to 'fruits'
            var merge_to_origin = {};
            for (let v of _.flatten(items)) {
                // set default
                merge_to_origin[v] = [v];
            }
            let merge_propt = `Given this array consisted of items, can you please merge items with the same meanings? Do not over merge. Give me a JSON dict showing the mapping between merged name and an array of original names. Return the JSON dict only with any other text. ${JSON.stringify(items.flat())}`;
            const merge_completion = await openai.chat.completions.create({
                messages: [{ role: "user", content: merge_propt}],
                model: "gpt-4",
                temperature: 0,
              });
            console.log(merge_completion.choices);

            try {
                // Try to parse the string as JSON
                merge_to_origin = JSON.parse(merge_completion.choices[0].message.content);
            } catch (error) {
                // If an error occurs during parsing, return an empty array instead
                console.error("JSON parsing error:", error);
            }

            var origin_to_merge = {};
            for (const [key, value] of Object.entries(merge_to_origin)) {
                for (let v of value) {
                    origin_to_merge[v] = key;
                }
            };
            console.log(merge_to_origin, origin_to_merge);

            setItems2group(origin_to_merge);
            setGroup2items(merge_to_origin);

            setKeyitems(items);
            setBusy(false)
        }
        console.log('useEffect postInterveneData', postInterveneData);
        console.log(ctx.history);

        if (previousPostInterveneData && JSON.stringify(previousPostInterveneData).length !== JSON.stringify(postInterveneData).length) { // dirty equal check
            console.log('update postInterveneData', postInterveneData);

            summarize();
        }
    }, [postInterveneData]);


    var messages = ctx.history.map((e, i) => (
        <Message key={i} bot={e.bot} loading={e.loading} regefunc={null} mid={i} removefunc={null}>{e.msg}</Message>
    ));

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

    function renderVisualizationContainer() {
        if (keyitems.length > 0) {
            // Return the SVG if visRef is not null
            return <svg id='vis' ref={visRef}></svg>;
        } else {
            // Return an image if visRef is null
            return <img src={instructionImage} alt="Placeholder" />;
        }
    }

    return (
        <main className="main">
            <div className="exploration-chat container">
                <div
                    className={`convo-container ${ctx.history.length > 0 ? "" : "empty"}`}
                    onScroll={onChatScroll} ref={convoRef}
                >
                    {ctx.history.length > 0 ?
                        messages :
                        <p id="no-messages">Start chatting below to get started</p>
                    }
                </div>
                    
                <div id='exploration-config'>
                    <UIToggleButtons
                        id="subattribute"
                        label="Attributes to explore! "
                        value={postInterveneSubAttr}
                        options={INTERVENEOPTIONS}
                        onChange={
                            (e) => {
                                setPostInterveneSubAttr(e.target.value);
                            }
                        }
                    />
                    <Button onClick={postIntervention} disabled={isBusy} id='exploration-button'>
                        Intervention
                    </Button>
                </div>
            </div>
            <div className="exploration-vis container" style={{borderStyle: "solid", borderWidth: "0 0 0 2px", borderRadius: "0"}}>
                    {
                        isBusy ? (
                            <div className="lds-facebook"><div></div><div></div><div></div><div></div></div>
                        ) : (
                            renderVisualizationContainer()
                        )
                    }
            </div>
        </main>

    )
}

export { Exploration }