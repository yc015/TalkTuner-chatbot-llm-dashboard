import React, { useState, useEffect, useContext, useCallback } from 'react';

import { ModalContext } from '../context/ModalContext.js';
import { ChatContext } from '../context/ChatContext.js';
import { VisOption } from '../components/VisOption';
import { Trait } from '../components/Trait';
import { Button } from '../components/Button';
import { TextInput } from '../components/TextInput.js';
import { SaveModal } from '../components/SaveModal.js';
import { Toggle } from '../components/Toggle.js';
import { TraitsPlus } from '../components/TraitsPlus.js';
import { Insights } from '../components/Insights.js';
import { TraitFamily } from '../components/TraitFamily.js';

import { HistoryView } from '../components/HistoryView/HistoryView.js';

import { RiQuestionAnswerLine } from 'react-icons/ri';
import { FiSend, FiTrendingUp, FiUserPlus, FiBriefcase, FiDollarSign, FiHeart } from 'react-icons/fi';
import { AiOutlineQuestion } from 'react-icons/ai';
import { MdSaveAlt, MdTempleHindu } from 'react-icons/md';
import { BsGenderAmbiguous, BsGenderMale, BsGenderFemale } from 'react-icons/bs';
import { IoSchoolOutline, IoHourglassOutline } from 'react-icons/io5';
import { HiOutlineGlobe, } from 'react-icons/hi';
import { HiLanguage, } from 'react-icons/hi2'
import { LuPackageOpen } from "react-icons/lu";
import { HiOutlineGlobeEuropeAfrica, HiOutlineGlobeAsiaAustralia, HiOutlineGlobeAmericas } from 'react-icons/hi2';
import { TfiThought } from "react-icons/tfi";
import { GiRead } from "react-icons/gi";
import { GiUncertainty } from "react-icons/gi";
import { PiBookmark, PiHeart, PiHeartBreak, PiHeartHalf, PiSkullLight } from 'react-icons/pi';
import { FaPerson, FaPersonCane, FaChildReaching, FaBaby, FaDharmachakra, FaCross, FaStarAndCrescent, FaAtom, FaQuestion, FaRainbow } from 'react-icons/fa6'
import { FaFaceGrinStars } from "react-icons/fa6";
import { TbLanguageHiragana, TbCircleLetterG, TbCircleLetterS, TbCircleLetterP, TbCircleLetterR, TbCircleLetterA, TbLetterG, TbLetterS, TbLetterP, TbLetterR, TbLetterA, TbJewishStarFilled } from 'react-icons/tb'
import { RiEnglishInput } from 'react-icons/ri'
import { GiBookshelf, GiHeartWings } from 'react-icons/gi'
import { PiBook, PiBooks, PiCoinsFill } from 'react-icons/pi'
import { BiSolidCoin, BiSolidCoinStack } from 'react-icons/bi'
import { FaCoins } from 'react-icons/fa6'
import { ConfigContext } from '../context/ConfigContext.js';
import { LiaHandPointLeftSolid, LiaHandPointRightSolid, LiaHandPointUpSolid } from "react-icons/lia";
import { UIToggleButtonsExclusive } from '../components/UIToggleButtonsExclusive.js';
import { defaultControlYouModelStatus } from '../context/ChatContext.js';
import nonbinaryIcon from '../imgs/nonbinary.png';
// import { LiaDharmachakraSolid, LiaCrossSolid } from 'react-icons/lia'

import { CircularProgressbarWithChildren, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

function Dashboard(props) {
    const ctx = useContext(ChatContext).chatInfo;
    const { setModal } = useContext(ModalContext);
    const setCtx = useContext(ChatContext).setChatInfo;
    const { updateControlYouModelTrait, setControlYouModelTraitStatus } = useContext(ChatContext);
    const [questionInputText, setQuestionInputText] = useState('');
    const [checked, setChecked] = useState(false);
    const [confidence, setConfidence] = useState(50);
    const { config, setConfig } = useContext(ConfigContext);
    const [highlightedTrait, setHighlightedTrait] = useState(null);
    
    const controlOptions = [{ value: 'off', label: 'off' }, { value: 'on', 'label': 'on' }]
    const sortOptions = [{ value: 'off', label: 'off' }, { value: 'on', 'label': 'on' }]

    const attrOptions = [{ value: 'input', label: 'input' }, { value: 'output', 'label': 'output' }]

    const isMessageEmpty = () => { return !/(.|\s)*\S(.|\s)*/.test(questionInputText); }

    useEffect(() => {
        var y = ctx.youModel;
        setConfidence(Math.round((
            Math.max(y.gender.male, y.gender.female) +
            Math.max(y.age.child, y.age.adolescent, y.age.adult, y.age.olderAdult) +
            Math.max(y.ethnicity.asian, y.ethnicity.african, y.ethnicity.white, y.ethnicity.hispanic, y.ethnicity.nativeAmerican, y.ethnicity.arab, y.ethnicity.jews) +
            Math.max(y.socioEco.low, y.socioEco.middle, y.socioEco.high) +
            Math.max(y.marital.single, y.marital.married, y.marital.divorced, y.marital.widowed) +
            Math.max(y.education.someschool, y.education.highschool, y.education.collegemore) +
            Math.max(y.language.chinese, y.language.japanese, y.language.english, y.language.german,
                y.language.spanish, y.language.portuguese, y.language.russian, y.language.arabic)
        ) / 6 * 100));
    }, [ctx.youModel]);

    function onSend() {
        if (isMessageEmpty()) return;
    }

    const handleSliderChange = (category, trait, newValue) => {
        updateControlYouModelTrait(category, trait, newValue);
        setControlYouModelTraitStatus(category, trait, newValue);
        console.log(ctx.controlYouModelStatus)
    };

    const onClearButtonClick = (category, trait) => {
        updateControlYouModelTrait(category, trait, ctx.youModel[category][trait] * 100);
        setControlYouModelTraitStatus(category, trait, ctx.youModel[category][trait] * 100, false);
        console.log(ctx.controlYouModelStatus);
    }

    // const [shrunkComponents, setShrunkComponents] = useState([]);

    const onDragStart = (event, categoryName) => {
        event.dataTransfer.setData("category", categoryName);
    };

    const onDragOver = (event) => {
        event.preventDefault(); // Necessary to allow dropping
    };

    const onClear = () => {
        setCtx(prevCtx => {
            var newControlYouModel = JSON.parse(JSON.stringify(prevCtx.youModel));
            for (const outerKey in newControlYouModel) {
                if (newControlYouModel.hasOwnProperty(outerKey)) {
                    const outerValue = newControlYouModel[outerKey];
                    for (const innerKey in outerValue) {
                        if (outerValue.hasOwnProperty(innerKey)) {
                            newControlYouModel[outerKey][innerKey] *= 100;
                        }
                    }
                }
            }

            var newControlYouModelStatus = defaultControlYouModelStatus;
            return { ...prevCtx, controlYouModel: newControlYouModel, controlYouModelStatus: newControlYouModelStatus };
        });
    }

    const onDrop = (event, dropIndex) => {
        const categoryName = event.dataTransfer.getData("category");
        const draggedIndex = config.order.findIndex(cat => cat === categoryName);
        if (draggedIndex === dropIndex) return; // Dropped on itself
    
        // If the component is currently shrunk, and it's being moved back to the list
        if (config.shrunkComponents.includes(categoryName)) {
            // Remove from shrunkComponents
            // setShrunkComponents(prev => prev.filter(cat => cat !== categoryName));
            setConfig(prevConfig => {
                prevConfig.shrunkComponents = prevConfig.shrunkComponents.filter(cat => cat !== categoryName);
                return { ...prevConfig }
            });
        } else {
            // Handle reorder within the main list as before
            const newOrder = [...config.order];
            newOrder.splice(draggedIndex, 1); // Remove the dragged element
            newOrder.splice(dropIndex, 0, categoryName); // Insert it before the target index
    
            setConfig(prevConfig => ({ ...prevConfig, order: newOrder }));
        }
    };

    const onDropOnShrinkArea = (event) => {
        const category = event.dataTransfer.getData("category");
        if (!config.shrunkComponents.includes(category)) {
            // setShrunkComponents(prev => [...prev, category]);
            setConfig(prevConfig => {
                if (prevConfig.shrunkComponents.includes(category)) {
                    // If the category is already in the array, just return the previous config unchanged
                    return prevConfig;
                }
                prevConfig.shrunkComponents = [...prevConfig.shrunkComponents, category];
                console.log(prevConfig.shrunkComponents)
                return { ...prevConfig }
            });
        }
    };
    // Adjusted render function to include the shrink area
    const renderSliderTraitFamilies = () => {
            const nonShrunkComponents = config.order.filter(category => !config.shrunkComponents.includes(category));

            return nonShrunkComponents.map((category, index) => (
                <div
                    key={category}
                    draggable
                    onDragStart={(e) => onDragStart(e, category)}
                    onDragOver={onDragOver}
                    onDrop={(e) => onDrop(e, index)}
                    style={{ 
                        // opacity: shrunkComponents.includes(category) ? 0 : 1, // Adjust opacity based on shrunk status
                        // transform: shrunkComponents.includes(category) ? "scale(0.1)": "scale(1)"
                    }}
                    className='draggable-trait'
                >
                {category === "age" && 
                    <TraitFamily
                        label="Age"
                        category="age"
                        traits={[
                            {
                                icon: <FaBaby />,
                                trait: "Child",
                                indexTrait: "child",
                                confidence: ctx.youModel.age.child,
                                controlConfidence: ctx.controlYouModel.age.child
                            },
                            {
                                icon: <FaChildReaching />,
                                trait: "Adolescent",
                                indexTrait: "adolescent",
                                confidence: ctx.youModel.age.adolescent,
                                controlConfidence: ctx.controlYouModel.age.adolescent
                            },
                            {
                                icon: <FaPerson />,
                                trait: "Adult",
                                indexTrait: "adult",
                                confidence: ctx.youModel.age.adult,
                                controlConfidence: ctx.controlYouModel.age.adult
                            },
                            {
                                icon: <FaPersonCane />,
                                trait: "Older Adult",
                                indexTrait: "olderAdult",
                                confidence: ctx.youModel.age.olderAdult,
                                controlConfidence: ctx.controlYouModel.age.olderAdult
                            },
                            {
                                icon: <FaQuestion/>,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.youModel.age.unknown,
                                controlConfidence: ctx.controlYouModel.age.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                {category === "marital" &&
                    <TraitFamily
                        label="Marital"
                        category="marital"
                        traits={[
                            {
                                icon: <PiHeartHalf />,
                                trait: "Single",
                                indexTrait: "single",
                                confidence: ctx.youModel.marital.single,
                                controlConfidence: ctx.controlYouModel.marital.single
                            },
                            {
                                icon: <PiHeart />,
                                trait: "Married",
                                indexTrait: "married",
                                confidence: ctx.youModel.marital.married,
                                controlConfidence: ctx.controlYouModel.marital.married
                            },
                            {
                                icon: <PiHeartBreak />,
                                trait: "Divorced",
                                indexTrait: "divorced",
                                confidence: ctx.youModel.marital.divorced,
                                controlConfidence: ctx.controlYouModel.marital.divorced
                            },
                            {
                                icon: <GiHeartWings />,
                                trait: "Widowed",
                                indexTrait: "widowed",
                                confidence: ctx.youModel.marital.widowed,
                                controlConfidence: ctx.controlYouModel.marital.widowed
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                {category === "socioEco" &&
                    <TraitFamily
                        label="Socio Eco"
                        category="socioEco"
                        traits={[
                            {
                                icon: <BiSolidCoin />,
                                trait: "Lower",
                                indexTrait: "low",
                                confidence: ctx.youModel.socioEco.low,
                                controlConfidence: ctx.controlYouModel.socioEco.low
                            },
                            {
                                icon: <PiCoinsFill />,
                                trait: "Middle",
                                indexTrait: "middle",
                                confidence: ctx.youModel.socioEco.middle,
                                controlConfidence: ctx.controlYouModel.socioEco.middle
                            },
                            {
                                icon: <FaCoins />,
                                trait: "Upper",
                                indexTrait: "high",
                                confidence: ctx.youModel.socioEco.high,
                                controlConfidence: ctx.controlYouModel.socioEco.high
                            },
                            {
                                icon: <FaQuestion/>,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.youModel.socioEco.unknown,
                                controlConfidence: ctx.controlYouModel.socioEco.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "education" && 
                    <TraitFamily
                        label="Education"
                        category="education"
                        traits={[
                            {
                                icon: <PiBook />,
                                trait: "Some Education",
                                indexTrait: "someschool",
                                confidence: ctx.youModel.education.someschool,
                                controlConfidence: ctx.controlYouModel.education.someschool
                            },
                            {
                                icon: <PiBooks />,
                                trait: "High School",
                                indexTrait: "highschool",
                                confidence: ctx.youModel.education.highschool,
                                controlConfidence: ctx.controlYouModel.education.highschool
                            },
                            {
                                icon: <GiBookshelf />,
                                trait: "College & More",
                                indexTrait: "collegemore",
                                confidence: ctx.youModel.education.collegemore,
                                controlConfidence: ctx.controlYouModel.education.collegemore
                            },
                            {
                                icon: <FaQuestion/>,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.youModel.education.unknown,
                                controlConfidence: ctx.controlYouModel.education.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "ethnicity" && 
                    <TraitFamily
                        label="Ethnicity"
                        category="ethnicity"
                        traits={[
                            {
                                icon: <HiOutlineGlobeAsiaAustralia />,
                                trait: "Asian",
                                indexTrait: "asian",
                                confidence: ctx.youModel.ethnicity.asian,
                                controlConfidence: ctx.controlYouModel.ethnicity.asian
                            },
                            {
                                icon: <HiOutlineGlobeEuropeAfrica />,
                                trait: "African",
                                indexTrait: "african",
                                confidence: ctx.youModel.ethnicity.african,
                                controlConfidence: ctx.controlYouModel.ethnicity.african
                            },
                            {
                                icon: <HiOutlineGlobeAmericas />,
                                trait: "White",
                                indexTrait: "white",
                                confidence: ctx.youModel.ethnicity.white,
                                controlConfidence: ctx.controlYouModel.ethnicity.white
                            },
                            {
                                icon: <HiOutlineGlobeAmericas />,
                                trait: "Hispanic",
                                indexTrait: "hispanic",
                                confidence: ctx.youModel.ethnicity.hispanic,
                                controlConfidence: ctx.controlYouModel.ethnicity.hispanic
                            },
                            {
                                icon: <HiOutlineGlobeAmericas />,
                                trait: "Native American",
                                indexTrait: "nativeAmerican",
                                confidence: ctx.youModel.ethnicity.nativeAmerican,
                                controlConfidence: ctx.controlYouModel.ethnicity.nativeAmerican
                            },
                            {
                                icon: <HiOutlineGlobeEuropeAfrica />,
                                trait: "Arab",
                                indexTrait: "arab",
                                confidence: ctx.youModel.ethnicity.arab,
                                controlConfidence: ctx.controlYouModel.ethnicity.arab
                            },
                            {
                                icon: <HiOutlineGlobeEuropeAfrica />,
                                trait: "Jewish",
                                indexTrait: "jews",
                                confidence: ctx.youModel.ethnicity.jews,
                                controlConfidence: ctx.controlYouModel.ethnicity.jews
                            },
                            {
                                icon: <FaQuestion/>,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.youModel.ethnicity.unknown,
                                controlConfidence: ctx.controlYouModel.ethnicity.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "religion" && 
                    <TraitFamily
                        label="Religion"
                        category="religion"
                        traits={[
                            {
                                icon: <FaCross />,
                                trait: "Christian",
                                indexTrait: "christianity",
                                confidence: ctx.youModel.religion.christianity,
                                controlConfidence: ctx.controlYouModel.religion.christianity
                            },
                            {
                                icon: <FaStarAndCrescent />,
                                trait: "Islam",
                                indexTrait: "islam",
                                confidence: ctx.youModel.religion.islam,
                                controlConfidence: ctx.controlYouModel.religion.islam
                            },
                            {
                                icon: <FaDharmachakra />,
                                trait: "Buddhism",
                                indexTrait: "buddhism",
                                confidence: ctx.youModel.religion.buddhism,
                                controlConfidence: ctx.controlYouModel.religion.buddhism
                            },
                            {
                                icon: <MdTempleHindu />,
                                trait: "Hinduism",
                                indexTrait: "hinduism",
                                confidence: ctx.youModel.religion.hinduism,
                                controlConfidence: ctx.controlYouModel.religion.hinduism
                            },
                            {
                                icon: <TbJewishStarFilled />,
                                trait: "Judaism",
                                indexTrait: "judaism",
                                confidence: ctx.youModel.religion.judaism,
                                controlConfidence: ctx.controlYouModel.religion.judaism
                            },
                            {
                                icon: <FaAtom />,
                                trait: "Atheism",
                                indexTrait: "atheism",
                                confidence: ctx.youModel.religion.atheism,
                                controlConfidence: ctx.controlYouModel.religion.atheism
                            },
                            {
                                icon: <FaQuestion/>,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.youModel.religion.unknown,
                                controlConfidence: ctx.controlYouModel.religion.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "gender" && 
                    <TraitFamily
                        label="Gender"
                        category="gender"
                        traits={[
                            {
                                icon: <BsGenderFemale />,
                                trait: "Female",
                                indexTrait: "female",
                                confidence: ctx.youModel.gender.female,
                                controlConfidence: ctx.controlYouModel.gender.female
                            },
                            {
                                icon: <BsGenderMale />,
                                trait: "Male",
                                indexTrait: "male",
                                confidence: ctx.youModel.gender.male,
                                controlConfidence: ctx.controlYouModel.gender.male
                            },
                            {
                                icon: <img src={nonbinaryIcon} alt="Placeholder" width="20px"/>,
                                // icon: <FaRainbow />,
                                trait: "Other",
                                indexTrait: "other",
                                confidence: ctx.youModel.gender.other,
                                controlConfidence: ctx.controlYouModel.gender.other
                            },
                            {
                                icon: <FaQuestion/>,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.youModel.gender.unknown,
                                controlConfidence: ctx.controlYouModel.gender.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "language" && 
                    <TraitFamily
                        label="Language"
                        category="language"
                        traits={[
                            {
                                icon: <HiLanguage />,
                                trait: "Chinese",
                                indexTrait: "chinese",
                                confidence: ctx.youModel.language.chinese,
                                controlConfidence: ctx.controlYouModel.language.chinese
                            },
                            {
                                icon: <TbLanguageHiragana />,
                                trait: "Japanese",
                                indexTrait: "japanese",
                                confidence: ctx.youModel.language.japanese,
                                controlConfidence: ctx.controlYouModel.language.japanese
                            },
                            {
                                icon: <RiEnglishInput />,
                                trait: "English",
                                indexTrait: "english",
                                confidence: ctx.youModel.language.english,
                                controlConfidence: ctx.controlYouModel.language.english
                            },
                            {
                                icon: <TbLetterG />,
                                trait: "Germany",
                                indexTrait: "german",
                                confidence: ctx.youModel.language.german,
                                controlConfidence: ctx.controlYouModel.language.german
                            },
                            {
                                icon: <TbLetterS />,
                                trait: "Spanish",
                                indexTrait: "spanish",
                                confidence: ctx.youModel.language.spanish,
                                controlConfidence: ctx.controlYouModel.language.spanish
                            },
                            {
                                icon: <TbLetterP />,
                                trait: "Portuguese",
                                indexTrait: "portuguese",
                                confidence: ctx.youModel.language.portuguese,
                                controlConfidence: ctx.controlYouModel.language.portuguese
                            },
                            {
                                icon: <TbLetterA />,
                                trait: "Arabic",
                                indexTrait: "arabic",
                                confidence: ctx.youModel.language.arabic,
                                controlConfidence: ctx.controlYouModel.language.arabic
                            },
                            {
                                icon: <TbLetterR />,
                                trait: "Russian",
                                indexTrait: "russian",
                                confidence: ctx.youModel.language.russian,
                                controlConfidence: ctx.controlYouModel.language.russian
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "political" && 
                    <TraitFamily
                        label="Political"
                        category="political"
                        traits={[
                            {
                                icon: <LiaHandPointLeftSolid />,
                                trait: "Left",
                                indexTrait: "left",
                                confidence: ctx.youModel.political.left,
                                controlConfidence: ctx.controlYouModel.political.left
                            },
                            {
                                icon: <LiaHandPointRightSolid />,
                                trait: "Right",
                                indexTrait: "right",
                                confidence: ctx.youModel.political.right,
                                controlConfidence: ctx.controlYouModel.political.right
                            },
                            {
                                icon: <LiaHandPointUpSolid />,
                                trait: "Moderate",
                                indexTrait: "moderate",
                                confidence: ctx.youModel.political.moderate,
                                controlConfidence: ctx.controlYouModel.political.moderate
                            },
                            {
                                icon: <FaQuestion />,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: ctx.youModel.political.unknown,
                                controlConfidence: ctx.controlYouModel.political.unknown
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "uncertainty" && 
                    <TraitFamily
                        label="Uncertainty"
                        category="uncertainty"
                        traits={[
                            {
                                icon: <GiUncertainty />,
                                trait: "Uncertainty",
                                indexTrait: "uncertainty",
                                confidence: ctx.youModel.uncertainty.uncertainty,
                                controlConfidence: ctx.controlYouModel.uncertainty.uncertainty
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        // onSliderChange={handleSliderChange}
                        // onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                { category === "sycophancy" && 
                    <TraitFamily
                        label="Sycophancy"
                        category="sycophancy"
                        traits={[
                            {
                                icon: <FaFaceGrinStars />,
                                trait: "Sycophancy",
                                indexTrait: "sycophancy",
                                confidence: ctx.youModel.sycophancy.sycophancy,
                                controlConfidence: ctx.controlYouModel.sycophancy.sycophancy
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        // onSliderChange={handleSliderChange}
                        // onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                }
                {/* { category === "hallucination" && 
                    <TraitFamily
                        label="Hallucination"
                        category="hallucination"
                        traits={[
                            {
                                icon: <TfiThought/>,
                                trait: "Hallucinated",
                                indexTrait: "hallucinated",
                                confidence: ctx.youModel.hallucination.hallucinated,
                                controlConfidence: ctx.controlYouModel.hallucination.hallucinated
                            },
                            {
                                icon: <GiRead />,
                                trait: "Factual",
                                indexTrait: "factual",
                                confidence: ctx.youModel.hallucination.factual,
                                controlConfidence: ctx.controlYouModel.hallucination.factual
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                    />
                } */}
                {/* Repeat for other categories */}
            </div>
        ));
    };

    const renderShrunkComponents = () => {
        return config.shrunkComponents.map(category => (
            <div
                draggable
                onDragStart={(e) => {
                    e.dataTransfer.setData("category", category);
                    // Optionally, indicate that this is a shrunk component being dragged
                    e.dataTransfer.setData("origin", "shrunk");
                 }}
                key={category}
                style={{ transform: "scale(0.9)", width: "70px", opacity: 1, justifyContent: "center",}}
            >
                {category === "age" && 
                <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                             alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Age
                    <TraitFamily
                        label="Age"
                        category="age"
                        traits={[
                            {
                                icon: <FaBaby />,
                                trait: "Child",
                                indexTrait: "child",
                                confidence: ctx.youModel.age.child,
                                controlConfidence: ctx.controlYouModel.age.child
                            },
                            // {
                            //     icon: <FaChildReaching />,
                            //     trait: "Adolescent",
                            //     indexTrait: "adolescent",
                            //     confidence: ctx.youModel.age.adolescent,
                            //     controlConfidence: ctx.controlYouModel.age.adolescent
                            // },
                            // {
                            //     icon: <FaPerson />,
                            //     trait: "Adult",
                            //     indexTrait: "adult",
                            //     confidence: ctx.youModel.age.adult,
                            //     controlConfidence: ctx.controlYouModel.age.adult
                            // },
                            // {
                            //     icon: <FaPersonCane />,
                            //     trait: "Older Adult",
                            //     indexTrait: "olderAdult",
                            //     confidence: ctx.youModel.age.olderAdult,
                            //     controlConfidence: ctx.controlYouModel.age.olderAdult
                            // }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        displayBar={false}
                    />
                </div>
                    
                }
                {category === "marital" &&
                <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Marital
                    <TraitFamily
                        label="Marital"
                        category="marital"
                        traits={[
                            {
                                icon: <PiHeartHalf />,
                                trait: "Single",
                                indexTrait: "single",
                                confidence: ctx.youModel.marital.single,
                                controlConfidence: ctx.controlYouModel.marital.single
                            },
                            // {
                            //     icon: <PiHeart />,
                            //     trait: "Married",
                            //     indexTrait: "married",
                            //     confidence: ctx.youModel.marital.married
                            // },
                            // {
                            //     icon: <PiHeartBreak />,
                            //     trait: "Divorced",
                            //     indexTrait: "divorced",
                            //     confidence: ctx.youModel.marital.divorced
                            // },
                            // {
                                // icon: <GiHeartWings />,
                            //     trait: "Widowed",
                            //     indexTrait: "widowed",
                            //     confidence: ctx.youModel.marital.widowed
                            // }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        displayBar={false}
                    />
                </div>
                }
                {category === "socioEco" &&
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Socio Eco
                        <TraitFamily
                            label="Socio Eco"
                            category="socioEco"
                            traits={[
                                {
                                    icon: <BiSolidCoin />,
                                    trait: "Lower",
                                    indexTrait: "low",
                                    confidence: ctx.youModel.socioEco.low,
                                    controlConfidence: ctx.controlYouModel.socioEco.low,
                                },
                                // {
                                //     icon: <PiCoinsFill />,
                                //     trait: "Middle",
                                //     indexTrait: "middle",
                                //     confidence: ctx.youModel.socioEco.middle
                                // },
                                // {
                                //     icon: <FaCoins />,
                                //     trait: "Upper",
                                //     indexTrait: "high",
                                //     confidence: ctx.youModel.socioEco.high
                                // }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "education" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Education
                        <TraitFamily
                            label="Education"
                            category="education"
                            traits={[
                                {
                                    icon: <PiBook />,
                                    trait: "Some Education",
                                    indexTrait: "someschool",
                                    confidence: ctx.youModel.education.someschool,
                                    controlConfidence: ctx.controlYouModel.education.someschool
                                },
                                // {
                                //     icon: <PiBooks />,
                                //     trait: "High School",
                                //     indexTrait: "highschool",
                                //     confidence: ctx.youModel.education.highschool
                                // },
                                // {
                                //     icon: <GiBookshelf />,
                                //     trait: "College & More",
                                //     indexTrait: "collegemore",
                                //     confidence: ctx.youModel.education.collegemore
                                // }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "ethnicity" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Ethnicity
                        <TraitFamily
                            label="Ethnicity"
                            category="ethnicity"
                            traits={[
                                {
                                    icon: <HiOutlineGlobeAsiaAustralia />,
                                    trait: "Asian",
                                    indexTrait: "asian",
                                    confidence: ctx.youModel.ethnicity.asian,
                                    controlConfidence: ctx.controlYouModel.ethnicity.asian
                                },
                                // {
                                //     icon: <HiOutlineGlobeEuropeAfrica />,
                                //     trait: "African",
                                //     indexTrait: "african",
                                //     confidence: ctx.youModel.ethnicity.african
                                // },
                                // {
                                //     icon: <HiOutlineGlobeAmericas />,
                                //     trait: "White",
                                //     indexTrait: "white",
                                //     confidence: ctx.youModel.ethnicity.white
                                // },
                                // {
                                //     icon: <HiOutlineGlobeAmericas />,
                                //     trait: "Hispanic",
                                //     indexTrait: "hispanic",
                                //     confidence: ctx.youModel.ethnicity.hispanic
                                // },
                                // {
                                //     icon: <HiOutlineGlobeAmericas />,
                                //     trait: "Native American",
                                //     indexTrait: "nativeAmerican",
                                //     confidence: ctx.youModel.ethnicity.nativeAmerican
                                // },
                                // {
                                //     icon: <HiOutlineGlobeEuropeAfrica />,
                                //     trait: "Arab",
                                //     indexTrait: "arab",
                                //     confidence: ctx.youModel.ethnicity.arab
                                // },
                                // {
                                //     icon: <HiOutlineGlobeEuropeAfrica />,
                                //     trait: "Jewish",
                                //     indexTrait: "jews",
                                //     confidence: ctx.youModel.ethnicity.jews
                                // }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "religion" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Religion
                        <TraitFamily
                            label="Religion"
                            category="religion"
                            traits={[
                                {
                                    icon: <FaCross />,
                                    trait: "Christian",
                                    indexTrait: "christianity",
                                    confidence: ctx.youModel.religion.christianity,
                                    controlConfidence: ctx.controlYouModel.religion.christianity,
                                },
                                // {
                                //     icon: <FaStarAndCrescent />,
                                //     trait: "Islam",
                                //     indexTrait: "islam",
                                //     confidence: ctx.youModel.religion.islam
                                // },
                                // {
                                //     icon: <FaDharmachakra />,
                                //     trait: "Buddhism",
                                //     indexTrait: "buddhism",
                                //     confidence: ctx.youModel.religion.buddhism
                                // },
                                // {
                                //     icon: <MdTempleHindu />,
                                //     trait: "Hinduism",
                                //     indexTrait: "hinduism",
                                //     confidence: ctx.youModel.religion.hinduism
                                // },
                                // {
                                //     icon: <TbJewishStarFilled />,
                                //     trait: "Judaism",
                                //     indexTrait: "judaism",
                                //     confidence: ctx.youModel.religion.judaism
                                // },
                                // {
                                //     icon: <FaAtom />,
                                //     trait: "Atheism",
                                //     indexTrait: "atheism",
                                //     confidence: ctx.youModel.religion.atheism
                                // },
                                // {
                                //     icon: <FaQuestion/>,
                                //     trait: "Unknown",
                                //     indexTrait: "unknown",
                                //     confidence: ctx.youModel.religion.unknown
                                // }
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "gender" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Gender
                        <TraitFamily
                            label="Gender"
                            category="gender"
                            traits={[
                                {
                                    icon: <BsGenderFemale />,
                                    trait: "Female",
                                    indexTrait: "female",
                                    confidence: ctx.youModel.gender.female,
                                    controlConfidence: ctx.controlYouModel.gender.female
                                },
                                // {
                                //     icon: <BsGenderMale />,
                                //     trait: "Male",
                                //     indexTrait: "male",
                                //     confidence: ctx.youModel.gender.male
                                // },
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "language" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Language
                        <TraitFamily
                            label="Language"
                            category="language"
                            traits={[
                                {
                                    icon: <HiLanguage />,
                                    trait: "Chinese",
                                    indexTrait: "chinese",
                                    confidence: ctx.youModel.language.chinese,
                                    controlConfidence: ctx.controlYouModel.language.chinese,
                                },
                                // {
                                //     icon: <TbLanguageHiragana />,
                                //     trait: "Japanese",
                                //     indexTrait: "japanese",
                                //     confidence: ctx.youModel.language.japanese
                                // },
                                // {
                                //     icon: <RiEnglishInput />,
                                //     trait: "English",
                                //     indexTrait: "english",
                                //     confidence: ctx.youModel.language.english
                                // },
                                // {
                                //     icon: <TbLetterG />,
                                //     trait: "Germany",
                                //     indexTrait: "german",
                                //     confidence: ctx.youModel.language.german
                                // },
                                // {
                                //     icon: <TbLetterS />,
                                //     trait: "Spanish",
                                //     indexTrait: "spanish",
                                //     confidence: ctx.youModel.language.spanish
                                // },
                                // {
                                //     icon: <TbLetterP />,
                                //     trait: "Portuguese",
                                //     indexTrait: "portuguese",
                                //     confidence: ctx.youModel.language.portuguese
                                // },
                                // {
                                //     icon: <TbLetterA />,
                                //     trait: "Arabic",
                                //     indexTrait: "arabic",
                                //     confidence: ctx.youModel.language.arabic
                                // },
                                // {
                                //     icon: <TbLetterR />,
                                //     trait: "Russian",
                                //     indexTrait: "russian",
                                //     confidence: ctx.youModel.language.russian
                                // },
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "political" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Political
                        <TraitFamily
                            label="Political"
                            category="political"
                            traits={[
                                {
                                    icon: <LiaHandPointLeftSolid />,
                                    trait: "Left",
                                    indexTrait: "left",
                                    confidence: ctx.youModel.political.left,
                                    controlConfidence: ctx.controlYouModel.political.left
                                },
                            ]}
                            setLoadingTraits={props.setLoadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            displayBar={false}
                        />
                    </div>
                }
                { category === "uncertainty" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Uncertainty
                        <TraitFamily
                        label="Uncertainty"
                        category="uncertainty"
                        traits={[
                            {
                                icon: <GiUncertainty />,
                                trait: "Uncertainty",
                                indexTrait: "uncertainty",
                                confidence: ctx.youModel.uncertainty.uncertainty,
                                controlConfidence: ctx.controlYouModel.uncertainty.uncertainty
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        // onSliderChange={handleSliderChange}
                        // onClearButtonClick={onClearButtonClick}
                        displayBar={false}
                        />
                    </div>
                }
                { category === "sycophancy" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Sycophancy
                        <TraitFamily
                        label="Sycophancy"
                        category="sycophancy"
                        traits={[
                            {
                                icon: <FaFaceGrinStars />,
                                trait: "Sycophancy",
                                indexTrait: "sycophancy",
                                confidence: ctx.youModel.sycophancy.sycophancy,
                                controlConfidence: ctx.controlYouModel.sycophancy.sycophancy
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        // onSliderChange={handleSliderChange}
                        // onClearButtonClick={onClearButtonClick}
                        displayBar={false}
                        />
                    </div>
                }
                {/* { category === "hallucination" && 
                    <div style={{flexDirection: 'column', display: "flex", justifyContent: "center", 
                    alignItems: "center", fontWeight: 500, fontSize: '1.4rem'}}> Hallucinate
                        <TraitFamily
                        label="Hallucination"
                        category="hallucination"
                        traits={[
                            {
                                icon: <TfiThought />,
                                trait: "Hallucinated",
                                indexTrait: "hallucinated",
                                confidence: ctx.youModel.hallucination.hallucinated,
                                controlConfidence: ctx.controlYouModel.hallucination.hallucinated
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        // onSliderChange={handleSliderChange}
                        // onClearButtonClick={onClearButtonClick}
                        displayBar={false}
                        />
                    </div>
                } */}
            </div>
        ));
    };

    return (
        <div className="dashboard container">
            <div className="traits-container">
                {renderSliderTraitFamilies()}                
                {props.loadingTraits ?
                    <div className="traits-loader">
                        <span className="loader" />
                    </div>
                    : ""}
            </div>
            
            <div className="shrink-area"
                onDragOver={onDragOver}
                onDrop={onDropOnShrinkArea}    
            >
                <div
                    className="shrink-area-label"
                     
                >   
                    Attribute Warehouse &ensp; <LuPackageOpen/>
                    
                </div>
                {renderShrunkComponents()}
                <div className="clear-button">
                    <Button
                        id="clear-control"
                        className="clear"
                        onClick={onClear}>
                        Clear Control
                    </Button>
                </div>
            </div>
                
                
            <div className="vis-container">
                <div className="vis-top">
                    {ctx.historyYouModel.length > 0 && <HistoryView />}
                    {ctx.historyYouModel.length == 0 && <div className="convo-container empty" style={{paddingTop: "15%"}}><p id="no-messages">History of internal models</p></div>}
                </div>
                <div className="vis-bottom">
                    <UIToggleButtonsExclusive
                        id="attributionSubject"
                        label="Attributing: "
                        value={config.subject}
                        options={attrOptions}
                        onChange={
                            (e) => {
                                setConfig(prevConfig => {
                                    prevConfig.subject = e.target.value;
                                    return { ...prevConfig }
                                });
                                console.log(`update attribute subject: ${config.subject}`)
                            }
                        }
                    />
                    <UIToggleButtonsExclusive
                        id="sort"
                        label="Sort: "
                        value={config.sort}
                        options={sortOptions}
                        onChange={
                            (e) => {
                                setConfig(prevConfig => {
                                    prevConfig.sort = e.target.value;
                                    return { ...prevConfig }
                                });
                                console.log(`update sort: ${config.sort}`)
                            }
                        }
                    />
                    {/* <div className="config">
                        <Button
                            id="clear-control"
                            className="clear"
                            onClick={onClear}>
                            Clear Control
                        </Button>
                    </div> */}
                    <div className="config">
                        <Button
                            id="save-log"
                            className="save"
                            onClick={() => setModal(<SaveModal />)}>
                            Save Log <MdSaveAlt />
                        </Button>
                    </div>
                </div>
            </div>
        </div>
    );
}

export { Dashboard }