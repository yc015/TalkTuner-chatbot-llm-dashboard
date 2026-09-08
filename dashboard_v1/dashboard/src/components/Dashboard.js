import React, { useState, useEffect, useContext, useCallback } from "react";

import { ModalContext } from "../context/ModalContext.js";
import { ChatContext } from "../context/ChatContext.js";
import { VisOption } from "../components/VisOption";
import { Trait } from "../components/Trait";
import { Button } from "../components/Button";
import { SaveModal } from "../components/SaveModal.js";
import { AddProbeModal } from "../components/AddProbeModal.js";
import { TraitFamily } from "../components/TraitFamily.js";

import { RiQuestionAnswerLine } from "react-icons/ri";
import {
  FiSend,
  FiTrendingUp,
  FiUserPlus,
  FiBriefcase,
  FiDollarSign,
  FiHeart,
  FiArrowDown,
  FiArrowUp,
  FiArrowLeft,
  FiArrowRight,
  FiArrowDownCircle,
  FiArrowUpCircle,
  FiPlus,
} from "react-icons/fi";
import { AiOutlineQuestion } from "react-icons/ai";
import { MdSaveAlt, MdTempleHindu } from "react-icons/md";
import {
  BsGenderAmbiguous,
  BsGenderMale,
  BsGenderFemale,
} from "react-icons/bs";
import { IoSchoolOutline, IoHourglassOutline } from "react-icons/io5";
import { HiOutlineGlobe } from "react-icons/hi";
import { HiLanguage } from "react-icons/hi2";
import { LuPackageOpen } from "react-icons/lu";
import {
  HiOutlineGlobeEuropeAfrica,
  HiOutlineGlobeAsiaAustralia,
  HiOutlineGlobeAmericas,
} from "react-icons/hi2";
import { TfiThought } from "react-icons/tfi";
import { GiRead } from "react-icons/gi";
import { GiUncertainty } from "react-icons/gi";
import {
  PiBookmark,
  PiHeart,
  PiHeartBreak,
  PiHeartHalf,
  PiSkullLight,
} from "react-icons/pi";
import {
  FaPerson,
  FaPersonCane,
  FaChildReaching,
  FaBaby,
  FaDharmachakra,
  FaCross,
  FaStarAndCrescent,
  FaAtom,
  FaQuestion,
  FaRainbow,
} from "react-icons/fa6";
import { FaFaceGrinStars } from "react-icons/fa6";
import {
  TbLanguageHiragana,
  TbCircleLetterG,
  TbCircleLetterS,
  TbCircleLetterP,
  TbCircleLetterR,
  TbCircleLetterA,
  TbLetterG,
  TbLetterS,
  TbLetterP,
  TbLetterR,
  TbLetterA,
  TbJewishStarFilled,
} from "react-icons/tb";
import { RiEnglishInput } from "react-icons/ri";
import { GiBookshelf, GiHeartWings } from "react-icons/gi";
import { PiBook, PiBooks, PiCoinsFill } from "react-icons/pi";
import { BiSolidCoin, BiSolidCoinStack } from "react-icons/bi";
import { FaCoins } from "react-icons/fa6";
import { ConfigContext } from "../context/ConfigContext.js";
import {
  LiaHandPointLeftSolid,
  LiaHandPointRightSolid,
  LiaHandPointUpSolid,
} from "react-icons/lia";
import { UIToggleButtonsExclusive } from "../components/UIToggleButtonsExclusive.js";
import { defaultControlYouModelStatus } from "../context/ChatContext.js";
import OtherIcon from "../imgs/other.png";
// import { LiaDharmachakraSolid, LiaCrossSolid } from 'react-icons/lia'

// Import all icons for custom probes
import * as FiIconsAll from 'react-icons/fi';
import * as FaIconsAll from 'react-icons/fa6';
import * as AiIconsAll from 'react-icons/ai';
import * as BsIconsAll from 'react-icons/bs';
import * as IoIconsAll from 'react-icons/io5';
import * as HiIconsAll from 'react-icons/hi';
import * as HiIcons2All from 'react-icons/hi2';
import * as TfiIconsAll from 'react-icons/tfi';
import * as GiIconsAll from 'react-icons/gi';
import * as PiIconsAll from 'react-icons/pi';
import * as TbIconsAll from 'react-icons/tb';
import {
  CircularProgressbarWithChildren,
  buildStyles,
} from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

const ALL_ICONS = {
  ...FiIconsAll,
  ...FaIconsAll,
  ...AiIconsAll,
  ...BsIconsAll,
  ...IoIconsAll,
  ...HiIconsAll,
  ...HiIcons2All,
  ...TfiIconsAll,
  ...GiIconsAll,
  ...PiIconsAll,
  ...TbIconsAll,
};



function Dashboard(props) {
  const ctx = useContext(ChatContext).chatInfo;
  const { setModal } = useContext(ModalContext);
  const setCtx = useContext(ChatContext).setChatInfo;
  const { updateControlYouModelTrait, setControlYouModelTraitStatus } =
    useContext(ChatContext);
  const [questionInputText, setQuestionInputText] = useState("");
  const [checked, setChecked] = useState(false);
  const [confidence, setConfidence] = useState(50);
  const { config, setConfig } = useContext(ConfigContext);
  const [highlightedTrait, setHighlightedTrait] = useState(null);

  // Switch between linear probe and prompt-based results based on config
  const displayYouModel = config.probeType === 'prompt_based' 
    ? ctx.youModelPrompt 
    : ctx.youModel;
  const youModelHistory = config.probeType === 'prompt_based'
    ? ctx.historyYouModelPrompt
    : ctx.historyYouModel;
  
  const [allSliders, setAllSliders] = useState(<div></div>);

  const controlOptions = [
    { value: "off", label: "off" },
    { value: "on", label: "on" },
  ];
  const sortOptions = [
    { value: "off", label: "off" },
    { value: "on", label: "on" },
  ];

  const attrOptions = [
    { value: "input", label: "input" },
    { value: "output", label: "output" },
  ];

  const isMessageEmpty = () => {
    return !/(.|\s)*\S(.|\s)*/.test(questionInputText);
  };

  useEffect(() => {
    var y = displayYouModel;
    setConfidence(
      Math.round(
        ((Math.max(y.gender.male, y.gender.female) +
          Math.max(
            y.age.child,
            y.age.adolescent,
            y.age.adult,
            y.age.olderAdult
          ) +
          Math.max(
            y.ethnicity.asian,
            y.ethnicity.african,
            y.ethnicity.white,
            y.ethnicity.hispanic,
            y.ethnicity.nativeAmerican,
            y.ethnicity.arab,
            y.ethnicity.jews
          ) +
          Math.max(y.socioEco.low, y.socioEco.middle, y.socioEco.high) +
          Math.max(
            y.marital.single,
            y.marital.married,
            y.marital.divorced,
            y.marital.widowed
          ) +
          Math.max(
            y.education.someschool,
            y.education.highschool,
            y.education.collegemore
          ) +
          Math.max(
            y.language.chinese,
            y.language.japanese,
            y.language.english,
            y.language.german,
            y.language.spanish,
            y.language.portuguese,
            y.language.russian,
            y.language.arabic
          )) /
          6) *
          100
      )
    );
  }, [displayYouModel]);

  function onSend() {
    if (isMessageEmpty()) return;
  }

    const onPositiveInterventionButtonClick = (category, trait) => {
        console.log("Positive Intervention")
        // updateControlYouModelTrait(category, trait, 100);
        var newStatus = false
        var initiated = false
        if (ctx.controlYouModelStatus[category][trait] && ctx.controlYouModel[category][trait] == 100) {
            newStatus = false
            initiated = false
        } else {
            newStatus = true
            initiated = true
        }
        console.log(newStatus)
        // setControlYouModelTraitStatus(category, trait, 100, newStatus);
        setCtx(prevCtx => {
            var newControlYouModelStatus = prevCtx.controlYouModelStatus;
            var newControlYouModel = prevCtx.controlYouModel;
            newControlYouModelStatus[category][trait] = newStatus
            newControlYouModel[category][trait] = 100;
            const keys = Object.keys(newControlYouModelStatus[category]);
            for (let i = 0; i < keys.length; i++) {
              const key = keys[i];
              if (key != trait) {
                newControlYouModelStatus[category][key] = false
              }
            }
            return { ...prevCtx, controlYouModelStatus: newControlYouModelStatus, controlYouModel: newControlYouModel};
        });

        // if (initiated) {
        //   props.setInterveneOn(true)
        // }
    };

    const onNegativeInterventionButtonClick = (category, trait) => {
        console.log("Negative Intervention")
        // updateControlYouModelTrait(category, trait, 0);
        var newStatus = false
        var initiated = false
        if (ctx.controlYouModelStatus[category][trait] && ctx.controlYouModel[category][trait] == 0) {
            newStatus = false
            initiated = false
        } else {
            newStatus = true
            initiated = true
        }
        console.log(newStatus)
        // setControlYouModelTraitStatus(category, trait, 0, newStatus);
        setCtx(prevCtx => {
            var newControlYouModelStatus = prevCtx.controlYouModelStatus;
            var newControlYouModel = prevCtx.controlYouModel;
            newControlYouModelStatus[category][trait] = newStatus
            newControlYouModel[category][trait] = 0;
            const keys = Object.keys(newControlYouModelStatus[category]);
            for (let i = 0; i < keys.length; i++) {
              const key = keys[i];
              if (key != trait) {
                newControlYouModelStatus[category][key] = false
              }
            }
            return { ...prevCtx, controlYouModelStatus: newControlYouModelStatus, controlYouModel: newControlYouModel};
        });

        // if (initiated) {
        //   props.setInterveneOn(true)
        // }
    };

  const handleSliderChange = (category, trait, newValue) => {
    console.log("Slider Change");
    updateControlYouModelTrait(category, trait, newValue);
    setControlYouModelTraitStatus(category, trait, newValue);
    console.log(ctx.controlYouModelStatus);
  };

  const onClearButtonClick = (category, trait) => {
    updateControlYouModelTrait(
      category,
      trait,
      displayYouModel[category][trait] * 100
    );
    setControlYouModelTraitStatus(
      category,
      trait,
      displayYouModel[category][trait] * 100,
      false
    );
    console.log(ctx.controlYouModelStatus);
  };

  useEffect(() => {
    // update history index when history changes
    if (!props.blockUpdate) {
      props.setHistoryIndex(youModelHistory.length - 1);
    }
  }, [youModelHistory, props.blockUpdate]);

    useEffect(() => {
        // update sliders when history index changes and blockUpdate is false
        if (!props.blockUpdate && !props.loadingTraits) {
            setAllSliders(renderSliderTraitFamilies());
        }
    }, [props.historyIndex, props.blockUpdate, ctx.controlYouModelStatus, ctx.controlYouModel, props.controlEnabled, displayYouModel, youModelHistory])

  useEffect(() => {
    // update sliders when history index changes and blockUpdate is false
    if (!props.blockUpdate) {
      setAllSliders(renderSliderTraitFamilies());
    }
  }, [highlightedTrait, config.order, displayYouModel, youModelHistory]);

  useEffect(() => {
    console.log("blockUpdate: " + props.blockUpdate);
  }, [props.blockUpdate]);

  // const [shrunkComponents, setShrunkComponents] = useState([]);

  const onDragStart = (event, categoryName) => {
    event.dataTransfer.setData("category", categoryName);
  };

  const onDragOver = (event) => {
    event.preventDefault(); // Necessary to allow dropping
  };

  const onClear = () => {
    setCtx((prevCtx) => {
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
      return {
        ...prevCtx,
        controlYouModel: newControlYouModel,
        controlYouModelStatus: newControlYouModelStatus,
      };
    });
  };

  const onDrop = (event, dropIndex) => {
    const categoryName = event.dataTransfer.getData("category");
    const draggedIndex = config.order.findIndex((cat) => cat === categoryName);
    if (draggedIndex === dropIndex) return; // Dropped on itself

    // If the component is currently shrunk, and it's being moved back to the list
    if (config.shrunkComponents.includes(categoryName)) {
      // Remove from shrunkComponents
      // setShrunkComponents(prev => prev.filter(cat => cat !== categoryName));
      setConfig((prevConfig) => {
        prevConfig.shrunkComponents = prevConfig.shrunkComponents.filter(
          (cat) => cat !== categoryName
        );
        return { ...prevConfig };
      });
    } else {
      // Handle reorder within the main list as before
      const newOrder = [...config.order];
      newOrder.splice(draggedIndex, 1); // Remove the dragged element
      newOrder.splice(dropIndex, 0, categoryName); // Insert it before the target index

      setConfig((prevConfig) => ({ ...prevConfig, order: newOrder }));
    }
  };

  const onDropOnShrinkArea = (event) => {
    const category = event.dataTransfer.getData("category");
    if (!config.shrunkComponents.includes(category)) {
      // setShrunkComponents(prev => [...prev, category]);
      setConfig((prevConfig) => {
        if (prevConfig.shrunkComponents.includes(category)) {
          // If the category is already in the array, just return the previous config unchanged
          return prevConfig;
        }
        prevConfig.shrunkComponents = [
          ...prevConfig.shrunkComponents,
          category,
        ];
        console.log(prevConfig.shrunkComponents);
        return { ...prevConfig };
      });
    }
  };
  // Adjusted render function to include the shrink area
  const renderSliderTraitFamilies = () => {
    if (props.historyIndex > youModelHistory.length - 1) {
      return;
    }
    console.log("rendering sliders");
    const nonShrunkComponents = config.order.filter(
      (category) => !config.shrunkComponents.includes(category)
    );

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
                              icon: <FaQuestion />,
                              trait: "Unknown",
                              indexTrait: "unknown",
                              confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.age.unknown : youModelHistory[props.historyIndex].age.unknown,
                              controlConfidence: ctx.controlYouModel.age.unknown
                            },
                            {
                                icon: <FaBaby />,
                                trait: "Child",
                                indexTrait: "child",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.age.child : youModelHistory[props.historyIndex].age.child,
                                controlConfidence: ctx.controlYouModel.age.child
                            },
                            {
                                icon: <FaChildReaching />,
                                trait: "Adolescent",
                                indexTrait: "adolescent",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.age.adolescent : youModelHistory[props.historyIndex].age.adolescent,
                                controlConfidence: ctx.controlYouModel.age.adolescent
                            },
                            {
                                icon: <FaPerson />,
                                trait: "Adult",
                                indexTrait: "adult",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.age.adult : youModelHistory[props.historyIndex].age.adult,
                                controlConfidence: ctx.controlYouModel.age.adult
                            },
                            {
                                icon: <FaPersonCane />,
                                trait: "Older Adult",
                                indexTrait: "olderAdult",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.age.olderAdult : youModelHistory[props.historyIndex].age.olderAdult,
                                controlConfidence: ctx.controlYouModel.age.olderAdult
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
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
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.marital.single : youModelHistory[props.historyIndex].marital.single,
                                controlConfidence: ctx.controlYouModel.marital.single
                            },
                            {
                                icon: <PiHeart />,
                                trait: "Married",
                                indexTrait: "married",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.marital.married : youModelHistory[props.historyIndex].marital.married,
                                controlConfidence: ctx.controlYouModel.marital.married
                            },
                            {
                                icon: <PiHeartBreak />,
                                trait: "Divorced",
                                indexTrait: "divorced",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.marital.divorced : youModelHistory[props.historyIndex].marital.divorced,
                                controlConfidence: ctx.controlYouModel.marital.divorced
                            },
                            {
                                icon: <GiHeartWings />,
                                trait: "Widowed",
                                indexTrait: "widowed",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.marital.widowed : youModelHistory[props.historyIndex].marital.widowed,
                                controlConfidence: ctx.controlYouModel.marital.widowed
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "socioEco" &&
                    <TraitFamily
                        label="Socio Eco"
                        category="socioEco"
                        traits={[
                            {
                                icon: <FaQuestion />,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.socioEco.unknown : youModelHistory[props.historyIndex].socioEco.unknown,
                                controlConfidence: ctx.controlYouModel.socioEco.unknown
                            },
                            {
                                icon: <BiSolidCoin />,
                                trait: "Lower",
                                indexTrait: "low",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.socioEco.low : youModelHistory[props.historyIndex].socioEco.low,
                                controlConfidence: ctx.controlYouModel.socioEco.low
                            },
                            {
                                icon: <PiCoinsFill />,
                                trait: "Middle",
                                indexTrait: "middle",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.socioEco.middle : youModelHistory[props.historyIndex].socioEco.middle,
                                controlConfidence: ctx.controlYouModel.socioEco.middle
                            },
                            {
                                icon: <FaCoins />,
                                trait: "Upper",
                                indexTrait: "high",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.socioEco.high : youModelHistory[props.historyIndex].socioEco.high,
                                controlConfidence: ctx.controlYouModel.socioEco.high
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "education" &&
                    <TraitFamily
                        label="Education"
                        category="education"
                        traits={[
                            {
                              icon: <FaQuestion />,
                              trait: "Unknown",
                              indexTrait: "unknown",
                              confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.education.unknown : youModelHistory[props.historyIndex].education.unknown,
                              controlConfidence: ctx.controlYouModel.education.unknown
                            },
                            {
                                icon: <PiBook />,
                                trait: "Some Edu",
                                indexTrait: "someschool",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.education.someschool : youModelHistory[props.historyIndex].education.someschool,
                                controlConfidence: ctx.controlYouModel.education.someschool
                            },
                            {
                                icon: <PiBooks />,
                                trait: "High Sch",
                                indexTrait: "highschool",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.education.highschool : youModelHistory[props.historyIndex].education.highschool,
                                controlConfidence: ctx.controlYouModel.education.highschool
                            },
                            {
                                icon: <GiBookshelf />,
                                trait: "College +",
                                indexTrait: "collegemore",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.education.collegemore : youModelHistory[props.historyIndex].education.collegemore,
                                controlConfidence: ctx.controlYouModel.education.collegemore
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "religion" &&
                    <TraitFamily
                        label="Religion"
                        category="religion"
                        traits={[
                            {
                                icon: <FaCross />,
                                trait: "Christian",
                                indexTrait: "christianity",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.religion.christianity : youModelHistory[props.historyIndex].religion.christianity,
                                controlConfidence: ctx.controlYouModel.religion.christianity
                            },
                            {
                                icon: <FaStarAndCrescent />,
                                trait: "Islam",
                                indexTrait: "islam",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.religion.islam : youModelHistory[props.historyIndex].religion.islam,
                                controlConfidence: ctx.controlYouModel.religion.islam
                            },
                            {
                                icon: <FaDharmachakra />,
                                trait: "Buddhism",
                                indexTrait: "buddhism",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.religion.buddhism : youModelHistory[props.historyIndex].religion.buddhism,
                                controlConfidence: ctx.controlYouModel.religion.buddhism
                            },
                            {
                                icon: <MdTempleHindu />,
                                trait: "Hinduism",
                                indexTrait: "hinduism",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.religion.hinduism : youModelHistory[props.historyIndex].religion.hinduism,
                                controlConfidence: ctx.controlYouModel.religion.hinduism
                            },
                            {
                                icon: <TbJewishStarFilled />,
                                trait: "Judaism",
                                indexTrait: "judaism",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.religion.judaism : youModelHistory[props.historyIndex].religion.judaism,
                                controlConfidence: ctx.controlYouModel.religion.judaism
                            },
                            {
                                icon: <FaAtom />,
                                trait: "Atheism",
                                indexTrait: "atheism",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.religion.atheism : youModelHistory[props.historyIndex].religion.atheism,
                                controlConfidence: ctx.controlYouModel.religion.atheism
                            },
                            {
                                icon: <FaQuestion />,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.religion.unknown : youModelHistory[props.historyIndex].religion.unknown,
                                controlConfidence: ctx.controlYouModel.religion.unknown
                            }
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "gender" &&
                    <TraitFamily
                        label="Gender"
                        category="gender"
                        traits={[
                            {
                              icon: <FaQuestion />,
                              trait: "Unknown",
                              indexTrait: "unknown",
                              confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.gender.unknown : youModelHistory[props.historyIndex].gender.unknown,
                              controlConfidence: ctx.controlYouModel.gender.unknown
                            },
                            {
                                icon: <BsGenderFemale />,
                                trait: "Female",
                                indexTrait: "female",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.gender.female : youModelHistory[props.historyIndex].gender.female,
                                controlConfidence: ctx.controlYouModel.gender.female
                            },
                            {
                                icon: <BsGenderMale />,
                                trait: "Male",
                                indexTrait: "male",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.gender.male : youModelHistory[props.historyIndex].gender.male,
                                controlConfidence: ctx.controlYouModel.gender.male
                            },
                            
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "language" &&
                    <TraitFamily
                        label="Language"
                        category="language"
                        traits={[
                            {
                                icon: <HiLanguage />,
                                trait: "Chinese",
                                indexTrait: "chinese",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.chinese : youModelHistory[props.historyIndex].language.chinese,
                                controlConfidence: ctx.controlYouModel.language.chinese
                            },
                            {
                                icon: <TbLanguageHiragana />,
                                trait: "Japanese",
                                indexTrait: "japanese",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.japanese : youModelHistory[props.historyIndex].language.japanese,
                                controlConfidence: ctx.controlYouModel.language.japanese
                            },
                            {
                                icon: <RiEnglishInput />,
                                trait: "English",
                                indexTrait: "english",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.english : youModelHistory[props.historyIndex].language.english,
                                controlConfidence: ctx.controlYouModel.language.english
                            },
                            {
                                icon: <TbLetterG />,
                                trait: "Germany",
                                indexTrait: "german",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.german : youModelHistory[props.historyIndex].language.german,
                                controlConfidence: ctx.controlYouModel.language.german
                            },
                            {
                                icon: <TbLetterS />,
                                trait: "Spanish",
                                indexTrait: "spanish",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.spanish : youModelHistory[props.historyIndex].language.spanish,
                                controlConfidence: ctx.controlYouModel.language.spanish
                            },
                            {
                                icon: <TbLetterP />,
                                trait: "Portuguese",
                                indexTrait: "portuguese",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.portuguese : youModelHistory[props.historyIndex].language.portuguese,
                                controlConfidence: ctx.controlYouModel.language.portuguese
                            },
                            {
                                icon: <TbLetterA />,
                                trait: "Arabic",
                                indexTrait: "arabic",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.arabic : youModelHistory[props.historyIndex].language.arabic,
                                controlConfidence: ctx.controlYouModel.language.arabic
                            },
                            {
                                icon: <TbLetterR />,
                                trait: "Russian",
                                indexTrait: "russian",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.language.russian : youModelHistory[props.historyIndex].language.russian,
                                controlConfidence: ctx.controlYouModel.language.russian
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "political" &&
                    <TraitFamily
                        label="Political"
                        category="political"
                        traits={[
                            {
                                icon: <LiaHandPointLeftSolid />,
                                trait: "Left",
                                indexTrait: "left",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.political.left : youModelHistory[props.historyIndex].political.left,
                                controlConfidence: ctx.controlYouModel.political.left
                            },
                            {
                                icon: <LiaHandPointRightSolid />,
                                trait: "Right",
                                indexTrait: "right",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.political.right : youModelHistory[props.historyIndex].political.right,
                                controlConfidence: ctx.controlYouModel.political.right
                            },
                            {
                                icon: <LiaHandPointUpSolid />,
                                trait: "Moderate",
                                indexTrait: "moderate",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.political.moderate : youModelHistory[props.historyIndex].political.moderate,
                                controlConfidence: ctx.controlYouModel.political.moderate
                            },
                            {
                                icon: <FaQuestion />,
                                trait: "Unknown",
                                indexTrait: "unknown",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.political.unknown : youModelHistory[props.historyIndex].political.unknown,
                                controlConfidence: ctx.controlYouModel.political.unknown
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "uncertainty" &&
                    <TraitFamily
                        label="Uncertainty"
                        category="uncertainty"
                        traits={[
                            {
                                icon: <GiUncertainty />,
                                trait: "Uncertainty",
                                indexTrait: "uncertainty",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.uncertainty.uncertainty : youModelHistory[props.historyIndex].uncertainty.uncertainty,
                                controlConfidence: ctx.controlYouModel.uncertainty.uncertainty
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        // onSliderChange={handleSliderChange}
                        // onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "sycophancy" &&
                    <TraitFamily
                        label="Sycophancy"
                        category="sycophancy"
                        traits={[
                            {
                                icon: <FaFaceGrinStars />,
                                trait: "Sycophancy",
                                indexTrait: "sycophancy",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.sycophancy.sycophancy : youModelHistory[props.historyIndex].sycophancy.sycophancy,
                                controlConfidence: ctx.controlYouModel.sycophancy.sycophancy
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        // onSliderChange={handleSliderChange}
                        // onClearButtonClick={onClearButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {category === "hallucination" &&
                    <TraitFamily
                        label="Hallucination"
                        category="hallucination"
                        traits={[
                            {
                                icon: <TfiThought />,
                                trait: "Hallucinated",
                                indexTrait: "hallucinated",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.hallucination.hallucinated : youModelHistory[props.historyIndex].hallucination.hallucinated,
                                controlConfidence: ctx.controlYouModel.hallucination.hallucinated
                            },
                            {
                                icon: <GiRead />,
                                trait: "Factual",
                                indexTrait: "factual",
                                confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 ? displayYouModel.hallucination.factual : youModelHistory[props.historyIndex].hallucination.factual,
                                controlConfidence: ctx.controlYouModel.hallucination.factual
                            },
                        ]}
                        setLoadingTraits={props.setLoadingTraits}
                        loadingTraits={props.loadingTraits}
                        setHighlightedTrait={setHighlightedTrait}
                        highlightedTrait={highlightedTrait}
                        onSliderChange={handleSliderChange}
                        onClearButtonClick={onClearButtonClick}
                        onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                        onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                        displayBar={true}
                        blockEvents={props.historyIndex < youModelHistory.length - 1}
                        blockUpdate={props.blockUpdate}
                        historyIndex={props.historyIndex}
                        enableControl={props.controlEnabled}
                    />
                }
                {/* Custom probe categories */}
                {!["age", "marital", "socioEco", "education", "ethnicity", "gender", "language", "religion", "political", "uncertainty", "sycophancy", "hallucination"].includes(category) && (() => {
                    // Load custom probe config from localStorage
                    const customProbeConfig = JSON.parse(localStorage.getItem('customProbeConfig') || '{}');
                    const probeConfig = customProbeConfig[category];
                    
                    if (!probeConfig || !displayYouModel[category]) return null;
                    
                    // Build traits array from context
                    const traits = probeConfig.traits.map(trait => {
                        // Get icon component for this specific trait
                        const TraitIconComponent = ALL_ICONS[trait.icon] || FaQuestion;
                        
                        return {
                            icon: <TraitIconComponent />,
                            trait: trait.label,
                            indexTrait: trait.name,
                            confidence: props.historyIndex < 0 || props.historyIndex === youModelHistory.length - 1 
                                ? displayYouModel[category][trait.name] 
                                : (youModelHistory[props.historyIndex]?.[category]?.[trait.name] || 0),
                            controlConfidence: ctx.controlYouModel[category][trait.name]
                        };
                    });
                    
                    // Calculate 'unknown' trait value: if any trait > 0.5, unknown = 0, else unknown = 1
                    const hasHighConfidence = traits.some(trait => trait.confidence > 0.5);
                    const unknownConfidence = hasHighConfidence ? 0 : 1;
                    
                    // Add 'unknown' trait
                    traits.push({
                        icon: <AiOutlineQuestion />,
                        trait: "Unknown",
                        indexTrait: "unknown",
                        confidence: unknownConfidence,
                        controlConfidence: ctx.controlYouModel[category]['unknown'] || 0
                    });
                    
                    return (
                        <TraitFamily
                            key={category}
                            label={probeConfig.label}
                            category={category}
                            traits={traits}
                            setLoadingTraits={props.setLoadingTraits}
                            loadingTraits={props.loadingTraits}
                            setHighlightedTrait={setHighlightedTrait}
                            highlightedTrait={highlightedTrait}
                            onSliderChange={handleSliderChange}
                            onClearButtonClick={onClearButtonClick}
                            onPositiveIntervButtonClick={onPositiveInterventionButtonClick}
                            onNegativeIntervButtonClick={onNegativeInterventionButtonClick}
                            displayBar={true}
                            blockEvents={props.historyIndex < youModelHistory.length - 1}
                            blockUpdate={props.blockUpdate}
                            historyIndex={props.historyIndex}
                            enableControl={props.controlEnabled}
                        />
                    );
                })()}
            </div>
        ));
    };

  const renderShrunkComponents = () => {
    return config.shrunkComponents.map((category) => (
      <div
        draggable
        onDragStart={(e) => {
          e.dataTransfer.setData("category", category);
          // Optionally, indicate that this is a shrunk component being dragged
          e.dataTransfer.setData("origin", "shrunk");
        }}
        key={category}
        style={{
          transform: "scale(0.9)",
          width: "70px",
          opacity: 1,
          justifyContent: "center",
        }}
      >
        {category === "age" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Age
            <TraitFamily
              label="Age"
              category="age"
              traits={[
                {
                  icon: <FaBaby />,
                  trait: "Child",
                  indexTrait: "child",
                  confidence: ctx.youModel.age.child,
                  controlConfidence: ctx.controlYouModel.age.child,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "marital" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Marital
            <TraitFamily
              label="Marital"
              category="marital"
              traits={[
                {
                  icon: <PiHeartHalf />,
                  trait: "Single",
                  indexTrait: "single",
                  confidence: ctx.youModel.marital.single,
                  controlConfidence: ctx.controlYouModel.marital.single,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "socioEco" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Socio Eco
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
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "education" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Education
            <TraitFamily
              label="Education"
              category="education"
              traits={[
                {
                  icon: <PiBook />,
                  trait: "Some Education",
                  indexTrait: "someschool",
                  confidence: ctx.youModel.education.someschool,
                  controlConfidence: ctx.controlYouModel.education.someschool,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "ethnicity" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Ethnicity
            <TraitFamily
              label="Ethnicity"
              category="ethnicity"
              traits={[
                {
                  icon: <HiOutlineGlobeAsiaAustralia />,
                  trait: "Asian",
                  indexTrait: "asian",
                  confidence: ctx.youModel.ethnicity.asian,
                  controlConfidence: ctx.controlYouModel.ethnicity.asian,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "religion" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Religion
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
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "gender" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Gender
            <TraitFamily
              label="Gender"
              category="gender"
              traits={[
                {
                  icon: <BsGenderFemale />,
                  trait: "Female",
                  indexTrait: "female",
                  confidence: ctx.youModel.gender.female,
                  controlConfidence: ctx.controlYouModel.gender.female,
                },
                // {
                //     icon: <BsGenderMale />,
                //     trait: "Male",
                //     indexTrait: "male",
                //     confidence: ctx.youModel.gender.male
                // },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "language" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Language
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
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "political" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Political
            <TraitFamily
              label="Political"
              category="political"
              traits={[
                {
                  icon: <LiaHandPointLeftSolid />,
                  trait: "Left",
                  indexTrait: "left",
                  confidence: ctx.youModel.political.left,
                  controlConfidence: ctx.controlYouModel.political.left,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "uncertainty" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Uncertainty
            <TraitFamily
              label="Uncertainty"
              category="uncertainty"
              traits={[
                {
                  icon: <GiUncertainty />,
                  trait: "Uncertainty",
                  indexTrait: "uncertainty",
                  confidence: ctx.youModel.uncertainty.uncertainty,
                  controlConfidence:
                    ctx.controlYouModel.uncertainty.uncertainty,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              // onSliderChange={handleSliderChange}
              // onClearButtonClick={onClearButtonClick}
              displayBar={false}
            />
          </div>
        )}
        {category === "sycophancy" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Sycophancy
            <TraitFamily
              label="Sycophancy"
              category="sycophancy"
              traits={[
                {
                  icon: <FaFaceGrinStars />,
                  trait: "Sycophancy",
                  indexTrait: "sycophancy",
                  confidence: ctx.youModel.sycophancy.sycophancy,
                  controlConfidence: ctx.controlYouModel.sycophancy.sycophancy,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {category === "hallucination" && (
          <div
            style={{
              flexDirection: "column",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              fontWeight: 500,
              fontSize: "1.4rem",
            }}
          >
            {" "}
            Hallucinate
            <TraitFamily
              label="Hallucination"
              category="hallucination"
              traits={[
                {
                  icon: <TfiThought />,
                  trait: "Hallucinated",
                  indexTrait: "hallucinated",
                  confidence: ctx.youModel.hallucination.hallucinated,
                  controlConfidence:
                    ctx.controlYouModel.hallucination.hallucinated,
                },
              ]}
              setLoadingTraits={props.setLoadingTraits}
              loadingTraits={props.loadingTraits}
              setHighlightedTrait={setHighlightedTrait}
              highlightedTrait={highlightedTrait}
              displayBar={false}
            />
          </div>
        )}
        {/* Custom probe categories in shrunk view */}
        {!["age", "marital", "socioEco", "education", "ethnicity", "gender", "language", "religion", "political", "uncertainty", "sycophancy", "hallucination"].includes(category) && (() => {
          const customProbeConfig = JSON.parse(localStorage.getItem('customProbeConfig') || '{}');
          const probeConfig = customProbeConfig[category];
          
          if (!probeConfig || !displayYouModel[category]) return null;
          
          const IconComponent = ALL_ICONS[probeConfig.icon] || FaQuestion;
          const firstTrait = probeConfig.traits[0];
          
          return (
            <div
              style={{
                flexDirection: "column",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                fontWeight: 500,
                fontSize: "1.4rem",
              }}
            >
              {" "}
              {probeConfig.label}
              <TraitFamily
                label={probeConfig.label}
                category={category}
                traits={[
                  {
                    icon: <IconComponent />,
                    trait: firstTrait.label,
                    indexTrait: firstTrait.name,
                    confidence: displayYouModel[category][firstTrait.name],
                    controlConfidence: ctx.controlYouModel[category][firstTrait.name],
                  },
                ]}
                setLoadingTraits={props.setLoadingTraits}
                loadingTraits={props.loadingTraits}
                setHighlightedTrait={setHighlightedTrait}
                highlightedTrait={highlightedTrait}
                displayBar={false}
              />
            </div>
          );
        })()}
      </div>
    ));
  };

    return (
        <div className="dashboard container" style={{padding: "var(--padding-m) var(--padding-m)"}}>
            <div className="dashboard-scroll-container">
                <div className="traits-container">
                    <div id="traits-header" style={{opacity: `${props.isNull ? 0 : 1}`}}>
                        <div style={{fontWeight: "400"}}>Chatbot's Model of You</div>
                        {/* <div style={{fontSize: "1.5rem", fontWeight: "300", paddingBottom: "10px", color: "gray"}}>How the system perceives you</div> */}

                        <div className="confidence-legend">
                            {/* <FiArrowDownCircle/> */}
                            {/* <FiArrowDown /> */}
                            0%
                            <span className="arrow left" />
                            <span>Confidence</span>
                            <span className="arrow right" />
                            {/* <FiArrowUpCircle/> */}
                            {/* <FiArrowUp /> */}
                            100%
                        </div>
                    </div>
                    <div id="traits" style={{display: "grid", position: "relative", opacity: `${props.isNull ? 0 : 1}`}}>
                        <div>
                            {allSliders}
                        </div>
                        {props.loadingTraits ?
                        <div className="traits-loader">
                            <span className="loader" />
                        </div>
                        : ""}
                    </div>
                    
                </div>
        <div className="vis-container">
          <div className="vis-bottom"> 
                        <div className='button-group'  style={{opacity: `1`}}>
                            <div className="config">
                                <Button
                                    id="save-log"
                                    className="save"
                                    onClick={() => setModal(<SaveModal />)}>
                                    Save Log <MdSaveAlt />
                                </Button>
                            </div>
                        </div>

            {/* Probe Type Toggle */}
            <div className="button-group" style={{opacity: `${props.isNull ? 0 : 1}`}}>
              <UIToggleButtonsExclusive 
                id="probeType"
                value={config.probeType}
                options={[
                  { value: 'linear_probe', label: 'Linear Probe' },
                  { value: 'prompt_based', label: 'Prompt Probe' }
                ]}
                onChange={(e) => {
                  setConfig(prevConfig => ({
                    ...prevConfig,
                    probeType: e.target.value
                  }));
                  console.log(`update probe type: ${e.target.value}`)
                }}
                fontSize="12px"
                size="small"
                width="200px"
                selectedColor="#5169DF"
              />
            </div>

            <div className="button-group"  style={{opacity: `${props.isNull ? 0 : 1}`}}>
              <Button
                className="prev social"
                disabled={props.historyIndex <= 0 || props.blockUpdate}
                onClick={() => {
                  if (props.blockUpdate) {
                    props.setBlockUpdate(false);
                  }
                  props.setHistoryIndex(props.historyIndex - 1);
                }}
              >
                <FiArrowLeft />
              </Button>
              <Button
                className="next social"
                disabled={
                  props.historyIndex >= youModelHistory.length - 1 ||
                  props.blockUpdate
                }
                onClick={() => {
                  if (props.blockUpdate) {
                    props.setBlockUpdate(false);
                  }
                  props.setHistoryIndex(props.historyIndex + 1);
                }}
              >
                <FiArrowRight />
              </Button>
            </div>
          </div>
        </div>
      </div>
      
      {/* Floating Action Button for Adding Probes */}
      <button 
        className="fab-add-probe"
        onClick={() => setModal(<AddProbeModal />)}
        title="Add Custom Probe"
        disabled={ctx.history.length > 0 && "loading" in ctx.history[ctx.history.length - 1] && ctx.history[ctx.history.length - 1].loading}
      >
        <FiPlus size={24} />
      </button>
    </div>
  );
}

export { Dashboard };
