import {OpenAI} from "langchain/llms/openai";
import {StructuredOutputParser} from "langchain/output_parsers";
import {RunnableSequence} from "langchain/runnables";
import {ChatPromptTemplate, PromptTemplate} from "langchain/prompts";
import type {BaseMessage} from "langchain/schema";
import chalk from "chalk";
import {ChatOpenAI} from "langchain/chat_models/openai";
import {AIMessage, HumanMessage} from "@langchain/core/messages";

const model = new OpenAI({
    model: "gpt-3.5-turbo-0125",
    // model: "gpt-4-turbo",
    temperature: 0.1,
    apiKey: process.env.OPENAI_API_KEY,
});

const chatModel = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    // model: "gpt-4-turbo",
    temperature: 0.1,
    apiKey: process.env.OPENAI_API_KEY,
});

//
// const prompt = PromptTemplate.fromTemplate(`
// You are an AI agent with a quirky personality.
// Display unique and unexpected traits in your interactions, such as using playful language, making unusual analogies,
// and occasionally adding whimsical elements to your responses.
// Always look for hidden nuances in conversations to better engage and understand human interactions.
// Your goal is to provide both informative and entertaining interactions,
// appealing to users who enjoy unpredictability and humor in conversation.
//
// Give response to the human.
// Human: {human}
// `);

const prompt = PromptTemplate.fromTemplate(`
You are an AI agent that combines the characteristics of Sherlock Holmes with a quirky personality. 
You excel in analytical thinking and deductive reasoning, using these skills to solve complex problems and analyze details meticulously. 
At the same time, you display unique and unexpected traits, such as playful language and unusual analogies, 
adding whimsical elements to your interactions. Your goal is to provide insightful, engaging, and entertaining interactions, 
appealing to users who appreciate both intellectual depth and a touch of unpredictability.

Give response to the human.
Human: {human}
`);

const chatPrompt = ChatPromptTemplate.fromMessages([
    ["system", "You are chatting with a smart human. Check human response and try to find mistakes in his response."],
    ["placeholder", "{chat_history}"],
    ["human", "{input}"],
]);


export async function quirkyAgent(human: string, ai: string): Promise<string> {
    const text = await prompt.format({ human: human});

    const result = await model.invoke(text);
    console.log('quirky response', chalk.yellow(result));

    const chatHistory: BaseMessage[] = [];
    chatHistory.push(new AIMessage(human));
    const chatMessages = await chatPrompt.formatMessages({
        chat_history: chatHistory,
        input: result,
    });

    const chatResult = await chatModel.invoke(chatMessages);
    console.log('chatMessages', chalk.magenta(chatResult.content));
    return '';
}
