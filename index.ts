import inquirer from "inquirer";
import chalk from "chalk";
import {ChatPromptTemplate, PromptTemplate} from "langchain/prompts";
import {DynamicStructuredTool, DynamicTool} from "langchain/tools";
import {z} from "zod";
import {AgentExecutor, createOpenAIFunctionsAgent} from "langchain/agents";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import type { BaseMessage } from "langchain/schema";
import {shellAgent} from "./shell-agent.ts";
import {$ as shell} from "bun";
import {fetchFromMemory, memoryAgent, saveInnerMemory} from "./memory-agent.ts";
import {chatAgent} from "./chat-agent.ts";


const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    // model: "gpt-4-turbo",
    temperature: 0.9,
    apiKey: process.env.OPENAI_API_KEY,
});

const chatHistory: BaseMessage[] = [];

const prompt = ChatPromptTemplate.fromMessages([
    ["system", "You are agent orchestrating the conversation, use functions to interact with the system. Also create plan to achieve the goal."],
    ["placeholder", "{chat_history}"],
    ["human", "{input}"],
    ["placeholder", "{agent_scratchpad}"],
]);
const tools = [
    // new DynamicStructuredTool({
    //     name: "get-knowledge",
    //     description: "get the knowledge about anything, that is not in the current conversation",
    //     schema: z.object({
    //         question: z.string().describe("The question to ask the oracle about"),
    //     }),
    //     func: async ({question}) => {
    //         const questions = [{
    //             type: 'input',
    //             name: 'answer',
    //             message: question + ' > ',
    //         }];
    //         const answers = await inquirer.prompt(questions);
    //         return answers.answer;
    //     }
    // }),
    new DynamicStructuredTool({
        name: "save-conversation",
        description: "save current conversation to memory",
        schema: z.object({
            conversation: z.string().describe("The conversation to save to memory"),
        }),
        func: async ({conversation}) => {
            console.log(chalk.yellow('Saving conversation to memory:'), conversation);
            await memoryAgent(conversation);
            return 'Conversation saved';
        }
    }),
    new DynamicStructuredTool({
        name: "check-wiki",
        description: "get the knowledge about anything, that is not in the current conversation",
        schema: z.object({
            query: z.string().describe("The query to fetch from memory"),
        }),
        func: async ({query}) => {
            console.log(chalk.yellow('Fetching from memory:'), query);
            return await fetchFromMemory(query);
        }
    }),
    new DynamicStructuredTool({
        name: "request-system-admin",
        description: "Write request to system admin, system admin can operate underlined operating system.",
        schema: z.object({
            request: z.string().describe("The request to system admin"),
        }),
        func: async ({request}) => {
            console.log(chalk.yellow('Asking system admin:'), request);
            return await shellAgent(request);
        }
    }),

];
const agent = await createOpenAIFunctionsAgent({
    tools: tools,
    llm: llm,
    prompt: prompt,
    streamRunnable: false,
});

const agentExecutor = new AgentExecutor({
    agent,
    tools,
});

async function conversation() {
    const questions = [{
        type: 'input',
        name: 'answer',
        message: '> ',
    }];
    const answers = await inquirer.prompt(questions);
    if (answers.answer === "exit") {
        await saveInnerMemory();
        return;
    } else if (answers.answer === "clear") {
        chatHistory.length = 0;
        console.log('Chat history cleared');
    } else if (answers.answer === "memory") {
        const res = await agentExecutor.invoke({
            input: "save short summary of the conversation",
            chat_history: chatHistory,
        });
        console.log(chalk.magenta(res.output));
    } else {
        const res = await agentExecutor.invoke({
            input: answers.answer,
            chat_history: chatHistory,
        });
        const proxyResp = await chatAgent(answers.answer, res.output);

        chatHistory.push(new HumanMessage(answers.answer));
        chatHistory.push(new AIMessage(proxyResp));

        // const res = await chain.call({input: answers.answer});
        console.log(chalk.magenta(proxyResp));
    }

    await conversation();
}

// add exit listener
process.on('SIGINT', async function () {
    console.log("Caught interrupt signal");
    await saveInnerMemory();
    process.exit();
});

await conversation();


