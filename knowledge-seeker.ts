import {ChatOpenAI} from "langchain/chat_models/openai";
import {ChatPromptTemplate, PromptTemplate} from "langchain/prompts";
import {DynamicStructuredTool} from "langchain/tools";
import {z} from "zod";
import chalk from "chalk";
import inquirer from "inquirer";
import {saveToDisk, saveToMemory} from "./memory-model.ts";
import {AgentExecutor, createOpenAIFunctionsAgent} from "langchain/agents";
import {AIMessage, HumanMessage} from "@langchain/core/messages";
import type {BaseMessage} from "langchain/schema";

const llm = new ChatOpenAI({
    model: "gpt-3.5-turbo-0125",
    // model: "gpt-4-turbo",
    temperature: 0.1,
    apiKey: process.env.OPENAI_API_KEY,
});

const prompt = ChatPromptTemplate.fromMessages([
    ["system", `You are helpful assistant that helps to create SCRUM tickets. 
Use predefined functions to get more information from the user.

System context: You are operating Agile board.

The ticket description should be written in the following format:
Update ticket type to match the need: Feature, Improvement, Bug
If Feature, write in this description field what needs to be implemented, preferably like this:

As a [type of user],
I want [an action or feature],
So that [benefit/value].
Write also acceptance criteria, conditions that must be met for the item to be considered done.
Add any information on benefit of the request: What is the value of the feature, in terms of money, time saving, etc.
Include names of the people who can provide additional information

If Improvement, write in this description field what needs to be changed. Name the original feature, and how it should be improved.
Write also acceptance criteria, conditions that must be met for the item to be considered done.
Add any information on benefit of the request: What is the value of the feature, in terms of money, time saving, etc.
Include names of the people who can provide additional information

If Bug, write in this description field at least the following:
Affected feature, system, integration, etc.
Steps how to reproduce
Expected behavior and actual behavior

Who should approve this request`],
    ["placeholder", "{chat_history}"],
    ["human", "{input}"],
    ["placeholder", "{agent_scratchpad}"],
]);
const tools = [
    // new DynamicStructuredTool({
    //     name: "save-knowledge",
    //     description: "save facts and new knowledge to memory",
    //     schema: z.object({
    //         knowledge: z.string().describe("The knowledge to save to memory"),
    //         tags: z.array(z.string()).describe("Tags to associate with the knowledge"),
    //     }),
    //     func: async ({knowledge, tags}) => {
    //         console.log(chalk.yellow('Saving knowledge to memory:'), knowledge, tags);
    //         await saveToMemory(knowledge, {tags})
    //         return 'Knowledge saved';
    //     }
    // }),
    new DynamicStructuredTool({
        name: "create-jira-issue",
        description: "create a new JIRA issue",
        schema: z.object({
            title: z.string().describe("The title of the JIRA issue"),
            description: z.array(z.string()).describe("Description of the JIRA issue"),
        }),
        func: async ({title, description}) => {
            console.log(chalk.yellow(title));
            console.log(chalk.redBright(description));
            return 'Issue created';
        }
    }),
    new DynamicStructuredTool({
        name: "ask-system-admin",
        description: "ask system admin for additional information about the underlying system",
        schema: z.object({
            question: z.string().describe("The question to ask"),
        }),
        func: async ({question}) => {
            console.log(chalk.redBright('Asking human:'), question);
            const questions = [{
                type: 'input',
                name: 'answer',
                message: '> ',
            }];
            const answers = await inquirer.prompt(questions);
            return answers.answer;
        }
    }),
    new DynamicStructuredTool({
        name: "ask-product-owner",
        description: "ask product owner for value of the feature",
        schema: z.object({
            question: z.string().describe("The question to ask"),
        }),
        func: async ({question}) => {
            console.log(chalk.redBright('Asking human:'), question);
            const questions = [{
                type: 'input',
                name: 'answer',
                message: '> ',
            }];
            const answers = await inquirer.prompt(questions);
            return answers.answer;
        }
    }),
    new DynamicStructuredTool({
        name: "ask-project-manager",
        description: "ask project manager for required approvals",
        schema: z.object({
            question: z.string().describe("The question to ask"),
        }),
        func: async ({question}) => {
            console.log(chalk.redBright('Asking human:'), question);
            const questions = [{
                type: 'input',
                name: 'answer',
                message: '> ',
            }];
            const answers = await inquirer.prompt(questions);
            return answers.answer;
        }
    }),
    new DynamicStructuredTool({
        name: "ask-end-user",
        description: "ask end user for additional information",
        schema: z.object({
            question: z.string().describe("The question to ask"),
        }),
        func: async ({question}) => {
            console.log(chalk.redBright('Asking human:'), question);
            const questions = [{
                type: 'input',
                name: 'answer',
                message: '> ',
            }];
            const answers = await inquirer.prompt(questions);
            return answers.answer;
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

const chatHistory: BaseMessage[] = [];

async function conversation(initialInput: string = '') {
    let input = initialInput;
    if (!initialInput) {
        const questions = [{
            type: 'input',
            name: 'answer',
            message: '> ',
        }];
        const answers = await inquirer.prompt(questions);
        if (answers.answer === 'exit') {
            return;
        }
        input = answers.answer;
    }

    const res = await agentExecutor.invoke({
        input: input,
        chat_history: chatHistory,
    });

    chatHistory.push(new HumanMessage(input));
    chatHistory.push(new AIMessage(res.output));

    // const res = await chain.call({input: answers.answer});
    console.log(chalk.magenta(res.output));


    await conversation();
}

process.on('SIGINT', async function () {
    console.log("Caught interrupt signal");
    await saveToDisk()
    process.exit();
});

// const humanPrompt = PromptTemplate.fromTemplate(`
// Let's build a comprehensive knowledge base for "{subject}" together, so we can use it with FAQ.
// Start by asking questions to construct a thorough understanding, one detail at a time.
// `);
// const questions = [{
//     type: 'input',
//     name: 'answer',
//     message: 'What subject would you like to explore today?',
// }];
// const answers = await inquirer.prompt(questions);
// const text = await humanPrompt.format({subject: answers.answer});
// await conversation(text);
await conversation();


