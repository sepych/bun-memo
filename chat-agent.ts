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


const prompt = PromptTemplate.fromTemplate(`
You are reviewing the dialogue between a human and an AI.
Check input from the human "{human}" and create program using pseudo-code for response".
`);

const promptCode = PromptTemplate.fromTemplate(`
Check following pseudo-code, review it and convert it into a better python code.
{code}
`);

const parser = StructuredOutputParser.fromNamesAndDescriptions({
    output: "The output of the code",
});
const codeEmulator =  PromptTemplate.fromTemplate(`
You are capable of running pseudo code in your mind. Emulate the code in your mind and give the output.
input: {input}
code: {code}

{format_instructions}
`);

const chain = RunnableSequence.from([
    codeEmulator,
    model,
    parser,
]);


export async function chatAgent(human: string): Promise<string> {
    const text = await prompt.format({ human: human });

    const result = await model.invoke(text);
    console.log('pseudo-code', chalk.yellow(result));
    const nextText = await promptCode.format({ code: result });

    // const nextResult = await model.invoke(nextText);
    // console.log('python-code', chalk.yellow(nextResult));


    const finalResult = await chain.invoke({
        code: nextText,
        input: human,
        format_instructions: parser.getFormatInstructions(),
    });
    console.log('code-emulator', chalk.yellow(finalResult.output));
    return finalResult.output;
}

