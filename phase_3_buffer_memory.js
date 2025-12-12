import { ChatOpenAI } from "@langchain/openai";
import { BufferMemory } from "@langchain/classic/memory";
import { HumanMessage } from "@langchain/core/messages";
import * as dotenv from "dotenv";
dotenv.config();

// 初始化模型（DeepSeek，可按需替换）
const model = new ChatOpenAI({
    apiKey: process.env.DEEPSEEK_API_KEY,
    model: "deepseek-chat",
    configuration: { baseURL: "https://api.deepseek.com/v1" },
    temperature: 0.7,
});

// 使用官方 BufferMemory 管理历史
const memory = new BufferMemory({
    returnMessages: true,  // 规则1：存入和取出的是“消息对象”，而非字符串。
    memoryKey: "history",   // 规则2：从这个仓库取东西时，用“history”这个标签。
    inputKey: "input",      // 规则3：你给“用户说的话”贴上“input”标签再存进来。
    outputKey: "response",  // 规则4：你给“AI的回复”贴上“response”标签再存进来。
});

async function chat(input) {
    // 加载当前内存变量
    const { history } = await memory.loadMemoryVariables();
    
    // 调用模型并生成响应
    const aiResponse = await model.invoke([...history, new HumanMessage(input)]);

    // 保存输入和模型的响应到内存
    await memory.saveContext({ input }, { response: aiResponse });

    return aiResponse;
}

async function main() {
    const response1 = await chat("你好，请叫我小王。");
    console.log(`你: 你好，请叫我小王。`);
    console.log(`AI: ${response1.content}\n`);

    const response2 = await chat("你还记得我的名字吗？");
    console.log(`你: 你还记得我的名字吗？`);
    console.log(`AI: ${response2.content}\n`);

    // 打印当前内存内容
    const { history } = await memory.loadMemoryVariables({});
    console.log("=== 当前记忆内容 ===");
    console.log(JSON.stringify(history, null, 2));
}

main().catch(console.error);
