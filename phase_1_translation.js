import { ChatOpenAI } from "@langchain/openai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import * as dotenv from 'dotenv';
dotenv.config();

// 1. 初始化模型
const model = new ChatOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  model: "deepseek-chat",
  configuration: { baseURL: "https://api.deepseek.com/v1" },
});

// 2. 创建提示词模板：定义输入变量
const promptTemplate = PromptTemplate.fromTemplate(
  `你是一位专业的翻译家。请将以下{sourceLang}文本翻译成{targetLang}，翻译成莎士比亚风格的英文，并确保译文流畅、自然。
原文：{text}
译文：`
);

// 3. 使用LCEL（LangChain表达式语言）将组件“链”起来
// LCEL语法： | 或 .pipe() 代表“将前者的输出传给后者”
const translationChain = promptTemplate
  .pipe(model) // 将填充好的提示词传给模型
  .pipe(new StringOutputParser()); // 将模型的复杂响应解析为纯文本字符串

// 4. 调用链    
async function main() {
  const result = await translationChain.invoke({
    sourceLang: "中文",
    targetLang: "英文",
    text: "青山不改，绿水长流，我们后会有期。"
  });
  console.log("翻译结果：\n", result);
}

main().catch(console.error);