// 使用新版模块化导入
import { ChatOpenAI } from "@langchain/openai";
import { DynamicTool } from "@langchain/core/tools";
import { initializeAgentExecutorWithOptions } from "@langchain/classic/agents";
import { BufferWindowMemory } from "@langchain/classic/memory";

import * as dotenv from 'dotenv';
dotenv.config();

const memory = new BufferWindowMemory({
    k: 3, // 只保留最近3轮 Thought/Action/Observation
    memoryKey: "chat_history",
    returnMessages: true,
});

// 1. 初始化模型（使用思考能力更强的模型效果更好，如 deepseek-chat）
const model = new ChatOpenAI({
    apiKey: process.env.DEEPSEEK_API_KEY,
    model: "deepseek-chat",
    configuration: { baseURL: "https://api.deepseek.com/v1" },
    temperature: 0, // 执行任务时，建议降低随机性
});

// 2. 定义工具：一个安全的计算器
const calculatorTool = new DynamicTool({
    name: "calculator",
    description: "用于执行数学计算。输入应该是一个完整的、可计算的数学表达式，例如：(12 + 5) * 3。请确保表达式是数字和运算符的组合。",
    func: async (input) => {
        console.log(`[工具调用] 计算器正在计算: ${input}`);
        try {
            // 使用Function构造函数在沙盒中安全地评估数学表达式，避免直接使用eval
            const safeEval = new Function(`return (${input})`);
            const result = safeEval();
            return `计算结果为: ${result}`;
        } catch (error) {
            return `计算失败：输入“${input}”不是有效的数学表达式。请确保只包含数字和运算符(+, -, *, /, %, (), .)。`;
        }
    },
});

// 3. 定义工具：一个模拟的天气查询工具
const weatherTool = new DynamicTool({
    name: "weather_query",
    description: "查询指定城市的当前天气。输入应该是城市名称，例如：北京。",
    func: async (input) => {
        console.log(`[工具调用] 正在查询城市天气: ${input}`);
        // 模拟一个天气API的返回
        const mockWeatherData = {
            "北京": "晴，15°C，西北风2级",
            "上海": "多云，18°C，东南风1级",
            "深圳": "阵雨，22°C，南风3级",
        };
        const weather = mockWeatherData[input] || `抱歉，未找到城市“${input}”的天气信息。`;
        return `城市 ${input} 的天气情况：${weather}`;
    },
});

async function main() {
    console.log('=== 初始化智能体，这可能需要几秒钟... ===\n');

    // 4. 创建智能体执行器
    // 方案一：在自定义前缀中移除 {tool_names} 变量引用（推荐）
    // 将工具名称直接写进提示词，或者用其他表述代替
    const customPrefix = `请按格式回答：
    Thought: 思考步骤
    Action: 工具名
    Action Input: 输入
    Observation: 结果
    (重复直到完成)
    Final Answer: 最终答案
    
    当前对话：
    `;

    const executor = await initializeAgentExecutorWithOptions(
        [calculatorTool, weatherTool],
        model,
        {
            agentType: "openai-functions",
            verbose: true,
            maxIterations: 5,
            memory: memory, // 添加记忆管理
            // 关键修改：将 agentArgs 结构改为传入 prefix
            agentArgs: {
                prefix: customPrefix // 使用我们定义好的、不含未填充变量的字符串
            }
        }
    );

    const questions = [
        "如果我有15个苹果，又买了3箱，每箱有12个，我现在总共有多少个苹果？",
        "北京和上海现在的天气怎么样？",
        "请先计算(25 * 4)等于多少，然后告诉我深圳的天气。"
    ];

    for (const question of questions) {
        console.log(`\n🤔 我的问题: ${question}`);
        console.log('-'.repeat(50));

        try {
            const result = await executor.invoke({ input: question });
            console.log(`\n💡 最终答案: ${result.output}`);
        } catch (error) {
            console.error(`❌ 执行出错: ${error.message}`);
        }
        console.log('='.repeat(60));
    }
}

main().catch(console.error);