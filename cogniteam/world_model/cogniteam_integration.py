from __future__ import annotations 
 
import asyncio 
from typing import Any, Dict, Optional 
 
try:
    from deterministic_cage import (
        DeterministicCage,
        CageResult,
        CageState,
        run_with_world_model,
    )
except ImportError:
    DeterministicCage = None  # type: ignore
    CageResult = None  # type: ignore
    CageState = None  # type: ignore
    run_with_world_model = None  # type: ignore
 
 
# 
# 1. ADAPTADOR DE CLIENTE LLM PARA COGNITEAM 
# 
 
class CogniTeamLLMAdapter: 
    """Adapta tu cliente LLM existente (Groq/Ollama/LiteLLM) a la interfaz que espera DeterministicCage.

    Reemplaza los métodos con tus llamadas reales.
 
    """

    def __init__(self, groq_client=None, ollama_client=None, fallback_to_ollama=True): 
        self.groq = groq_client 
        self.ollama = ollama_client 
        self.fallback = fallback_to_ollama 
        self.groq_calls_today = 0 
        self.groq_limit = 14400  # Límite free tier Groq 
 
    async def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048) -> str:
        """Genera texto usando Groq si hay quota, o Ollama local si no.""" 
        # Intentar Groq primero (70B para razonamiento) 
        if self.groq and self.groq_calls_today < self.groq_limit: 
            try: 
                response = await self._call_groq(prompt, temperature, max_tokens) 
                self.groq_calls_today += 1 
                return response 
            except Exception as e: 
                print(f"[WARN] Groq falló: {e}. Fallback a Ollama.") 
 
        # Fallback a Ollama local 
        if self.ollama and self.fallback: 
            return await self._call_ollama(prompt, temperature, max_tokens) 
 
        raise RuntimeError("No hay proveedor LLM disponible") 
 
    async def _call_groq(self, prompt: str, temperature: float, max_tokens: int) -> str: 
        """Llama a Groq API. Reemplazar con tu implementación real.""" 
        # Ejemplo con tu cliente Groq: 
        # response = await self.groq.chat.completions.create( 
        #     model="llama-3.3-70b-versatile", 
        #     messages=[{"role": "user", "content": prompt}], 
        #     temperature=temperature, 
        #     max_tokens=max_tokens, 
        # ) 
        # return response.choices[0].message.content 
        raise NotImplementedError("Implementa con tu cliente Groq real") 
 
    async def _call_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str: 
        """Llama a Ollama local. Reemplazar con tu implementación real.""" 
        # Ejemplo: 
        # response = await self.ollama.generate( 
        #     model="qwen2.5:7b", 
        #     prompt=prompt, 
        #     options={"temperature": temperature, "num_predict": max_tokens}, 
        # ) 
        # return response["response"] 
        raise NotImplementedError("Implementa con tu cliente Ollama real") 
 
 
# 
# 2. WRAPPER DEL ORQUESTADOR DE COGNITEAM 
# 
 
class CageOrchestrator: 
    """Reemplaza o envuelve tu Orchestrator de CogniTeam.
    Inyecta la jaula determinista en el flujo existente. 
  """
 
    def __init__( 
        self, 
        llm_client: CogniTeamLLMAdapter, 
        world_model_layer=None, 
        calibration_store=None, 
        original_orchestrator=None,  # Tu Orchestrator actual 
    ): 
        self.cage = DeterministicCage( 
            llm_client=llm_client, 
            world_model_layer=world_model_layer, 
            calibration_store=calibration_store, 
        ) 
        self.original = original_orchestrator 
        self.stats = { 
            "tasks_processed": 0, 
            "tasks_succeeded": 0, 
            "tasks_reclarified": 0, 
            "tasks_failed": 0, 
            "total_llm_calls": 0, 
            "avg_execution_time_ms": 0, 
        } 
 
    async def process_task(self, task_description: str, user_id: str = "default") -> Dict[str, Any]: 
        """Punto de entrada principal. Reemplaza tu main.py actual. print(f"\n[🚀] Procesando tarea para usuario {user_id}")"""

        print(f"[📝] Tarea: {task_description[:100]}...") 
 
        # Ejecutar jaula 
        result = await self.cage.run(task_description) 
 
        # Actualizar estadísticas 
        self._update_stats(result) 
 
        # Decidir flujo según resultado 
        if result.reclarification_needed: 
            return await self._handle_reclarification(result, task_description) 
 
        if result.success: 
            return await self._handle_success(result) 
 
        return await self._handle_failure(result) 
 
    async def _handle_reclarification(self, result: CageResult, original_task: str) -> Dict[str, 
Any]: 
        """Devuelve preguntas al usuario para clarificar.""" 
        self.stats["tasks_reclarified"] += 1 
 
        print(f"[❓] Reclarificación necesaria") 
        for q in result.reclarification_questions: 
            print(f"   - {q}") 
 
        return { 
            "status": "RECLARIFY", 
            "questions": result.reclarification_questions, 
            "original_task": original_task, 
            "llm_calls_used": result.llm_calls_used, 
        } 
 
    async def _handle_success(self, result: CageResult) -> Dict[str, Any]: 
        """Tarea completada. Guardar artefactos.""" 
        self.stats["tasks_succeeded"] += 1 
 
        output = result.output 
        print(f"[✅] Tarea completada: {output['domain']}/{output['archetype']}") 
        print(f"[📊] LLM calls: {result.llm_calls_used} | Tiempo: {result.execution_time_ms}ms") 
 
        # Aquí integrarías con tu sistema de archivos actual 
        # await self._save_artifacts(output["code"], output["domain"], output["archetype"]) 
 
        return { 
            "status": "SUCCESS", 
            "code": output["code"], 
            "domain": output["domain"], 
            "archetype": output["archetype"], 
            "confidence": output["classification_confidence"], 
            "llm_calls_used": result.llm_calls_used, 
            "execution_time_ms": result.execution_time_ms, 
        } 
 
    async def _handle_failure(self, result: CageResult) -> Dict[str, Any]: 
        """Tarea fallida. Intentar recuperación o reportar.""" 
        self.stats["tasks_failed"] += 1 
 
        print(f"[❌] Tarea fallida en estado {result.state.name}: {result.error}") 
 
        # Si tenemos orchestrator original, intentar fallback 
        if self.original: 
            print("[🔄] Intentando fallback con orchestrator original...") 
            # return await self.original.process_task(...) 
 
        return { 
            "status": "FAILED", 
            "error": result.error, 
            "state": result.state.name, 
            "llm_calls_used": result.llm_calls_used, 
        } 
 
    def _update_stats(self, result: CageResult): 
        """Actualiza estadísticas de ejecución.""" 
        self.stats["tasks_processed"] += 1 
        self.stats["total_llm_calls"] += result.llm_calls_used 
 
        # Promedio móvil de tiempo 
        n = self.stats["tasks_processed"] 
        old_avg = self.stats["avg_execution_time_ms"] 
        new_time = result.execution_time_ms or 0 
        self.stats["avg_execution_time_ms"] = (old_avg * (n - 1) + new_time) / n 
 
    def get_stats(self) -> Dict[str, Any]: 
        """Retorna estadísticas de uso.""" 
        return { 
            **self.stats, 
            "success_rate": ( 
                self.stats["tasks_succeeded"] / max(self.stats["tasks_processed"], 1) 
            ), 
            "avg_llm_calls_per_task": ( 
                self.stats["total_llm_calls"] / max(self.stats["tasks_processed"], 1) 
            ), 
        } 
 
 
# 
# 3. INTEGRACIÓN CON MEMORIA HÍBRIDA DE COGNITEAM 
# 
 
class CageMemoryBridge: 
    """Conecta la jaula con H-MEM, GraphRAG, MATM y Fast-Slow de CogniTeam."""
 
    def __init__(self, h_mem=None, graph_rag=None, matm=None, fast_slow=None): 
        self.h_mem = h_mem 
        self.graph_rag = graph_rag 
        self.matm = matm 
        self.fast_slow = fast_slow 
 
    async def store_task_memory(self, result: CageResult, task_description: str): 
        """Almacena resultado en la memoria híbrida."""
        if not result.success or not result.output:
            return 
 
        memory_entry = { 
            "task": task_description, 
            "domain": result.output["domain"], 
            "archetype": result.output["archetype"], 
            "confidence": result.output["classification_confidence"], 
            "llm_calls": result.llm_calls_used, 
            "execution_time": result.execution_time_ms, 
            "success": result.success, 
        } 
 
        # Guardar en MATM (memoria episódica) 
        if self.matm: 
            await self.matm.store(memory_entry) 
 
        # Indexar en GraphRAG para búsqueda semántica futura 
        if self.graph_rag: 
            await self.graph_rag.index( 
                text=f"{task_description} -> {result.output['archetype']}", 
                metadata=memory_entry, 
            ) 
 
        # Actualizar Fast-Slow (refuerzo) 
        if self.fast_slow: 
            reward = 1.0 if result.success else -1.0 
            await self.fast_slow.update( 
                state=result.output["archetype"], 
                action="execute", 
                reward=reward, 
            ) 
 
    async def get_similar_tasks(self, task_description: str, k: int = 3) -> list: 
        """Recupera tareas similares para few-shot learning.""" 
        if not self.graph_rag:
            return [] 
        return await self.graph_rag.search(task_description, top_k=k) 
 
 
# 
# 4. FUNCIÓN DE ARRANQUE (reemplaza tu main.py) 
# 
 
async def main_cogniteam_cage(): 
    """Punto de entrada principal para CogniTeam con jaula determinista. Reemplaza tu main.py actual con esta función."""

 
    # 1. Inicializar cliente LLM 
    llm_adapter = CogniTeamLLMAdapter( 
        groq_client=None,      # Tu cliente Groq real 
        ollama_client=None,    # Tu cliente Ollama real 
        fallback_to_ollama=True, 
    ) 
 
    # 2. Inicializar memoria (opcional) 
    memory = CageMemoryBridge( 
        h_mem=None,      # Tu H-MEM real 
        graph_rag=None,  # Tu GraphRAG real 
        matm=None,       # Tu MATM real 
        fast_slow=None,  # Tu Fast-Slow real 
    ) 
 
    # 3. Inicializar orquestador con jaula 
    orchestrator = CageOrchestrator( 
        llm_client=llm_adapter, 
        world_model_layer=None,  # Tu world_model_layer si lo implementas 
        calibration_store=None,  # Tu CalibrationStore 
    ) 
 
    # 4. Loop principal 
    print("=" * 60) 
    print("🧠 CogniTeam + DeterministicCage") 
    print("=" * 60) 
 
    while True: 
        print("\nIntroduce la tarea (multilínea, 'FIN_TAREA' para terminar):") 
        lines = [] 
        while True: 
            line = input("TAREA> ") 
            if line.strip() == "FIN_TAREA": 
                break 
            lines.append(line) 
 
        task = "\n".join(lines) 
        if not task.strip(): 
            continue 
 
        # Procesar tarea 
        result = await orchestrator.process_task(task) 
 
        # Mostrar resultado 
        print(f"\n📋 RESULTADO: {result['status']}") 
 
        if result["status"] == "RECLARIFY": 
            print("🔍 Se necesita más información:") 
            for q in result["questions"]: 
                print(f"   ❓ {q}") 
 
        elif result["status"] == "SUCCESS": 
            print(f"✅ Código generado ({len(result['code'])} caracteres)") 
            print(f"📁 Guardar en: proyectos_finalizados/{result['domain']}/{result['archetype']}/")
            # Aquí guardarías los archivos 
 
        elif result["status"] == "FAILED": 
            print(f"❌ Error: {result.get('error', 'Desconocido')}") 
 
        # Mostrar estadísticas 
        stats = orchestrator.get_stats() 
        print(f"\n📊 Stats: {stats['tasks_processed']} tareas | " 
              f"{stats['success_rate']*100:.1f}% éxito | " 
              f"{stats['avg_llm_calls_per_task']:.1f} LLM calls/tarea") 
 
 
if __name__ == "__main__": 
    asyncio.run(main_cogniteam_cage())

