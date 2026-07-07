# Voice Agent Proof of Concept

The voice-agent-poc project demonstrates a zero-cost voice agent pipeline for
appointment booking and call routing. Speech synthesis uses the real Piper TTS engine
and speech recognition uses faster-whisper, both running locally on CPU, so the whole
loop works without any paid voice API.

The agent handles a booking conversation end to end: it transcribes the caller,
extracts intent and slots (service, date, time), confirms the booking, and can route
the call to the right department. The design maps directly onto hosted stacks such as
LiveKit Agents, Vapi or Retell if a client wants a managed deployment. The test suite
contains 53 passing tests covering transcription, intent extraction and dialog flow.
