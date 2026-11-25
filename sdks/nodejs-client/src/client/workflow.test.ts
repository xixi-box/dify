import { beforeEach, describe, expect, it, vi } from 'vite-plus/test'
import { createHttpClient, createHttpClientWithSpies } from '../../tests/test-utils'
import { WorkflowClient } from './workflow'

describe('WorkflowClient', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('runs workflows with blocking and streaming modes', async () => {
    const { client, request, requestStream } = createHttpClientWithSpies()
    const workflow = new WorkflowClient(client)

    await workflow.run({ inputs: { input: 'x' }, user: 'user' })
    await workflow.run({ input: 'x' }, 'user', true)

    expect(request).toHaveBeenCalledWith({
      method: 'POST',
      path: '/workflows/run',
      data: {
        inputs: { input: 'x' },
        user: 'user',
      },
    })
    expect(requestStream).toHaveBeenCalledWith({
      method: 'POST',
      path: '/workflows/run',
      data: {
        inputs: { input: 'x' },
        user: 'user',
        response_mode: 'streaming',
      },
    })
  })

  it('runs workflow by id', async () => {
    const { client, request, requestStream } = createHttpClientWithSpies()
    const workflow = new WorkflowClient(client)

    await workflow.runById('wf', {
      inputs: { input: 'x' },
      user: 'user',
      response_mode: 'blocking',
    })
    await workflow.runById('wf', {
      inputs: { input: 'x' },
      user: 'user',
      response_mode: 'streaming',
    })

    expect(request).toHaveBeenCalledWith({
      method: 'POST',
      path: '/workflows/wf/run',
      data: {
        inputs: { input: 'x' },
        user: 'user',
        response_mode: 'blocking',
      },
    })
    expect(requestStream).toHaveBeenCalledWith({
      method: 'POST',
      path: '/workflows/wf/run',
      data: {
        inputs: { input: 'x' },
        user: 'user',
        response_mode: 'streaming',
      },
    })
  })

  it.each([
    ['run', 'blocking', '/workflows/run'],
    ['run', 'streaming', '/workflows/run'],
    ['runById', 'blocking', '/workflows/wf/run'],
    ['runById', 'streaming', '/workflows/wf/run'],
  ] as const)(
    'serializes omitted workflow inputs as an empty object for %s in %s mode',
    async (method, responseMode, path) => {
      const { client, fetchMock } = createHttpClient({ baseUrl: 'https://example.com/v1' })
      const workflow = new WorkflowClient(client)
      const streaming = responseMode === 'streaming'
      fetchMock.mockImplementation(async () =>
        streaming
          ? new Response('data: {"event":"workflow_finished"}\n\n', {
              headers: { 'content-type': 'text/event-stream' },
            })
          : Response.json({}),
      )
      const payload = { user: 'user', response_mode: responseMode, files: null }
      const inputs = { input: 'x' }
      const payloadWithInputs = { ...payload, inputs }

      for (const request of [payload, payload, payloadWithInputs]) {
        const result =
          method === 'run' ? await workflow.run(request) : await workflow.runById('wf', request)
        expect(Symbol.asyncIterator in result).toBe(streaming)
      }

      expect(fetchMock).toHaveBeenCalledTimes(3)
      const bodies = fetchMock.mock.calls.map(([url, init]) => {
        expect(url).toBe(`https://example.com/v1${path}`)
        expect(init.method).toBe('POST')
        return JSON.parse(init.body)
      })
      const expectedPayload = { user: 'user', response_mode: responseMode, files: null }
      expect(bodies).toEqual([
        { ...expectedPayload, inputs: {} },
        { ...expectedPayload, inputs: {} },
        { ...expectedPayload, inputs: { input: 'x' } },
      ])
      expect(payload).toStrictEqual(expectedPayload)
      expect(payloadWithInputs).toStrictEqual({ ...expectedPayload, inputs: { input: 'x' } })
    },
  )

  it('gets run details and stops workflow', async () => {
    const { client, request } = createHttpClientWithSpies()
    const workflow = new WorkflowClient(client)

    await workflow.getRun('run')
    await workflow.stop('task', 'user')

    expect(request).toHaveBeenCalledWith({
      method: 'GET',
      path: '/workflows/run/run',
    })
    expect(request).toHaveBeenCalledWith({
      method: 'POST',
      path: '/workflows/tasks/task/stop',
      data: { user: 'user' },
    })
  })

  it('fetches workflow logs', async () => {
    const { client, request } = createHttpClientWithSpies()
    const workflow = new WorkflowClient(client)

    await workflow.getLogs({
      keyword: 'k',
      status: 'succeeded',
      startTime: '2024-01-01',
      endTime: '2024-01-02',
      createdByEndUserSessionId: 'session-123',
      page: 1,
      limit: 20,
    })

    expect(request).toHaveBeenCalledWith({
      method: 'GET',
      path: '/workflows/logs',
      query: {
        keyword: 'k',
        status: 'succeeded',
        created_at__before: '2024-01-02',
        created_at__after: '2024-01-01',
        created_by_end_user_session_id: 'session-123',
        created_by_account: undefined,
        page: 1,
        limit: 20,
      },
    })
  })
})
