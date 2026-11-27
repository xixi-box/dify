import type { ComponentProps } from 'react'
import { useRef } from 'react'
import { render } from 'vitest-browser-react'
import { guardUnexpectedFetch } from '@/test/unexpected-fetch'
import { PublisherSummarySection } from '../built-in-publisher/summary-section'
import WorkflowToolAction from '../workflow-tool-action'

beforeEach(() => onTestFinished(guardUnexpectedFetch()))

function PublishSummary({
  onPublish,
}: {
  onPublish: ComponentProps<typeof PublisherSummarySection>['handlePublish']
}) {
  const keyboardTargetRef = useRef<HTMLDivElement>(null)

  return (
    <div ref={keyboardTargetRef}>
      <PublisherSummarySection
        formatTimeFromNow={() => 'just now'}
        handlePublish={onPublish}
        handleRestore={async () => {}}
        isChatApp={false}
        isPublishing={false}
        isWorkflowApp
        keyboardTarget={keyboardTargetRef}
        published={false}
        publishedAt={1_710_000_000_000}
        showSyncWorkflowTool
        upgradeHighlightStyle={{}}
      />
    </div>
  )
}

it('requests tool sync only when selected for this publish', async () => {
  const onPublish = vi.fn().mockResolvedValue(undefined)
  const screen = await render(<PublishSummary onPublish={onPublish} />)
  const sync = screen.getByRole('checkbox', { name: /common\.syncWorkflowToolOnPublish/ })
  const publish = screen.getByRole('button', { name: /common\.publishUpdate/ })

  await expect.element(sync).not.toBeChecked()
  await publish.click()
  expect(onPublish).toHaveBeenNthCalledWith(1, undefined)

  await sync.click()
  await expect.element(sync).toBeChecked()
  await publish.click()
  expect(onPublish).toHaveBeenNthCalledWith(2, undefined, { syncWorkflowTool: true })
})

it('warns that the saved tool is behind even when its parameters are current', async () => {
  const screen = await render(
    <WorkflowToolAction
      disabled={false}
      isLoading={false}
      outdated={false}
      published
      updatePending
      onConfigure={vi.fn()}
    />,
  )

  await expect
    .element(screen.getByRole('status', { name: /common\.workflowToolUpdatePending/ }))
    .toBeVisible()
  await expect.element(screen.getByRole('button', { name: /common\.configure\b/ })).toBeEnabled()
})
