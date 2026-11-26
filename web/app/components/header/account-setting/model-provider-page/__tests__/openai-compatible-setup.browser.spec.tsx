import { render } from 'vitest-browser-react'
import {
  createPluginInstallPermissionStore,
  PluginInstallPermissionContext,
} from '@/app/components/plugins/install-plugin/hooks/use-plugin-install-permission'
import { PluginCategoryEnum } from '@/app/components/plugins/types'
import { marketplaceLinks } from '@/features/marketplace/links'
import {
  createProviderResponseFixture,
  createProviderSummaryFixture,
  seedModelProviderQueries,
} from '@/test/console/model-provider'
import { createConsoleQueryWrapper } from '@/test/console/query-data'
import { setupFakeApi } from '@/test/fake-api'
import { createMarketplacePluginFixture } from '@/test/fixtures/plugin'
import { createNuqsTestWrapper } from '@/test/nuqs-testing'
import InstallFromMarketplace from '../install-from-marketplace'
import { ModelCredentialDialog } from '../model-modal/dialog'
import LazyCustomModelActions from '../provider-added-card/lazy-custom-model-actions'

const compatible = createMarketplacePluginFixture({
  org: 'langgenius',
  name: 'openai_api_compatible',
  plugin_id: 'langgenius/openai_api_compatible',
  label: { en_US: 'OpenAI-API-compatible' },
  category: PluginCategoryEnum.model,
  tags: [],
})
const other = createMarketplacePluginFixture({
  name: 'another_provider',
  label: { en_US: 'Another provider' },
  category: PluginCategoryEnum.model,
  tags: [],
})

const fake = setupFakeApi(() => ({
  console: {
    workspaces: {
      current: {
        plugin: {
          permission: {
            fetch: {
              get: () => ({ install_permission: 'everyone', debug_permission: 'everyone' }),
            },
          },
          installedIds: { get: () => ({ plugin_ids: [] }) },
          list: { installations: { ids: { post: () => ({ plugins: [] }) } } },
        },
      },
    },
  },
  marketplace: {
    collectionPlugins: () => ({ data: { plugins: [], total: 0 } }),
    searchAdvanced: ({ body }) => ({
      data: {
        plugins: body.query === compatible.name ? [compatible] : [other],
        total: 1,
      },
    }),
  },
}))

async function renderCatalog(searchText = '') {
  const { wrapper: Providers } = createConsoleQueryWrapper({
    workspacePermissionKeys: ['plugin.install'],
    systemFeatures: { rbac_enabled: true },
  })
  const { wrapper: UrlState } = createNuqsTestWrapper()
  const permissions = createPluginInstallPermissionStore({ canInstallPlugin: true })
  return render(
    <Providers>
      <PluginInstallPermissionContext value={permissions}>
        <UrlState>
          <InstallFromMarketplace marketplace={marketplaceLinks} searchText={searchText} />
        </UrlState>
      </PluginInstallPermissionContext>
    </Providers>,
  )
}

it('offers OpenAI-API-compatible installation when the default catalog omits it', async () => {
  const screen = await renderCatalog()
  await expect.element(screen.getByRole('heading', { name: 'Another provider' })).toBeVisible()
  const card = screen.getByRole('article', { name: 'OpenAI-API-compatible' })
  await expect.element(card, { timeout: 2000 }).toBeVisible()
  await expect
    .element(card.getByRole('button', { name: /install OpenAI-API-compatible/i }))
    .toBeInTheDocument()
})

it('keeps an installed OpenAI-API-compatible plugin out of the install catalog', async () => {
  fake.use({
    console: {
      workspaces: {
        current: {
          plugin: { installedIds: { get: () => ({ plugin_ids: [compatible.plugin_id] }) } },
        },
      },
    },
  })
  const screen = await renderCatalog()
  await expect.element(screen.getByRole('heading', { name: 'Another provider' })).toBeVisible()
  await expect
    .element(screen.getByRole('article', { name: 'OpenAI-API-compatible' }))
    .not.toBeInTheDocument()
})

it('honors the active provider search', async () => {
  const screen = await renderCatalog('Another provider')
  await expect.element(screen.getByRole('heading', { name: 'Another provider' })).toBeVisible()
  await expect
    .element(screen.getByRole('article', { name: 'OpenAI-API-compatible' }))
    .not.toBeInTheDocument()
})

it('uses a saved connection to register another exact model ID', async () => {
  const { queryClient, wrapper } = createConsoleQueryWrapper({
    workspacePermissionKeys: ['credential.use', 'credential.create', 'credential.manage'],
  })
  seedModelProviderQueries(queryClient)
  const savedConnection = {
    credential_id: '89ebfb35-f67e-4fcb-913c-405d10ebd5da',
    credential_name: 'Model Studio',
  }
  const provider = 'langgenius/openai_api_compatible/openai_api_compatible'
  const baseUrl = 'https://model-studio.example/compatible-mode/v1'
  const firstModel = 'qwen-plus'
  const secondModel = 'Qwen/Model-2026.02'
  const providerDetails = createProviderResponseFixture({
    provider,
    label: { en_US: 'OpenAI-API-compatible' },
    configurate_methods: ['customizable-model'],
    model_credential_schema: {
      model: { label: { en_US: 'Model Name' } },
      credential_form_schemas: [
        {
          variable: 'endpoint_url',
          label: { en_US: 'API Base URL' },
          type: 'text-input',
          required: true,
          max_length: 0,
          show_on: [],
        },
        {
          variable: 'api_key',
          label: { en_US: 'API Key' },
          type: 'secret-input',
          required: false,
          max_length: 0,
          show_on: [],
        },
      ],
    },
    custom_configuration: { status: 'no-configure', custom_models: [] },
  })
  const { console: workspaceApi } = fake.use({
    console: {
      workspaces: {
        current: {
          modelProviders: {
            get: () => ({ data: [providerDetails] }),
            byProvider: {
              models: {
                credentials: {
                  post: ({ body }) => {
                    if (body.model === firstModel) {
                      providerDetails.custom_configuration = {
                        status: 'active',
                        custom_models: [
                          {
                            model: firstModel,
                            model_type: 'llm',
                            credentials: null,
                            unadded_to_model_list: false,
                            available_model_credentials: [savedConnection],
                            current_credential_id: savedConnection.credential_id,
                            current_credential_name: savedConnection.credential_name,
                          },
                        ],
                      }
                    }
                    return { result: 'success' }
                  },
                },
              },
            },
          },
        },
      },
    },
  })
  const screen = await render(
    <>
      <LazyCustomModelActions
        provider={createProviderSummaryFixture({
          provider,
          configurate_methods: ['customizable-model'],
          is_configured: false,
        })}
      />
      <ModelCredentialDialog />
    </>,
    { wrapper },
  )
  const open = screen.getByRole('button', { name: 'modelProvider.modelProvider.addModel' })
  const dialog = screen.getByRole('dialog', { name: 'modelProvider.modelProvider.auth.addModel' })
  const add = dialog.getByRole('button', { name: 'common.operation.add' })
  const selectModelType = async () => {
    await dialog.getByRole('combobox', { name: 'Model Type' }).click()
    await screen.getByRole('option', { name: 'LLM' }).click()
  }
  const post = workspaceApi.workspaces.current.modelProviders.byProvider.models.credentials.post

  await open.click()
  await dialog.getByRole('textbox', { name: 'Model Name' }).fill(firstModel)
  await selectModelType()
  await dialog.getByLabelText('plugin.auth.authorizationName').fill(savedConnection.credential_name)
  await dialog.getByLabelText('API Base URL').fill(baseUrl)
  await dialog.getByLabelText('API Key').fill('test-secret')
  await add.click()
  await expect.element(dialog).not.toBeInTheDocument()
  expect(post).toHaveBeenCalledExactlyOnceWith(
    expect.objectContaining({
      params: { provider },
      body: expect.objectContaining({
        model: firstModel,
        model_type: 'llm',
        name: savedConnection.credential_name,
        credentials: expect.objectContaining({ endpoint_url: baseUrl, api_key: 'test-secret' }),
      }),
    }),
  )

  await open.click()
  await expect.element(dialog.getByLabelText('API Base URL')).toBeVisible()
  await dialog
    .getByRole('button', { name: 'modelProvider.modelProvider.auth.selectModelCredential' })
    .click({ timeout: 2000 })
  await screen.getByRole('button', { name: /Model Studio/ }).click()
  await expect.element(dialog.getByLabelText('API Key')).not.toBeInTheDocument()
  await expect.element(dialog.getByLabelText('API Base URL')).not.toBeInTheDocument()
  await dialog.getByRole('textbox', { name: 'Model Name' }).fill(secondModel)
  await selectModelType()
  await add.click()
  await expect.element(dialog).not.toBeInTheDocument()
  expect(post).toHaveBeenCalledTimes(2)
  expect(post).toHaveBeenLastCalledWith(
    expect.objectContaining({
      params: { provider },
      body: expect.objectContaining({
        model: secondModel,
        model_type: 'llm',
        source_credential_id: savedConnection.credential_id,
      }),
    }),
  )
  expect(post.mock.calls[1]?.[0].body.credentials).not.toHaveProperty('api_key')
  expect(post.mock.calls[1]?.[0].body.credentials).not.toHaveProperty('endpoint_url')
})
