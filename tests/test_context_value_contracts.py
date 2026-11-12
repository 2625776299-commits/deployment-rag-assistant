"""Deployment field boundaries and provenance, without provider/SQL calls."""
from copy import deepcopy
from dataclasses import asdict
import unittest

from app.context_proposal import validate_context_value_contracts
from app.conversation_service import deployment_registry


class ContextValueContractTests(unittest.TestCase):
    def setUp(self):
        self.registry = deployment_registry('C1', 'M1')

    def item(self, entity, field, value, quote):
        target = next(r.target for r in self.registry if r.target.entity_id == entity and r.target.field == field)
        return {'operation': 'add', 'target': asdict(target), 'previous_fact_id': None,
                'value': value, 'source': {'message_id': 'M1', 'quote': quote}}

    def check(self, item, **kwargs):
        return validate_context_value_contracts({'updates': [item]}, self.registry, **kwargs)

    def test_concrete_values_and_separate_error_availability_clauses(self):
        availability = {'updates': [{'availability': 'unavailable',
                        'source': {'message_id': 'M1', 'quote': '历史日志已经清理'}}]}
        values = [('database', 'service_name', 'db.prod_2', '服务名是db.prod_2'),
                  ('database', 'container_port', '5432', '端口映射5433:5432'),
                  ('database', 'published_host_port', '65535', '端口65535'),
                  ('application', 'runtime_location', 'host', '应用位于宿主机'),
                  ('database', 'runtime_location', 'container', '数据库在容器里'),
                  ('application', 'network_relation', 'different_network', '明确不在同一个网络'),
                  ('application', 'error_text', 'connection refused', 'connection refused'),
                  ('application', 'configuration_text', 'DB_HOST=db', '当前配置DB_HOST=db')]
        for entity, field, value, quote in values:
            with self.subTest(field=field):
                self.check(self.item(entity, field, value, quote), evidence_request=availability)

    def test_names_ports_and_literals_cannot_be_invented_or_relational(self):
        cases = [('database', 'service_name', '未变化', '服务名未变化', 'service_name'),
                 ('database', 'service_name', 'db', '服务名没有改', 'literal_not_in_source'),
                 ('database', 'container_port', '5432', '未提供端口', 'literal_not_in_source'),
                 ('database', 'container_port', '5432', '端口15432', 'literal_not_in_source'),
                 ('database', 'container_port', '65536', '端口65536', 'not_a_port'),
                 ('database', 'container_port', '0', '端口0', 'not_a_port'),
                 ('database', 'container_port', '５４３２', '端口５４３２', 'not_a_port'),
                 ('application', 'runtime_location', '可能是容器', '可能是容器', 'registered_enum'),
                 ('application', 'error_text', 'timeout', '暂时没有错误原文', 'literal_not_in_source')]
        for entity, field, value, quote, error in cases:
            with self.subTest(value=value, quote=quote), self.assertRaisesRegex(ValueError, error):
                self.check(self.item(entity, field, value, quote))

    def test_unavailability_cannot_double_as_actual_error_but_acquired_error_can(self):
        item = self.item('application', 'error_text', '报错无法取得', '报错无法取得')
        evidence = {'updates': [{'availability': 'unavailable', 'source': deepcopy(item['source'])}]}
        before = deepcopy((item, evidence))
        with self.assertRaisesRegex(ValueError, 'availability_is_not_error'):
            self.check(item, evidence_request=evidence)
        self.assertEqual((item, evidence), before)
        actual = self.item('application', 'error_text', 'timeout', 'timeout')
        evidence = {'updates': [{'availability': 'available', 'source': deepcopy(actual['source'])}]}
        self.check(actual, evidence_request=evidence)

    def test_hypotheses_share_value_contracts_without_rechecking_untouched_history(self):
        item = self.item('database', 'service_name', '未变化', '服务名未变化')
        del item['operation']; del item['previous_fact_id']
        with self.assertRaisesRegex(ValueError, 'service_name'):
            validate_context_value_contracts({'updates': []}, self.registry,
                analysis_request={'mode': 'hypothetical', 'assumptions': [item]})
        item['value'] = None
        validate_context_value_contracts({'updates': []}, self.registry,
            analysis_request={'mode': 'inherit', 'assumption_updates': [item]})
