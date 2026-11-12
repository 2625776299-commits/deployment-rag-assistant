import json
import unittest
from app.query_planning import build_query_plan_messages, parse_query_plan


class QueryPlanningTests(unittest.TestCase):
    def test_general_only_is_allowed(self):
        plan={'specific_query':None,'mechanism_query':'operation outcome without a response', 'technical_anchors':[]}
        self.assertEqual(parse_query_plan(json.dumps(plan),'request timed out'),plan)

    def test_specific_only_with_exact_anchor_is_allowed(self):
        q='PostgreSQL 18 protocol completion'
        plan={'specific_query':'PostgreSQL 18 ReadyForQuery meaning','mechanism_query':None,
              'technical_anchors':[{'term':'PostgreSQL 18','question_quote':q}]}
        self.assertEqual(parse_query_plan(json.dumps(plan),q),plan)

    def test_both_absent_rejected(self):
        with self.assertRaisesRegex(ValueError,'invalid_queries'):
            parse_query_plan(json.dumps({'specific_query':None,'mechanism_query':None,'technical_anchors':[]}), 'q')

    def test_fabricated_anchor_rejected(self):
        plan={'specific_query':'JDBC commit outcome','mechanism_query':None,
              'technical_anchors':[{'term':'JDBC','question_quote':'JDBC commit'}]}
        with self.assertRaisesRegex(ValueError,'technical_anchor_not_in_question'):
            parse_query_plan(json.dumps(plan),'commit interrupted')

    def test_normalized_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate_queries'):
            parse_query_plan(json.dumps({'specific_query':' Same Query ','mechanism_query':'same  query','technical_anchors':[]}), 'q')

    def test_existing_quote_cannot_support_an_unrelated_technical_term(self):
        plan={'specific_query':'JDBC commit outcome','mechanism_query':None,
              'technical_anchors':[{'term':'JDBC','question_quote':'commit interrupted'}]}
        with self.assertRaisesRegex(ValueError,'technical_anchor_not_in_question'):
            parse_query_plan(json.dumps(plan),'commit interrupted')

    def test_missing_gap_is_not_a_planning_request(self):
        with self.assertRaises(ValueError):
            build_query_plan_messages('question',[])


if __name__=='__main__':
    unittest.main()
