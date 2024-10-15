import Mathlib

namespace NUMINAMATH_GPT_min_height_required_kingda_ka_l2426_242687

-- Definitions of the given conditions
def brother_height : ℕ := 180
def mary_relative_height : ℚ := 2 / 3
def growth_needed : ℕ := 20

-- Definition and statement of the problem
def marys_height : ℚ := mary_relative_height * brother_height
def minimum_height_required : ℚ := marys_height + growth_needed

theorem min_height_required_kingda_ka :
  minimum_height_required = 140 := by
  sorry

end NUMINAMATH_GPT_min_height_required_kingda_ka_l2426_242687


namespace NUMINAMATH_GPT_determine_h_l2426_242639

theorem determine_h (x : ℝ) (h : ℝ → ℝ) :
  2 * x ^ 5 + 4 * x ^ 3 + h x = 7 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 →
  h x = -2 * x ^ 5 + 3 * x ^ 3 - 5 * x ^ 2 + 9 * x + 3 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_determine_h_l2426_242639


namespace NUMINAMATH_GPT_evaluate_expression_l2426_242626

theorem evaluate_expression : (120 / 6 * 2 / 3 = (40 / 3)) := 
by sorry

end NUMINAMATH_GPT_evaluate_expression_l2426_242626


namespace NUMINAMATH_GPT_somu_present_age_l2426_242623

variable (S F : ℕ)

-- Conditions from the problem
def condition1 : Prop := S = F / 3
def condition2 : Prop := S - 10 = (F - 10) / 5

-- The statement we need to prove
theorem somu_present_age (h1 : condition1 S F) (h2 : condition2 S F) : S = 20 := 
by sorry

end NUMINAMATH_GPT_somu_present_age_l2426_242623


namespace NUMINAMATH_GPT_simplify_fraction_l2426_242652

variable {a b m : ℝ}

theorem simplify_fraction (h : a + b ≠ 0) : (ma/a + b) + (mb/a + b) = m :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2426_242652


namespace NUMINAMATH_GPT_cube_root_of_64_is_4_l2426_242690

theorem cube_root_of_64_is_4 (x : ℝ) (h1 : 0 < x) (h2 : x^3 = 64) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_cube_root_of_64_is_4_l2426_242690


namespace NUMINAMATH_GPT_value_of_b_l2426_242613

theorem value_of_b (x y b : ℝ) (h1: 7^(3 * x - 1) * b^(4 * y - 3) = 49^x * 27^y) (h2: x + y = 4) : b = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l2426_242613


namespace NUMINAMATH_GPT_james_spends_252_per_week_l2426_242677

noncomputable def cost_pistachios_per_ounce := 10 / 5
noncomputable def cost_almonds_per_ounce := 8 / 4
noncomputable def cost_walnuts_per_ounce := 12 / 6

noncomputable def daily_consumption_pistachios := 30 / 5
noncomputable def daily_consumption_almonds := 24 / 4
noncomputable def daily_consumption_walnuts := 18 / 3

noncomputable def weekly_consumption_pistachios := daily_consumption_pistachios * 7
noncomputable def weekly_consumption_almonds := daily_consumption_almonds * 7
noncomputable def weekly_consumption_walnuts := daily_consumption_walnuts * 7

noncomputable def weekly_cost_pistachios := weekly_consumption_pistachios * cost_pistachios_per_ounce
noncomputable def weekly_cost_almonds := weekly_consumption_almonds * cost_almonds_per_ounce
noncomputable def weekly_cost_walnuts := weekly_consumption_walnuts * cost_walnuts_per_ounce

noncomputable def total_weekly_cost := weekly_cost_pistachios + weekly_cost_almonds + weekly_cost_walnuts

theorem james_spends_252_per_week :
  total_weekly_cost = 252 := by
  sorry

end NUMINAMATH_GPT_james_spends_252_per_week_l2426_242677


namespace NUMINAMATH_GPT_pow_mod_sub_remainder_l2426_242646

theorem pow_mod_sub_remainder :
  (10^23 - 7) % 6 = 3 :=
sorry

end NUMINAMATH_GPT_pow_mod_sub_remainder_l2426_242646


namespace NUMINAMATH_GPT_cylinder_surface_area_l2426_242604

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area (h r : ℕ) (h_eq : h = 8) (r_eq : r = 3) :
  2 * Real.pi * r * h + 2 * Real.pi * r ^ 2 = 66 * Real.pi := by
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l2426_242604


namespace NUMINAMATH_GPT_function_matches_table_values_l2426_242638

variable (f : ℤ → ℤ)

theorem function_matches_table_values (h1 : f (-1) = -2) (h2 : f 0 = 0) (h3 : f 1 = 2) (h4 : f 2 = 4) : 
  ∀ x : ℤ, f x = 2 * x := 
by
  -- Prove that the function satisfying the given table values is f(x) = 2x
  sorry

end NUMINAMATH_GPT_function_matches_table_values_l2426_242638


namespace NUMINAMATH_GPT_sum_of_first_n_terms_l2426_242606

-- Definitions for the sequences and the problem conditions.
def a (n : ℕ) : ℕ := 2 ^ n
def b (n : ℕ) : ℕ := 2 * n - 1
def c (n : ℕ) : ℕ := a n * b n
def T (n : ℕ) : ℕ := (2 * n - 3) * 2 ^ (n + 1) + 6

-- The theorem statement
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range n).sum c = T n :=
  sorry

end NUMINAMATH_GPT_sum_of_first_n_terms_l2426_242606


namespace NUMINAMATH_GPT_shopper_total_payment_l2426_242607

theorem shopper_total_payment :
  let original_price := 150
  let discount_rate := 0.25
  let coupon_discount := 10
  let sales_tax_rate := 0.10
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price := price_after_coupon * (1 + sales_tax_rate)
  final_price = 112.75 := by
{
  sorry
}

end NUMINAMATH_GPT_shopper_total_payment_l2426_242607


namespace NUMINAMATH_GPT_min_initial_bags_l2426_242620

theorem min_initial_bags :
  ∃ x : ℕ, (∃ y : ℕ, (y + 90 = 2 * (x - 90) ∧ x + (11 * x - 1620) / 7 = 6 * (2 * x - 270 - (11 * x - 1620) / 7))
             ∧ x = 153) :=
by { sorry }

end NUMINAMATH_GPT_min_initial_bags_l2426_242620


namespace NUMINAMATH_GPT_cube_volume_is_27_l2426_242628

noncomputable def original_volume (s : ℝ) : ℝ := s^3
noncomputable def new_solid_volume (s : ℝ) : ℝ := (s + 2) * (s + 2) * (s - 2)

theorem cube_volume_is_27 (s : ℝ) (h : original_volume s - new_solid_volume s = 10) :
  original_volume s = 27 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_is_27_l2426_242628


namespace NUMINAMATH_GPT_find_expression_l2426_242679

theorem find_expression (E a : ℝ) (h1 : (E + (3 * a - 8)) / 2 = 84) (h2 : a = 32) : E = 80 :=
by
  -- Proof to be filled in here
  sorry

end NUMINAMATH_GPT_find_expression_l2426_242679


namespace NUMINAMATH_GPT_intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l2426_242627

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + x^2 - 2 * a * x + a^2

-- Question Ⅰ
theorem intervals_of_monotonicity_when_a_eq_2 :
  (∀ x : ℝ, 0 < x ∧ x < (2 - Real.sqrt 2) / 2 → f x 2 > 0) ∧
  (∀ x : ℝ, (2 - Real.sqrt 2) / 2 < x ∧ x < (2 + Real.sqrt 2) / 2 → f x 2 < 0) ∧
  (∀ x : ℝ, (2 + Real.sqrt 2) / 2 < x → f x 2 > 0) := sorry

-- Question Ⅱ
theorem no_increasing_intervals_on_1_3_implies_a_ge_19_over_6 (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → f x a ≤ 0) → a ≥ (19 / 6) := sorry

end NUMINAMATH_GPT_intervals_of_monotonicity_when_a_eq_2_no_increasing_intervals_on_1_3_implies_a_ge_19_over_6_l2426_242627


namespace NUMINAMATH_GPT_factor_expression_l2426_242650

theorem factor_expression:
  ∀ (x : ℝ), (10 * x^3 + 50 * x^2 - 4) - (3 * x^3 - 5 * x^2 + 2) = 7 * x^3 + 55 * x^2 - 6 :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2426_242650


namespace NUMINAMATH_GPT_center_of_circle_l2426_242617

theorem center_of_circle (x y : ℝ) (h : x^2 + y^2 - 10 * x + 4 * y = -40) : 
  x + y = 3 := 
sorry

end NUMINAMATH_GPT_center_of_circle_l2426_242617


namespace NUMINAMATH_GPT_middle_managers_to_be_selected_l2426_242675

def total_employees : ℕ := 160
def senior_managers : ℕ := 10
def middle_managers : ℕ := 30
def staff_members : ℕ := 120
def total_to_be_selected : ℕ := 32

theorem middle_managers_to_be_selected : 
  (middle_managers * total_to_be_selected / total_employees) = 6 := by
  sorry

end NUMINAMATH_GPT_middle_managers_to_be_selected_l2426_242675


namespace NUMINAMATH_GPT_tan_alpha_value_l2426_242634

variables (α β : ℝ)

theorem tan_alpha_value
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) :
  Real.tan α = 13 / 16 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_l2426_242634


namespace NUMINAMATH_GPT_relationship_abc_l2426_242683

theorem relationship_abc (a b c : ℕ) (ha : a = 2^555) (hb : b = 3^444) (hc : c = 6^222) : a < c ∧ c < b := by
  sorry

end NUMINAMATH_GPT_relationship_abc_l2426_242683


namespace NUMINAMATH_GPT_value_of_2a_minus_1_l2426_242614

theorem value_of_2a_minus_1 (a : ℝ) (h : ∀ x : ℝ, (x = 2 → (3 / 2) * x - 2 * a = 0)) : 2 * a - 1 = 2 :=
sorry

end NUMINAMATH_GPT_value_of_2a_minus_1_l2426_242614


namespace NUMINAMATH_GPT_sin_cos_inequality_l2426_242625

theorem sin_cos_inequality (α : ℝ) 
  (h1 : 0 ≤ α) (h2 : α < 2 * Real.pi) 
  (h3 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  (Real.pi / 3 < α ∧ α < 4 * Real.pi / 3) :=
sorry

end NUMINAMATH_GPT_sin_cos_inequality_l2426_242625


namespace NUMINAMATH_GPT_maximum_watchman_demand_l2426_242693

theorem maximum_watchman_demand (bet_loss : ℕ) (bet_win : ℕ) (x : ℕ) 
  (cond_bet_loss : bet_loss = 100)
  (cond_bet_win : bet_win = 100) :
  x < 200 :=
by
  have h₁ : bet_loss = 100 := cond_bet_loss
  have h₂ : bet_win = 100 := cond_bet_win
  sorry

end NUMINAMATH_GPT_maximum_watchman_demand_l2426_242693


namespace NUMINAMATH_GPT_divisor_is_three_l2426_242603

noncomputable def find_divisor (n : ℕ) (reduction : ℕ) (result : ℕ) : ℕ :=
  n / result

theorem divisor_is_three (x : ℝ) : 
  (original : ℝ) → (reduction : ℝ) → (new_result : ℝ) → 
  original = 45 → new_result = 45 - 30 → (original / x = new_result) → 
  x = 3 := by 
  intros original reduction new_result h1 h2 h3
  sorry

end NUMINAMATH_GPT_divisor_is_three_l2426_242603


namespace NUMINAMATH_GPT_product_of_first_three_terms_l2426_242673

/--
  The eighth term of an arithmetic sequence is 20.
  If the difference between two consecutive terms is 2,
  prove that the product of the first three terms of the sequence is 480.
-/
theorem product_of_first_three_terms (a d : ℕ) (h_d : d = 2) (h_eighth_term : a + 7 * d = 20) :
  (a * (a + d) * (a + 2 * d) = 480) :=
by
  sorry

end NUMINAMATH_GPT_product_of_first_three_terms_l2426_242673


namespace NUMINAMATH_GPT_new_plan_cost_correct_l2426_242669

def oldPlanCost : ℝ := 150
def rateIncrease : ℝ := 0.3
def newPlanCost : ℝ := oldPlanCost * (1 + rateIncrease) 

theorem new_plan_cost_correct : newPlanCost = 195 := by
  sorry

end NUMINAMATH_GPT_new_plan_cost_correct_l2426_242669


namespace NUMINAMATH_GPT_original_number_is_144_l2426_242671

theorem original_number_is_144 (x : ℕ) (h : x - x / 3 = x - 48) : x = 144 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_144_l2426_242671


namespace NUMINAMATH_GPT_graduation_graduates_l2426_242647

theorem graduation_graduates :
  ∃ G : ℕ, (∀ (chairs_for_parents chairs_for_teachers chairs_for_admins : ℕ),
    chairs_for_parents = 2 * G ∧
    chairs_for_teachers = 20 ∧
    chairs_for_admins = 10 ∧
    G + chairs_for_parents + chairs_for_teachers + chairs_for_admins = 180) ↔ G = 50 :=
by
  sorry

end NUMINAMATH_GPT_graduation_graduates_l2426_242647


namespace NUMINAMATH_GPT_next_shared_meeting_day_l2426_242665

-- Definitions based on the conditions:
def dramaClubMeetingInterval : ℕ := 3
def choirMeetingInterval : ℕ := 5
def debateTeamMeetingInterval : ℕ := 7

-- Statement to prove:
theorem next_shared_meeting_day : Nat.lcm (Nat.lcm dramaClubMeetingInterval choirMeetingInterval) debateTeamMeetingInterval = 105 := by
  sorry

end NUMINAMATH_GPT_next_shared_meeting_day_l2426_242665


namespace NUMINAMATH_GPT_no_solution_a_squared_plus_b_squared_eq_2023_l2426_242608

theorem no_solution_a_squared_plus_b_squared_eq_2023 :
  ∀ (a b : ℤ), a^2 + b^2 ≠ 2023 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_a_squared_plus_b_squared_eq_2023_l2426_242608


namespace NUMINAMATH_GPT_solution_set_ineq_min_value_sum_l2426_242643

-- Part (1)
theorem solution_set_ineq (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|) :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} :=
sorry

-- Part (2)
theorem min_value_sum (f : ℝ → ℝ) (h : ∀ x, f x = |2 * x - 1| + |x - 2|)
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (hx : ∀ x, f x ≥ (1 / m) + (1 / n)) :
  m + n = 8 / 3 :=
sorry

end NUMINAMATH_GPT_solution_set_ineq_min_value_sum_l2426_242643


namespace NUMINAMATH_GPT_min_value_l2426_242696

-- Conditions
variables {x y : ℝ}
variable (hx : x > 0)
variable (hy : y > 0)
variable (hxy : x + y = 2)

-- Theorem
theorem min_value (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) : 
  ∃ x y, (x > 0) ∧ (y > 0) ∧ (x + y = 2) ∧ (1/x + 4/y = 9/2) := 
by
  sorry

end NUMINAMATH_GPT_min_value_l2426_242696


namespace NUMINAMATH_GPT_sum_of_eight_numbers_l2426_242692

theorem sum_of_eight_numbers (avg : ℚ) (n : ℕ) (sum : ℚ) 
  (h_avg : avg = 5.3) (h_n : n = 8) : sum = 42.4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eight_numbers_l2426_242692


namespace NUMINAMATH_GPT_total_cakes_served_l2426_242685

def L : Nat := 5
def D : Nat := 6
def Y : Nat := 3
def T : Nat := L + D + Y

theorem total_cakes_served : T = 14 := by
  sorry

end NUMINAMATH_GPT_total_cakes_served_l2426_242685


namespace NUMINAMATH_GPT_matinee_receipts_l2426_242657

theorem matinee_receipts :
  let child_ticket_cost := 4.50
  let adult_ticket_cost := 6.75
  let num_children := 48
  let num_adults := num_children - 20
  total_receipts = num_children * child_ticket_cost + num_adults * adult_ticket_cost :=
by 
  sorry

end NUMINAMATH_GPT_matinee_receipts_l2426_242657


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l2426_242689

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), (a > 0) ∧ (b > 0) → 3 * a^2 ≠ b^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l2426_242689


namespace NUMINAMATH_GPT_martian_calendar_months_l2426_242663

theorem martian_calendar_months (x y : ℕ) 
  (h1 : 100 * x + 77 * y = 5882) : x + y = 74 :=
sorry

end NUMINAMATH_GPT_martian_calendar_months_l2426_242663


namespace NUMINAMATH_GPT_canteen_distance_l2426_242691

theorem canteen_distance (r G B : ℝ) (d_g d_b : ℝ) (h_g : G = 600) (h_b : B = 800) (h_dg_db : d_g = d_b) : 
  d_g = 781 :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_canteen_distance_l2426_242691


namespace NUMINAMATH_GPT_distinct_real_roots_k_root_condition_k_l2426_242629

-- Part (1) condition: The quadratic equation has two distinct real roots
theorem distinct_real_roots_k (k : ℝ) : (∃ x : ℝ, x^2 + 2*x + k = 0) ∧ (∀ x y : ℝ, x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0 → x ≠ y) → k < 1 := 
sorry

-- Part (2) condition: m is a root and satisfies m^2 + 2m = 2
theorem root_condition_k (m k : ℝ) : m^2 + 2*m = 2 → m^2 + 2*m + k = 0 → k = -2 := 
sorry

end NUMINAMATH_GPT_distinct_real_roots_k_root_condition_k_l2426_242629


namespace NUMINAMATH_GPT_total_visitors_l2426_242641

noncomputable def visitors_questionnaire (V E U : ℕ) : Prop :=
  (130 ≠ E ∧ E ≠ U) ∧ 
  (E = U) ∧ 
  (3 * V = 4 * E) ∧ 
  (V = 130 + 3 / 4 * V)

theorem total_visitors (V : ℕ) : visitors_questionnaire V V V → V = 520 :=
by sorry

end NUMINAMATH_GPT_total_visitors_l2426_242641


namespace NUMINAMATH_GPT_alcohol_percentage_after_additions_l2426_242609

/-
Problem statement:
A 40-liter solution of alcohol and water is 5% alcohol. If 4.5 liters of alcohol and 5.5 liters of water are added to this solution, what percent of the solution produced is alcohol?

Conditions:
1. Initial solution volume = 40 liters
2. Initial percentage of alcohol = 5%
3. Volume of alcohol added = 4.5 liters
4. Volume of water added = 5.5 liters

Correct answer:
The percent of the solution that is alcohol after the additions is 13%.
-/

theorem alcohol_percentage_after_additions (initial_volume : ℝ) (initial_percentage : ℝ) 
  (alcohol_added : ℝ) (water_added : ℝ) :
  initial_volume = 40 ∧ initial_percentage = 5 ∧ alcohol_added = 4.5 ∧ water_added = 5.5 →
  ((initial_percentage / 100 * initial_volume + alcohol_added) / (initial_volume + alcohol_added + water_added) * 100) = 13 :=
by simp; sorry

end NUMINAMATH_GPT_alcohol_percentage_after_additions_l2426_242609


namespace NUMINAMATH_GPT_union_of_A_and_B_l2426_242666

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2426_242666


namespace NUMINAMATH_GPT_outstanding_student_awards_l2426_242678

theorem outstanding_student_awards :
  ∃ n : ℕ, 
  (n = Nat.choose 9 7) ∧ 
  (∀ (awards : ℕ) (classes : ℕ), awards = 10 → classes = 8 → n = 36) := 
by
  sorry

end NUMINAMATH_GPT_outstanding_student_awards_l2426_242678


namespace NUMINAMATH_GPT_min_keychains_to_reach_profit_l2426_242694

theorem min_keychains_to_reach_profit :
  let cost_per_keychain := 0.15
  let sell_price_per_keychain := 0.45
  let total_keychains := 1200
  let target_profit := 180
  let total_cost := total_keychains * cost_per_keychain
  let total_revenue := total_cost + target_profit
  let min_keychains_to_sell := total_revenue / sell_price_per_keychain
  min_keychains_to_sell = 800 := 
by
  sorry

end NUMINAMATH_GPT_min_keychains_to_reach_profit_l2426_242694


namespace NUMINAMATH_GPT_total_percentage_of_failed_candidates_l2426_242615

theorem total_percentage_of_failed_candidates :
  ∀ (total_candidates girls boys : ℕ) (passed_boys passed_girls : ℝ),
    total_candidates = 2000 →
    girls = 900 →
    boys = total_candidates - girls →
    passed_boys = 0.34 * boys →
    passed_girls = 0.32 * girls →
    (total_candidates - (passed_boys + passed_girls)) / total_candidates * 100 = 66.9 :=
by
  intros total_candidates girls boys passed_boys passed_girls
  intro h_total_candidates
  intro h_girls
  intro h_boys
  intro h_passed_boys
  intro h_passed_girls
  sorry

end NUMINAMATH_GPT_total_percentage_of_failed_candidates_l2426_242615


namespace NUMINAMATH_GPT_probability_two_points_one_unit_apart_l2426_242659

theorem probability_two_points_one_unit_apart :
  let total_points := 10
  let total_ways := (total_points * (total_points - 1)) / 2
  let favorable_horizontal_pairs := 8
  let favorable_vertical_pairs := 5
  let favorable_pairs := favorable_horizontal_pairs + favorable_vertical_pairs
  let probability := (favorable_pairs : ℚ) / total_ways
  probability = 13 / 45 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_points_one_unit_apart_l2426_242659


namespace NUMINAMATH_GPT_miniature_tank_height_l2426_242602

-- Given conditions
def actual_tank_height : ℝ := 50
def actual_tank_volume : ℝ := 200000
def model_tank_volume : ℝ := 0.2

-- Theorem: Calculate the height of the miniature water tank
theorem miniature_tank_height :
  (model_tank_volume / actual_tank_volume) ^ (1/3 : ℝ) * actual_tank_height = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_miniature_tank_height_l2426_242602


namespace NUMINAMATH_GPT_compute_g_ggg2_l2426_242610

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 1
  else if n < 5 then 2 * n + 2
  else 4 * n - 3

theorem compute_g_ggg2 : g (g (g 2)) = 65 :=
by
  sorry

end NUMINAMATH_GPT_compute_g_ggg2_l2426_242610


namespace NUMINAMATH_GPT_candies_count_l2426_242695

variable (m_and_m : Nat) (starbursts : Nat)
variable (ratio_m_and_m_to_starbursts : Nat → Nat → Prop)

-- Definition of the ratio condition
def ratio_condition : Prop :=
  ∃ (k : Nat), (m_and_m = 7 * k) ∧ (starbursts = 4 * k)

-- The main theorem to prove
theorem candies_count (h : m_and_m = 56) (r : ratio_condition m_and_m starbursts) : starbursts = 32 :=
  by
  sorry

end NUMINAMATH_GPT_candies_count_l2426_242695


namespace NUMINAMATH_GPT_option_d_correct_l2426_242612

theorem option_d_correct (m n : ℝ) : (m + n) * (m - 2 * n) = m^2 - m * n - 2 * n^2 :=
by
  sorry

end NUMINAMATH_GPT_option_d_correct_l2426_242612


namespace NUMINAMATH_GPT_convert_base_7_to_base_10_l2426_242681

theorem convert_base_7_to_base_10 : 
  ∀ n : ℕ, (n = 3 * 7^2 + 2 * 7^1 + 1 * 7^0) → n = 162 :=
by
  intros n h
  rw [pow_zero, pow_one, pow_two] at h
  norm_num at h
  exact h

end NUMINAMATH_GPT_convert_base_7_to_base_10_l2426_242681


namespace NUMINAMATH_GPT_possible_values_of_n_l2426_242660

-- Conditions: Definition of equilateral triangles and squares with side length 1
def equilateral_triangle_side_length_1 : Prop := ∀ (a : ℕ), 
  ∃ (triangle : ℕ), triangle * 60 = 180 * (a - 2)

def square_side_length_1 : Prop := ∀ (b : ℕ), 
  ∃ (square : ℕ), square * 90 = 180 * (b - 2)

-- Definition of convex n-sided polygon formed using these pieces
def convex_polygon_formed (n : ℕ) : Prop := 
  ∃ (a b c d : ℕ), 
    a + b + c + d = n ∧ 
    60 * a + 90 * b + 120 * c + 150 * d = 180 * (n - 2)

-- Equivalent proof problem
theorem possible_values_of_n :
  ∃ (n : ℕ), (5 ≤ n ∧ n ≤ 12) ∧ convex_polygon_formed n :=
sorry

end NUMINAMATH_GPT_possible_values_of_n_l2426_242660


namespace NUMINAMATH_GPT_ellipse_equation_correct_l2426_242676

theorem ellipse_equation_correct :
  ∃ (a b h k : ℝ), 
    h = 4 ∧ 
    k = 0 ∧ 
    a = 10 + 2 * Real.sqrt 10 ∧ 
    b = Real.sqrt (101 + 20 * Real.sqrt 10) ∧ 
    (∀ x y : ℝ, (x, y) = (9, 6) → 
    ((x - h)^2 / a^2 + y^2 / b^2 = 1)) ∧
    (dist (4 - 3, 0) (4 + 3, 0) = 6) := 
sorry

end NUMINAMATH_GPT_ellipse_equation_correct_l2426_242676


namespace NUMINAMATH_GPT_ducks_and_geese_difference_l2426_242645

variable (d g d' l : ℕ)
variables (hd : d = 25)
variables (hg : g = 2 * d - 10)
variables (hd' : d' = d + 4)
variables (hl : l = 15 - 5)

theorem ducks_and_geese_difference :
  let geese_remain := g - l
  let ducks_remain := d'
  geese_remain - ducks_remain = 1 :=
by
  sorry

end NUMINAMATH_GPT_ducks_and_geese_difference_l2426_242645


namespace NUMINAMATH_GPT_seq_a_n_100th_term_l2426_242632

theorem seq_a_n_100th_term :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ 
  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) ∧ 
  a 100 = -3 := 
sorry

end NUMINAMATH_GPT_seq_a_n_100th_term_l2426_242632


namespace NUMINAMATH_GPT_parabola_focus_value_of_a_l2426_242640

theorem parabola_focus_value_of_a :
  (∀ a : ℝ, (∃ y : ℝ, y = a * (0^2) ∧ (0, y) = (0, 3 / 8)) → a = 2 / 3) := by
sorry

end NUMINAMATH_GPT_parabola_focus_value_of_a_l2426_242640


namespace NUMINAMATH_GPT_sequence_formula_l2426_242654

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n →  1 / a (n + 1) = 1 / a n + 1) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by {
  sorry
}

end NUMINAMATH_GPT_sequence_formula_l2426_242654


namespace NUMINAMATH_GPT_number_of_sheep_l2426_242672

variable (S H C : ℕ)

def ratio_constraint : Prop := 4 * H = 7 * S ∧ 5 * S = 4 * C

def horse_food_per_day (H : ℕ) : ℕ := 230 * H
def sheep_food_per_day (S : ℕ) : ℕ := 150 * S
def cow_food_per_day (C : ℕ) : ℕ := 300 * C

def total_horse_food : Prop := horse_food_per_day H = 12880
def total_sheep_food : Prop := sheep_food_per_day S = 9750
def total_cow_food : Prop := cow_food_per_day C = 15000

theorem number_of_sheep (h1 : ratio_constraint S H C)
                        (h2 : total_horse_food H)
                        (h3 : total_sheep_food S)
                        (h4 : total_cow_food C) :
  S = 98 :=
sorry

end NUMINAMATH_GPT_number_of_sheep_l2426_242672


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l2426_242699

theorem sufficient_and_necessary_condition (x : ℝ) : 
  2 * x - 4 ≥ 0 ↔ x ≥ 2 :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l2426_242699


namespace NUMINAMATH_GPT_factorization_eq_l2426_242670

variable (x y : ℝ)

theorem factorization_eq : 9 * y - 25 * x^2 * y = y * (3 + 5 * x) * (3 - 5 * x) :=
by sorry 

end NUMINAMATH_GPT_factorization_eq_l2426_242670


namespace NUMINAMATH_GPT_total_legs_in_park_l2426_242616

theorem total_legs_in_park :
  let dogs := 109
  let cats := 37
  let birds := 52
  let spiders := 19
  let dog_legs := 4
  let cat_legs := 4
  let bird_legs := 2
  let spider_legs := 8
  dogs * dog_legs + cats * cat_legs + birds * bird_legs + spiders * spider_legs = 840 := by
  sorry

end NUMINAMATH_GPT_total_legs_in_park_l2426_242616


namespace NUMINAMATH_GPT_find_prime_A_l2426_242648

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_prime_A (A : ℕ) :
  is_prime A ∧ is_prime (A + 14) ∧ is_prime (A + 18) ∧ is_prime (A + 32) ∧ is_prime (A + 36) → A = 5 := by
  sorry

end NUMINAMATH_GPT_find_prime_A_l2426_242648


namespace NUMINAMATH_GPT_factor_expression_l2426_242697

variable {a : ℝ}

theorem factor_expression :
  ((10 * a^4 - 160 * a^3 - 32) - (-2 * a^4 - 16 * a^3 + 32)) = 4 * (3 * a^3 * (a - 12) - 16) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2426_242697


namespace NUMINAMATH_GPT_factor_expression_l2426_242611

theorem factor_expression (b : ℝ) :
  (8 * b^4 - 100 * b^3 + 14 * b^2) - (3 * b^4 - 10 * b^3 + 14 * b^2) = 5 * b^3 * (b - 18) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2426_242611


namespace NUMINAMATH_GPT_initial_cakes_l2426_242622

variable (friend_bought : Nat) (baker_has : Nat)

theorem initial_cakes (h1 : friend_bought = 140) (h2 : baker_has = 15) : 
  (friend_bought + baker_has = 155) := 
by
  sorry

end NUMINAMATH_GPT_initial_cakes_l2426_242622


namespace NUMINAMATH_GPT_photo_album_slots_l2426_242605

def photos_from_cristina : Nat := 7
def photos_from_john : Nat := 10
def photos_from_sarah : Nat := 9
def photos_from_clarissa : Nat := 14

theorem photo_album_slots :
  photos_from_cristina + photos_from_john + photos_from_sarah + photos_from_clarissa = 40 :=
by
  sorry

end NUMINAMATH_GPT_photo_album_slots_l2426_242605


namespace NUMINAMATH_GPT_probability_not_blue_marble_l2426_242662

-- Define the conditions
def odds_for_blue_marble : ℕ := 5
def odds_for_not_blue_marble : ℕ := 6
def total_outcomes := odds_for_blue_marble + odds_for_not_blue_marble

-- Define the question and statement to be proven
theorem probability_not_blue_marble :
  (odds_for_not_blue_marble : ℚ) / total_outcomes = 6 / 11 :=
by
  -- skipping the proof step as per instruction
  sorry

end NUMINAMATH_GPT_probability_not_blue_marble_l2426_242662


namespace NUMINAMATH_GPT_linear_valid_arrangements_circular_valid_arrangements_l2426_242601

def word := "EFFERVESCES"
def multiplicities := [("E", 4), ("F", 2), ("S", 2), ("R", 1), ("V", 1), ("C", 1)]

-- Number of valid linear arrangements
def linear_arrangements_no_adj_e : ℕ := 88200

-- Number of valid circular arrangements
def circular_arrangements_no_adj_e : ℕ := 6300

theorem linear_valid_arrangements : 
  ∃ n, n = linear_arrangements_no_adj_e := 
  by
    sorry 

theorem circular_valid_arrangements :
  ∃ n, n = circular_arrangements_no_adj_e :=
  by
    sorry

end NUMINAMATH_GPT_linear_valid_arrangements_circular_valid_arrangements_l2426_242601


namespace NUMINAMATH_GPT_train_pass_time_correct_l2426_242661

noncomputable def train_time_to_pass_post (length_of_train : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  length_of_train / speed_mps

theorem train_pass_time_correct :
  train_time_to_pass_post 60 36 = 6 := by
  sorry

end NUMINAMATH_GPT_train_pass_time_correct_l2426_242661


namespace NUMINAMATH_GPT_exists_two_people_with_property_l2426_242633

theorem exists_two_people_with_property (n : ℕ) (P : Fin (2 * n + 2) → Fin (2 * n + 2) → Prop) :
  ∃ A B : Fin (2 * n + 2), 
    A ≠ B ∧
    (∃ S : Finset (Fin (2 * n + 2)), 
      S.card = n ∧
      ∀ C ∈ S, (P C A ∧ P C B) ∨ (¬P C A ∧ ¬P C B)) :=
sorry

end NUMINAMATH_GPT_exists_two_people_with_property_l2426_242633


namespace NUMINAMATH_GPT_solve_trig_eq_l2426_242651

open Real

theorem solve_trig_eq (k : ℤ) : 
  (∃ x : ℝ, 
    (|cos x| + cos (3 * x)) / (sin x * cos (2 * x)) = -2 * sqrt 3 
    ∧ (x = -π/6 + 2 * k * π ∨ x = 2 * π/3 + 2 * k * π ∨ x = 7 * π/6 + 2 * k * π)) :=
sorry

end NUMINAMATH_GPT_solve_trig_eq_l2426_242651


namespace NUMINAMATH_GPT_gcd_max_value_l2426_242653

theorem gcd_max_value (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + b = 1005) : ∃ d, d = Int.gcd a b ∧ d = 335 :=
by {
  sorry
}

end NUMINAMATH_GPT_gcd_max_value_l2426_242653


namespace NUMINAMATH_GPT_blanket_rate_l2426_242621

/-- 
A man purchased 4 blankets at Rs. 100 each, 
5 blankets at Rs. 150 each, 
and two blankets at an unknown rate x. 
If the average price of the blankets was Rs. 150, 
prove that the unknown rate x is 250. 
-/
theorem blanket_rate (x : ℝ) 
  (h1 : 4 * 100 + 5 * 150 + 2 * x = 11 * 150) : 
  x = 250 := 
sorry

end NUMINAMATH_GPT_blanket_rate_l2426_242621


namespace NUMINAMATH_GPT_abc_sum_is_twelve_l2426_242674

theorem abc_sum_is_twelve
  (f : ℤ → ℤ)
  (a b c : ℕ)
  (h1 : f 1 = 10)
  (h2 : f 0 = 8)
  (h3 : f (-3) = -28)
  (h4 : ∀ x, x > 0 → f x = 2 * a * x + 6)
  (h5 : f 0 = a^2 * b)
  (h6 : ∀ x, x < 0 → f x = 2 * b * x + 2 * c)
  : a + b + c = 12 := sorry

end NUMINAMATH_GPT_abc_sum_is_twelve_l2426_242674


namespace NUMINAMATH_GPT_john_finishes_fourth_task_at_12_18_PM_l2426_242618

theorem john_finishes_fourth_task_at_12_18_PM :
  let start_time := 8 * 60 + 45 -- Start time in minutes from midnight
  let third_task_time := 11 * 60 + 25 -- End time of the third task in minutes from midnight
  let total_time_three_tasks := third_task_time - start_time -- Total time in minutes to complete three tasks
  let time_per_task := total_time_three_tasks / 3 -- Time per task in minutes
  let fourth_task_end_time := third_task_time + time_per_task -- End time of the fourth task in minutes from midnight
  fourth_task_end_time = 12 * 60 + 18 := -- Expected end time in minutes from midnight
  sorry

end NUMINAMATH_GPT_john_finishes_fourth_task_at_12_18_PM_l2426_242618


namespace NUMINAMATH_GPT_distinct_domino_paths_l2426_242688

/-- Matt will arrange five identical, dotless dominoes (1 by 2 rectangles) 
on a 6 by 4 grid so that a path is formed from the upper left-hand corner 
(0, 0) to the lower right-hand corner (4, 5). Prove that the number of 
distinct arrangements is 126. -/
theorem distinct_domino_paths : 
  let m := 4
  let n := 5
  let total_moves := m + n
  let right_moves := m
  let down_moves := n
  (total_moves.choose right_moves) = 126 := by
{ 
  sorry 
}

end NUMINAMATH_GPT_distinct_domino_paths_l2426_242688


namespace NUMINAMATH_GPT_center_circle_sum_l2426_242668

theorem center_circle_sum (x y : ℝ) (h : x^2 + y^2 = 4 * x + 10 * y - 12) : x + y = 7 := 
sorry

end NUMINAMATH_GPT_center_circle_sum_l2426_242668


namespace NUMINAMATH_GPT_min_value_of_f_range_of_a_l2426_242682

noncomputable def f (x : ℝ) : ℝ := 2 * x * Real.log x - 1

theorem min_value_of_f : ∃ x ∈ Set.Ioi 0, ∀ y ∈ Set.Ioi 0, f y ≥ f x ∧ f x = -2 * Real.exp (-1) - 1 := 
  sorry

theorem range_of_a {a : ℝ} : (∀ x > 0, f x ≤ 3 * x^2 + 2 * a * x) ↔ a ∈ Set.Ici (-2) := 
  sorry

end NUMINAMATH_GPT_min_value_of_f_range_of_a_l2426_242682


namespace NUMINAMATH_GPT_older_brother_catches_up_in_half_hour_l2426_242644

-- Defining the parameters according to the conditions
def speed_younger_brother := 4 -- kilometers per hour
def speed_older_brother := 20 -- kilometers per hour
def initial_distance := 8 -- kilometers

-- Calculate the relative speed difference
def speed_difference := speed_older_brother - speed_younger_brother

theorem older_brother_catches_up_in_half_hour:
  ∃ t : ℝ, initial_distance = speed_difference * t ∧ t = 0.5 := by
  use 0.5
  sorry

end NUMINAMATH_GPT_older_brother_catches_up_in_half_hour_l2426_242644


namespace NUMINAMATH_GPT_range_of_m_l2426_242624

theorem range_of_m (m x : ℝ) :
  (m-1 < x ∧ x < m+1) → (2 < x ∧ x < 6) → (3 ≤ m ∧ m ≤ 5) :=
by
  intros hp hq
  sorry

end NUMINAMATH_GPT_range_of_m_l2426_242624


namespace NUMINAMATH_GPT_sum_of_distinct_integers_l2426_242636

theorem sum_of_distinct_integers (a b c d : ℤ) (h : (a - 1) * (b - 1) * (c - 1) * (d - 1) = 25) (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) : a + b + c + d = 4 :=
by
    sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_l2426_242636


namespace NUMINAMATH_GPT_total_nails_needed_l2426_242667

-- Definitions based on problem conditions
def nails_per_plank : ℕ := 2
def planks_needed : ℕ := 2

-- Theorem statement: Prove that the total number of nails John needs is 4.
theorem total_nails_needed : nails_per_plank * planks_needed = 4 := by
  sorry

end NUMINAMATH_GPT_total_nails_needed_l2426_242667


namespace NUMINAMATH_GPT_number_of_children_l2426_242684

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l2426_242684


namespace NUMINAMATH_GPT_Damien_jogs_miles_over_three_weeks_l2426_242658

theorem Damien_jogs_miles_over_three_weeks :
  (5 * 5) * 3 = 75 :=
by sorry

end NUMINAMATH_GPT_Damien_jogs_miles_over_three_weeks_l2426_242658


namespace NUMINAMATH_GPT_sqrt_10_bounds_l2426_242664

theorem sqrt_10_bounds : 10 > 9 ∧ 10 < 16 → 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_10_bounds_l2426_242664


namespace NUMINAMATH_GPT_map_distance_l2426_242656

theorem map_distance (scale_cm : ℝ) (scale_km : ℝ) (actual_distance_km : ℝ) 
  (h1 : scale_cm = 0.4) (h2 : scale_km = 5.3) (h3 : actual_distance_km = 848) :
  actual_distance_km / (scale_km / scale_cm) = 64 :=
by
  rw [h1, h2, h3]
  -- Further steps would follow here, but to ensure code compiles
  -- and there is no assumption directly from solution steps, we use sorry.
  sorry

end NUMINAMATH_GPT_map_distance_l2426_242656


namespace NUMINAMATH_GPT_problem_xyz_l2426_242686

noncomputable def distance_from_intersection_to_side_CD (s : ℝ) : ℝ :=
  s * ((8 - Real.sqrt 15) / 8)

theorem problem_xyz
  (s : ℝ)
  (ABCD_is_square : (0 ≤ s))
  (X_is_intersection: ∃ (X : ℝ × ℝ), (X.1^2 + X.2^2 = s^2) ∧ ((X.1 - s)^2 + X.2^2 = (s / 2)^2))
  : distance_from_intersection_to_side_CD s = (s * (8 - Real.sqrt 15) / 8) :=
sorry

end NUMINAMATH_GPT_problem_xyz_l2426_242686


namespace NUMINAMATH_GPT_part1_part2_l2426_242619

variable (A B C : ℝ) (a b c : ℝ)
variable (h1 : a = 5) (h2 : c = 6) (h3 : Real.sin B = 3 / 5) (h4 : b < a)

-- Part 1: Prove b = sqrt(13) and sin A = (3 * sqrt(13)) / 13
theorem part1 : b = Real.sqrt 13 ∧ Real.sin A = (3 * Real.sqrt 13) / 13 := sorry

-- Part 2: Prove sin (2A + π / 4) = 7 * sqrt(2) / 26
theorem part2 (h5 : b = Real.sqrt 13) (h6 : Real.sin A = (3 * Real.sqrt 13) / 13) : 
  Real.sin (2 * A + Real.pi / 4) = (7 * Real.sqrt 2) / 26 := sorry

end NUMINAMATH_GPT_part1_part2_l2426_242619


namespace NUMINAMATH_GPT_employee_salary_proof_l2426_242600

variable (x : ℝ) (M : ℝ) (P : ℝ)

theorem employee_salary_proof (h1 : x + 1.2 * x + 1.8 * x = 1500)
(h2 : M = 1.2 * x)
(h3 : P = 1.8 * x)
: x = 375 ∧ M = 450 ∧ P = 675 :=
sorry

end NUMINAMATH_GPT_employee_salary_proof_l2426_242600


namespace NUMINAMATH_GPT_men_to_complete_work_l2426_242631

theorem men_to_complete_work (x : ℕ) (h1 : 10 * 80 = x * 40) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_men_to_complete_work_l2426_242631


namespace NUMINAMATH_GPT_digit_distribution_l2426_242680

theorem digit_distribution (n: ℕ) : 
(1 / 2) * n + (1 / 5) * n + (1 / 5) * n + (1 / 10) * n = n → 
n = 10 :=
by
  sorry

end NUMINAMATH_GPT_digit_distribution_l2426_242680


namespace NUMINAMATH_GPT_at_least_one_not_solved_l2426_242635

theorem at_least_one_not_solved (p q : Prop) : (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by sorry

end NUMINAMATH_GPT_at_least_one_not_solved_l2426_242635


namespace NUMINAMATH_GPT_solution_l2426_242642

def g (x : ℝ) : ℝ := x^2 - 4 * x

theorem solution (x : ℝ) : g (g x) = g x ↔ x = 0 ∨ x = 4 ∨ x = 5 ∨ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_solution_l2426_242642


namespace NUMINAMATH_GPT_probability_shots_result_l2426_242630

open ProbabilityTheory

noncomputable def P_A := 3 / 4
noncomputable def P_B := 4 / 5
noncomputable def P_not_A := 1 - P_A
noncomputable def P_not_B := 1 - P_B

theorem probability_shots_result :
    (P_not_A * P_not_B * P_A) + (P_not_A * P_not_B * P_not_A * P_B) = 19 / 400 :=
    sorry

end NUMINAMATH_GPT_probability_shots_result_l2426_242630


namespace NUMINAMATH_GPT_simplify_expression_l2426_242655

theorem simplify_expression (x y : ℝ) : 
  (x - y) * (x + y) + (x - y) ^ 2 = 2 * x ^ 2 - 2 * x * y :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2426_242655


namespace NUMINAMATH_GPT_algebraic_expression_value_l2426_242698

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) :
  x^2 - 6 * x * y + 9 * y^2 = 4 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2426_242698


namespace NUMINAMATH_GPT_ratio_geometric_sequence_of_arithmetic_l2426_242637

variable {d : ℤ}
variable {a : ℕ → ℤ}

-- definition of an arithmetic sequence with common difference d
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- definition of a geometric sequence for a_5, a_9, a_{15}
def geometric_sequence (a : ℕ → ℤ) : Prop :=
  a 9 * a 9 = a 5 * a 15

theorem ratio_geometric_sequence_of_arithmetic
  (h_arith : arithmetic_sequence a d) (h_nonzero : d ≠ 0) (h_geom : geometric_sequence a) :
  a 15 / a 9 = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_geometric_sequence_of_arithmetic_l2426_242637


namespace NUMINAMATH_GPT_total_gold_coins_l2426_242649

/--
An old man distributed all the gold coins he had to his two sons into 
two different numbers such that the difference between the squares 
of the two numbers is 49 times the difference between the two numbers. 
Prove that the total number of gold coins the old man had is 49.
-/
theorem total_gold_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 49 * (x - y)) : x + y = 49 :=
sorry

end NUMINAMATH_GPT_total_gold_coins_l2426_242649
