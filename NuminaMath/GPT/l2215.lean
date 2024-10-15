import Mathlib

namespace NUMINAMATH_GPT_donut_selection_count_l2215_221595

def num_donut_selections : ℕ :=
  Nat.choose 9 3

theorem donut_selection_count : num_donut_selections = 84 := 
by
  sorry

end NUMINAMATH_GPT_donut_selection_count_l2215_221595


namespace NUMINAMATH_GPT_total_rowing_time_l2215_221537

theorem total_rowing_time (s_b : ℕ) (s_s : ℕ) (d : ℕ) : 
  s_b = 9 → s_s = 6 → d = 170 → 
  (d / (s_b + s_s) + d / (s_b - s_s)) = 68 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_total_rowing_time_l2215_221537


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l2215_221596

-- Define the first equation as a condition
def equation1 (x : ℝ) : Prop :=
  3 * x + 20 = 4 * x - 25

-- Prove that x = 45 satisfies equation1
theorem solve_equation1 : equation1 45 :=
by 
  -- Proof steps would go here
  sorry

-- Define the second equation as a condition
def equation2 (x : ℝ) : Prop :=
  (2 * x - 1) / 3 = 1 - (2 * x - 1) / 6

-- Prove that x = 3/2 satisfies equation2
theorem solve_equation2 : equation2 (3 / 2) :=
by 
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l2215_221596


namespace NUMINAMATH_GPT_mildred_heavier_than_carol_l2215_221515

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem mildred_heavier_than_carol : mildred_weight - carol_weight = 50 := 
by
  sorry

end NUMINAMATH_GPT_mildred_heavier_than_carol_l2215_221515


namespace NUMINAMATH_GPT_find_natural_pairs_l2215_221582

-- Definitions
def is_natural (n : ℕ) : Prop := n > 0
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def satisfies_equation (x y : ℕ) : Prop := 2 * x^2 + 5 * x * y + 3 * y^2 = 41 * x + 62 * y + 21

-- Problem statement
theorem find_natural_pairs (x y : ℕ) (hx : is_natural x) (hy : is_natural y) (hrel : relatively_prime x y) :
  satisfies_equation x y ↔ (x = 2 ∧ y = 19) ∨ (x = 19 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_natural_pairs_l2215_221582


namespace NUMINAMATH_GPT_difference_of_numbers_l2215_221568

noncomputable def larger_num : ℕ := 1495
noncomputable def quotient : ℕ := 5
noncomputable def remainder : ℕ := 4

theorem difference_of_numbers :
  ∃ S : ℕ, larger_num = quotient * S + remainder ∧ (larger_num - S = 1197) :=
by 
  sorry

end NUMINAMATH_GPT_difference_of_numbers_l2215_221568


namespace NUMINAMATH_GPT_find_vidya_age_l2215_221552

theorem find_vidya_age (V M : ℕ) (h1: M = 3 * V + 5) (h2: M = 44) : V = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_vidya_age_l2215_221552


namespace NUMINAMATH_GPT_shaltaev_boltaev_proof_l2215_221566

variable (S B : ℝ)

axiom cond1 : 175 * S > 125 * B
axiom cond2 : 175 * S < 126 * B

theorem shaltaev_boltaev_proof : 3 * S + B ≥ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_shaltaev_boltaev_proof_l2215_221566


namespace NUMINAMATH_GPT_david_tips_l2215_221520

noncomputable def avg_tips_resort (tips_other_months : ℝ) (months : ℕ) := tips_other_months / months

theorem david_tips 
  (tips_march_to_july_september : ℝ)
  (tips_august_resort : ℝ)
  (total_tips_delivery_driver : ℝ)
  (total_tips_resort : ℝ)
  (total_tips : ℝ)
  (fraction_august : ℝ)
  (avg_tips := avg_tips_resort tips_march_to_july_september 6):
  tips_august_resort = 4 * avg_tips →
  total_tips_delivery_driver = 2 * avg_tips →
  total_tips_resort = tips_march_to_july_september + tips_august_resort →
  total_tips = total_tips_resort + total_tips_delivery_driver →
  fraction_august = tips_august_resort / total_tips →
  fraction_august = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_david_tips_l2215_221520


namespace NUMINAMATH_GPT_area_of_right_triangle_l2215_221529

variables {x y : ℝ} (r : ℝ)

theorem area_of_right_triangle (hx : ∀ r, r * (x + y + r) = x * y) :
  1 / 2 * (x + r) * (y + r) = x * y :=
by sorry

end NUMINAMATH_GPT_area_of_right_triangle_l2215_221529


namespace NUMINAMATH_GPT_triangle_is_isosceles_l2215_221506

theorem triangle_is_isosceles (A B C : ℝ) (a b c : ℝ) 
  (h1 : c = 2 * a * Real.cos B) 
  (h2 : a = b) :
  ∃ (isIsosceles : Bool), isIsosceles := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l2215_221506


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l2215_221579

noncomputable def a : ℝ := Real.log (7 / 2) / Real.log 3
noncomputable def b : ℝ := (1 / 4)^(1 / 3)
noncomputable def c : ℝ := -Real.log 5 / Real.log 3

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l2215_221579


namespace NUMINAMATH_GPT_bill_profit_difference_l2215_221546

theorem bill_profit_difference (P : ℝ) 
  (h1 : 1.10 * P = 549.9999999999995)
  (h2 : ∀ NP NSP, NP = 0.90 * P ∧ NSP = 1.30 * NP →
  NSP - 549.9999999999995 = 35) :
  true :=
by {
  sorry
}

end NUMINAMATH_GPT_bill_profit_difference_l2215_221546


namespace NUMINAMATH_GPT_no_solution_in_natural_numbers_l2215_221576

theorem no_solution_in_natural_numbers :
  ¬ ∃ (x y : ℕ), 2^x + 21^x = y^3 :=
sorry

end NUMINAMATH_GPT_no_solution_in_natural_numbers_l2215_221576


namespace NUMINAMATH_GPT_perfect_square_sequence_l2215_221502

theorem perfect_square_sequence (x : ℕ → ℤ) (h₀ : x 0 = 0) (h₁ : x 1 = 3) 
  (h₂ : ∀ n, x (n + 1) + x (n - 1) = 4 * x n) : 
  ∀ n, ∃ k : ℤ, x (n + 1) * x (n - 1) + 9 = k^2 :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_sequence_l2215_221502


namespace NUMINAMATH_GPT_max_value_fraction_l2215_221530

theorem max_value_fraction (x : ℝ) : 
  ∃ (n : ℤ), n = 3 ∧ 
  ∃ (y : ℝ), y = (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ∧ 
  y ≤ n := 
sorry

end NUMINAMATH_GPT_max_value_fraction_l2215_221530


namespace NUMINAMATH_GPT_fraction_of_raisins_l2215_221597

-- Define the cost of a single pound of raisins
variables (R : ℝ) -- R represents the cost of one pound of raisins

-- Conditions
def mixed_raisins := 5 -- Chris mixed 5 pounds of raisins
def mixed_nuts := 4 -- with 4 pounds of nuts
def nuts_cost_ratio := 3 -- A pound of nuts costs 3 times as much as a pound of raisins

-- Statement to prove
theorem fraction_of_raisins
  (R_pos : R > 0) : (5 * R) / ((5 * R) + (4 * (3 * R))) = 5 / 17 :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_fraction_of_raisins_l2215_221597


namespace NUMINAMATH_GPT_find_abc_squared_sum_l2215_221594

theorem find_abc_squared_sum (a b c : ℕ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : a^3 + 32 * b + 2 * c = 2018) (h₃ : b^3 + 32 * a + 2 * c = 1115) :
  a^2 + b^2 + c^2 = 226 :=
sorry

end NUMINAMATH_GPT_find_abc_squared_sum_l2215_221594


namespace NUMINAMATH_GPT_apples_left_l2215_221531

def Mike_apples : ℝ := 7.0
def Nancy_apples : ℝ := 3.0
def Keith_ate_apples : ℝ := 6.0

theorem apples_left : Mike_apples + Nancy_apples - Keith_ate_apples = 4.0 := by
  sorry

end NUMINAMATH_GPT_apples_left_l2215_221531


namespace NUMINAMATH_GPT_bowling_team_score_ratio_l2215_221545

theorem bowling_team_score_ratio :
  ∀ (F S T : ℕ),
  F + S + T = 810 →
  F = (1 / 3 : ℚ) * S →
  T = 162 →
  S / T = 3 := 
by
  intros F S T h1 h2 h3
  sorry

end NUMINAMATH_GPT_bowling_team_score_ratio_l2215_221545


namespace NUMINAMATH_GPT_rowan_distance_downstream_l2215_221588

-- Conditions
def speed_still : ℝ := 9.75
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4

-- Statement to prove
theorem rowan_distance_downstream : ∃ (d : ℝ) (c : ℝ), 
  d / (speed_still + c) = downstream_time ∧
  d / (speed_still - c) = upstream_time ∧
  d = 26 := by
    sorry

end NUMINAMATH_GPT_rowan_distance_downstream_l2215_221588


namespace NUMINAMATH_GPT_determine_F_l2215_221555

def f1 (x : ℝ) : ℝ := x^2 + x
def f2 (x : ℝ) : ℝ := 2 * x^2 - x
def f3 (x : ℝ) : ℝ := x^2 + x

def g1 (x : ℝ) : ℝ := x - 2
def g2 (x : ℝ) : ℝ := 2 * x
def g3 (x : ℝ) : ℝ := x + 2

def h (x : ℝ) : ℝ := x

theorem determine_F (F1 F2 F3 : ℕ) : 
  (F1 = 0 ∧ F2 = 0 ∧ F3 = 1) :=
by
  sorry

end NUMINAMATH_GPT_determine_F_l2215_221555


namespace NUMINAMATH_GPT_subcommittee_count_l2215_221559

theorem subcommittee_count :
  let total_members := 12
  let teachers := 5
  let total_subcommittees := (Nat.choose total_members 4)
  let subcommittees_with_zero_teachers := (Nat.choose 7 4)
  let subcommittees_with_one_teacher := (Nat.choose teachers 1) * (Nat.choose 7 3)
  let subcommittees_with_fewer_than_two_teachers := subcommittees_with_zero_teachers + subcommittees_with_one_teacher
  let subcommittees_with_at_least_two_teachers := total_subcommittees - subcommittees_with_fewer_than_two_teachers
  subcommittees_with_at_least_two_teachers = 285 := by
  sorry

end NUMINAMATH_GPT_subcommittee_count_l2215_221559


namespace NUMINAMATH_GPT_odds_against_C_l2215_221583

theorem odds_against_C (pA pB : ℚ) (hA : pA = 1 / 5) (hB : pB = 2 / 3) :
  (1 - (1 - pA + 1 - pB)) / (1 - pA - pB) = 13 / 2 := 
sorry

end NUMINAMATH_GPT_odds_against_C_l2215_221583


namespace NUMINAMATH_GPT_num_positive_integer_solutions_l2215_221590

theorem num_positive_integer_solutions : 
  ∃ n : ℕ, (∀ x : ℕ, x ≤ n → x - 1 < Real.sqrt 5) ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_num_positive_integer_solutions_l2215_221590


namespace NUMINAMATH_GPT_principal_invested_years_l2215_221524

-- Define the given conditions
def principal : ℕ := 9200
def rate : ℕ := 12
def interest_deficit : ℤ := 5888

-- Define the time to be proved
def time_invested : ℤ := 3

-- Define the simple interest formula
def simple_interest (P R t : ℕ) : ℕ :=
  (P * R * t) / 100

-- Define the problem statement
theorem principal_invested_years :
  ∃ t : ℕ, principal - interest_deficit = simple_interest principal rate t ∧ t = time_invested := 
by
  sorry

end NUMINAMATH_GPT_principal_invested_years_l2215_221524


namespace NUMINAMATH_GPT_negative_solution_range_l2215_221528

theorem negative_solution_range (m x : ℝ) (h : (2 * x + m) / (x - 1) = 1) (hx : x < 0) : m > -1 :=
  sorry

end NUMINAMATH_GPT_negative_solution_range_l2215_221528


namespace NUMINAMATH_GPT_f_f_2_equals_l2215_221549

def f (x : ℕ) : ℕ := 4 * x ^ 3 - 6 * x + 2

theorem f_f_2_equals :
  f (f 2) = 42462 :=
by
  sorry

end NUMINAMATH_GPT_f_f_2_equals_l2215_221549


namespace NUMINAMATH_GPT_number_of_planters_l2215_221560

variable (a b : ℕ)

-- Conditions
def tree_planting_condition_1 : Prop := a * b = 2013
def tree_planting_condition_2 : Prop := (a - 5) * (b + 2) < 2013
def tree_planting_condition_3 : Prop := (a - 5) * (b + 3) > 2013

-- Theorem stating the number of people who participated in the planting is 61
theorem number_of_planters (h1 : tree_planting_condition_1 a b) 
                           (h2 : tree_planting_condition_2 a b) 
                           (h3 : tree_planting_condition_3 a b) : 
                           a = 61 := 
sorry

end NUMINAMATH_GPT_number_of_planters_l2215_221560


namespace NUMINAMATH_GPT_integer_solutions_of_log_inequality_l2215_221567

def log_inequality_solution_set : Set ℤ := {0, 1, 2}

theorem integer_solutions_of_log_inequality (x : ℤ) (h : 2 < Real.log (x + 5) / Real.log 2 ∧ Real.log (x + 5) / Real.log 2 < 3) :
    x ∈ log_inequality_solution_set :=
sorry

end NUMINAMATH_GPT_integer_solutions_of_log_inequality_l2215_221567


namespace NUMINAMATH_GPT_sum_of_other_endpoint_l2215_221542

theorem sum_of_other_endpoint (x y : ℝ) (h₁ : (9 + x) / 2 = 5) (h₂ : (-6 + y) / 2 = -8) :
  x + y = -9 :=
sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_l2215_221542


namespace NUMINAMATH_GPT_mary_baseball_cards_count_l2215_221564

def mary_initial_cards := 18
def mary_torn_cards := 8
def fred_gift_cards := 26
def mary_bought_cards := 40
def exchange_with_tom := 0
def mary_lost_cards := 5
def trade_with_lisa_gain := 1
def exchange_with_alex_loss := 2

theorem mary_baseball_cards_count : 
  mary_initial_cards - mary_torn_cards
  + fred_gift_cards
  + mary_bought_cards 
  + exchange_with_tom
  - mary_lost_cards
  + trade_with_lisa_gain 
  - exchange_with_alex_loss 
  = 70 := 
by
  sorry

end NUMINAMATH_GPT_mary_baseball_cards_count_l2215_221564


namespace NUMINAMATH_GPT_cream_ratio_l2215_221547

variable (servings : ℕ) (fat_per_serving : ℕ) (fat_per_cup : ℕ)
variable (h_servings : servings = 4) (h_fat_per_serving : fat_per_serving = 11) (h_fat_per_cup : fat_per_cup = 88)

theorem cream_ratio (total_fat : ℕ) (h_total_fat : total_fat = fat_per_serving * servings) :
  (total_fat : ℚ) / fat_per_cup = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cream_ratio_l2215_221547


namespace NUMINAMATH_GPT_arithmetic_geometric_sum_l2215_221586

theorem arithmetic_geometric_sum {n : ℕ} (a : ℕ → ℤ) (S : ℕ → ℚ) 
  (h1 : ∀ k, a (k + 1) = a k + 2) 
  (h2 : (a 1) * (a 1 + a 4) = (a 1 + a 2) ^ 2 / 2) :
  S n = 6 - (4 * n + 6) / 2^n :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_sum_l2215_221586


namespace NUMINAMATH_GPT_range_of_a_l2215_221517

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (2 - a) * x - 3 * a + 3 
  else Real.log x / Real.log a

-- Main statement to prove
theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (5 / 4 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2215_221517


namespace NUMINAMATH_GPT_find_number_l2215_221554

variable (x : ℕ)

theorem find_number (h : (10 + 20 + x) / 3 = ((10 + 40 + 25) / 3) + 5) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2215_221554


namespace NUMINAMATH_GPT_dogs_remaining_end_month_l2215_221599

theorem dogs_remaining_end_month :
  let initial_dogs := 200
  let dogs_arrive_w1 := 30
  let dogs_adopt_w1 := 40
  let dogs_arrive_w2 := 40
  let dogs_adopt_w2 := 50
  let dogs_arrive_w3 := 30
  let dogs_adopt_w3 := 30
  let dogs_adopt_w4 := 70
  let dogs_return_w4 := 20
  initial_dogs + (dogs_arrive_w1 - dogs_adopt_w1) + 
  (dogs_arrive_w2 - dogs_adopt_w2) +
  (dogs_arrive_w3 - dogs_adopt_w3) + 
  (-dogs_adopt_w4 - dogs_return_w4) = 90 := by
  sorry

end NUMINAMATH_GPT_dogs_remaining_end_month_l2215_221599


namespace NUMINAMATH_GPT_candy_bars_total_l2215_221509

theorem candy_bars_total :
  let people : ℝ := 3.0;
  let candy_per_person : ℝ := 1.66666666699999;
  people * candy_per_person = 5.0 :=
by
  let people : ℝ := 3.0
  let candy_per_person : ℝ := 1.66666666699999
  show people * candy_per_person = 5.0
  sorry

end NUMINAMATH_GPT_candy_bars_total_l2215_221509


namespace NUMINAMATH_GPT_set_intersections_l2215_221505

open Set Nat

def I : Set ℕ := univ

def A : Set ℕ := { x | ∃ n, x = 3 * n ∧ ∃ k, n = 2 * k }

def B : Set ℕ := { y | ∃ m, y = m ∧ 24 % m = 0 }

theorem set_intersections :
  A ∩ B = {6, 12, 24} ∧ (I \ A) ∩ B = {1, 2, 3, 4, 8} :=
by
  sorry

end NUMINAMATH_GPT_set_intersections_l2215_221505


namespace NUMINAMATH_GPT_find_percentage_of_other_investment_l2215_221522

theorem find_percentage_of_other_investment
  (total_investment : ℝ) (specific_investment : ℝ) (specific_rate : ℝ) (total_interest : ℝ) 
  (other_investment : ℝ) (other_interest : ℝ) (P : ℝ) :
  total_investment = 17000 ∧
  specific_investment = 12000 ∧
  specific_rate = 0.04 ∧
  total_interest = 1380 ∧
  other_investment = total_investment - specific_investment ∧
  other_interest = total_interest - specific_rate * specific_investment ∧ 
  other_interest = (P / 100) * other_investment
  → P = 18 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_percentage_of_other_investment_l2215_221522


namespace NUMINAMATH_GPT_range_of_a_l2215_221553

def A : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a < x ∧ x < a + 1}

theorem range_of_a (a : ℝ)
  (h₀ : a < 1)
  (h₁ : B a ⊆ A) :
  a ∈ {x : ℝ | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2215_221553


namespace NUMINAMATH_GPT_volume_of_revolved_region_l2215_221565

theorem volume_of_revolved_region :
  let R := {p : ℝ × ℝ | |8 - p.1| + p.2 ≤ 10 ∧ 3 * p.2 - p.1 ≥ 15}
  let volume := (1 / 3) * Real.pi * (7 / Real.sqrt 10)^2 * (7 * Real.sqrt 10 / 4)
  let m := 343
  let n := 12
  let p := 10
  m + n + p = 365 := by
  sorry

end NUMINAMATH_GPT_volume_of_revolved_region_l2215_221565


namespace NUMINAMATH_GPT_gcd_eq_55_l2215_221598

theorem gcd_eq_55 : Nat.gcd 5280 12155 = 55 := sorry

end NUMINAMATH_GPT_gcd_eq_55_l2215_221598


namespace NUMINAMATH_GPT_work_rate_B_l2215_221510

theorem work_rate_B :
  (∀ A B : ℝ, A = 30 → (1 / A + 1 / B = 1 / 19.411764705882355) → B = 55) := by 
    intro A B A_cond combined_rate
    have hA : A = 30 := A_cond
    rw [hA] at combined_rate
    sorry

end NUMINAMATH_GPT_work_rate_B_l2215_221510


namespace NUMINAMATH_GPT_find_original_expenditure_l2215_221513

def original_expenditure (x : ℝ) := 35 * x
def new_expenditure (x : ℝ) := 42 * (x - 1)

theorem find_original_expenditure :
  ∃ x, 35 * x + 42 = 42 * (x - 1) ∧ original_expenditure x = 420 :=
by
  sorry

end NUMINAMATH_GPT_find_original_expenditure_l2215_221513


namespace NUMINAMATH_GPT_percentage_of_x_is_2x_minus_y_l2215_221562

variable (x y : ℝ)
variable (h1 : x / y = 4)
variable (h2 : y ≠ 0)

theorem percentage_of_x_is_2x_minus_y :
  (2 * x - y) / x * 100 = 175 := by
  sorry

end NUMINAMATH_GPT_percentage_of_x_is_2x_minus_y_l2215_221562


namespace NUMINAMATH_GPT_problem_f_2010_l2215_221521

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1 / 4
axiom f_eq : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem problem_f_2010 : f 2010 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_problem_f_2010_l2215_221521


namespace NUMINAMATH_GPT_triangle_is_isosceles_l2215_221571

theorem triangle_is_isosceles {a b c : ℝ} {A B C : ℝ} (h1 : b * Real.cos A = a * Real.cos B) : 
  a = b ∨ a = c ∨ b = c :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l2215_221571


namespace NUMINAMATH_GPT_exists_reals_condition_l2215_221577

-- Define the conditions in Lean
theorem exists_reals_condition (n : ℕ) (h₁ : n ≥ 3) : 
  (∃ a : Fin (n + 2) → ℝ, a 0 = a n ∧ a 1 = a (n + 1) ∧ 
  ∀ i : Fin n, a i * a (i + 1) + 1 = a (i + 2)) ↔ 3 ∣ n := 
sorry

end NUMINAMATH_GPT_exists_reals_condition_l2215_221577


namespace NUMINAMATH_GPT_overall_percentage_gain_l2215_221573

theorem overall_percentage_gain
    (original_price : ℝ)
    (first_increase : ℝ)
    (first_discount : ℝ)
    (second_discount : ℝ)
    (third_discount : ℝ)
    (final_increase : ℝ)
    (final_price : ℝ)
    (overall_gain : ℝ)
    (overall_percentage_gain : ℝ)
    (h1 : original_price = 100)
    (h2 : first_increase = original_price * 1.5)
    (h3 : first_discount = first_increase * 0.9)
    (h4 : second_discount = first_discount * 0.85)
    (h5 : third_discount = second_discount * 0.8)
    (h6 : final_increase = third_discount * 1.1)
    (h7 : final_price = final_increase)
    (h8 : overall_gain = final_price - original_price)
    (h9 : overall_percentage_gain = (overall_gain / original_price) * 100) :
  overall_percentage_gain = 0.98 := by
  sorry

end NUMINAMATH_GPT_overall_percentage_gain_l2215_221573


namespace NUMINAMATH_GPT_greatest_odd_factors_below_200_l2215_221534

theorem greatest_odd_factors_below_200 : ∃ n : ℕ, (n < 200) ∧ (n = 196) ∧ (∃ k : ℕ, n = k^2) ∧ ∀ m : ℕ, (m < 200) ∧ (∃ j : ℕ, m = j^2) → m ≤ n := by
  sorry

end NUMINAMATH_GPT_greatest_odd_factors_below_200_l2215_221534


namespace NUMINAMATH_GPT_rectangle_dimensions_l2215_221561

theorem rectangle_dimensions (a b : ℝ) 
  (h_area : a * b = 12) 
  (h_perimeter : 2 * (a + b) = 26) : 
  (a = 1 ∧ b = 12) ∨ (a = 12 ∧ b = 1) :=
sorry

end NUMINAMATH_GPT_rectangle_dimensions_l2215_221561


namespace NUMINAMATH_GPT_exactly_one_absent_l2215_221518

variables (B K Z : Prop)

theorem exactly_one_absent (h1 : B ∨ K) (h2 : K ∨ Z) (h3 : Z ∨ B)
    (h4 : ¬B ∨ ¬K ∨ ¬Z) : (¬B ∧ K ∧ Z) ∨ (B ∧ ¬K ∧ Z) ∨ (B ∧ K ∧ ¬Z) :=
by
  sorry

end NUMINAMATH_GPT_exactly_one_absent_l2215_221518


namespace NUMINAMATH_GPT_championship_outcomes_l2215_221570

theorem championship_outcomes (students events : ℕ) (hs : students = 5) (he : events = 3) :
  ∃ outcomes : ℕ, outcomes = 5 ^ 3 := by
  sorry

end NUMINAMATH_GPT_championship_outcomes_l2215_221570


namespace NUMINAMATH_GPT_male_population_half_total_l2215_221540

theorem male_population_half_total (total_population : ℕ) (segments : ℕ) (male_segment : ℕ) :
  total_population = 800 ∧ segments = 4 ∧ male_segment = 1 ∧ male_segment = segments / 2 →
  total_population / 2 = 400 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_male_population_half_total_l2215_221540


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2215_221584

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a + b > 4) ↔ (¬ (a > 2 ∧ b > 2)) ∧ ((a > 2 ∧ b > 2) → (a + b > 4)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2215_221584


namespace NUMINAMATH_GPT_power_mod_equivalence_l2215_221501

theorem power_mod_equivalence : (7^700) % 100 = 1 := 
by 
  -- Given that (7^4) % 100 = 1
  have h : 7^4 % 100 = 1 := by sorry
  -- Use this equivalence to prove the statement
  sorry

end NUMINAMATH_GPT_power_mod_equivalence_l2215_221501


namespace NUMINAMATH_GPT_solve_equation1_solve_equation2_l2215_221587

-- Define the first equation and state the theorem that proves its roots
def equation1 (x : ℝ) : Prop := 2 * x^2 + 1 = 3 * x

theorem solve_equation1 (x : ℝ) : equation1 x ↔ (x = 1 ∨ x = 1/2) :=
by sorry

-- Define the second equation and state the theorem that proves its roots
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 = (3 - x)^2

theorem solve_equation2 (x : ℝ) : equation2 x ↔ (x = -2 ∨ x = 4 / 3) :=
by sorry

end NUMINAMATH_GPT_solve_equation1_solve_equation2_l2215_221587


namespace NUMINAMATH_GPT_parabola_directrix_standard_eq_l2215_221500

theorem parabola_directrix_standard_eq (p : ℝ) (h : p = 2) :
  ∀ y x : ℝ, (x = -1) → (y^2 = 4 * x) :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_standard_eq_l2215_221500


namespace NUMINAMATH_GPT_digit_in_base_l2215_221539

theorem digit_in_base (t : ℕ) (h1 : t ≤ 9) (h2 : 5 * 7 + t = t * 9 + 3) : t = 4 := by
  sorry

end NUMINAMATH_GPT_digit_in_base_l2215_221539


namespace NUMINAMATH_GPT_problem_statement_l2215_221556

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-1, 1)

-- Define vector addition
def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define perpendicular condition
def perp (u v : ℝ × ℝ) : Prop := dot_prod u v = 0

theorem problem_statement : perp (vec_add a b) a :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2215_221556


namespace NUMINAMATH_GPT_sin_75_is_sqrt_6_add_sqrt_2_div_4_l2215_221507

noncomputable def sin_75_angle (a : Real) (b : Real) : Real :=
  Real.sin (75 * Real.pi / 180)

theorem sin_75_is_sqrt_6_add_sqrt_2_div_4 :
  sin_75_angle π (π / 6) = (Real.sqrt 6 + Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_75_is_sqrt_6_add_sqrt_2_div_4_l2215_221507


namespace NUMINAMATH_GPT_equation_of_chord_l2215_221511

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

def is_midpoint_of_chord (P M N : ℝ × ℝ) : Prop :=
  ∃ (C : ℝ × ℝ), circle_eq (C.1) (C.2) ∧ (P.1, P.2) = ((M.1 + N.1) / 2, (M.2 + N.2) / 2)

theorem equation_of_chord (P : ℝ × ℝ) (M N : ℝ × ℝ) (h : P = (4, 2)) (h_mid : is_midpoint_of_chord P M N) :
  ∀ (x y : ℝ), (2 * y) - (8 : ℝ) = (-(1 / 2) * (x - 4)) →
  x + 2 * y - 8 = 0 :=
by
  intro x y H
  sorry

end NUMINAMATH_GPT_equation_of_chord_l2215_221511


namespace NUMINAMATH_GPT_expression_evaluates_to_one_l2215_221525

noncomputable def a := Real.sqrt 2 + 0.8
noncomputable def b := Real.sqrt 2 - 0.2

theorem expression_evaluates_to_one : 
  ( (2 - b) / (b - 1) + 2 * (a - 1) / (a - 2) ) / ( b * (a - 1) / (b - 1) + a * (2 - b) / (a - 2) ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluates_to_one_l2215_221525


namespace NUMINAMATH_GPT_credit_extended_l2215_221548

noncomputable def automobile_installment_credit (total_consumer_credit : ℝ) : ℝ :=
  0.43 * total_consumer_credit

noncomputable def extended_by_finance_companies (auto_credit : ℝ) : ℝ :=
  0.25 * auto_credit

theorem credit_extended (total_consumer_credit : ℝ) (h : total_consumer_credit = 465.1162790697675) :
  extended_by_finance_companies (automobile_installment_credit total_consumer_credit) = 50.00 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_credit_extended_l2215_221548


namespace NUMINAMATH_GPT_find_possible_values_of_b_l2215_221592

def good_number (x : ℕ) : Prop :=
  ∃ p n : ℕ, Nat.Prime p ∧ n ≥ 2 ∧ x = p^n

theorem find_possible_values_of_b (b : ℕ) : 
  (b ≥ 4) ∧ good_number (b^2 - 2 * b - 3) ↔ b = 87 := sorry

end NUMINAMATH_GPT_find_possible_values_of_b_l2215_221592


namespace NUMINAMATH_GPT_sum_first_8_terms_64_l2215_221581

-- Define the problem conditions
def isArithmeticSeq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def isGeometricSeq (a : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → n < k → (a n)^2 = a m * a k

-- Given arithmetic sequence with a common difference 2
def arithmeticSeqWithDiff2 (a : ℕ → ℤ) : Prop :=
  isArithmeticSeq a ∧ (∃ d : ℤ, d = 2 ∧ ∀ (n : ℕ), a (n + 1) = a n + d)

-- Given a₁, a₂, a₅ form a geometric sequence
def a1_a2_a5_formGeometricSeq (a: ℕ → ℤ) : Prop :=
  (a 2)^2 = (a 1) * (a 5)

-- Sum of the first 8 terms of the arithmetic sequence
def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  n * (a 1 + a (n - 1)) / 2

-- Main statement
theorem sum_first_8_terms_64 (a : ℕ → ℤ) (h1 : arithmeticSeqWithDiff2 a) (h2 : a1_a2_a5_formGeometricSeq a) : 
  sum_of_first_n_terms a 8 = 64 := 
sorry

end NUMINAMATH_GPT_sum_first_8_terms_64_l2215_221581


namespace NUMINAMATH_GPT_average_candies_correct_l2215_221512

noncomputable def Eunji_candies : ℕ := 35
noncomputable def Jimin_candies : ℕ := Eunji_candies + 6
noncomputable def Jihyun_candies : ℕ := Eunji_candies - 3
noncomputable def Total_candies : ℕ := Eunji_candies + Jimin_candies + Jihyun_candies
noncomputable def Average_candies : ℚ := Total_candies / 3

theorem average_candies_correct :
  Average_candies = 36 := by
  sorry

end NUMINAMATH_GPT_average_candies_correct_l2215_221512


namespace NUMINAMATH_GPT_inequality_pow4_geq_sum_l2215_221504

theorem inequality_pow4_geq_sum (a b c d e : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) :
  (a / b) ^ 4 + (b / c) ^ 4 + (c / d) ^ 4 + (d / e) ^ 4 + (e / a) ^ 4 ≥ 
  (a / b) + (b / c) + (c / d) + (d / e) + (e / a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_pow4_geq_sum_l2215_221504


namespace NUMINAMATH_GPT_hockey_games_per_month_calculation_l2215_221558

-- Define the given conditions
def months_in_season : Nat := 14
def total_hockey_games : Nat := 182

-- Prove the number of hockey games played each month
theorem hockey_games_per_month_calculation :
  total_hockey_games / months_in_season = 13 := by
  sorry

end NUMINAMATH_GPT_hockey_games_per_month_calculation_l2215_221558


namespace NUMINAMATH_GPT_parabola_latus_rectum_l2215_221532

theorem parabola_latus_rectum (x p y : ℝ) (hp : p > 0) (h_eq : x^2 = 2 * p * y) (hl : y = -3) :
  p = 6 :=
by
  sorry

end NUMINAMATH_GPT_parabola_latus_rectum_l2215_221532


namespace NUMINAMATH_GPT_inequality_chain_l2215_221535

theorem inequality_chain (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b) (h₃ : b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end NUMINAMATH_GPT_inequality_chain_l2215_221535


namespace NUMINAMATH_GPT_smallest_possible_c_minus_a_l2215_221578

theorem smallest_possible_c_minus_a :
  ∃ (a b c : ℕ), 
    a < b ∧ b < c ∧ a * b * c = Nat.factorial 9 ∧ c - a = 216 := 
by
  sorry

end NUMINAMATH_GPT_smallest_possible_c_minus_a_l2215_221578


namespace NUMINAMATH_GPT_sum_pairwise_relatively_prime_integers_eq_160_l2215_221574

theorem sum_pairwise_relatively_prime_integers_eq_160
  (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_prod : a * b * c = 27000)
  (h_coprime_ab : Nat.gcd a b = 1)
  (h_coprime_bc : Nat.gcd b c = 1)
  (h_coprime_ac : Nat.gcd a c = 1) :
  a + b + c = 160 :=
by
  sorry

end NUMINAMATH_GPT_sum_pairwise_relatively_prime_integers_eq_160_l2215_221574


namespace NUMINAMATH_GPT_max_dot_product_of_points_on_ellipses_l2215_221519

theorem max_dot_product_of_points_on_ellipses :
  let C1 (M : ℝ × ℝ) := M.1^2 / 25 + M.2^2 / 9 = 1
  let C2 (N : ℝ × ℝ) := N.1^2 / 9 + N.2^2 / 25 = 1
  ∃ M N : ℝ × ℝ,
    C1 M ∧ C2 N ∧
    (∀ M N, C1 M ∧ C2 N → M.1 * N.1 + M.2 * N.2 ≤ 15 ∧ 
      (∃ θ φ, M = (5 * Real.cos θ, 3 * Real.sin θ) ∧ N = (3 * Real.cos φ, 5 * Real.sin φ) ∧ (M.1 * N.1 + M.2 * N.2 = 15))) :=
by
  sorry

end NUMINAMATH_GPT_max_dot_product_of_points_on_ellipses_l2215_221519


namespace NUMINAMATH_GPT_find_age_of_b_l2215_221503

variable (a b : ℤ)

-- Conditions
axiom cond1 : a + 10 = 2 * (b - 10)
axiom cond2 : a = b + 9

-- Goal
theorem find_age_of_b : b = 39 :=
sorry

end NUMINAMATH_GPT_find_age_of_b_l2215_221503


namespace NUMINAMATH_GPT_juniper_remaining_bones_l2215_221569

-- Conditions
def initial_bones : ℕ := 4
def doubled_bones (b : ℕ) : ℕ := 2 * b
def stolen_bones (b : ℕ) : ℕ := b - 2

-- Theorem Statement
theorem juniper_remaining_bones : stolen_bones (doubled_bones initial_bones) = 6 := by
  -- Proof is omitted, only the statement is required as per instructions
  sorry

end NUMINAMATH_GPT_juniper_remaining_bones_l2215_221569


namespace NUMINAMATH_GPT_find_m_l2215_221575

theorem find_m (m : ℤ) : m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_l2215_221575


namespace NUMINAMATH_GPT_water_park_admission_l2215_221514

def adult_admission_charge : ℝ := 1
def child_admission_charge : ℝ := 0.75
def children_accompanied : ℕ := 3
def total_admission_charge (adults : ℝ) (children : ℝ) : ℝ := adults + children

theorem water_park_admission :
  let adult_charge := adult_admission_charge
  let children_charge := children_accompanied * child_admission_charge
  total_admission_charge adult_charge children_charge = 3.25 :=
by sorry

end NUMINAMATH_GPT_water_park_admission_l2215_221514


namespace NUMINAMATH_GPT_product_of_differing_inputs_equal_l2215_221536

theorem product_of_differing_inputs_equal (a b : ℝ) (h₁ : a ≠ b)
(h₂ : |Real.log a - (1 / 2)| = |Real.log b - (1 / 2)|) : a * b = Real.exp 1 :=
sorry

end NUMINAMATH_GPT_product_of_differing_inputs_equal_l2215_221536


namespace NUMINAMATH_GPT_min_value_expression_l2215_221516

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  8 * x^3 + 27 * y^3 + 64 * z^3 + (1 / (8 * x * y * z)) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l2215_221516


namespace NUMINAMATH_GPT_valid_param_a_valid_param_c_l2215_221538

/-
The task is to prove that the goals provided are valid parameterizations of the given line.
-/

def line_eqn (x y : ℝ) : Prop := y = -7/4 * x + 21/4

def is_valid_param (p₀ : ℝ × ℝ) (d : ℝ × ℝ) : Prop :=
  ∀ t : ℝ, line_eqn ((p₀.1 + t * d.1) : ℝ) ((p₀.2 + t * d.2) : ℝ)

theorem valid_param_a : is_valid_param (7, 0) (4, -7) :=
by
  sorry

theorem valid_param_c : is_valid_param (0, 21/4) (-4, 7) :=
by
  sorry


end NUMINAMATH_GPT_valid_param_a_valid_param_c_l2215_221538


namespace NUMINAMATH_GPT_arithmetic_seq_a9_l2215_221580

theorem arithmetic_seq_a9 (a : ℕ → ℤ) (h1 : a 3 - a 2 = -2) (h2 : a 7 = -2) : a 9 = -6 := 
by sorry

end NUMINAMATH_GPT_arithmetic_seq_a9_l2215_221580


namespace NUMINAMATH_GPT_weight_of_each_pack_l2215_221572

-- Definitions based on conditions
def total_sugar : ℕ := 3020
def leftover_sugar : ℕ := 20
def number_of_packs : ℕ := 12

-- Definition of sugar used for packs
def sugar_used_for_packs : ℕ := total_sugar - leftover_sugar

-- Proof statement to be verified
theorem weight_of_each_pack : sugar_used_for_packs / number_of_packs = 250 := by
  sorry

end NUMINAMATH_GPT_weight_of_each_pack_l2215_221572


namespace NUMINAMATH_GPT_train_length_l2215_221527

theorem train_length (speed_kmh : ℕ) (cross_time : ℕ) (h_speed : speed_kmh = 54) (h_time : cross_time = 9) :
  let speed_ms := speed_kmh * (1000 / 3600)
  let length_m := speed_ms * cross_time
  length_m = 135 := by
  sorry

end NUMINAMATH_GPT_train_length_l2215_221527


namespace NUMINAMATH_GPT_distinct_points_count_l2215_221557

-- Definitions based on conditions
def eq1 (x y : ℝ) : Prop := (x + y = 7) ∨ (2 * x - 3 * y = -7)
def eq2 (x y : ℝ) : Prop := (x - y = 3) ∨ (3 * x + 2 * y = 18)

-- The statement combining conditions and requiring the proof of 3 distinct solutions
theorem distinct_points_count : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (eq1 p1.1 p1.2 ∧ eq2 p1.1 p1.2) ∧ 
    (eq1 p2.1 p2.2 ∧ eq2 p2.1 p2.2) ∧ 
    (eq1 p3.1 p3.2 ∧ eq2 p3.1 p3.2) ∧ 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 :=
sorry

end NUMINAMATH_GPT_distinct_points_count_l2215_221557


namespace NUMINAMATH_GPT_chemistry_class_students_l2215_221585

theorem chemistry_class_students (total_students both_classes biology_class only_chemistry_class : ℕ)
    (h1: total_students = 100)
    (h2 : both_classes = 10)
    (h3 : total_students = biology_class + only_chemistry_class + both_classes)
    (h4 : only_chemistry_class = 4 * (biology_class + both_classes)) : 
    only_chemistry_class = 80 :=
by
  sorry

end NUMINAMATH_GPT_chemistry_class_students_l2215_221585


namespace NUMINAMATH_GPT_canada_population_l2215_221544

theorem canada_population 
    (M : ℕ) (B : ℕ) (H : ℕ)
    (hM : M = 1000000)
    (hB : B = 2 * M)
    (hH : H = 19 * B) : 
    H = 38000000 := by
  sorry

end NUMINAMATH_GPT_canada_population_l2215_221544


namespace NUMINAMATH_GPT_clive_money_l2215_221593

noncomputable def clive_initial_money : ℝ  :=
  let total_olives := 80
  let olives_per_jar := 20
  let cost_per_jar := 1.5
  let change := 4
  let jars_needed := total_olives / olives_per_jar
  let total_cost := jars_needed * cost_per_jar
  total_cost + change

theorem clive_money (h1 : clive_initial_money = 10) : clive_initial_money = 10 :=
by sorry

end NUMINAMATH_GPT_clive_money_l2215_221593


namespace NUMINAMATH_GPT_bhanu_income_problem_l2215_221541

-- Define the total income
def total_income (I : ℝ) : Prop :=
  let petrol_spent := 300
  let house_rent := 70
  (0.10 * (I - petrol_spent) = house_rent)

-- Define the percentage of income spent on petrol
def petrol_percentage (P : ℝ) (I : ℝ) : Prop :=
  0.01 * P * I = 300

-- The theorem we aim to prove
theorem bhanu_income_problem : 
  ∃ I P, total_income I ∧ petrol_percentage P I ∧ P = 30 :=
by
  sorry

end NUMINAMATH_GPT_bhanu_income_problem_l2215_221541


namespace NUMINAMATH_GPT_find_total_grade10_students_l2215_221526

/-
Conditions:
1. The school has a total of 1800 students in grades 10 and 11.
2. 90 students are selected as a sample for a survey.
3. The sample contains 42 grade 10 students.
-/

variables (total_students sample_size sample_grade10 total_grade10 : ℕ)

axiom total_students_def : total_students = 1800
axiom sample_size_def : sample_size = 90
axiom sample_grade10_def : sample_grade10 = 42

theorem find_total_grade10_students : total_grade10 = 840 :=
by
  have h : (sample_size : ℚ) / (total_students : ℚ) = (sample_grade10 : ℚ) / (total_grade10 : ℚ) :=
    sorry
  sorry

end NUMINAMATH_GPT_find_total_grade10_students_l2215_221526


namespace NUMINAMATH_GPT_circle_radius_l2215_221551

theorem circle_radius
  (area_sector : ℝ)
  (arc_length : ℝ)
  (h_area : area_sector = 8.75)
  (h_arc : arc_length = 3.5) :
  ∃ r : ℝ, r = 5 :=
by
  let r := 5
  use r
  sorry

end NUMINAMATH_GPT_circle_radius_l2215_221551


namespace NUMINAMATH_GPT_symmetry_proof_l2215_221543

-- Define the coordinates of point A
def A : ℝ × ℝ := (-1, 8)

-- Define the reflection property across the y-axis
def is_reflection_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

-- Define the point B which we need to prove
def B : ℝ × ℝ := (1, 8)

-- The proof statement
theorem symmetry_proof :
  is_reflection_y_axis A B :=
by
  sorry

end NUMINAMATH_GPT_symmetry_proof_l2215_221543


namespace NUMINAMATH_GPT_percentage_increase_is_200_l2215_221508

noncomputable def total_cost : ℝ := 300
noncomputable def rate_per_sq_m : ℝ := 5
noncomputable def length : ℝ := 13.416407864998739
noncomputable def area : ℝ := total_cost / rate_per_sq_m
noncomputable def breadth : ℝ := area / length
noncomputable def percentage_increase : ℝ := (length - breadth) / breadth * 100

theorem percentage_increase_is_200 :
  percentage_increase = 200 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_is_200_l2215_221508


namespace NUMINAMATH_GPT_book_pages_total_l2215_221589

-- Define the conditions
def pagesPerNight : ℝ := 120.0
def nights : ℝ := 10.0

-- State the theorem to prove
theorem book_pages_total : pagesPerNight * nights = 1200.0 := by
  sorry

end NUMINAMATH_GPT_book_pages_total_l2215_221589


namespace NUMINAMATH_GPT_remainder_div_8_l2215_221523

theorem remainder_div_8 (x : ℤ) (h : ∃ k : ℤ, x = 63 * k + 27) : x % 8 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_div_8_l2215_221523


namespace NUMINAMATH_GPT_number_of_single_rooms_l2215_221563

theorem number_of_single_rooms (S : ℕ) : 
  (S + 13 * 2 = 40) ∧ (S * 10 + 13 * 2 * 10 = 400) → S = 14 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_single_rooms_l2215_221563


namespace NUMINAMATH_GPT_ab_fraction_l2215_221533

theorem ab_fraction (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 9) (h2 : a * b = 20) : 
  (1 / a + 1 / b) = 9 / 20 := 
by 
  sorry

end NUMINAMATH_GPT_ab_fraction_l2215_221533


namespace NUMINAMATH_GPT_find_cuboid_length_l2215_221591

theorem find_cuboid_length
  (b : ℝ) (h : ℝ) (S : ℝ)
  (hb : b = 10) (hh : h = 12) (hS : S = 960) :
  ∃ l : ℝ, 2 * (l * b + b * h + h * l) = S ∧ l = 16.36 :=
by
  sorry

end NUMINAMATH_GPT_find_cuboid_length_l2215_221591


namespace NUMINAMATH_GPT_calculate_new_measure_l2215_221550

noncomputable def equilateral_triangle_side_length : ℝ := 7.5

theorem calculate_new_measure :
  3 * (equilateral_triangle_side_length ^ 2) = 168.75 :=
by
  sorry

end NUMINAMATH_GPT_calculate_new_measure_l2215_221550
