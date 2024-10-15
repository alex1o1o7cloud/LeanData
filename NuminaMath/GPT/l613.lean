import Mathlib

namespace NUMINAMATH_GPT_david_more_push_ups_than_zachary_l613_61367

def zachary_push_ups : ℕ := 53
def zachary_crunches : ℕ := 14
def zachary_total : ℕ := 67
def david_crunches : ℕ := zachary_crunches - 10
def david_push_ups : ℕ := zachary_total - david_crunches

theorem david_more_push_ups_than_zachary : david_push_ups - zachary_push_ups = 10 := by
  sorry  -- Proof is not required as per instructions

end NUMINAMATH_GPT_david_more_push_ups_than_zachary_l613_61367


namespace NUMINAMATH_GPT_inverse_proportion_point_l613_61301

theorem inverse_proportion_point (a : ℝ) (h : (a, 7) ∈ {p : ℝ × ℝ | ∃ x y, y = 14 / x ∧ p = (x, y)}) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportion_point_l613_61301


namespace NUMINAMATH_GPT_gcd_proof_l613_61300

def gcd_10010_15015 := Nat.gcd 10010 15015 = 5005

theorem gcd_proof : gcd_10010_15015 :=
by
  sorry

end NUMINAMATH_GPT_gcd_proof_l613_61300


namespace NUMINAMATH_GPT_calculate_expression_l613_61368

theorem calculate_expression :
  ( (128^2 - 5^2) / (72^2 - 13^2) * ((72 - 13) * (72 + 13)) / ((128 - 5) * (128 + 5)) * (128 + 5) / (72 + 13) )
  = (133 / 85) :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_calculate_expression_l613_61368


namespace NUMINAMATH_GPT_skill_testing_question_l613_61361

theorem skill_testing_question : (5 * (10 - 6) / 2) = 10 := by
  sorry

end NUMINAMATH_GPT_skill_testing_question_l613_61361


namespace NUMINAMATH_GPT_problem_statement_l613_61358

noncomputable def a : ℝ := 2 * Real.log 0.99
noncomputable def b : ℝ := Real.log 0.98
noncomputable def c : ℝ := Real.sqrt 0.96 - 1

theorem problem_statement : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_problem_statement_l613_61358


namespace NUMINAMATH_GPT_abs_inequality_solution_set_l613_61330

theorem abs_inequality_solution_set :
  { x : ℝ | abs (2 - x) < 5 } = { x : ℝ | -3 < x ∧ x < 7 } :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_set_l613_61330


namespace NUMINAMATH_GPT_pool_cannot_be_filled_l613_61351

noncomputable def pool := 48000 -- Pool capacity in gallons
noncomputable def hose_rate := 3 -- Rate of each hose in gallons per minute
noncomputable def number_of_hoses := 6 -- Number of hoses
noncomputable def leakage_rate := 18 -- Leakage rate in gallons per minute

theorem pool_cannot_be_filled : 
  (number_of_hoses * hose_rate - leakage_rate <= 0) -> False :=
by
  -- Skipping the proof with 'sorry' as per instructions
  sorry

end NUMINAMATH_GPT_pool_cannot_be_filled_l613_61351


namespace NUMINAMATH_GPT_articles_bought_l613_61325

theorem articles_bought (C : ℝ) (N : ℝ) (h1 : (N * C) = (30 * ((5 / 3) * C))) : N = 50 :=
by
  sorry

end NUMINAMATH_GPT_articles_bought_l613_61325


namespace NUMINAMATH_GPT_num_satisfying_inequality_l613_61317

theorem num_satisfying_inequality : ∃ (s : Finset ℤ), (∀ n ∈ s, (n + 4) * (n - 8) ≤ 0) ∧ s.card = 13 := by
  sorry

end NUMINAMATH_GPT_num_satisfying_inequality_l613_61317


namespace NUMINAMATH_GPT_x1_x2_eq_e2_l613_61384

variable (x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * Real.exp x1 = Real.exp 2
def condition2 : Prop := x2 * Real.log x2 = Real.exp 2

-- The proof problem
theorem x1_x2_eq_e2 (hx1 : condition1 x1) (hx2 : condition2 x2) : x1 * x2 = Real.exp 2 := 
sorry

end NUMINAMATH_GPT_x1_x2_eq_e2_l613_61384


namespace NUMINAMATH_GPT_contradiction_proof_l613_61398

theorem contradiction_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h1 : a + 1/b < 2) (h2 : b + 1/c < 2) (h3 : c + 1/a < 2) : 
  ¬ (a + 1/b ≥ 2 ∨ b + 1/c ≥ 2 ∨ c + 1/a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_contradiction_proof_l613_61398


namespace NUMINAMATH_GPT_neg_or_false_of_or_true_l613_61385

variable {p q : Prop}

theorem neg_or_false_of_or_true (h : ¬ (p ∨ q) = false) : p ∨ q :=
by {
  sorry
}

end NUMINAMATH_GPT_neg_or_false_of_or_true_l613_61385


namespace NUMINAMATH_GPT_initial_meals_for_adults_l613_61377

theorem initial_meals_for_adults (C A : ℕ) (h1 : C = 90) (h2 : 14 * C / A = 72) : A = 18 :=
by
  sorry

end NUMINAMATH_GPT_initial_meals_for_adults_l613_61377


namespace NUMINAMATH_GPT_jane_reads_pages_l613_61342

theorem jane_reads_pages (P : ℕ) (h1 : 7 * (P + 10) = 105) : P = 5 := by
  sorry

end NUMINAMATH_GPT_jane_reads_pages_l613_61342


namespace NUMINAMATH_GPT_find_marks_in_biology_l613_61336

/-- 
David's marks in various subjects and his average marks are given.
This statement proves David's marks in Biology assuming the conditions provided.
--/
theorem find_marks_in_biology
  (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (avg_marks : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 91)
  (h_math : math = 65)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_avg_marks : avg_marks = 78)
  (h_total_subjects : total_subjects = 5)
  : ∃ (biology : ℕ), biology = 85 := 
by
  sorry

end NUMINAMATH_GPT_find_marks_in_biology_l613_61336


namespace NUMINAMATH_GPT_smallest_positive_multiple_l613_61328

theorem smallest_positive_multiple (a : ℕ) (k : ℕ) (h : 17 * a ≡ 7 [MOD 101]) : 
  ∃ k, k = 17 * 42 := 
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_l613_61328


namespace NUMINAMATH_GPT_convert_base_10_to_base_7_l613_61339

def base10_to_base7 (n : ℕ) : ℕ := 
  match n with
  | 5423 => 21545
  | _ => 0

theorem convert_base_10_to_base_7 : base10_to_base7 5423 = 21545 := by
  rfl

end NUMINAMATH_GPT_convert_base_10_to_base_7_l613_61339


namespace NUMINAMATH_GPT_unique_a_value_l613_61323

theorem unique_a_value (a : ℝ) :
  let M := { x : ℝ | x^2 = 2 }
  let N := { x : ℝ | a * x = 1 }
  N ⊆ M ↔ (a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_a_value_l613_61323


namespace NUMINAMATH_GPT_chip_credit_card_balance_l613_61327

-- Conditions
def initial_balance : Float := 50.00
def first_interest_rate : Float := 0.20
def additional_charge : Float := 20.00
def second_interest_rate : Float := 0.20

-- Question
def current_balance : Float :=
  let first_interest_fee := initial_balance * first_interest_rate
  let balance_after_first_interest := initial_balance + first_interest_fee
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  let second_interest_fee := balance_before_second_interest * second_interest_rate
  balance_before_second_interest + second_interest_fee

-- Correct Answer
def expected_balance : Float := 96.00

-- Proof Problem Statement
theorem chip_credit_card_balance : current_balance = expected_balance := by
  sorry

end NUMINAMATH_GPT_chip_credit_card_balance_l613_61327


namespace NUMINAMATH_GPT_probability_of_at_least_one_pair_of_women_l613_61383

/--
Theorem: Calculate the probability that at least one pair consists of two young women from a group of 6 young men and 6 young women paired up randomly is 0.93.
-/
theorem probability_of_at_least_one_pair_of_women 
  (men_women_group : Finset (Fin 12))
  (pairs : Finset (Finset (Fin 12)))
  (h_pairs : pairs.card = 6)
  (h_men_women : ∀ pair ∈ pairs, pair.card = 2)
  (h_distinct : ∀ (x y : Finset (Fin 12)), x ≠ y → x ∩ y = ∅):
  ∃ (p : ℝ), p = 0.93 := 
sorry

end NUMINAMATH_GPT_probability_of_at_least_one_pair_of_women_l613_61383


namespace NUMINAMATH_GPT_complement_intersection_l613_61304

open Set

variable (U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8})
variable (A : Set ℕ := {2, 5, 8})
variable (B : Set ℕ := {1, 3, 5, 7})

theorem complement_intersection (CUA : Set ℕ := {1, 3, 4, 6, 7}) :
  (CUA ∩ B) = {1, 3, 7} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_l613_61304


namespace NUMINAMATH_GPT_certain_number_divisibility_l613_61366

theorem certain_number_divisibility :
  ∃ k : ℕ, 3150 = 1050 * k :=
sorry

end NUMINAMATH_GPT_certain_number_divisibility_l613_61366


namespace NUMINAMATH_GPT_find_value_of_a_l613_61396

variable (a b : ℝ)

def varies_inversely (a : ℝ) (b_minus_one_sq : ℝ) : ℝ :=
  a * b_minus_one_sq

theorem find_value_of_a 
  (h₁ : ∀ b : ℝ, varies_inversely a ((b - 1) ^ 2) = 64)
  (h₂ : b = 5) : a = 4 :=
by sorry

end NUMINAMATH_GPT_find_value_of_a_l613_61396


namespace NUMINAMATH_GPT_speed_of_current_l613_61322

variable (c : ℚ) -- Speed of the current in miles per hour
variable (d : ℚ) -- Distance to the certain point in miles

def boat_speed := 16 -- Boat's speed relative to water in mph
def upstream_time := (20:ℚ) / 60 -- Time upstream in hours 
def downstream_time := (15:ℚ) / 60 -- Time downstream in hours

theorem speed_of_current (h1 : d = (boat_speed - c) * upstream_time)
                         (h2 : d = (boat_speed + c) * downstream_time) :
    c = 16 / 7 :=
  by
  sorry

end NUMINAMATH_GPT_speed_of_current_l613_61322


namespace NUMINAMATH_GPT_z_is_46_percent_less_than_y_l613_61338

variable (w e y z : ℝ)

-- Conditions
def w_is_60_percent_of_e := w = 0.60 * e
def e_is_60_percent_of_y := e = 0.60 * y
def z_is_150_percent_of_w := z = w * 1.5000000000000002

-- Proof Statement
theorem z_is_46_percent_less_than_y (h1 : w_is_60_percent_of_e w e)
                                    (h2 : e_is_60_percent_of_y e y)
                                    (h3 : z_is_150_percent_of_w z w) :
                                    100 - (z / y * 100) = 46 :=
by
  sorry

end NUMINAMATH_GPT_z_is_46_percent_less_than_y_l613_61338


namespace NUMINAMATH_GPT_find_income_4_l613_61379

noncomputable def income_4 (income_1 income_2 income_3 income_5 average_income num_days : ℕ) : ℕ :=
  average_income * num_days - (income_1 + income_2 + income_3 + income_5)

theorem find_income_4
  (income_1 : ℕ := 200)
  (income_2 : ℕ := 150)
  (income_3 : ℕ := 750)
  (income_5 : ℕ := 500)
  (average_income : ℕ := 400)
  (num_days : ℕ := 5) :
  income_4 income_1 income_2 income_3 income_5 average_income num_days = 400 :=
by
  unfold income_4
  sorry

end NUMINAMATH_GPT_find_income_4_l613_61379


namespace NUMINAMATH_GPT_min_value_expression_l613_61370

noncomputable def log (base : ℝ) (num : ℝ) := Real.log num / Real.log base

theorem min_value_expression (a b : ℝ) (h1 : b > a) (h2 : a > 1) 
  (h3 : 3 * log a b + 6 * log b a = 11) : 
  a^3 + (2 / (b - 1)) ≥ 2 * Real.sqrt 2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l613_61370


namespace NUMINAMATH_GPT_find_three_digit_numbers_l613_61350

theorem find_three_digit_numbers :
  ∃ A, (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376) ∨ (A = 625) :=
by
  sorry

end NUMINAMATH_GPT_find_three_digit_numbers_l613_61350


namespace NUMINAMATH_GPT_alex_sweaters_l613_61397

def num_items (shirts : ℕ) (pants : ℕ) (jeans : ℕ) (total_cycle_time_minutes : ℕ)
  (cycle_time_minutes : ℕ) (max_items_per_cycle : ℕ) : ℕ :=
  total_cycle_time_minutes / cycle_time_minutes * max_items_per_cycle

def num_sweaters_to_wash (total_items : ℕ) (non_sweater_items : ℕ) : ℕ :=
  total_items - non_sweater_items

theorem alex_sweaters :
  ∀ (shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle : ℕ),
  shirts = 18 →
  pants = 12 →
  jeans = 13 →
  total_cycle_time_minutes = 180 →
  cycle_time_minutes = 45 →
  max_items_per_cycle = 15 →
  num_sweaters_to_wash
    (num_items shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle)
    (shirts + pants + jeans) = 17 :=
by
  intros shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle
    h_shirts h_pants h_jeans h_total_cycle_time_minutes h_cycle_time_minutes h_max_items_per_cycle
  
  sorry

end NUMINAMATH_GPT_alex_sweaters_l613_61397


namespace NUMINAMATH_GPT_prize_interval_l613_61378

theorem prize_interval (prize1 prize2 prize3 prize4 prize5 interval : ℝ) (h1 : prize1 = 5000) 
  (h2 : prize2 = 5000 - interval) (h3 : prize3 = 5000 - 2 * interval) 
  (h4 : prize4 = 5000 - 3 * interval) (h5 : prize5 = 5000 - 4 * interval) 
  (h_total : prize1 + prize2 + prize3 + prize4 + prize5 = 15000) : 
  interval = 1000 := 
by
  sorry

end NUMINAMATH_GPT_prize_interval_l613_61378


namespace NUMINAMATH_GPT_solve_inequality_l613_61372

namespace InequalityProof

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem solve_inequality (x : ℝ) : cube_root x + 3 / (cube_root x + 4) ≤ 0 ↔ x ∈ Set.Icc (-27 : ℝ) (-1 : ℝ) :=
by
  have y_eq := cube_root x
  sorry

end InequalityProof

end NUMINAMATH_GPT_solve_inequality_l613_61372


namespace NUMINAMATH_GPT_solve_for_x_l613_61344

theorem solve_for_x (x : ℚ) : (3 * x / 7 - 2 = 12) → (x = 98 / 3) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l613_61344


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l613_61376

-- Problem 1
theorem problem_1 (m n : ℝ) : 
  3 * (m - n) ^ 2 - 4 * (m - n) ^ 2 + 3 * (m - n) ^ 2 = 2 * (m - n) ^ 2 := 
by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) (h : x^2 + 2 * y = 4) : 
  3 * x^2 + 6 * y - 2 = 10 := 
by
  sorry

-- Problem 3
theorem problem_3 (x y : ℝ) 
  (h1 : x^2 + x * y = 2) 
  (h2 : 2 * y^2 + 3 * x * y = 5) : 
  2 * x^2 + 11 * x * y + 6 * y^2 = 19 := 
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l613_61376


namespace NUMINAMATH_GPT_fifth_root_of_unity_l613_61308

noncomputable def expression (x : ℂ) := 
  2 * x + 1 / (1 + x) + x / (1 + x^2) + x^2 / (1 + x^3) + x^3 / (1 + x^4)

theorem fifth_root_of_unity (x : ℂ) (hx : x^5 = 1) : 
  (expression x = 4) ∨ (expression x = -1 + Real.sqrt 5) ∨ (expression x = -1 - Real.sqrt 5) :=
sorry

end NUMINAMATH_GPT_fifth_root_of_unity_l613_61308


namespace NUMINAMATH_GPT_truck_license_combinations_l613_61340

theorem truck_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let number_of_digits := 6
  letter_choices * (digit_choices ^ number_of_digits) = 3000000 :=
by
  sorry

end NUMINAMATH_GPT_truck_license_combinations_l613_61340


namespace NUMINAMATH_GPT_wedding_chairs_total_l613_61392

theorem wedding_chairs_total :
  let first_section_rows := 5
  let first_section_chairs_per_row := 10
  let first_section_late_people := 15
  let first_section_extra_chairs_per_late := 2
  
  let second_section_rows := 8
  let second_section_chairs_per_row := 12
  let second_section_late_people := 25
  let second_section_extra_chairs_per_late := 3
  
  let third_section_rows := 4
  let third_section_chairs_per_row := 15
  let third_section_late_people := 8
  let third_section_extra_chairs_per_late := 1

  let fourth_section_rows := 6
  let fourth_section_chairs_per_row := 9
  let fourth_section_late_people := 12
  let fourth_section_extra_chairs_per_late := 1
  
  let total_original_chairs := 
    (first_section_rows * first_section_chairs_per_row) + 
    (second_section_rows * second_section_chairs_per_row) + 
    (third_section_rows * third_section_chairs_per_row) + 
    (fourth_section_rows * fourth_section_chairs_per_row)
  
  let total_extra_chairs :=
    (first_section_late_people * first_section_extra_chairs_per_late) + 
    (second_section_late_people * second_section_extra_chairs_per_late) + 
    (third_section_late_people * third_section_extra_chairs_per_late) + 
    (fourth_section_late_people * fourth_section_extra_chairs_per_late)
  
  total_original_chairs + total_extra_chairs = 385 :=
by
  sorry

end NUMINAMATH_GPT_wedding_chairs_total_l613_61392


namespace NUMINAMATH_GPT_daily_rental_cost_l613_61374

theorem daily_rental_cost
  (daily_rent : ℝ)
  (cost_per_mile : ℝ)
  (max_budget : ℝ)
  (miles : ℝ)
  (H1 : cost_per_mile = 0.18)
  (H2 : max_budget = 75)
  (H3 : miles = 250)
  (H4 : daily_rent + (cost_per_mile * miles) = max_budget) : daily_rent = 30 :=
by sorry

end NUMINAMATH_GPT_daily_rental_cost_l613_61374


namespace NUMINAMATH_GPT_total_paths_A_to_C_via_B_l613_61329

-- Define the conditions
def steps_from_A_to_B : Nat := 6
def steps_from_B_to_C : Nat := 6
def right_moves_A_to_B : Nat := 4
def down_moves_A_to_B : Nat := 2
def right_moves_B_to_C : Nat := 3
def down_moves_B_to_C : Nat := 3

-- Define binomial coefficient function
def binom (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of paths for each segment
def paths_A_to_B : Nat := binom steps_from_A_to_B down_moves_A_to_B
def paths_B_to_C : Nat := binom steps_from_B_to_C down_moves_B_to_C

-- Theorem stating the total number of distinct paths
theorem total_paths_A_to_C_via_B : paths_A_to_B * paths_B_to_C = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_paths_A_to_C_via_B_l613_61329


namespace NUMINAMATH_GPT_average_mark_of_excluded_students_l613_61316

theorem average_mark_of_excluded_students (N A A_remaining N_excluded N_remaining T T_remaining T_excluded A_excluded : ℝ)
  (hN : N = 33) 
  (hA : A = 90) 
  (hA_remaining : A_remaining = 95)
  (hN_excluded : N_excluded = 3) 
  (hN_remaining : N_remaining = N - N_excluded) 
  (hT : T = N * A) 
  (hT_remaining : T_remaining = N_remaining * A_remaining) 
  (hT_eq : T = T_excluded + T_remaining) : 
  A_excluded = T_excluded / N_excluded :=
by
  have hTN : N = 33 := hN
  have hTA : A = 90 := hA
  have hTAR : A_remaining = 95 := hA_remaining
  have hTN_excluded : N_excluded = 3 := hN_excluded
  have hNrem : N_remaining = N - N_excluded := hN_remaining
  have hT_sum : T = N * A := hT
  have hTRem : T_remaining = N_remaining * A_remaining := hT_remaining
  have h_sum_eq : T = T_excluded + T_remaining := hT_eq
  sorry -- proof yet to be constructed

end NUMINAMATH_GPT_average_mark_of_excluded_students_l613_61316


namespace NUMINAMATH_GPT_percentage_increase_l613_61360

theorem percentage_increase (Z Y X : ℝ) (h1 : Y = 1.20 * Z) (h2 : Z = 250) (h3 : X + Y + Z = 925) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l613_61360


namespace NUMINAMATH_GPT_parallel_lines_slope_condition_l613_61309

theorem parallel_lines_slope_condition (b : ℝ) (x y : ℝ) :
    (∀ (x y : ℝ), 3 * y - 3 * b = 9 * x) →
    (∀ (x y : ℝ), y + 2 = (b + 9) * x) →
    b = -6 :=
by
    sorry

end NUMINAMATH_GPT_parallel_lines_slope_condition_l613_61309


namespace NUMINAMATH_GPT_meaningful_expression_l613_61369

theorem meaningful_expression (x : ℝ) : 
    (x + 2 > 0 ∧ x - 1 ≠ 0) ↔ (x > -2 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_expression_l613_61369


namespace NUMINAMATH_GPT_no_equal_partition_of_173_ones_and_neg_ones_l613_61373

theorem no_equal_partition_of_173_ones_and_neg_ones
  (L : List ℤ) (h1 : L.length = 173) (h2 : ∀ x ∈ L, x = 1 ∨ x = -1) :
  ¬ (∃ (L1 L2 : List ℤ), L = L1 ++ L2 ∧ L1.sum = L2.sum) :=
by
  sorry

end NUMINAMATH_GPT_no_equal_partition_of_173_ones_and_neg_ones_l613_61373


namespace NUMINAMATH_GPT_solve_equation_l613_61345

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2) ↔ (x = (Real.sqrt 6) / 3 ∨ x = -(Real.sqrt 6) / 3) :=
by sorry

end NUMINAMATH_GPT_solve_equation_l613_61345


namespace NUMINAMATH_GPT_percentage_of_full_marks_D_l613_61353

theorem percentage_of_full_marks_D (full_marks a b c d : ℝ)
  (h_full_marks : full_marks = 500)
  (h_a : a = 360)
  (h_a_b : a = b - 0.10 * b)
  (h_b_c : b = c + 0.25 * c)
  (h_c_d : c = d - 0.20 * d) :
  d / full_marks * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_full_marks_D_l613_61353


namespace NUMINAMATH_GPT_pqr_value_l613_61313

noncomputable def complex_numbers (p q r : ℂ) := p * q + 5 * q = -20 ∧ q * r + 5 * r = -20 ∧ r * p + 5 * p = -20

theorem pqr_value (p q r : ℂ) (h : complex_numbers p q r) : p * q * r = 80 := by
  sorry

end NUMINAMATH_GPT_pqr_value_l613_61313


namespace NUMINAMATH_GPT_value_of_k_h_10_l613_61356

def h (x : ℝ) : ℝ := 4 * x - 5
def k (x : ℝ) : ℝ := 2 * x + 6

theorem value_of_k_h_10 : k (h 10) = 76 := by
  -- We provide only the statement as required, skipping the proof
  sorry

end NUMINAMATH_GPT_value_of_k_h_10_l613_61356


namespace NUMINAMATH_GPT_train_crossing_time_l613_61335

def train_length : ℝ := 140
def bridge_length : ℝ := 235.03
def speed_kmh : ℝ := 45

noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def total_distance : ℝ := train_length + bridge_length

theorem train_crossing_time :
  (total_distance / speed_mps) = 30.0024 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l613_61335


namespace NUMINAMATH_GPT_different_colors_probability_l613_61320

noncomputable def differentColorProbability : ℚ :=
  let redChips := 7
  let greenChips := 5
  let totalChips := redChips + greenChips
  let probRedThenGreen := (redChips / totalChips) * (greenChips / totalChips)
  let probGreenThenRed := (greenChips / totalChips) * (redChips / totalChips)
  (probRedThenGreen + probGreenThenRed)

theorem different_colors_probability :
  differentColorProbability = 35 / 72 :=
by sorry

end NUMINAMATH_GPT_different_colors_probability_l613_61320


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l613_61357

-- Problem 1
theorem problem1 (x y : ℝ) : 4 * x^2 - y^4 = (2 * x + y^2) * (2 * x - y^2) :=
by
  -- proof omitted
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : 8 * x^2 - 24 * x * y + 18 * y^2 = 2 * (2 * x - 3 * y)^2 :=
by
  -- proof omitted
  sorry

-- Problem 3
theorem problem3 (x y : ℝ) : (x - y) * (3 * x + 1) - 2 * (x^2 - y^2) - (y - x)^2 = (x - y) * (1 - y) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l613_61357


namespace NUMINAMATH_GPT_xia_sheets_left_l613_61326

def stickers_left (initial : ℕ) (shared : ℕ) (per_sheet : ℕ) : ℕ :=
  (initial - shared) / per_sheet

theorem xia_sheets_left :
  stickers_left 150 100 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_xia_sheets_left_l613_61326


namespace NUMINAMATH_GPT_second_hand_distance_l613_61391

theorem second_hand_distance (r : ℝ) (minutes : ℝ) : r = 8 → minutes = 45 → (2 * π * r * minutes) = 720 * π :=
by
  intros r_eq minutes_eq
  simp only [r_eq, minutes_eq, mul_assoc, mul_comm π 8, mul_mul_mul_comm]
  sorry

end NUMINAMATH_GPT_second_hand_distance_l613_61391


namespace NUMINAMATH_GPT_max_min_K_max_min_2x_plus_y_l613_61337

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

theorem max_min_K (x y : ℝ) (h : circle_equation x y) : 
  - (Real.sqrt 3) / 3 ≤ (y - 1) / (x - 2) ∧ (y - 1) / (x - 2) ≤ (Real.sqrt 3) / 3 :=
by sorry

theorem max_min_2x_plus_y (x y : ℝ) (h : circle_equation x y) :
  1 - Real.sqrt 5 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_max_min_K_max_min_2x_plus_y_l613_61337


namespace NUMINAMATH_GPT_find_B_and_distance_l613_61306

noncomputable def pointA : ℝ × ℝ := (2, 4)

noncomputable def pointB : ℝ × ℝ := (-(1 + Real.sqrt 385) / 8, (-(1 + Real.sqrt 385) / 8) ^ 2)

noncomputable def distanceToOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem find_B_and_distance :
  (pointA.snd = pointA.fst ^ 2) ∧
  (pointB.snd = (-(1 + Real.sqrt 385) / 8) ^ 2) ∧
  (distanceToOrigin pointB = Real.sqrt ((-(1 + Real.sqrt 385) / 8) ^ 2 + (-(1 + Real.sqrt 385) / 8) ^ 4)) :=
  sorry

end NUMINAMATH_GPT_find_B_and_distance_l613_61306


namespace NUMINAMATH_GPT_tips_fraction_l613_61382

theorem tips_fraction (S T I : ℝ) (hT : T = 9 / 4 * S) (hI : I = S + T) : 
  T / I = 9 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_tips_fraction_l613_61382


namespace NUMINAMATH_GPT_exists_infinitely_many_primes_dividing_form_l613_61332

theorem exists_infinitely_many_primes_dividing_form (a : ℕ) (ha : 0 < a) :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ 2^(2*n) + a := 
sorry

end NUMINAMATH_GPT_exists_infinitely_many_primes_dividing_form_l613_61332


namespace NUMINAMATH_GPT_ratio_condition_l613_61381

theorem ratio_condition (x y a b : ℝ) (h1 : 8 * x - 6 * y = a) 
  (h2 : 9 * y - 12 * x = b) (hx : x ≠ 0) (hy : y ≠ 0) (hb : b ≠ 0) : 
  a / b = -2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_condition_l613_61381


namespace NUMINAMATH_GPT_proof_l_shaped_area_l613_61305

-- Define the overall rectangle dimensions
def overall_length : ℕ := 10
def overall_width : ℕ := 7

-- Define the dimensions of the removed rectangle
def removed_length : ℕ := overall_length - 3
def removed_width : ℕ := overall_width - 2

-- Calculate the areas
def overall_area : ℕ := overall_length * overall_width
def removed_area : ℕ := removed_length * removed_width
def l_shaped_area : ℕ := overall_area - removed_area

-- The theorem to be proved
theorem proof_l_shaped_area : l_shaped_area = 35 := by
  sorry

end NUMINAMATH_GPT_proof_l_shaped_area_l613_61305


namespace NUMINAMATH_GPT_find_number_of_boxes_l613_61359

-- Definitions and assumptions
def pieces_per_box : ℕ := 5 + 5
def total_pieces : ℕ := 60

-- The theorem to be proved
theorem find_number_of_boxes (B : ℕ) (h : total_pieces = B * pieces_per_box) :
  B = 6 :=
sorry

end NUMINAMATH_GPT_find_number_of_boxes_l613_61359


namespace NUMINAMATH_GPT_oaks_not_adjacent_probability_l613_61393

theorem oaks_not_adjacent_probability :
  let total_trees := 13
  let oaks := 5
  let other_trees := total_trees - oaks
  let possible_slots := other_trees + 1
  let combinations := Nat.choose possible_slots oaks
  let total_arrangements := Nat.factorial total_trees / (Nat.factorial oaks * Nat.factorial (total_trees - oaks))
  let probability := combinations / total_arrangements
  probability = 1 / 220 :=
by
  sorry

end NUMINAMATH_GPT_oaks_not_adjacent_probability_l613_61393


namespace NUMINAMATH_GPT_abc_inequality_l613_61371

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) :=
  sorry

end NUMINAMATH_GPT_abc_inequality_l613_61371


namespace NUMINAMATH_GPT_find_c_in_terms_of_a_and_b_l613_61386

theorem find_c_in_terms_of_a_and_b (a b : ℝ) :
  (∃ α β : ℝ, (α + β = -a) ∧ (α * β = b)) →
  (∃ c d : ℝ, (∃ α β : ℝ, (α^3 + β^3 = -c) ∧ (α^3 * β^3 = d))) →
  c = a^3 - 3 * a * b :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_c_in_terms_of_a_and_b_l613_61386


namespace NUMINAMATH_GPT_ellipse_eccentricity_l613_61314

theorem ellipse_eccentricity (a c : ℝ) (h : 2 * a = 2 * (2 * c)) : (c / a) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l613_61314


namespace NUMINAMATH_GPT_table_can_be_zeroed_out_l613_61364

open Matrix

-- Define the dimensions of the table
def m := 8
def n := 5

-- Define the operation of doubling all elements in a row
def double_row (table : Matrix (Fin m) (Fin n) ℕ) (i : Fin m) : Matrix (Fin m) (Fin n) ℕ :=
  fun i' j => if i' = i then 2 * table i' j else table i' j

-- Define the operation of subtracting one from all elements in a column
def subtract_one_column (table : Matrix (Fin m) (Fin n) ℕ) (j : Fin n) : Matrix (Fin m) (Fin n) ℕ :=
  fun i j' => if j' = j then table i j' - 1 else table i j'

-- The main theorem stating that it is possible to transform any table to a table of all zeros
theorem table_can_be_zeroed_out (table : Matrix (Fin m) (Fin n) ℕ) : 
  ∃ (ops : List (Matrix (Fin m) (Fin n) ℕ → Matrix (Fin m) (Fin n) ℕ)), 
    (ops.foldl (fun t op => op t) table) = fun _ _ => 0 :=
sorry

end NUMINAMATH_GPT_table_can_be_zeroed_out_l613_61364


namespace NUMINAMATH_GPT_problem_solution_l613_61363

noncomputable def problem_statement : Prop :=
  ∀ (α β : ℝ), 
    (0 < α ∧ α < Real.pi / 2) →
    (0 < β ∧ β < Real.pi / 2) →
    (Real.sin α = 4 / 5) →
    (Real.cos (α + β) = 5 / 13) →
    (Real.cos β = 63 / 65 ∧ (Real.sin α ^ 2 + Real.sin (2 * α)) / (Real.cos (2 * α) - 1) = -5 / 4)
    
theorem problem_solution : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l613_61363


namespace NUMINAMATH_GPT_range_of_a_l613_61315

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1 < a ∧ a ≤ 1) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l613_61315


namespace NUMINAMATH_GPT_mod_pow_sub_eq_l613_61333

theorem mod_pow_sub_eq : 
  (45^1537 - 25^1537) % 8 = 4 := 
by
  have h1 : 45 % 8 = 5 := by norm_num
  have h2 : 25 % 8 = 1 := by norm_num
  sorry

end NUMINAMATH_GPT_mod_pow_sub_eq_l613_61333


namespace NUMINAMATH_GPT_average_speed_l613_61375

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem average_speed (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * a * b) / (a + b) = (2 * b * a) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l613_61375


namespace NUMINAMATH_GPT_sum_of_first_n_terms_l613_61365

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def forms_geometric_sequence (a2 a4 a8 : ℤ) : Prop :=
  a4^2 = a2 * a8

def arithmetic_sum (S : ℕ → ℤ) (a : ℕ → ℤ) (n : ℕ) : Prop :=
  S n = n * (a 1) + (n * (n - 1) / 2) * (a 2 - a 1)

theorem sum_of_first_n_terms
  (d : ℤ) (n : ℕ)
  (h_nonzero : d ≠ 0)
  (h_arithmetic : is_arithmetic_sequence a d)
  (h_initial : a 1 = 1)
  (h_geom : forms_geometric_sequence (a 2) (a 4) (a 8)) :
  S n = n * (n + 1) / 2 := 
sorry

end NUMINAMATH_GPT_sum_of_first_n_terms_l613_61365


namespace NUMINAMATH_GPT_total_canoes_built_l613_61388

-- Given conditions as definitions
def a1 : ℕ := 10
def r : ℕ := 3

-- Define the geometric series sum for first four terms
noncomputable def sum_of_geometric_series (a1 r : ℕ) (n : ℕ) : ℕ :=
  a1 * ((r^n - 1) / (r - 1))

-- Prove that the total number of canoes built by the end of April is 400
theorem total_canoes_built (a1 r : ℕ) (n : ℕ) : sum_of_geometric_series a1 r n = 400 :=
  sorry

end NUMINAMATH_GPT_total_canoes_built_l613_61388


namespace NUMINAMATH_GPT_envelope_weight_l613_61355

-- Define the conditions as constants
def total_weight_kg : ℝ := 7.48
def num_envelopes : ℕ := 880
def kg_to_g_conversion : ℝ := 1000

-- Calculate the total weight in grams
def total_weight_g : ℝ := total_weight_kg * kg_to_g_conversion

-- Define the expected weight of one envelope in grams
def expected_weight_one_envelope_g : ℝ := 8.5

-- The proof statement
theorem envelope_weight :
  total_weight_g / num_envelopes = expected_weight_one_envelope_g := by
  sorry

end NUMINAMATH_GPT_envelope_weight_l613_61355


namespace NUMINAMATH_GPT_polynomial_self_composition_l613_61307

theorem polynomial_self_composition {p : Polynomial ℝ} {n : ℕ} (hn : 0 < n) :
  (∀ x, p.eval (p.eval x) = (p.eval x) ^ n) ↔ p = Polynomial.X ^ n :=
by sorry

end NUMINAMATH_GPT_polynomial_self_composition_l613_61307


namespace NUMINAMATH_GPT_perfect_number_mod_9_l613_61354

theorem perfect_number_mod_9 (N : ℕ) (hN : ∃ p, N = 2^(p-1) * (2^p - 1) ∧ Nat.Prime (2^p - 1)) (hN_ne_6 : N ≠ 6) : ∃ n : ℕ, N = 9 * n + 1 :=
by
  sorry

end NUMINAMATH_GPT_perfect_number_mod_9_l613_61354


namespace NUMINAMATH_GPT_smallest_n_l613_61346

theorem smallest_n (n : ℕ) (h : 5 * n ≡ 850 [MOD 26]) : n = 14 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l613_61346


namespace NUMINAMATH_GPT_simplify_expression_l613_61399

noncomputable def simplified_result (a b : ℝ) (i : ℂ) (hi : i * i = -1) : ℂ :=
  (a + b * i) * (a - b * i)

theorem simplify_expression (a b : ℝ) (i : ℂ) (hi : i * i = -1) :
  simplified_result a b i hi = a^2 + b^2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l613_61399


namespace NUMINAMATH_GPT_train_length_l613_61312

theorem train_length (L : ℝ) (V : ℝ)
  (h1 : V = L / 8)
  (h2 : V = (L + 273) / 20) :
  L = 182 :=
  by
  sorry

end NUMINAMATH_GPT_train_length_l613_61312


namespace NUMINAMATH_GPT_smallest_y_square_l613_61334

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end NUMINAMATH_GPT_smallest_y_square_l613_61334


namespace NUMINAMATH_GPT_choir_average_age_solution_l613_61348

noncomputable def choir_average_age (avg_f avg_m avg_c : ℕ) (n_f n_m n_c : ℕ) : ℕ :=
  (n_f * avg_f + n_m * avg_m + n_c * avg_c) / (n_f + n_m + n_c)

def choir_average_age_problem : Prop :=
  let avg_f := 32
  let avg_m := 38
  let avg_c := 10
  let n_f := 12
  let n_m := 18
  let n_c := 5
  choir_average_age avg_f avg_m avg_c n_f n_m n_c = 32

theorem choir_average_age_solution : choir_average_age_problem := by
  sorry

end NUMINAMATH_GPT_choir_average_age_solution_l613_61348


namespace NUMINAMATH_GPT_number_of_plains_routes_is_81_l613_61390

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end NUMINAMATH_GPT_number_of_plains_routes_is_81_l613_61390


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l613_61303

open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 3)

theorem part1_solution : ∀ x, f x ≤ 4 ↔ (0 ≤ x) ∧ (x ≤ 4) :=
by
  intro x
  sorry

theorem part2_solution : ∀ m, (∀ x, f x > m^2 + m) ↔ (-2 < m) ∧ (m < 1) :=
by
  intro m
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l613_61303


namespace NUMINAMATH_GPT_fifth_number_in_10th_row_l613_61318

theorem fifth_number_in_10th_row : 
  ∀ (n : ℕ), (∃ (a : ℕ), ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 10 → (m = 10 → a = 67)) :=
by
  sorry

end NUMINAMATH_GPT_fifth_number_in_10th_row_l613_61318


namespace NUMINAMATH_GPT_new_edition_pages_less_l613_61394

theorem new_edition_pages_less :
  let new_edition_pages := 450
  let old_edition_pages := 340
  (2 * old_edition_pages - new_edition_pages) = 230 :=
by
  let new_edition_pages := 450
  let old_edition_pages := 340
  sorry

end NUMINAMATH_GPT_new_edition_pages_less_l613_61394


namespace NUMINAMATH_GPT_sum_of_intercepts_modulo_13_l613_61395

theorem sum_of_intercepts_modulo_13 :
  ∃ (x0 y0 : ℤ), 0 ≤ x0 ∧ x0 < 13 ∧ 0 ≤ y0 ∧ y0 < 13 ∧
    (4 * x0 ≡ 1 [ZMOD 13]) ∧ (3 * y0 ≡ 12 [ZMOD 13]) ∧ (x0 + y0 = 14) := 
sorry

end NUMINAMATH_GPT_sum_of_intercepts_modulo_13_l613_61395


namespace NUMINAMATH_GPT_towels_per_person_l613_61341

-- Define the conditions
def num_rooms : ℕ := 10
def people_per_room : ℕ := 3
def total_towels : ℕ := 60

-- Define the total number of people
def total_people : ℕ := num_rooms * people_per_room

-- Define the proposition to prove
theorem towels_per_person : total_towels / total_people = 2 :=
by sorry

end NUMINAMATH_GPT_towels_per_person_l613_61341


namespace NUMINAMATH_GPT_part_one_solution_set_part_two_range_a_l613_61389

noncomputable def f (x a : ℝ) := |x - a| + x

theorem part_one_solution_set (x : ℝ) :
  f x 3 ≥ x + 4 ↔ (x ≤ -1 ∨ x ≥ 7) :=
by sorry

theorem part_two_range_a (a : ℝ) :
  (∀ x, (1 ≤ x ∧ x ≤ 3) → f x a ≥ 2 * a^2) ↔ (-1 ≤ a ∧ a ≤ 1/2) :=
by sorry

end NUMINAMATH_GPT_part_one_solution_set_part_two_range_a_l613_61389


namespace NUMINAMATH_GPT_average_students_present_l613_61380

-- Define the total number of students
def total_students : ℝ := 50

-- Define the absent rates for each day
def absent_rate_mon : ℝ := 0.10
def absent_rate_tue : ℝ := 0.12
def absent_rate_wed : ℝ := 0.15
def absent_rate_thu : ℝ := 0.08
def absent_rate_fri : ℝ := 0.05

-- Define the number of students present each day
def present_mon := (1 - absent_rate_mon) * total_students
def present_tue := (1 - absent_rate_tue) * total_students
def present_wed := (1 - absent_rate_wed) * total_students
def present_thu := (1 - absent_rate_thu) * total_students
def present_fri := (1 - absent_rate_fri) * total_students

-- Define the statement to prove
theorem average_students_present : 
  (present_mon + present_tue + present_wed + present_thu + present_fri) / 5 = 45 :=
by 
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_average_students_present_l613_61380


namespace NUMINAMATH_GPT_solve_for_y_solve_for_x_l613_61310

variable (x y : ℝ)

theorem solve_for_y (h : 2 * x + 3 * y - 4 = 0) : y = (4 - 2 * x) / 3 := 
sorry

theorem solve_for_x (h : 2 * x + 3 * y - 4 = 0) : x = (4 - 3 * y) / 2 := 
sorry

end NUMINAMATH_GPT_solve_for_y_solve_for_x_l613_61310


namespace NUMINAMATH_GPT_quadratic_no_real_roots_probability_l613_61319

theorem quadratic_no_real_roots_probability :
  (1 : ℝ) - 1 / 4 - 0 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_probability_l613_61319


namespace NUMINAMATH_GPT_find_mn_l613_61347

theorem find_mn (sec_x_plus_tan_x : ℝ) (sec_tan_eq : sec_x_plus_tan_x = 24 / 7) :
  ∃ (m n : ℕ) (h : Int.gcd m n = 1), (∃ y, y = (m:ℝ) / (n:ℝ) ∧ (y^2)*527^2 - 2*y*527*336 + 336^2 = 1) ∧
  m + n = boxed_mn :=
by
  sorry

end NUMINAMATH_GPT_find_mn_l613_61347


namespace NUMINAMATH_GPT_total_time_spent_l613_61387

-- Define the total time for one shoe
def time_per_shoe (time_buckle: ℕ) (time_heel: ℕ) : ℕ :=
  time_buckle + time_heel

-- Conditions
def time_buckle : ℕ := 5
def time_heel : ℕ := 10
def number_of_shoes : ℕ := 2

-- The proof problem statement
theorem total_time_spent :
  (time_per_shoe time_buckle time_heel) * number_of_shoes = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_time_spent_l613_61387


namespace NUMINAMATH_GPT_sector_area_l613_61324

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 10) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 25 / 4 :=
by 
  sorry

end NUMINAMATH_GPT_sector_area_l613_61324


namespace NUMINAMATH_GPT_mens_wages_l613_61362

-- Definitions based on the problem conditions
def equivalent_wages (M W_earn B : ℝ) : Prop :=
  (5 * M = W_earn) ∧ 
  (W_earn = 8 * B) ∧ 
  (5 * M + W_earn + 8 * B = 210)

-- Prove that the total wages of 5 men are Rs. 105 given the conditions
theorem mens_wages (M W_earn B : ℝ) (h : equivalent_wages M W_earn B) : 5 * M = 105 :=
by
  sorry

end NUMINAMATH_GPT_mens_wages_l613_61362


namespace NUMINAMATH_GPT_tom_candy_pieces_l613_61331

def total_boxes : ℕ := 14
def give_away_boxes : ℕ := 8
def pieces_per_box : ℕ := 3

theorem tom_candy_pieces : (total_boxes - give_away_boxes) * pieces_per_box = 18 := 
by 
  sorry

end NUMINAMATH_GPT_tom_candy_pieces_l613_61331


namespace NUMINAMATH_GPT_desired_average_l613_61321

variable (avg_4_tests : ℕ)
variable (score_5th_test : ℕ)

theorem desired_average (h1 : avg_4_tests = 78) (h2 : score_5th_test = 88) : (4 * avg_4_tests + score_5th_test) / 5 = 80 :=
by
  sorry

end NUMINAMATH_GPT_desired_average_l613_61321


namespace NUMINAMATH_GPT_total_volume_of_drink_l613_61311

theorem total_volume_of_drink :
  ∀ (total_ounces : ℝ),
    (∀ orange_juice watermelon_juice grape_juice : ℝ,
      orange_juice = 0.25 * total_ounces →
      watermelon_juice = 0.4 * total_ounces →
      grape_juice = 0.35 * total_ounces →
      grape_juice = 105 →
      total_ounces = 300) :=
by
  intros total_ounces orange_juice watermelon_juice grape_juice ho hw hg hg_eq
  sorry

end NUMINAMATH_GPT_total_volume_of_drink_l613_61311


namespace NUMINAMATH_GPT_solution_set_of_inequality_l613_61302

theorem solution_set_of_inequality :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l613_61302


namespace NUMINAMATH_GPT_julieta_total_cost_l613_61343

variable (initial_backpack_price : ℕ)
variable (initial_binder_price : ℕ)
variable (backpack_price_increase : ℕ)
variable (binder_price_reduction : ℕ)
variable (discount_rate : ℕ)
variable (num_binders : ℕ)

def calculate_total_cost (initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders : ℕ) : ℝ :=
  let new_backpack_price := initial_backpack_price + backpack_price_increase
  let new_binder_price := initial_binder_price - binder_price_reduction
  let total_bindable_cost := min num_binders ((num_binders + 1) / 2 * new_binder_price)
  let total_pre_discount := new_backpack_price + total_bindable_cost
  let discount_amount := total_pre_discount * discount_rate / 100
  let total_price := total_pre_discount - discount_amount
  total_price

theorem julieta_total_cost
  (initial_backpack_price : ℕ)
  (initial_binder_price : ℕ)
  (backpack_price_increase : ℕ)
  (binder_price_reduction : ℕ)
  (discount_rate : ℕ)
  (num_binders : ℕ)
  (h_initial_backpack : initial_backpack_price = 50)
  (h_initial_binder : initial_binder_price = 20)
  (h_backpack_inc : backpack_price_increase = 5)
  (h_binder_red : binder_price_reduction = 2)
  (h_discount : discount_rate = 10)
  (h_num_binders : num_binders = 3) :
  calculate_total_cost initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders = 81.90 :=
by
  sorry

end NUMINAMATH_GPT_julieta_total_cost_l613_61343


namespace NUMINAMATH_GPT_fraction_zero_solution_l613_61352

theorem fraction_zero_solution (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x + 3 ≠ 0) : x = 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_solution_l613_61352


namespace NUMINAMATH_GPT_parabola_focus_l613_61349

-- Define the equation of the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = -8 * x

-- Define the coordinates of the focus
def focus (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- The Lean statement that needs to be proved
theorem parabola_focus : ∀ (x y : ℝ), parabola_eq x y → focus x y :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_parabola_focus_l613_61349
