import Mathlib

namespace NUMINAMATH_GPT_quadratic_inequality_solution_l737_73742

theorem quadratic_inequality_solution 
  (a : ℝ) 
  (h : ∀ x : ℝ, -1 < x ∧ x < a → -x^2 + 2 * a * x + a + 1 > a + 1) : -1 < a ∧ a ≤ -1/2 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l737_73742


namespace NUMINAMATH_GPT_roots_sum_l737_73717

theorem roots_sum (a b : ℝ) 
  (h₁ : 3^(a-1) = 6 - a)
  (h₂ : 3^(6-b) = b - 1) : 
  a + b = 7 := 
by sorry

end NUMINAMATH_GPT_roots_sum_l737_73717


namespace NUMINAMATH_GPT_solve_for_x_l737_73700

theorem solve_for_x (x : ℝ) (d : ℝ) (h1 : x > 0) (h2 : x^2 = 4 + d) (h3 : 25 = x^2 + d) : x = Real.sqrt 14.5 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l737_73700


namespace NUMINAMATH_GPT_trigonometric_identity_l737_73712

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  (1 + Real.cos α ^ 2) / (Real.sin α * Real.cos α + Real.sin α ^ 2) = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l737_73712


namespace NUMINAMATH_GPT_equation_of_line_l_l737_73720

-- Define the conditions for the parabola and the line
def parabola_vertex : Prop := 
  ∃ C : ℝ × ℝ, C = (0, 0)

def parabola_symmetry_axis : Prop := 
  ∃ l : ℝ → ℝ, ∀ x, l x = -1

def midpoint_of_AB (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1

def parabola_equation (A B : ℝ × ℝ) : Prop :=
  A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1

-- State the theorem to be proven
theorem equation_of_line_l (A B : ℝ × ℝ) :
  parabola_vertex ∧ parabola_symmetry_axis ∧ midpoint_of_AB A B ∧ parabola_equation A B →
  ∃ l : ℝ → ℝ, ∀ x, l x = 2 * x - 3 :=
by sorry

end NUMINAMATH_GPT_equation_of_line_l_l737_73720


namespace NUMINAMATH_GPT_periodic_function_property_l737_73733

theorem periodic_function_property
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_period : ∀ x, f (x + 2) = f x)
  (h_def1 : ∀ x, -1 ≤ x ∧ x < 0 → f x = a * x + 1)
  (h_def2 : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = (b * x + 2) / (x + 1))
  (h_eq : f (1 / 2) = f (3 / 2)) :
  3 * a + 2 * b = -8 := by
  sorry

end NUMINAMATH_GPT_periodic_function_property_l737_73733


namespace NUMINAMATH_GPT_find_lesser_number_l737_73757

theorem find_lesser_number (x y : ℕ) (h₁ : x + y = 60) (h₂ : x - y = 10) : y = 25 := by
  sorry

end NUMINAMATH_GPT_find_lesser_number_l737_73757


namespace NUMINAMATH_GPT_clock1_runs_10_months_longer_l737_73793

noncomputable def battery_a_charge (C_B : ℝ) := 6 * C_B
noncomputable def clock1_total_charge (C_B : ℝ) := 2 * battery_a_charge C_B
noncomputable def clock2_total_charge (C_B : ℝ) := 2 * C_B
noncomputable def clock2_operating_time := 2
noncomputable def clock1_operating_time (C_B : ℝ) := clock1_total_charge C_B / C_B
noncomputable def operating_time_difference (C_B : ℝ) := clock1_operating_time C_B - clock2_operating_time

theorem clock1_runs_10_months_longer (C_B : ℝ) :
  operating_time_difference C_B = 10 :=
by
  unfold operating_time_difference clock1_operating_time clock2_operating_time clock1_total_charge battery_a_charge
  sorry

end NUMINAMATH_GPT_clock1_runs_10_months_longer_l737_73793


namespace NUMINAMATH_GPT_john_twice_sam_in_years_l737_73779

noncomputable def current_age_sam : ℕ := 9
noncomputable def current_age_john : ℕ := 27

theorem john_twice_sam_in_years (Y : ℕ) :
  (current_age_john + Y = 2 * (current_age_sam + Y)) → Y = 9 := 
by 
  sorry

end NUMINAMATH_GPT_john_twice_sam_in_years_l737_73779


namespace NUMINAMATH_GPT_base8_digits_sum_l737_73749

-- Define digits and their restrictions
variables {A B C : ℕ}

-- Main theorem
theorem base8_digits_sum (h1 : 0 < A ∧ A < 8)
                         (h2 : 0 < B ∧ B < 8)
                         (h3 : 0 < C ∧ C < 8)
                         (distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C)
                         (condition : (8^2 * A + 8 * B + C) + (8^2 * B + 8 * C + A) + (8^2 * C + 8 * A + B) = (8^2 + 8 + 1) * 8 * A) :
  A + B + C = 8 := 
sorry

end NUMINAMATH_GPT_base8_digits_sum_l737_73749


namespace NUMINAMATH_GPT_commodity_price_l737_73719

-- Define the variables for the prices of the commodities.
variable (x y : ℝ)

-- Conditions given in the problem.
def total_cost (x y : ℝ) : Prop := x + y = 827
def price_difference (x y : ℝ) : Prop := x = y + 127

-- The main statement to prove.
theorem commodity_price (x y : ℝ) (h1 : total_cost x y) (h2 : price_difference x y) : x = 477 :=
by
  sorry

end NUMINAMATH_GPT_commodity_price_l737_73719


namespace NUMINAMATH_GPT_percentage_of_valid_votes_l737_73722

theorem percentage_of_valid_votes 
  (total_votes : ℕ) 
  (invalid_percentage : ℕ) 
  (candidate_valid_votes : ℕ)
  (percentage_invalid : invalid_percentage = 15)
  (total_votes_eq : total_votes = 560000)
  (candidate_votes_eq : candidate_valid_votes = 380800) 
  : (candidate_valid_votes : ℝ) / (total_votes * (0.85 : ℝ)) * 100 = 80 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_valid_votes_l737_73722


namespace NUMINAMATH_GPT_find_a6_a7_l737_73704

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Given Conditions
axiom cond1 : arithmetic_sequence a d
axiom cond2 : a 2 + a 4 + a 9 + a 11 = 32

-- Proof Problem
theorem find_a6_a7 : a 6 + a 7 = 16 :=
  sorry

end NUMINAMATH_GPT_find_a6_a7_l737_73704


namespace NUMINAMATH_GPT_nelly_bid_l737_73743

theorem nelly_bid (joe_bid sarah_bid : ℕ) (h1 : joe_bid = 160000) (h2 : sarah_bid = 50000)
  (h3 : ∀ nelly_bid, nelly_bid = 3 * joe_bid + 2000) (h4 : ∀ nelly_bid, nelly_bid = 4 * sarah_bid + 1500) :
  ∃ nelly_bid, nelly_bid = 482000 :=
by
  -- Skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_nelly_bid_l737_73743


namespace NUMINAMATH_GPT_part1_proof_l737_73710

def a : ℚ := 1 / 2
def b : ℚ := -2
def expr : ℚ := 2 * (3 * a^2 * b - a * b^2) - 3 * (2 * a^2 * b - a * b^2 + a * b)

theorem part1_proof : expr = 5 := by
  unfold expr
  unfold a
  unfold b
  sorry

end NUMINAMATH_GPT_part1_proof_l737_73710


namespace NUMINAMATH_GPT_solve_for_C_days_l737_73740

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 15
noncomputable def C_work_rate : ℚ := 1 / 50
noncomputable def total_work_done_by_A_B : ℚ := 6 * (A_work_rate + B_work_rate)
noncomputable def remaining_work : ℚ := 1 - total_work_done_by_A_B

theorem solve_for_C_days : ∃ d : ℚ, d * C_work_rate = remaining_work ∧ d = 15 :=
by
  use 15
  simp [C_work_rate, remaining_work, total_work_done_by_A_B, A_work_rate, B_work_rate]
  sorry

end NUMINAMATH_GPT_solve_for_C_days_l737_73740


namespace NUMINAMATH_GPT_amber_max_ounces_l737_73797

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end NUMINAMATH_GPT_amber_max_ounces_l737_73797


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l737_73716

def A (x : ℝ) : Prop := x^2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := x > 1

theorem intersection_of_A_and_B :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l737_73716


namespace NUMINAMATH_GPT_units_digit_of_8_pow_2022_l737_73759

theorem units_digit_of_8_pow_2022 : (8 ^ 2022) % 10 = 4 := 
by
  -- We here would provide the proof of this theorem
  sorry

end NUMINAMATH_GPT_units_digit_of_8_pow_2022_l737_73759


namespace NUMINAMATH_GPT_peter_ate_7_over_48_l737_73783

-- Define the initial conditions
def total_slices : ℕ := 16
def slices_peter_ate : ℕ := 2
def shared_slice : ℚ := 1/3

-- Define the first part of the problem
def fraction_peter_ate_alone : ℚ := slices_peter_ate / total_slices

-- Define the fraction Peter ate from sharing one slice
def fraction_peter_ate_shared : ℚ := shared_slice / total_slices

-- Define the total fraction Peter ate
def total_fraction_peter_ate : ℚ := fraction_peter_ate_alone + fraction_peter_ate_shared

-- Create the theorem to be proved (statement only)
theorem peter_ate_7_over_48 :
  total_fraction_peter_ate = 7 / 48 :=
by
  sorry

end NUMINAMATH_GPT_peter_ate_7_over_48_l737_73783


namespace NUMINAMATH_GPT_largest_share_received_l737_73790

theorem largest_share_received (total_profit : ℝ) (ratios : List ℝ) (h_ratios : ratios = [1, 2, 2, 3, 4, 5]) 
  (h_profit : total_profit = 51000) : 
  let parts := ratios.sum 
  let part_value := total_profit / parts
  let largest_share := 5 * part_value 
  largest_share = 15000 := 
by 
  sorry

end NUMINAMATH_GPT_largest_share_received_l737_73790


namespace NUMINAMATH_GPT_fractional_to_decimal_l737_73794

theorem fractional_to_decimal : (7 / 16 : ℚ) = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fractional_to_decimal_l737_73794


namespace NUMINAMATH_GPT_minimize_pollution_park_distance_l737_73735

noncomputable def pollution_index (x : ℝ) : ℝ :=
  (1 / x) + (4 / (30 - x))

theorem minimize_pollution_park_distance : ∃ x : ℝ, (0 < x ∧ x < 30) ∧ pollution_index x = 10 :=
by
  sorry

end NUMINAMATH_GPT_minimize_pollution_park_distance_l737_73735


namespace NUMINAMATH_GPT_no_solution_l737_73777

def is_digit (B : ℕ) : Prop := B < 10

def divisible_by (n m : ℕ) : Prop := ∃ k, n = m * k

def satisfies_conditions (B : ℕ) : Prop :=
  is_digit B ∧
  divisible_by (12345670 + B) 2 ∧
  divisible_by (12345670 + B) 5 ∧
  divisible_by (12345670 + B) 11

theorem no_solution (B : ℕ) : ¬ satisfies_conditions B :=
sorry

end NUMINAMATH_GPT_no_solution_l737_73777


namespace NUMINAMATH_GPT_degrees_to_radians_l737_73726

theorem degrees_to_radians : (800 : ℝ) * (Real.pi / 180) = (40 / 9) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_degrees_to_radians_l737_73726


namespace NUMINAMATH_GPT_min_dist_of_PQ_l737_73708

open Real

theorem min_dist_of_PQ :
  ∀ (P Q : ℝ × ℝ),
    (P.fst - 3)^2 + (P.snd + 1)^2 = 4 →
    Q.fst = -3 →
    ∃ (min_dist : ℝ), min_dist = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_dist_of_PQ_l737_73708


namespace NUMINAMATH_GPT_mario_age_is_4_l737_73702

-- Define the conditions
def sum_of_ages (mario maria : ℕ) : Prop := mario + maria = 7
def mario_older_by_one (mario maria : ℕ) : Prop := mario = maria + 1

-- State the theorem to prove Mario's age is 4 given the conditions
theorem mario_age_is_4 (mario maria : ℕ) (h1 : sum_of_ages mario maria) (h2 : mario_older_by_one mario maria) : mario = 4 :=
sorry -- Proof to be completed later

end NUMINAMATH_GPT_mario_age_is_4_l737_73702


namespace NUMINAMATH_GPT_function_properties_l737_73730

-- Define the function f
def f (x p q : ℝ) : ℝ := x^3 + p * x^2 + 9 * q * x + p + q + 3

-- Stating the main theorem
theorem function_properties (p q : ℝ) :
  ( ∀ x : ℝ, f (-x) p q = -f x p q ) →
  (p = 0 ∧ q = -3 ∧ ∀ x : ℝ, f x 0 (-3) = x^3 - 27 * x ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≤ 26 ) ∧
   ( ∀ x ∈ Set.Icc (-1 : ℝ) 4, f x 0 (-3) ≥ -54 )) := 
sorry

end NUMINAMATH_GPT_function_properties_l737_73730


namespace NUMINAMATH_GPT_proof_problem_l737_73725

-- Definitions for the solution sets
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def intersection : Set ℝ := {x | -1 < x ∧ x < 2}

-- The quadratic inequality solution sets
def solution_set (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- The main theorem statement
theorem proof_problem (a b : ℝ) (h : solution_set a b = intersection) : a + b = -3 :=
sorry

end NUMINAMATH_GPT_proof_problem_l737_73725


namespace NUMINAMATH_GPT_log_expression_l737_73738

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression :
  log_base 4 16 - (log_base 2 3 * log_base 3 2) = 1 := by
  sorry

end NUMINAMATH_GPT_log_expression_l737_73738


namespace NUMINAMATH_GPT_solution_set_for_inequality_l737_73731

theorem solution_set_for_inequality 
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_mono_dec : ∀ x y, 0 < x → x < y → f y ≤ f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l737_73731


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l737_73715

variable {x : ℝ}

theorem necessary_not_sufficient_condition (h : x > 2) : x > 1 :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l737_73715


namespace NUMINAMATH_GPT_min_AB_plus_five_thirds_BF_l737_73707

theorem min_AB_plus_five_thirds_BF 
  (A : ℝ × ℝ) (onEllipse : ℝ × ℝ → Prop) (F : ℝ × ℝ)
  (B : ℝ × ℝ) (minFunction : ℝ)
  (hf : F = (-3, 0)) (hA : A = (-2,2))
  (hB : onEllipse B) :
  (∀ B', onEllipse B' → (dist A B' + 5/3 * dist B' F) ≥ minFunction) →
  minFunction = (dist A B + 5/3 * dist B F) →
  B = (-(5 * Real.sqrt 3) / 2, 2) := by
  sorry

def onEllipse (B : ℝ × ℝ) : Prop := (B.1^2) / 25 + (B.2^2) / 16 = 1

end NUMINAMATH_GPT_min_AB_plus_five_thirds_BF_l737_73707


namespace NUMINAMATH_GPT_combined_total_time_l737_73734

theorem combined_total_time
  (Katherine_time : Real := 20)
  (Naomi_time : Real := Katherine_time * (1 + 1 / 4))
  (Lucas_time : Real := Katherine_time * (1 + 1 / 3))
  (Isabella_time : Real := Katherine_time * (1 + 1 / 2))
  (Naomi_total : Real := Naomi_time * 10)
  (Lucas_total : Real := Lucas_time * 10)
  (Isabella_total : Real := Isabella_time * 10) :
  Naomi_total + Lucas_total + Isabella_total = 816.7 := sorry

end NUMINAMATH_GPT_combined_total_time_l737_73734


namespace NUMINAMATH_GPT_matchsticks_left_l737_73701

theorem matchsticks_left (total_matchsticks elvis_match_per_square ralph_match_per_square elvis_squares ralph_squares : ℕ)
  (h1 : total_matchsticks = 50)
  (h2 : elvis_match_per_square = 4)
  (h3 : ralph_match_per_square = 8)
  (h4 : elvis_squares = 5)
  (h5 : ralph_squares = 3) :
  total_matchsticks - (elvis_match_per_square * elvis_squares + ralph_match_per_square * ralph_squares) = 6 := 
by
  sorry

end NUMINAMATH_GPT_matchsticks_left_l737_73701


namespace NUMINAMATH_GPT_complex_number_properties_l737_73789

open Complex

noncomputable def z : ℂ := (1 - I) / I

theorem complex_number_properties :
  z ^ 2 = 2 * I ∧ Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_complex_number_properties_l737_73789


namespace NUMINAMATH_GPT_fraction_of_sum_l737_73799

theorem fraction_of_sum (l : List ℝ) (hl : l.length = 51)
  (n : ℝ) (hn : n ∈ l)
  (h : n = 7 * (l.erase n).sum / 50) :
  n / l.sum = 7 / 57 := by
  sorry

end NUMINAMATH_GPT_fraction_of_sum_l737_73799


namespace NUMINAMATH_GPT_clothing_value_is_correct_l737_73795

-- Define the value of the clothing to be C and the correct answer
def value_of_clothing (C : ℝ) : Prop :=
  (C + 2) = (7 / 12) * (C + 10)

-- Statement of the problem
theorem clothing_value_is_correct :
  ∃ (C : ℝ), value_of_clothing C ∧ C = 46 / 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_clothing_value_is_correct_l737_73795


namespace NUMINAMATH_GPT_pair_C_does_not_produce_roots_l737_73787

theorem pair_C_does_not_produce_roots (x : ℝ) :
  (x = 0 ∨ x = 2) ↔ (∃ x, y = x ∧ y = x - 2) = false :=
by
  sorry

end NUMINAMATH_GPT_pair_C_does_not_produce_roots_l737_73787


namespace NUMINAMATH_GPT_xy_zero_l737_73763

theorem xy_zero (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 54) : x * y = 0 :=
by
  sorry

end NUMINAMATH_GPT_xy_zero_l737_73763


namespace NUMINAMATH_GPT_sum_of_solutions_eq_l737_73771

theorem sum_of_solutions_eq (x : ℝ) : (5 * x - 7) * (4 * x + 11) = 0 ->
  -((27 : ℝ) / (20 : ℝ)) =
  - ((5 * - 7) * (4 * x + 11)) / ((5 * x - 7) * 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_l737_73771


namespace NUMINAMATH_GPT_fraction_exponentiation_l737_73706

theorem fraction_exponentiation : (3/4 : ℚ)^3 = 27/64 := by
  sorry

end NUMINAMATH_GPT_fraction_exponentiation_l737_73706


namespace NUMINAMATH_GPT_perpendicular_lines_a_eq_0_or_neg1_l737_73784

theorem perpendicular_lines_a_eq_0_or_neg1 (a : ℝ) :
  (∃ (k₁ k₂: ℝ), (k₁ = a ∧ k₂ = (2 * a - 1)) ∧ ∃ (k₃ k₄: ℝ), (k₃ = 3 ∧ k₄ = a) ∧ k₁ * k₃ + k₂ * k₄ = 0) →
  (a = 0 ∨ a = -1) := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_a_eq_0_or_neg1_l737_73784


namespace NUMINAMATH_GPT_man_is_older_l737_73744

-- Define present age of the son
def son_age : ℕ := 26

-- Define present age of the man (father)
axiom man_age : ℕ

-- Condition: in two years, the man's age will be twice the age of his son
axiom age_condition : man_age + 2 = 2 * (son_age + 2)

-- Prove that the man is 28 years older than his son
theorem man_is_older : man_age - son_age = 28 := sorry

end NUMINAMATH_GPT_man_is_older_l737_73744


namespace NUMINAMATH_GPT_find_last_week_rope_l737_73755

/-- 
Description: Mr. Sanchez bought 4 feet of rope less than he did the previous week. 
Given that he bought 96 inches in total, find how many feet he bought last week.
--/
theorem find_last_week_rope (F : ℕ) :
  12 * (F - 4) = 96 → F = 12 := by
  sorry

end NUMINAMATH_GPT_find_last_week_rope_l737_73755


namespace NUMINAMATH_GPT_points_can_move_on_same_line_l737_73728

variable {A B C x y x' y' : ℝ}

def transform_x (x y : ℝ) : ℝ := 3 * x + 2 * y + 1
def transform_y (x y : ℝ) : ℝ := x + 4 * y - 3

noncomputable def points_on_same_line (A B C : ℝ) (x y : ℝ) : Prop :=
  A*x + B*y + C = 0 ∧
  A*(transform_x x y) + B*(transform_y x y) + C = 0

theorem points_can_move_on_same_line :
  ∃ (A B C : ℝ), ∀ (x y : ℝ), points_on_same_line A B C x y :=
sorry

end NUMINAMATH_GPT_points_can_move_on_same_line_l737_73728


namespace NUMINAMATH_GPT_area_triangle_MNR_l737_73782

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

/-- Given the quadrilateral PQRS with the midpoints M and N of PQ and QR 
and specified lengths, prove the calculated area of triangle MNR. -/
theorem area_triangle_MNR : 
  let P : (ℝ × ℝ) := (0, 5)
  let Q : (ℝ × ℝ) := (10, 5)
  let R : (ℝ × ℝ) := (14, 0)
  let S : (ℝ × ℝ) := (7, 0)
  let M : (ℝ × ℝ) := (5, 5)  -- Midpoint of PQ
  let N : (ℝ × ℝ) := (12, 2.5) -- Midpoint of QR
  distance M.fst M.snd N.fst N.snd = 7.435 →
  ((5 - 0 : ℝ) / 2 = 2.5) →
  (1 / 2 * 7.435 * 2.5) = 9.294375 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_MNR_l737_73782


namespace NUMINAMATH_GPT_total_votes_l737_73713

theorem total_votes (bob_votes total_votes : ℕ) (h1 : bob_votes = 48) (h2 : (2 : ℝ) / 5 * total_votes = bob_votes) :
  total_votes = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l737_73713


namespace NUMINAMATH_GPT_geom_seq_sum_elems_l737_73766

theorem geom_seq_sum_elems (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end NUMINAMATH_GPT_geom_seq_sum_elems_l737_73766


namespace NUMINAMATH_GPT_product_with_zero_is_zero_l737_73718

theorem product_with_zero_is_zero :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 0) = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_with_zero_is_zero_l737_73718


namespace NUMINAMATH_GPT_cube_increasing_on_reals_l737_73756

theorem cube_increasing_on_reals (a b : ℝ) (h : a < b) : a^3 < b^3 :=
sorry

end NUMINAMATH_GPT_cube_increasing_on_reals_l737_73756


namespace NUMINAMATH_GPT_symmetric_point_Q_l737_73765

-- Definitions based on conditions
def P : ℝ × ℝ := (-3, 2)
def symmetric_with_respect_to_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.fst, -point.snd)

-- Theorem stating that the coordinates of point Q (symmetric to P with respect to the x-axis) are (-3, -2)
theorem symmetric_point_Q : symmetric_with_respect_to_x_axis P = (-3, -2) := 
sorry

end NUMINAMATH_GPT_symmetric_point_Q_l737_73765


namespace NUMINAMATH_GPT_john_runs_with_dog_for_half_hour_l737_73746

noncomputable def time_with_dog_in_hours (t : ℝ) : Prop := 
  let d1 := 6 * t          -- Distance run with the dog
  let d2 := 4 * (1 / 2)    -- Distance run alone
  (d1 + d2 = 5) ∧ (t = 1 / 2)

theorem john_runs_with_dog_for_half_hour : ∃ t : ℝ, time_with_dog_in_hours t := 
by
  use (1 / 2)
  sorry

end NUMINAMATH_GPT_john_runs_with_dog_for_half_hour_l737_73746


namespace NUMINAMATH_GPT_cuberoot_3375_sum_l737_73788

theorem cuberoot_3375_sum (a b : ℕ) (h : 3375 = 3^3 * 5^3) (h1 : a = 15) (h2 : b = 1) : a + b = 16 := by
  sorry

end NUMINAMATH_GPT_cuberoot_3375_sum_l737_73788


namespace NUMINAMATH_GPT_buns_per_student_correct_l737_73776

variables (packages_per_bun : Nat) (num_packages : Nat)
           (num_classes : Nat) (students_per_class : Nat)

def total_buns (packages_per_bun : Nat) (num_packages : Nat) : Nat :=
  packages_per_bun * num_packages

def total_students (num_classes : Nat) (students_per_class : Nat) : Nat :=
  num_classes * students_per_class

def buns_per_student (total_buns : Nat) (total_students : Nat) : Nat :=
  total_buns / total_students

theorem buns_per_student_correct :
  packages_per_bun = 8 →
  num_packages = 30 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student (total_buns packages_per_bun num_packages) 
                  (total_students num_classes students_per_class) = 2 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_buns_per_student_correct_l737_73776


namespace NUMINAMATH_GPT_scientific_notation_of_19400000000_l737_73792

theorem scientific_notation_of_19400000000 :
  ∃ a n, 1 ≤ |a| ∧ |a| < 10 ∧ (19400000000 : ℝ) = a * 10^n ∧ a = 1.94 ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_19400000000_l737_73792


namespace NUMINAMATH_GPT_winning_candidate_votes_l737_73751

theorem winning_candidate_votes (T W : ℕ) (d1 d2 d3 : ℕ) 
  (hT : T = 963)
  (hd1 : d1 = 53) 
  (hd2 : d2 = 79) 
  (hd3 : d3 = 105) 
  (h_sum : T = W + (W - d1) + (W - d2) + (W - d3)) :
  W = 300 := 
by
  sorry

end NUMINAMATH_GPT_winning_candidate_votes_l737_73751


namespace NUMINAMATH_GPT_problem_stmt_l737_73705

variable (a b : ℝ)

theorem problem_stmt (ha : a > 0) (hb : b > 0) (a_plus_b : a + b = 2):
  3 * a^2 + b^2 ≥ 3 ∧ 4 / (a + 1) + 1 / b ≥ 3 := by
  sorry

end NUMINAMATH_GPT_problem_stmt_l737_73705


namespace NUMINAMATH_GPT_find_integers_l737_73770

theorem find_integers (n : ℤ) : (6 ∣ (n - 4)) ∧ (10 ∣ (n - 8)) ↔ (n % 30 = 28) :=
by
  sorry

end NUMINAMATH_GPT_find_integers_l737_73770


namespace NUMINAMATH_GPT_product_of_consecutive_numbers_with_25_is_perfect_square_l737_73764

theorem product_of_consecutive_numbers_with_25_is_perfect_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n * (n + 1)) + 25 = k^2 := 
by
  -- Proof body omitted
  sorry

end NUMINAMATH_GPT_product_of_consecutive_numbers_with_25_is_perfect_square_l737_73764


namespace NUMINAMATH_GPT_cost_of_milk_l737_73741

theorem cost_of_milk (x : ℝ) (h1 : 10 * 0.1 = 1) (h2 : 11 = 1 + x + 3 * x) : x = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_milk_l737_73741


namespace NUMINAMATH_GPT_constant_function_of_inequality_l737_73703

theorem constant_function_of_inequality (f : ℝ → ℝ) 
  (h : ∀ x y z : ℝ, f (x + y) + f (y + z) + f (z + x) ≥ 3 * f (x + 2 * y + 3 * z)) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end NUMINAMATH_GPT_constant_function_of_inequality_l737_73703


namespace NUMINAMATH_GPT_girl_from_grade_4_probability_l737_73729

-- Number of girls and boys in grade 3
def girls_grade_3 := 28
def boys_grade_3 := 35
def total_grade_3 := girls_grade_3 + boys_grade_3

-- Number of girls and boys in grade 4
def girls_grade_4 := 45
def boys_grade_4 := 42
def total_grade_4 := girls_grade_4 + boys_grade_4

-- Number of girls and boys in grade 5
def girls_grade_5 := 38
def boys_grade_5 := 51
def total_grade_5 := girls_grade_5 + boys_grade_5

-- Total number of children in playground
def total_children := total_grade_3 + total_grade_4 + total_grade_5

-- Probability that a randomly selected child is a girl from grade 4
def probability_girl_grade_4 := (girls_grade_4: ℚ) / total_children

theorem girl_from_grade_4_probability :
  probability_girl_grade_4 = 45 / 239 := by
  sorry

end NUMINAMATH_GPT_girl_from_grade_4_probability_l737_73729


namespace NUMINAMATH_GPT_car_speed_15_seconds_less_l737_73768

theorem car_speed_15_seconds_less (v : ℝ) : 
  (∀ v, 75 = 3600 / v + 15) → v = 60 :=
by
  intro H
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_car_speed_15_seconds_less_l737_73768


namespace NUMINAMATH_GPT_minimum_number_of_tiles_l737_73778

-- Define the measurement conversion and area calculations.
def tile_width := 2
def tile_length := 6
def region_width_feet := 3
def region_length_feet := 4

-- Convert feet to inches.
def region_width_inches := region_width_feet * 12
def region_length_inches := region_length_feet * 12

-- Calculate areas.
def tile_area := tile_width * tile_length
def region_area := region_width_inches * region_length_inches

-- Lean 4 statement to prove the minimum number of tiles required.
theorem minimum_number_of_tiles : region_area / tile_area = 144 := by
  sorry

end NUMINAMATH_GPT_minimum_number_of_tiles_l737_73778


namespace NUMINAMATH_GPT_find_positive_integer_tuples_l737_73775

theorem find_positive_integer_tuples
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hz_prime : Prime z) :
  z ^ x = y ^ 3 + 1 →
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ (x = 2 ∧ y = 2 ∧ z = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_positive_integer_tuples_l737_73775


namespace NUMINAMATH_GPT_willie_exchange_rate_l737_73724

theorem willie_exchange_rate :
  let euros := 70
  let normal_exchange_rate := 1 / 5 -- euros per dollar
  let airport_exchange_rate := 5 / 7
  let dollars := euros * normal_exchange_rate * airport_exchange_rate
  dollars = 10 := by
  sorry

end NUMINAMATH_GPT_willie_exchange_rate_l737_73724


namespace NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_l737_73791

-- Define the statement for the first problem
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 49 = 0 → x = 7 ∨ x = -7 :=
by
  sorry

-- Define the statement for the second problem
theorem solve_quadratic_eq2 (x : ℝ) : 2 * (x + 1)^2 - 49 = 1 → x = 4 ∨ x = -6 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_l737_73791


namespace NUMINAMATH_GPT_solve_for_x_l737_73721

theorem solve_for_x (x t : ℝ)
  (h₁ : t = 9)
  (h₂ : (3 * (x + 5)) / 4 = t + (3 - 3 * x) / 2) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l737_73721


namespace NUMINAMATH_GPT_trapezoid_base_length_sets_l737_73767

open Nat

theorem trapezoid_base_length_sets :
  ∃ (sets : Finset (ℕ × ℕ)), sets.card = 5 ∧ 
    (∀ p ∈ sets, ∃ (b1 b2 : ℕ), b1 = 10 * p.1 ∧ b2 = 10 * p.2 ∧ b1 + b2 = 90) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_base_length_sets_l737_73767


namespace NUMINAMATH_GPT_max_f_5_value_l737_73737

noncomputable def f (x : ℝ) : ℝ := x ^ 2 + 2 * x

noncomputable def f_1 (x : ℝ) : ℝ := f x
noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0       => x -- Not used, as n starts from 1
  | (n + 1) => f (f_n n x)

noncomputable def max_f_5 : ℝ := 3 ^ 32 - 1

theorem max_f_5_value : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f_n 5 x ≤ max_f_5 :=
by
  intro x hx
  have := hx
  -- The detailed proof would go here,
  -- but for the statement, we end with sorry.
  sorry

end NUMINAMATH_GPT_max_f_5_value_l737_73737


namespace NUMINAMATH_GPT_tangent_line_circle_l737_73714

theorem tangent_line_circle (m : ℝ) (h : ∀ x y : ℝ, (x + y = 0) → ((x - m)^2 + y^2 = 2)) : m = 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_circle_l737_73714


namespace NUMINAMATH_GPT_sum_of_numerator_and_denominator_l737_73773

def repeating_decimal_to_fraction_sum (x : ℚ) := 
  let numerator := 710
  let denominator := 99
  numerator + denominator

theorem sum_of_numerator_and_denominator : repeating_decimal_to_fraction_sum (71/10 + 7/990) = 809 := by
  sorry

end NUMINAMATH_GPT_sum_of_numerator_and_denominator_l737_73773


namespace NUMINAMATH_GPT_work_completion_l737_73754

theorem work_completion (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : a + b = 1/10) (h2 : a = 1/14) : a + b = 1/10 := 
by {
  sorry
}

end NUMINAMATH_GPT_work_completion_l737_73754


namespace NUMINAMATH_GPT_decreasing_implies_b_geq_4_l737_73711

-- Define the function and its derivative
def function (x : ℝ) (b : ℝ) : ℝ := x^3 - 3*b*x + 1

def derivative (x : ℝ) (b : ℝ) : ℝ := 3*x^2 - 3*b

theorem decreasing_implies_b_geq_4 (b : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 → derivative x b ≤ 0) → b ≥ 4 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_decreasing_implies_b_geq_4_l737_73711


namespace NUMINAMATH_GPT_equation_solution_l737_73769

theorem equation_solution (x : ℤ) (h : 3 * x - 2 * x + x = 3 - 2 + 1) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_equation_solution_l737_73769


namespace NUMINAMATH_GPT_parity_of_f_and_h_l737_73760

-- Define function f
def f (x : ℝ) : ℝ := x^2

-- Define function h
def h (x : ℝ) : ℝ := x

-- Define even and odd function
def even_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def odd_fun (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = - g x

-- Theorem statement
theorem parity_of_f_and_h :
  even_fun f ∧ odd_fun h :=
by {
  sorry
}

end NUMINAMATH_GPT_parity_of_f_and_h_l737_73760


namespace NUMINAMATH_GPT_ratio_of_pieces_l737_73781

def total_length (len: ℕ) := len = 35
def longer_piece (len: ℕ) := len = 20

theorem ratio_of_pieces (shorter len_shorter : ℕ) : 
  total_length 35 →
  longer_piece 20 →
  shorter = 35 - 20 →
  len_shorter = 15 →
  (20:ℚ) / (len_shorter:ℚ) = (4:ℚ) / (3:ℚ) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_pieces_l737_73781


namespace NUMINAMATH_GPT_spring_compression_l737_73798

theorem spring_compression (s F : ℝ) (h : F = 16 * s^2) (hF : F = 4) : s = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_spring_compression_l737_73798


namespace NUMINAMATH_GPT_intersect_sets_l737_73739

def A := {x : ℝ | x > -1}
def B := {x : ℝ | x ≤ 5}

theorem intersect_sets : (A ∩ B) = {x : ℝ | -1 < x ∧ x ≤ 5} := 
by 
  sorry

end NUMINAMATH_GPT_intersect_sets_l737_73739


namespace NUMINAMATH_GPT_rectangle_area_k_value_l737_73750

theorem rectangle_area_k_value (d : ℝ) (length width : ℝ) (h1 : 5 * width = 2 * length) (h2 : d^2 = length^2 + width^2) :
  ∃ (k : ℝ), A = k * d^2 ∧ k = 10 / 29 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_k_value_l737_73750


namespace NUMINAMATH_GPT_mat_pow_four_eq_l737_73761

open Matrix

def mat : Matrix (Fin 2) (Fin 2) ℤ :=
  !![2, -2; 1, 1]

def mat_fourth_power : Matrix (Fin 2) (Fin 2) ℤ :=
  !![-14, -6; 3, -17]

theorem mat_pow_four_eq :
  mat ^ 4 = mat_fourth_power :=
by
  sorry

end NUMINAMATH_GPT_mat_pow_four_eq_l737_73761


namespace NUMINAMATH_GPT_merchant_loss_l737_73752

theorem merchant_loss
  (sp : ℝ)
  (profit_percent: ℝ)
  (loss_percent:  ℝ)
  (sp1 : ℝ)
  (sp2 : ℝ)
  (cp1 cp2 : ℝ)
  (net_loss : ℝ) :
  
  sp = 990 → 
  profit_percent = 0.1 → 
  loss_percent = 0.1 →
  sp1 = sp → 
  sp2 = sp → 
  cp1 = sp1 / (1 + profit_percent) →
  cp2 = sp2 / (1 - loss_percent) →
  net_loss = (cp2 - sp2) - (sp1 - cp1) →
  net_loss = 20 :=
by 
  intros _ _ _ _ _ _ _ _ 
  -- placeholders for intros to bind variables
  sorry

end NUMINAMATH_GPT_merchant_loss_l737_73752


namespace NUMINAMATH_GPT_option_B_is_correct_l737_73796

-- Definitions and Conditions
variable {Line : Type} {Plane : Type}
variable (m n : Line) (α β γ : Plane)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Conditions
axiom m_perp_β : perpendicular m β
axiom m_parallel_α : parallel m α

-- Statement to prove
theorem option_B_is_correct : perpendicular_planes α β :=
by
  sorry

end NUMINAMATH_GPT_option_B_is_correct_l737_73796


namespace NUMINAMATH_GPT_degree_measure_of_subtracted_angle_l737_73762

def angle := 30

theorem degree_measure_of_subtracted_angle :
  let supplement := 180 - angle
  let complement_of_supplement := 90 - supplement
  let twice_complement := 2 * (90 - angle)
  twice_complement - complement_of_supplement = 180 :=
by
  sorry

end NUMINAMATH_GPT_degree_measure_of_subtracted_angle_l737_73762


namespace NUMINAMATH_GPT_greatest_sum_of_consecutive_integers_l737_73727

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end NUMINAMATH_GPT_greatest_sum_of_consecutive_integers_l737_73727


namespace NUMINAMATH_GPT_A_finishes_in_20_days_l737_73745

-- Define the rates and the work
variable (A B W : ℝ)

-- First condition: A and B together can finish the work in 12 days
axiom together_rate : (A + B) * 12 = W

-- Second condition: B alone can finish the work in 30.000000000000007 days
axiom B_rate : B * 30.000000000000007 = W

-- Prove that A alone can finish the work in 20 days
theorem A_finishes_in_20_days : (1 / A) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_A_finishes_in_20_days_l737_73745


namespace NUMINAMATH_GPT_three_lines_form_triangle_l737_73780

/-- Theorem to prove that for three lines x + y = 0, x - y = 0, and x + ay = 3 to form a triangle, the value of a cannot be ±1. -/
theorem three_lines_form_triangle (a : ℝ) : ¬ (a = 1 ∨ a = -1) :=
sorry

end NUMINAMATH_GPT_three_lines_form_triangle_l737_73780


namespace NUMINAMATH_GPT_total_cash_realized_correct_l737_73747

structure Stock where
  value : ℝ
  return_rate : ℝ
  brokerage_fee_rate : ℝ

def stockA : Stock := { value := 10000, return_rate := 0.14, brokerage_fee_rate := 0.0025 }
def stockB : Stock := { value := 20000, return_rate := 0.10, brokerage_fee_rate := 0.005 }
def stockC : Stock := { value := 30000, return_rate := 0.07, brokerage_fee_rate := 0.0075 }

def cash_realized (s : Stock) : ℝ :=
  let total_with_return := s.value * (1 + s.return_rate)
  total_with_return - (total_with_return * s.brokerage_fee_rate)

noncomputable def total_cash_realized : ℝ :=
  cash_realized stockA + cash_realized stockB + cash_realized stockC

theorem total_cash_realized_correct :
  total_cash_realized = 65120.75 :=
    sorry

end NUMINAMATH_GPT_total_cash_realized_correct_l737_73747


namespace NUMINAMATH_GPT_like_terms_l737_73748

theorem like_terms (x y : ℕ) (h1 : x + 1 = 2) (h2 : x + y = 2) : x = 1 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_like_terms_l737_73748


namespace NUMINAMATH_GPT_sum_of_m_and_n_l737_73772

theorem sum_of_m_and_n :
  ∃ m n : ℝ, (∀ x : ℝ, (x = 2 → m = 6 / x) ∧ (x = -2 → n = 6 / x)) ∧ (m + n = 0) :=
by
  let m := 6 / 2
  let n := 6 / (-2)
  use m, n
  simp
  sorry -- Proof omitted

end NUMINAMATH_GPT_sum_of_m_and_n_l737_73772


namespace NUMINAMATH_GPT_solution_set_quadratic_l737_73709

theorem solution_set_quadratic (a x : ℝ) (h : a < 0) : 
  (x^2 - 2 * a * x - 3 * a^2 < 0) ↔ (3 * a < x ∧ x < -a) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_quadratic_l737_73709


namespace NUMINAMATH_GPT_min_number_of_4_dollar_frisbees_l737_73732

theorem min_number_of_4_dollar_frisbees 
  (x y : ℕ) 
  (h1 : x + y = 60)
  (h2 : 3 * x + 4 * y = 200) 
  : y = 20 :=
sorry

end NUMINAMATH_GPT_min_number_of_4_dollar_frisbees_l737_73732


namespace NUMINAMATH_GPT_correct_product_l737_73786

namespace SarahsMultiplication

theorem correct_product (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100)
  (hx' : ∃ (a b : ℕ), x = 10 * a + b ∧ b * 10 + a = x' ∧ 221 = x' * y) : (x * y = 527 ∨ x * y = 923) := by
  sorry

end SarahsMultiplication

end NUMINAMATH_GPT_correct_product_l737_73786


namespace NUMINAMATH_GPT_second_container_sand_capacity_l737_73785

def volume (h: ℕ) (w: ℕ) (l: ℕ) : ℕ := h * w * l

def sand_capacity (v1: ℕ) (s1: ℕ) (v2: ℕ) : ℕ := (s1 * v2) / v1

theorem second_container_sand_capacity:
  let h1 := 3
  let w1 := 4
  let l1 := 6
  let s1 := 72
  let h2 := 3 * h1
  let w2 := 2 * w1
  let l2 := l1
  let v1 := volume h1 w1 l1
  let v2 := volume h2 w2 l2
  sand_capacity v1 s1 v2 = 432 :=
by {
  sorry
}

end NUMINAMATH_GPT_second_container_sand_capacity_l737_73785


namespace NUMINAMATH_GPT_hyperbola_sum_l737_73753

theorem hyperbola_sum
  (h k a b : ℝ)
  (center : h = 3 ∧ k = 1)
  (vertex : ∃ (v : ℝ), (v = 4 ∧ h = 3 ∧ a = |k - v|))
  (focus : ∃ (f : ℝ), (f = 10 ∧ h = 3 ∧ (f - k) = 9 ∧ ∃ (c : ℝ), c = |k - f|))
  (relationship : ∀ (c : ℝ), c = 9 → c^2 = a^2 + b^2): 
  h + k + a + b = 7 + 6 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_sum_l737_73753


namespace NUMINAMATH_GPT_derivative_at_pi_over_2_l737_73774

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem derivative_at_pi_over_2 : 
  (deriv f (π / 2)) = Real.exp (π / 2) :=
by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_over_2_l737_73774


namespace NUMINAMATH_GPT_staplers_left_l737_73723

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end NUMINAMATH_GPT_staplers_left_l737_73723


namespace NUMINAMATH_GPT_remaining_pie_proportion_l737_73736

def carlos_portion : ℝ := 0.6
def maria_share_of_remainder : ℝ := 0.25

theorem remaining_pie_proportion: 
  (1 - carlos_portion) - maria_share_of_remainder * (1 - carlos_portion) = 0.3 := 
by
  -- proof to be implemented here
  sorry

end NUMINAMATH_GPT_remaining_pie_proportion_l737_73736


namespace NUMINAMATH_GPT_graph_passes_through_2_2_l737_73758

theorem graph_passes_through_2_2 (a : ℝ) (h : a > 0) (h_ne : a ≠ 1) : (2, 2) ∈ { p : ℝ × ℝ | ∃ x, p = (x, a^(x-2) + 1) } :=
sorry

end NUMINAMATH_GPT_graph_passes_through_2_2_l737_73758
