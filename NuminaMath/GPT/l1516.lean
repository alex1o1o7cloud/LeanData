import Mathlib

namespace simplify_and_evaluate_expression_l1516_151640

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = Real.sqrt 2 - 3) : 
  (a - 3) / (a^2 + 6 * a + 9) / (1 - 6 / (a + 3)) = Real.sqrt 2 / 2 :=
by sorry

end simplify_and_evaluate_expression_l1516_151640


namespace point_in_fourth_quadrant_l1516_151658

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a^2 + 1 > 0) (h2 : -1 - b^2 < 0) : 
  (a^2 + 1 > 0 ∧ -1 - b^2 < 0) ∧ (0 < a^2 + 1) ∧ (-1 - b^2 < 0) :=
by
  sorry

end point_in_fourth_quadrant_l1516_151658


namespace polynomial_coefficients_sum_l1516_151687

theorem polynomial_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 160 :=
by
  sorry

end polynomial_coefficients_sum_l1516_151687


namespace negation_of_proposition_l1516_151665

-- Define the proposition P(x)
def P (x : ℝ) : Prop := x + Real.log x > 0

-- Translate the problem into lean
theorem negation_of_proposition :
  (¬ ∀ x : ℝ, P x) ↔ ∃ x : ℝ, ¬ P x := by
  sorry

end negation_of_proposition_l1516_151665


namespace find_ab_sum_l1516_151680

theorem find_ab_sum
  (a b : ℝ)
  (h₁ : a^3 - 3 * a^2 + 5 * a - 1 = 0)
  (h₂ : b^3 - 3 * b^2 + 5 * b - 5 = 0) :
  a + b = 2 := by
  sorry

end find_ab_sum_l1516_151680


namespace problem_equiv_l1516_151638

theorem problem_equiv (a b : ℝ) (h : a ≠ b) : 
  (a^2 - 4 * a + 5 > 0) ∧ (a^2 + b^2 ≥ 2 * (a - b - 1)) :=
by {
  sorry
}

end problem_equiv_l1516_151638


namespace convince_jury_l1516_151621

def not_guilty : Prop := sorry  -- definition indicating the defendant is not guilty
def not_liar : Prop := sorry    -- definition indicating the defendant is not a liar
def innocent_knight_statement : Prop := sorry  -- statement "I am an innocent knight"

theorem convince_jury (not_guilty : not_guilty) (not_liar : not_liar) : innocent_knight_statement :=
sorry

end convince_jury_l1516_151621


namespace inequality_solution_l1516_151650

theorem inequality_solution (a x : ℝ) (h : 0 ≤ a ∧ a ≤ 4) :
  (x^2 + a * x > 4 * x + a - 3) ↔ (x < -1 ∨ x > 3)
:=
sorry

end inequality_solution_l1516_151650


namespace math_problem_l1516_151623

theorem math_problem : 
  ∀ n : ℕ, 
  n = 5 * 96 → 
  ((n + 17) * 69) = 34293 := 
by
  intros n h
  sorry

end math_problem_l1516_151623


namespace find_divisor_l1516_151670

theorem find_divisor (d q r : ℕ) (h1 : d = 265) (h2 : q = 12) (h3 : r = 1) :
  ∃ x : ℕ, d = (x * q) + r ∧ x = 22 :=
by {
  sorry
}

end find_divisor_l1516_151670


namespace converse_even_sum_l1516_151614

variable (a b : ℤ)

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem converse_even_sum (h : is_even (a + b)) : is_even a ∧ is_even b :=
sorry

end converse_even_sum_l1516_151614


namespace power_of_negative_base_l1516_151641

theorem power_of_negative_base : (-64 : ℤ)^(7 / 6) = -128 := by
  sorry

end power_of_negative_base_l1516_151641


namespace largest_common_number_in_range_l1516_151648

theorem largest_common_number_in_range (n1 d1 n2 d2 : ℕ) (h1 : n1 = 2) (h2 : d1 = 4) (h3 : n2 = 5) (h4 : d2 = 6) :
  ∃ k : ℕ, k ≤ 200 ∧ (∀ n3 : ℕ, n3 = n1 + d1 * k) ∧ (∀ n4 : ℕ, n4 = n2 + d2 * k) ∧ n3 = 190 ∧ n4 = 190 := 
by {
  sorry
}

end largest_common_number_in_range_l1516_151648


namespace simplify_exponents_l1516_151606

theorem simplify_exponents (x : ℝ) : x^5 * x^3 = x^8 :=
by
  sorry

end simplify_exponents_l1516_151606


namespace add_to_divisible_l1516_151616

theorem add_to_divisible (n d x : ℕ) (h : n = 987654) (h1 : d = 456) (h2 : x = 222) : 
  (n + x) % d = 0 := 
by {
  sorry
}

end add_to_divisible_l1516_151616


namespace value_of_a_b_c_l1516_151662

theorem value_of_a_b_c 
  (a b c : ℤ) 
  (h1 : x^2 + 12*x + 35 = (x + a)*(x + b)) 
  (h2 : x^2 - 15*x + 56 = (x - b)*(x - c)) : 
  a + b + c = 20 := 
sorry

end value_of_a_b_c_l1516_151662


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l1516_151610

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l1516_151610


namespace shaded_region_area_eq_l1516_151636

noncomputable def areaShadedRegion : ℝ :=
  let side_square := 14
  let side_triangle := 18
  let height := 14
  let H := 9 * Real.sqrt 3
  let BF := (side_square + side_triangle, height - H)
  let base_BF := BF.1 - 0
  let height_BF := BF.2
  let area_triangle_BFH := 0.5 * base_BF * height_BF
  let total_triangle_area := 0.5 * side_triangle * height
  let area_half_BFE := 0.5 * total_triangle_area
  area_half_BFE - area_triangle_BFH

theorem shaded_region_area_eq :
  areaShadedRegion = 9 * Real.sqrt 3 :=
by 
 sorry

end shaded_region_area_eq_l1516_151636


namespace common_factor_of_right_triangle_l1516_151630

theorem common_factor_of_right_triangle (d : ℝ) 
  (h_triangle : (2*d)^2 + (4*d)^2 = (5*d)^2) 
  (h_side : 2*d = 45 ∨ 4*d = 45 ∨ 5*d = 45) : 
  d = 9 :=
sorry

end common_factor_of_right_triangle_l1516_151630


namespace find_b_l1516_151604

theorem find_b (x : ℝ) (b : ℝ) :
  (∃ t u : ℝ, (bx^2 + 18 * x + 9) = (t * x + u)^2 ∧ u^2 = 9 ∧ 2 * t * u = 18 ∧ t^2 = b) →
  b = 9 :=
by
  sorry

end find_b_l1516_151604


namespace last_three_digits_7_pow_103_l1516_151667

theorem last_three_digits_7_pow_103 : (7 ^ 103) % 1000 = 60 := sorry

end last_three_digits_7_pow_103_l1516_151667


namespace log_equation_l1516_151696

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_equation (x : ℝ) (h1 : x > 1) (h2 : (log_base_10 x)^2 - log_base_10 (x^4) = 32) :
  (log_base_10 x)^4 - log_base_10 (x^4) = 4064 :=
by
  sorry

end log_equation_l1516_151696


namespace mod_exponent_problem_l1516_151697

theorem mod_exponent_problem : (11 ^ 2023) % 100 = 31 := by
  sorry

end mod_exponent_problem_l1516_151697


namespace trigonometric_identity_l1516_151620

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 1 / 13 := 
by
-- The proof goes here
sorry

end trigonometric_identity_l1516_151620


namespace train_crosses_signal_pole_in_12_seconds_l1516_151632

noncomputable def time_to_cross_signal_pole (length_train : ℕ) (time_to_cross_platform : ℕ) (length_platform : ℕ) : ℕ :=
  let distance_train_platform := length_train + length_platform
  let speed_train := distance_train_platform / time_to_cross_platform
  let time_to_cross_pole := length_train / speed_train
  time_to_cross_pole

theorem train_crosses_signal_pole_in_12_seconds :
  time_to_cross_signal_pole 300 39 675 = 12 :=
by
  -- expected proof in the interactive mode
  sorry

end train_crosses_signal_pole_in_12_seconds_l1516_151632


namespace range_of_a_l1516_151631

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, a * x ^ 2 + 2 * a * x + 1 ≤ 0) →
  0 ≤ a ∧ a < 1 :=
by
  -- sorry to skip the proof
  sorry

end range_of_a_l1516_151631


namespace sum_of_integers_l1516_151617

theorem sum_of_integers : (∀ (x y : ℤ), x = -4 ∧ y = -5 ∧ x - y = 1 → x + y = -9) := 
by 
  intros x y
  sorry

end sum_of_integers_l1516_151617


namespace calc_ratio_of_d_to_s_l1516_151618

theorem calc_ratio_of_d_to_s {n s d : ℝ} (h_n_eq_24 : n = 24)
    (h_tiles_area_64_pct : (576 * s^2) = 0.64 * (n * s + d)^2) : 
    d / s = 6 / 25 :=
by
  sorry

end calc_ratio_of_d_to_s_l1516_151618


namespace first_term_of_geometric_series_l1516_151642

/-- An infinite geometric series with common ratio -1/3 has a sum of 24.
    Prove that the first term of the series is 32. -/
theorem first_term_of_geometric_series (r : ℝ) (S : ℝ) (a : ℝ) 
  (h1 : r = -1/3) 
  (h2 : S = 24) 
  (h3 : S = a / (1 - r)) : 
  a = 32 := 
sorry

end first_term_of_geometric_series_l1516_151642


namespace percentage_40_number_l1516_151626

theorem percentage_40_number (x y z P : ℝ) (hx : x = 93.75) (hy : y = 0.40 * x) (hz : z = 6) (heq : (P / 100) * y = z) :
  P = 16 :=
sorry

end percentage_40_number_l1516_151626


namespace sufficient_and_necessary_condition_l1516_151663

theorem sufficient_and_necessary_condition (a : ℝ) : 
  (0 < a ∧ a < 4) ↔ ∀ x : ℝ, (x^2 - a * x + a) > 0 :=
by sorry

end sufficient_and_necessary_condition_l1516_151663


namespace equalities_imply_forth_l1516_151602

variables {a b c d e f g h S1 S2 S3 O2 O3 : ℕ}

def S1_def := S1 = a + b + c
def S2_def := S2 = d + e + f
def S3_def := S3 = b + c + g + h - d
def O2_def := O2 = b + e + g
def O3_def := O3 = c + f + h

theorem equalities_imply_forth (h1 : S1 = S2) (h2 : S1 = S3) (h3 : S1 = O2) : S1 = O3 :=
  by sorry

end equalities_imply_forth_l1516_151602


namespace vertex_of_parabola_l1516_151601

theorem vertex_of_parabola (a b : ℝ) (roots_condition : ∀ x, -x^2 + a * x + b ≤ 0 ↔ (x ≤ -3 ∨ x ≥ 5)) :
  ∃ v : ℝ × ℝ, v = (1, 16) :=
by
  sorry

end vertex_of_parabola_l1516_151601


namespace arithmetic_sequence_inequality_l1516_151685

theorem arithmetic_sequence_inequality 
  (a b c : ℝ) 
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : b - a = d)
  (h3 : c - b = d) :
  ¬ (a^3 * b + b^3 * c + c^3 * a ≥ a^4 + b^4 + c^4) :=
sorry

end arithmetic_sequence_inequality_l1516_151685


namespace sanjay_homework_fraction_l1516_151675

theorem sanjay_homework_fraction :
  let original := 1
  let done_on_monday := 3 / 5
  let remaining_after_monday := original - done_on_monday
  let done_on_tuesday := 1 / 3 * remaining_after_monday
  let remaining_after_tuesday := remaining_after_monday - done_on_tuesday
  remaining_after_tuesday = 4 / 15 :=
by
  -- original := 1
  -- done_on_monday := 3 / 5
  -- remaining_after_monday := 1 - 3 / 5
  -- done_on_tuesday := 1 / 3 * (1 - 3 / 5)
  -- remaining_after_tuesday := (1 - 3 / 5) - (1 / 3 * (1 - 3 / 5))
  sorry

end sanjay_homework_fraction_l1516_151675


namespace christmas_tree_seller_l1516_151693

theorem christmas_tree_seller 
  (cost_spruce : ℕ := 220) 
  (cost_pine : ℕ := 250) 
  (cost_fir : ℕ := 330) 
  (total_revenue : ℕ := 36000) 
  (equal_trees: ℕ) 
  (h_costs : cost_spruce + cost_pine + cost_fir = 800) 
  (h_revenue : equal_trees * 800 = total_revenue):
  3 * equal_trees = 135 :=
sorry

end christmas_tree_seller_l1516_151693


namespace num_five_digit_ints_l1516_151672

open Nat

theorem num_five_digit_ints : 
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  num_ways = 10 :=
by
  let num_ways := (factorial 5) / (factorial 3 * factorial 2)
  sorry

end num_five_digit_ints_l1516_151672


namespace a_fraction_of_capital_l1516_151639

theorem a_fraction_of_capital (T : ℝ) (B : ℝ) (C : ℝ) (D : ℝ)
  (profit_A : ℝ) (total_profit : ℝ)
  (h1 : B = T * (1 / 4))
  (h2 : C = T * (1 / 5))
  (h3 : D = T - (T * (1 / 4) + T * (1 / 5) + T * x))
  (h4 : profit_A = 805)
  (h5 : total_profit = 2415) :
  x = 161 / 483 :=
by
  sorry

end a_fraction_of_capital_l1516_151639


namespace system_of_equations_solution_l1516_151679

theorem system_of_equations_solution
  (a b c d e f g : ℝ)
  (x y z : ℝ)
  (h1 : a * x = b * y)
  (h2 : b * y = c * z)
  (h3 : d * x + e * y + f * z = g) :
  (x = g * b * c / (d * b * c + e * a * c + f * a * b)) ∧
  (y = g * a * c / (d * b * c + e * a * c + f * a * b)) ∧
  (z = g * a * b / (d * b * c + e * a * c + f * a * b)) :=
by
  sorry

end system_of_equations_solution_l1516_151679


namespace age_difference_l1516_151647

variable (A B : ℕ)

-- Given conditions
def B_is_95 : Prop := B = 95
def A_after_30_years : Prop := A + 30 = 2 * (B - 30)

-- Theorem to prove
theorem age_difference (h1 : B_is_95 B) (h2 : A_after_30_years A B) : A - B = 5 := 
by
  sorry

end age_difference_l1516_151647


namespace opposite_of_neg_two_l1516_151684

theorem opposite_of_neg_two : ∃ x : Int, (-2 + x = 0) ∧ x = 2 :=
by
  use 2
  constructor
  . simp
  . rfl

end opposite_of_neg_two_l1516_151684


namespace starting_number_unique_l1516_151624

-- Definitions based on conditions
def has_two_threes (n : ℕ) : Prop :=
  (n / 10 = 3 ∧ n % 10 = 3)

def is_starting_number (n m : ℕ) : Prop :=
  ∃ k, n + k = m ∧ k < (m - n) ∧ has_two_threes m

-- Theorem stating the proof problem
theorem starting_number_unique : ∃ n, is_starting_number n 30 ∧ n = 32 := 
sorry

end starting_number_unique_l1516_151624


namespace amount_of_benzene_l1516_151645

-- Definitions of the chemical entities involved
def Benzene := Type
def Methane := Type
def Toluene := Type
def Hydrogen := Type

-- The balanced chemical equation as a condition
axiom balanced_equation : ∀ (C6H6 CH4 C7H8 H2 : ℕ), C6H6 + CH4 = C7H8 + H2

-- The proof problem: Prove the amount of Benzene required
theorem amount_of_benzene (moles_methane : ℕ) (moles_toluene : ℕ) (moles_hydrogen : ℕ) :
  moles_methane = 2 → moles_toluene = 2 → moles_hydrogen = 2 → 
  ∃ moles_benzene : ℕ, moles_benzene = 2 := by
  sorry

end amount_of_benzene_l1516_151645


namespace BoatCrafters_l1516_151661

/-
  Let J, F, M, A represent the number of boats built in January, February,
  March, and April respectively.

  Conditions:
  1. J = 4
  2. F = J / 2
  3. M = F * 3
  4. A = M * 3

  Goal:
  Prove that J + F + M + A = 30.
-/

def BoatCrafters.total_boats_built : Nat := 4 + (4 / 2) + ((4 / 2) * 3) + (((4 / 2) * 3) * 3)

theorem BoatCrafters.boats_built_by_end_of_April : 
  BoatCrafters.total_boats_built = 30 :=   
by 
  sorry

end BoatCrafters_l1516_151661


namespace part1_part2_l1516_151683

-- Part (1)
theorem part1 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) (opposite : m * n < 0) :
  m + n = -3 ∨ m + n = 3 :=
sorry

-- Part (2)
theorem part2 (m n : ℝ) (hm : |m| = 1) (hn : |n| = 4) :
  (m - n) ≤ 5 :=
sorry

end part1_part2_l1516_151683


namespace find_common_difference_l1516_151649

variable {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) = a n + d)
variable (a7_minus_2a4_eq_6 : a 7 - 2 * a 4 = 6)
variable (a3_eq_2 : a 3 = 2)

theorem find_common_difference (d : ℝ) : d = 4 :=
by
  -- Proof would go here
  sorry

end find_common_difference_l1516_151649


namespace compound_analysis_l1516_151677

noncomputable def molecular_weight : ℝ := 18
noncomputable def atomic_weight_nitrogen : ℝ := 14.01
noncomputable def atomic_weight_hydrogen : ℝ := 1.01

theorem compound_analysis :
  ∃ (n : ℕ) (element : String), element = "hydrogen" ∧ n = 4 ∧
  (∃ remaining_weight : ℝ, remaining_weight = molecular_weight - atomic_weight_nitrogen ∧
   ∃ k, remaining_weight / atomic_weight_hydrogen = k ∧ k = n) :=
by
  sorry

end compound_analysis_l1516_151677


namespace red_tickets_for_one_yellow_l1516_151625

-- Define the conditions given in the problem
def yellow_needed := 10
def red_for_yellow (R : ℕ) := R -- This function defines the number of red tickets for one yellow
def blue_for_red := 10

def toms_yellow := 8
def toms_red := 3
def toms_blue := 7
def blue_needed := 163

-- Define the target function that converts the given conditions into a statement.
def red_tickets_for_yellow_proof : Prop :=
  ∀ R : ℕ, (2 * R = 14) → (R = 7)

-- Statement for proof where the condition leads to conclusion
theorem red_tickets_for_one_yellow : red_tickets_for_yellow_proof :=
by
  intros R h
  rw [← h, mul_comm] at h
  sorry

end red_tickets_for_one_yellow_l1516_151625


namespace return_trip_time_l1516_151634

-- Define the given conditions
def run_time : ℕ := 20
def jog_time : ℕ := 10
def trip_time := run_time + jog_time
def multiplier: ℕ := 3

-- State the theorem
theorem return_trip_time : trip_time * multiplier = 90 := by
  sorry

end return_trip_time_l1516_151634


namespace number_of_boys_l1516_151686

-- Definitions for the given conditions
def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := 20
def total_girls := 41
def happy_boys := 6
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

-- Define the total number of boys
def total_boys := total_children - total_girls

-- Proof statement
theorem number_of_boys : total_boys = 19 :=
  by
    sorry

end number_of_boys_l1516_151686


namespace probability_same_color_socks_l1516_151613

-- Define the total number of socks and the groups
def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

-- Define combinatorial functions to calculate combinations
def comb (n m : ℕ) : ℕ := n.choose m

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  comb blue_socks 2 +
  comb green_socks 2 +
  comb red_socks 2

-- Calculate the total number of possible outcomes
def total_outcomes : ℕ := comb total_socks 2

-- Calculate the probability as a ratio of favorable outcomes to total outcomes
def probability := favorable_outcomes / total_outcomes

-- Prove the probability is 19/45
theorem probability_same_color_socks : probability = 19 / 45 := by
  sorry

end probability_same_color_socks_l1516_151613


namespace three_digit_numbers_with_repeated_digits_l1516_151652

theorem three_digit_numbers_with_repeated_digits :
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  total_three_digit_numbers - without_repeats = 252 := by
{
  let total_three_digit_numbers := 900
  let without_repeats := 9 * 9 * 8
  show total_three_digit_numbers - without_repeats = 252
  sorry
}

end three_digit_numbers_with_repeated_digits_l1516_151652


namespace goldfish_count_15_weeks_l1516_151605

def goldfish_count_after_weeks (initial : ℕ) (weeks : ℕ) : ℕ :=
  let deaths := λ n => 10 + 2 * (n - 1)
  let purchases := λ n => 5 + 2 * (n - 1)
  let rec update_goldfish (current : ℕ) (week : ℕ) :=
    if week = 0 then current
    else 
      let new_count := current - deaths week + purchases week
      update_goldfish new_count (week - 1)
  update_goldfish initial weeks

theorem goldfish_count_15_weeks : goldfish_count_after_weeks 35 15 = 15 :=
  by
  sorry

end goldfish_count_15_weeks_l1516_151605


namespace factorize_expression_l1516_151668

theorem factorize_expression (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) :=
by
  sorry

end factorize_expression_l1516_151668


namespace vehicles_sent_l1516_151690

theorem vehicles_sent (x y : ℕ) (h1 : x + y < 18) (h2 : y < 2 * x) (h3 : x + 4 < y) :
  x = 6 ∧ y = 11 := by
  sorry

end vehicles_sent_l1516_151690


namespace fraction_of_rotten_is_one_third_l1516_151646

def total_berries (blueberries cranberries raspberries : Nat) : Nat :=
  blueberries + cranberries + raspberries

def fresh_berries (berries_to_sell berries_to_keep : Nat) : Nat :=
  berries_to_sell + berries_to_keep

def rotten_berries (total fresh : Nat) : Nat :=
  total - fresh

def fraction_rot (rotten total : Nat) : Rat :=
  (rotten : Rat) / (total : Rat)

theorem fraction_of_rotten_is_one_third :
  ∀ (blueberries cranberries raspberries berries_to_sell : Nat),
    blueberries = 30 →
    cranberries = 20 →
    raspberries = 10 →
    berries_to_sell = 20 →
    fraction_rot (rotten_berries (total_berries blueberries cranberries raspberries) 
                  (fresh_berries berries_to_sell berries_to_sell))
                  (total_berries blueberries cranberries raspberries) = 1 / 3 :=
by
  intros blueberries cranberries raspberries berries_to_sell
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end fraction_of_rotten_is_one_third_l1516_151646


namespace probability_obtuse_triangle_is_one_fourth_l1516_151619

-- Define the set of possible integers
def S : Set ℤ := {1, 2, 3, 4, 5, 6}

-- Condition for forming an obtuse triangle
def is_obtuse_triangle (a b c : ℤ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b ∧ 
  (a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2)

-- List of valid triples that can form an obtuse triangle
def valid_obtuse_triples : List (ℤ × ℤ × ℤ) :=
  [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 6), (3, 5, 6)]

-- Total number of combinations
def total_combinations : Nat := 20

-- Number of valid combinations for obtuse triangles
def valid_combinations : Nat := 5

-- Calculate the probability
def probability_obtuse_triangle : ℚ := valid_combinations / total_combinations

theorem probability_obtuse_triangle_is_one_fourth :
  probability_obtuse_triangle = 1 / 4 :=
by
  sorry

end probability_obtuse_triangle_is_one_fourth_l1516_151619


namespace second_school_more_students_l1516_151688

theorem second_school_more_students (S1 S2 S3 : ℕ) 
  (hS3 : S3 = 200) 
  (hS1 : S1 = 2 * S2) 
  (h_total : S1 + S2 + S3 = 920) : 
  S2 - S3 = 40 :=
by
  sorry

end second_school_more_students_l1516_151688


namespace minimize_quadratic_l1516_151660

theorem minimize_quadratic (x : ℝ) : (∃ x, x = 3 ∧ ∀ y, 3 * (y ^ 2) - 18 * y + 7 ≥ 3 * (x ^ 2) - 18 * x + 7) :=
by
  sorry

end minimize_quadratic_l1516_151660


namespace min_value_quadratic_l1516_151682

theorem min_value_quadratic (x : ℝ) : x = -1 ↔ (∀ y : ℝ, x^2 + 2*x + 4 ≤ y) := by
  sorry

end min_value_quadratic_l1516_151682


namespace simplify_expression_l1516_151689

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

theorem simplify_expression (ha : a > 0) (hb : b > 0) (hab : a^3 + b^3 = a + b) :
  (a / b) + (b / a) - (1 / (a * b)) = 1 :=
by sorry

end simplify_expression_l1516_151689


namespace range_of_k_l1516_151671
noncomputable def quadratic_nonnegative (k : ℝ) : Prop :=
  ∀ x : ℝ, k * x^2 - 4 * x + 3 ≥ 0

theorem range_of_k (k : ℝ) : quadratic_nonnegative k ↔ k ∈ Set.Ici (4 / 3) :=
by
  sorry

end range_of_k_l1516_151671


namespace area_of_triangle_l1516_151644

noncomputable def findAreaOfTriangle (a b : ℝ) (cosAOF : ℝ) : ℝ := sorry

theorem area_of_triangle (a b cosAOF : ℝ)
  (ha : a = 15 / 7)
  (hb : b = Real.sqrt 21)
  (hcos : cosAOF = 2 / 5) :
  findAreaOfTriangle a b cosAOF = 6 := by
  rw [ha, hb, hcos]
  sorry

end area_of_triangle_l1516_151644


namespace monotonic_increasing_intervals_l1516_151664

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 2*x + 1)
noncomputable def f' (x : ℝ) : ℝ := Real.exp x * (x^2 + 4*x + 3)

theorem monotonic_increasing_intervals :
  ∀ x, f' x > 0 ↔ (x < -3 ∨ x > -1) :=
by
  intro x
  -- proof omitted
  sorry

end monotonic_increasing_intervals_l1516_151664


namespace trigonometric_expression_proof_l1516_151612

theorem trigonometric_expression_proof :
  (Real.cos (76 * Real.pi / 180) * Real.cos (16 * Real.pi / 180) +
   Real.cos (14 * Real.pi / 180) * Real.cos (74 * Real.pi / 180) -
   2 * Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)) = 0 :=
by
  sorry

end trigonometric_expression_proof_l1516_151612


namespace candle_height_relation_l1516_151603

theorem candle_height_relation : 
  ∀ (h : ℝ) (t : ℝ), h = 1 → (∀ (h1_burn_rate : ℝ), h1_burn_rate = 1 / 5) → (∀ (h2_burn_rate : ℝ), h2_burn_rate = 1 / 6) →
  (1 - t * 1 / 5 = 3 * (1 - t * 1 / 6)) → t = 20 / 3 :=
by
  intros h t h_init h1_burn_rate h2_burn_rate height_eq
  sorry

end candle_height_relation_l1516_151603


namespace probability_of_roots_condition_l1516_151699

theorem probability_of_roots_condition :
  let k := 6 -- Lower bound of the interval
  let k' := 10 -- Upper bound of the interval
  let interval_length := k' - k
  let satisfying_interval_length := (22 / 3) - 6
  -- The probability that the roots of the quadratic equation satisfy x₁ ≤ 2x₂
  (satisfying_interval_length / interval_length) = (1 / 3) := by
    sorry

end probability_of_roots_condition_l1516_151699


namespace percentage_apples_basket_l1516_151681

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end percentage_apples_basket_l1516_151681


namespace triangle_projection_inequality_l1516_151627

variable (a b c t r μ : ℝ)
variable (h1 : AC_1 = 2 * t * AB)
variable (h2 : BA_1 = 2 * r * BC)
variable (h3 : CB_1 = 2 * μ * AC)
variable (h4 : AB = c)
variable (h5 : AC = b)
variable (h6 : BC = a)

theorem triangle_projection_inequality
  (h1 : AC_1 = 2 * t * AB)  -- condition AC_1 = 2t * AB
  (h2 : BA_1 = 2 * r * BC)  -- condition BA_1 = 2r * BC
  (h3 : CB_1 = 2 * μ * AC)  -- condition CB_1 = 2μ * AC
  (h4 : AB = c)             -- side AB
  (h5 : AC = b)             -- side AC
  (h6 : BC = a)             -- side BC
  : (a^2 / b^2) * (t / (1 - 2 * t))^2 
  + (b^2 / c^2) * (r / (1 - 2 * r))^2 
  + (c^2 / a^2) * (μ / (1 - 2 * μ))^2 
  + 16 * t * r * μ ≥ 1 := 
  sorry

end triangle_projection_inequality_l1516_151627


namespace total_pencils_l1516_151659

/-- The conditions defining the number of pencils Sarah buys each day. -/
def pencils_monday : ℕ := 20
def pencils_tuesday : ℕ := 18
def pencils_wednesday : ℕ := 3 * pencils_tuesday

/-- The hypothesis that the total number of pencils bought by Sarah is 92. -/
theorem total_pencils : pencils_monday + pencils_tuesday + pencils_wednesday = 92 :=
by
  -- calculations skipped
  sorry

end total_pencils_l1516_151659


namespace highest_score_not_necessarily_12_l1516_151674

-- Define the structure of the round-robin tournament setup
structure RoundRobinTournament :=
  (teams : ℕ)
  (matches_per_team : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (points_draw : ℕ)

-- Tournament conditions
def tournament : RoundRobinTournament :=
  { teams := 12,
    matches_per_team := 11,
    points_win := 2,
    points_loss := 0,
    points_draw := 1 }

-- The statement we want to prove
theorem highest_score_not_necessarily_12 (T : RoundRobinTournament) :
  ∃ team_highest_score : ℕ, team_highest_score < 12 :=
by
  -- Provide a proof here
  sorry

end highest_score_not_necessarily_12_l1516_151674


namespace max_value_of_y_no_min_value_l1516_151666

noncomputable def function_y (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

theorem max_value_of_y_no_min_value :
  (∃ x, -2 < x ∧ x < 2 ∧ function_y x = 5) ∧
  (∀ y, ∃ x, -2 < x ∧ x < 2 ∧ function_y x >= y) :=
by
  sorry

end max_value_of_y_no_min_value_l1516_151666


namespace union_set_subset_range_intersection_empty_l1516_151635

-- Define the sets A and B
def A : Set ℝ := { x | 1 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 2 * m < x ∧ x < 1 - m }

-- Question 1: When m = -1, prove A ∪ B = { x | -2 < x < 3 }
theorem union_set (m : ℝ) (h : m = -1) : A ∪ B m = { x | -2 < x ∧ x < 3 } := by
  sorry

-- Question 2: If A ⊆ B, prove m ∈ (-∞, -2]
theorem subset_range (m : ℝ) (h : A ⊆ B m) : m ∈ Set.Iic (-2) := by
  sorry

-- Question 3: If A ∩ B = ∅, prove m ∈ [0, +∞)
theorem intersection_empty (m : ℝ) (h : A ∩ B m = ∅) : m ∈ Set.Ici 0 := by
  sorry

end union_set_subset_range_intersection_empty_l1516_151635


namespace S15_constant_l1516_151600

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Given condition: a_5 + a_8 + a_11 is constant
axiom const_sum : ∀ (a1 d : ℤ), a 5 a1 d + a 8 a1 d + a 11 a1 d = 3 * a1 + 21 * d

-- The equivalent proof problem
theorem S15_constant (a1 d : ℤ) : S 15 a1 d = 5 * (3 * a1 + 21 * d) :=
by
  sorry

end S15_constant_l1516_151600


namespace ratio_perimeters_l1516_151698

noncomputable def rectangle_length : ℝ := 3
noncomputable def rectangle_width : ℝ := 2
noncomputable def triangle_hypotenuse : ℝ := Real.sqrt ((rectangle_length / 2) ^ 2 + rectangle_width ^ 2)
noncomputable def perimeter_rectangle : ℝ := 2 * (rectangle_length + rectangle_width)
noncomputable def perimeter_rhombus : ℝ := 4 * triangle_hypotenuse

theorem ratio_perimeters (h1 : rectangle_length = 3) (h2 : rectangle_width = 2) :
  (perimeter_rectangle / perimeter_rhombus) = 1 :=
by
  /- proof would go here -/
  sorry

end ratio_perimeters_l1516_151698


namespace loads_ratio_l1516_151678

noncomputable def loads_wednesday : ℕ := 6
noncomputable def loads_friday (T : ℕ) : ℕ := T / 2
noncomputable def loads_saturday : ℕ := loads_wednesday / 3
noncomputable def total_loads_week (T : ℕ) : ℕ := loads_wednesday + T + loads_friday T + loads_saturday

theorem loads_ratio (T : ℕ) (h : total_loads_week T = 26) : T / loads_wednesday = 2 := 
by 
  -- proof steps would go here
  sorry

end loads_ratio_l1516_151678


namespace polygon_with_largest_area_l1516_151691

noncomputable def area_of_polygon_A : ℝ := 6
noncomputable def area_of_polygon_B : ℝ := 4
noncomputable def area_of_polygon_C : ℝ := 4 + 2 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_D : ℝ := 3 + 3 * (1 / 2 * 1 * 1)
noncomputable def area_of_polygon_E : ℝ := 7

theorem polygon_with_largest_area : 
  area_of_polygon_E > area_of_polygon_A ∧ 
  area_of_polygon_E > area_of_polygon_B ∧ 
  area_of_polygon_E > area_of_polygon_C ∧ 
  area_of_polygon_E > area_of_polygon_D :=
by
  sorry

end polygon_with_largest_area_l1516_151691


namespace find_S_30_l1516_151643

variable (S : ℕ → ℚ)
variable (a : ℕ → ℚ)
variable (d : ℚ)

-- Definitions based on conditions
def arithmetic_sum (n : ℕ) : ℚ := (n / 2) * (a 1 + a n)
def a_n (n : ℕ) : ℚ := a 1 + (n - 1) * d

-- Given conditions
axiom h1 : S 10 = 20
axiom h2 : S 20 = 15

-- Required Proof (the final statement to be proven)
theorem find_S_30 : S 30 = -15 := sorry

end find_S_30_l1516_151643


namespace intersection_of_sets_l1516_151694

def setA : Set ℝ := {x | x^2 - 1 ≥ 0}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets : (setA ∩ setB) = {x | 1 ≤ x ∧ x < 4} := 
by 
  sorry

end intersection_of_sets_l1516_151694


namespace line_through_intersection_and_parallel_l1516_151655

theorem line_through_intersection_and_parallel
  (x y : ℝ)
  (l1 : 3 * x + 4 * y - 2 = 0)
  (l2 : 2 * x + y + 2 = 0)
  (l3 : ∃ k : ℝ, k * x + y + 2 = 0 ∧ k = -(4 / 3)) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 4 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end line_through_intersection_and_parallel_l1516_151655


namespace no_solution_exists_l1516_151609

theorem no_solution_exists : ¬ ∃ n : ℕ, (n^2 ≡ 1 [MOD 5]) ∧ (n^3 ≡ 3 [MOD 5]) := 
sorry

end no_solution_exists_l1516_151609


namespace sufficient_not_necessary_condition_l1516_151653

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, abs (x - 1) < 3 → (x + 2) * (x + a) < 0) ∧ 
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ ¬(abs (x - 1) < 3)) →
  a < -4 :=
by
  sorry

end sufficient_not_necessary_condition_l1516_151653


namespace polynomial_abc_l1516_151633

theorem polynomial_abc {a b c : ℝ} (h : a * x^2 + b * x + c = x^2 - 3 * x + 2) : a * b * c = -6 := by
  sorry

end polynomial_abc_l1516_151633


namespace frustum_midsection_area_l1516_151608

theorem frustum_midsection_area (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) :
  let r_mid := (r1 + r2) / 2
  let area_mid := Real.pi * r_mid^2
  area_mid = 25 * Real.pi / 4 := by
  sorry

end frustum_midsection_area_l1516_151608


namespace length_of_CD_l1516_151692

theorem length_of_CD
  (radius : ℝ)
  (length : ℝ)
  (total_volume : ℝ)
  (cylinder_volume : ℝ := π * radius^2 * length)
  (hemisphere_volume : ℝ := (2 * (2/3) * π * radius^3))
  (h1 : radius = 4)
  (h2 : total_volume = 432 * π)
  (h3 : total_volume = cylinder_volume + hemisphere_volume) :
  length = 22 := by
sorry

end length_of_CD_l1516_151692


namespace chromium_first_alloy_percentage_l1516_151637

-- Defining the conditions
def percentage_chromium_first_alloy : ℝ := 10 
def percentage_chromium_second_alloy : ℝ := 6
def mass_first_alloy : ℝ := 15
def mass_second_alloy : ℝ := 35
def percentage_chromium_new_alloy : ℝ := 7.2

-- Proving the percentage of chromium in the first alloy is 10%
theorem chromium_first_alloy_percentage : percentage_chromium_first_alloy = 10 :=
by
  sorry

end chromium_first_alloy_percentage_l1516_151637


namespace find_letter_l1516_151628

def consecutive_dates (A B C D E F G : ℕ) : Prop :=
  B = A + 1 ∧ C = A + 2 ∧ D = A + 3 ∧ E = A + 4 ∧ F = A + 5 ∧ G = A + 6

theorem find_letter (A B C D E F G : ℕ) 
  (h_consecutive : consecutive_dates A B C D E F G) 
  (h_condition : ∃ y, (B + y = 2 * A + 6)) :
  y = F :=
by
  sorry

end find_letter_l1516_151628


namespace specific_gravity_cylinder_l1516_151654

noncomputable def specific_gravity_of_cylinder (r m : ℝ) : ℝ :=
  (1 / 3) - (Real.sqrt 3 / (4 * Real.pi))

theorem specific_gravity_cylinder
  (r m : ℝ) 
  (cylinder_floats : r > 0 ∧ m > 0)
  (submersion_depth : r / 2 = r / 2) :
  specific_gravity_of_cylinder r m = 0.1955 :=
sorry

end specific_gravity_cylinder_l1516_151654


namespace tangent_line_at_point_l1516_151695

noncomputable def f : ℝ → ℝ := λ x => 2 * Real.log x + x^2 

def tangent_line_equation (x y : ℝ) : Prop :=
  4 * x - y - 3 = 0 

theorem tangent_line_at_point {x y : ℝ} (h : f 1 = 1) : 
  tangent_line_equation 1 1 ∧
  y = 4 * (x - 1) + 1 := 
sorry

end tangent_line_at_point_l1516_151695


namespace increasing_inverse_relation_l1516_151615

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry -- This is the inverse function f^-1

theorem increasing_inverse_relation {a b c : ℝ} 
  (h_inc_f : ∀ x y, x < y → f x < f y)
  (h_inc_f_inv : ∀ x y, x < y → f_inv x < f_inv y)
  (h_f3 : f 3 = 0)
  (h_f2 : f 2 = a)
  (h_f_inv2 : f_inv 2 = b)
  (h_f_inv0 : f_inv 0 = c) :
  b > c ∧ c > a := sorry

end increasing_inverse_relation_l1516_151615


namespace largest_n_arithmetic_sequences_l1516_151622

theorem largest_n_arithmetic_sequences
  (a : ℕ → ℤ) (b : ℕ → ℤ) (x y : ℤ)
  (a_1 : a 1 = 2) (b_1 : b 1 = 3)
  (a_formula : ∀ n : ℕ, a n = 2 + (n - 1) * x)
  (b_formula : ∀ n : ℕ, b n = 3 + (n - 1) * y)
  (x_lt_y : x < y)
  (product_condition : ∃ n : ℕ, a n * b n = 1638) :
  ∃ n : ℕ, a n * b n = 1638 ∧ n = 35 := 
sorry

end largest_n_arithmetic_sequences_l1516_151622


namespace difference_of_x_values_l1516_151657

theorem difference_of_x_values : 
  ∀ x y : ℝ, ( (x + 3) ^ 2 / (3 * x + 29) = 2 ∧ (y + 3) ^ 2 / (3 * y + 29) = 2 ) → |x - y| = 14 := 
sorry

end difference_of_x_values_l1516_151657


namespace simplify_fraction_l1516_151651

theorem simplify_fraction (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c :=
by
  sorry

end simplify_fraction_l1516_151651


namespace cannot_determine_x_l1516_151629

theorem cannot_determine_x
  (n m : ℝ) (x : ℝ)
  (h1 : n + m = 8) 
  (h2 : n * x + m * (1/5) = 1) : true :=
by {
  sorry
}

end cannot_determine_x_l1516_151629


namespace problem_1_problem_2_l1516_151607

-- Problem (1)
theorem problem_1 (a c : ℝ) (h1 : ∀ x, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c > 0) :
  ∃ s, s = { x | -2 < x ∧ x < 3 } ∧ (∀ x, x ∈ s → cx^2 - 2*x + a < 0) := 
sorry

-- Problem (2)
theorem problem_2 (m : ℝ) (h : ∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) :
  m < 4 := 
sorry

end problem_1_problem_2_l1516_151607


namespace locus_of_midpoint_l1516_151673

theorem locus_of_midpoint {P Q M : ℝ × ℝ} (hP_on_circle : P.1^2 + P.2^2 = 13)
  (hQ_perpendicular_to_y_axis : Q.1 = P.1) (h_midpoint : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) :
  (M.1^2 / (13 / 4)) + (M.2^2 / 13) = 1 := 
sorry

end locus_of_midpoint_l1516_151673


namespace xy_value_l1516_151656

variable (a b x y : ℝ)
variable (h1 : 2 * a^x * b^3 = - a^2 * b^(1 - y))
variable (hx : x = 2)
variable (hy : y = -2)

theorem xy_value : x * y = -4 := 
by
  sorry

end xy_value_l1516_151656


namespace correct_proposition_l1516_151669

theorem correct_proposition :
  (∃ x₀ : ℤ, x₀^2 = 1) ∧ ¬(∃ x₀ : ℤ, x₀^2 < 0) ∧ ¬(∀ x : ℤ, x^2 ≤ 0) ∧ ¬(∀ x : ℤ, x^2 ≥ 1) :=
by
  sorry

end correct_proposition_l1516_151669


namespace z_value_l1516_151676

theorem z_value (x y z : ℝ) (h : 1 / x + 1 / y = 2 / z) : z = (x * y) / 2 :=
by
  sorry

end z_value_l1516_151676


namespace sequence_diff_l1516_151611

theorem sequence_diff (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hSn : ∀ n, S n = n^2)
  (hS1 : a 1 = S 1)
  (ha_n : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 3 - a 2 = 2 := sorry

end sequence_diff_l1516_151611
