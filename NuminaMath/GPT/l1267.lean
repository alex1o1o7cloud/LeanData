import Mathlib

namespace NUMINAMATH_GPT_minimal_ab_l1267_126709

theorem minimal_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
(h : 1 / (a : ℝ) + 1 / (3 * b : ℝ) = 1 / 9) : a * b = 60 :=
sorry

end NUMINAMATH_GPT_minimal_ab_l1267_126709


namespace NUMINAMATH_GPT_value_bounds_of_expression_l1267_126752

theorem value_bounds_of_expression
  (a b c : ℝ)
  (ha : 0 ≤ a)
  (hab : a ≤ b)
  (hbc : b ≤ c)
  (triangle_ineq1 : a + b > c)
  (triangle_ineq2 : a + c > b)
  (triangle_ineq3 : b + c > a)
  : 4 ≤ (a+b+c)^2 / (b*c) ∧ (a+b+c)^2 / (b*c) ≤ 9 := sorry

end NUMINAMATH_GPT_value_bounds_of_expression_l1267_126752


namespace NUMINAMATH_GPT_segment_AC_length_l1267_126787

noncomputable def circle_radius := 8
noncomputable def chord_length_AB := 10
noncomputable def arc_length_AC (circumference : ℝ) := circumference / 3

theorem segment_AC_length :
  ∀ (C : ℝ) (r : ℝ) (AB : ℝ) (AC : ℝ),
    r = circle_radius →
    AB = chord_length_AB →
    C = 2 * Real.pi * r →
    AC = arc_length_AC C →
    AC = 8 * Real.sqrt 3 :=
by
  intros C r AB AC hr hAB hC hAC
  sorry

end NUMINAMATH_GPT_segment_AC_length_l1267_126787


namespace NUMINAMATH_GPT_smallest_percent_both_l1267_126753

theorem smallest_percent_both (S J : ℝ) (hS : S = 0.9) (hJ : J = 0.8) : 
  ∃ B, B = S + J - 1 ∧ B = 0.7 :=
by
  sorry

end NUMINAMATH_GPT_smallest_percent_both_l1267_126753


namespace NUMINAMATH_GPT_tangent_lines_to_circle_l1267_126712

theorem tangent_lines_to_circle 
  (x y : ℝ) 
  (circle : (x - 2) ^ 2 + (y + 1) ^ 2 = 1) 
  (point : x = 3 ∧ y = 3) : 
  (x = 3 ∨ 15 * x - 8 * y - 21 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_lines_to_circle_l1267_126712


namespace NUMINAMATH_GPT_math_problem_l1267_126707

theorem math_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a + 2) * (b + 2) = 18) :
  (∀ x, (x = 3 / (a + 2) + 3 / (b + 2)) → x ≥ Real.sqrt 2) ∧
  ¬(∃ y, (y = a * b) ∧ y ≤ 11 - 6 * Real.sqrt 2) ∧
  (∀ z, (z = 2 * a + b) → z ≥ 6) ∧
  (∀ w, (w = (a + 1) * b) → w ≤ 8) :=
sorry

end NUMINAMATH_GPT_math_problem_l1267_126707


namespace NUMINAMATH_GPT_find_larger_number_l1267_126705

theorem find_larger_number (x y : ℕ) (h1 : y = 2 * x - 3) (h2 : x + y = 51) : y = 33 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l1267_126705


namespace NUMINAMATH_GPT_smallest_pos_int_ending_in_9_divisible_by_13_l1267_126797

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_pos_int_ending_in_9_divisible_by_13_l1267_126797


namespace NUMINAMATH_GPT_car_speeds_l1267_126765

-- Definitions and conditions
def distance_AB : ℝ := 200
def distance_meet : ℝ := 80
def car_A_speed : ℝ := sorry -- To Be Proved
def car_B_speed : ℝ := sorry -- To Be Proved

axiom car_B_faster (x : ℝ) : car_B_speed = car_A_speed + 30
axiom time_equal (x : ℝ) : (distance_meet / car_A_speed) = ((distance_AB - distance_meet) / car_B_speed)

-- Proof (only statement, without steps)
theorem car_speeds : car_A_speed = 60 ∧ car_B_speed = 90 :=
  by
  have car_A_speed := 60
  have car_B_speed := 90
  sorry

end NUMINAMATH_GPT_car_speeds_l1267_126765


namespace NUMINAMATH_GPT_predicted_yield_of_rice_l1267_126726

theorem predicted_yield_of_rice (x : ℝ) (h : x = 80) : 5 * x + 250 = 650 :=
by {
  sorry -- proof will be given later
}

end NUMINAMATH_GPT_predicted_yield_of_rice_l1267_126726


namespace NUMINAMATH_GPT_average_yield_per_tree_l1267_126733

theorem average_yield_per_tree :
  let t1 := 3
  let t2 := 2
  let t3 := 1
  let nuts1 := 60
  let nuts2 := 120
  let nuts3 := 180
  let total_nuts := t1 * nuts1 + t2 * nuts2 + t3 * nuts3
  let total_trees := t1 + t2 + t3
  let average_yield := total_nuts / total_trees
  average_yield = 100 := 
by
  sorry

end NUMINAMATH_GPT_average_yield_per_tree_l1267_126733


namespace NUMINAMATH_GPT_sum_transformed_roots_l1267_126703

theorem sum_transformed_roots :
  ∀ (a b c : ℝ),
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
  (45 * a^3 - 75 * a^2 + 33 * a - 2 = 0) →
  (45 * b^3 - 75 * b^2 + 33 * b - 2 = 0) →
  (45 * c^3 - 75 * c^2 + 33 * c - 2 = 0) →
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 60) :=
by
  intros a b c h_bounds h_poly_a h_poly_b h_poly_c h_distinct
  sorry

end NUMINAMATH_GPT_sum_transformed_roots_l1267_126703


namespace NUMINAMATH_GPT_find_f_neg_5pi_over_6_l1267_126768

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined_on_R : ∀ x : ℝ, ∃ y : ℝ, f y = f x
axiom f_periodic : ∀ x : ℝ, f (x + (3 * Real.pi / 2)) = f x
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x → x ≤ Real.pi → f x = Real.cos x

theorem find_f_neg_5pi_over_6 : f (-5 * Real.pi / 6) = -1 / 2 := 
by 
  -- use the axioms to prove the result 
  sorry

end NUMINAMATH_GPT_find_f_neg_5pi_over_6_l1267_126768


namespace NUMINAMATH_GPT_whitewash_all_planks_not_whitewash_all_planks_l1267_126761

open Finset

variable {N : ℕ} (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1))

def f (n : ℤ) : ℤ := n^2 + 3*n - 2

def f_equiv (x y : ℤ) : Prop := 2^(Nat.log2 (2 * N)) ∣ (f x - f y)

theorem whitewash_all_planks (N : ℕ) (is_power_of_two : ∃ (k : ℕ), N = 2^(k + 1)) : 
  ∀ n ∈ range N, ∃ m ∈ range N, f m = n :=
by {
  sorry
}

theorem not_whitewash_all_planks (N : ℕ) (not_power_of_two : ¬(∃ (k : ℕ), N = 2^(k + 1))) : 
  ∃ n ∈ range N, ∀ m ∈ range N, f m ≠ n :=
by {
  sorry
}

end NUMINAMATH_GPT_whitewash_all_planks_not_whitewash_all_planks_l1267_126761


namespace NUMINAMATH_GPT_probability_circle_or_square_l1267_126774

theorem probability_circle_or_square (total_figures : ℕ)
    (num_circles : ℕ) (num_squares : ℕ) (num_triangles : ℕ)
    (total_figures_eq : total_figures = 10)
    (num_circles_eq : num_circles = 3)
    (num_squares_eq : num_squares = 4)
    (num_triangles_eq : num_triangles = 3) :
    (num_circles + num_squares) / total_figures = 7 / 10 :=
by sorry

end NUMINAMATH_GPT_probability_circle_or_square_l1267_126774


namespace NUMINAMATH_GPT_probability_neither_red_nor_purple_correct_l1267_126745

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def yellow_balls : ℕ := 7
def red_balls : ℕ := 15
def purple_balls : ℕ := 6

def neither_red_nor_purple_balls : ℕ := total_balls - (red_balls + purple_balls)
def probability_neither_red_nor_purple : ℚ := (neither_red_nor_purple_balls : ℚ) / (total_balls : ℚ)

theorem probability_neither_red_nor_purple_correct : 
  probability_neither_red_nor_purple = 13 / 20 := 
by sorry

end NUMINAMATH_GPT_probability_neither_red_nor_purple_correct_l1267_126745


namespace NUMINAMATH_GPT_initial_mean_corrected_l1267_126720

theorem initial_mean_corrected
  (M : ℝ)
  (h : 30 * M + 10 = 30 * 140.33333333333334) :
  M = 140 :=
by
  sorry

end NUMINAMATH_GPT_initial_mean_corrected_l1267_126720


namespace NUMINAMATH_GPT_average_investment_per_km_in_scientific_notation_l1267_126799

-- Definitions based on the conditions of the problem
def total_investment : ℝ := 29.6 * 10^9
def upgraded_distance : ℝ := 6000

-- A theorem to be proven
theorem average_investment_per_km_in_scientific_notation :
  (total_investment / upgraded_distance) = 4.9 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_average_investment_per_km_in_scientific_notation_l1267_126799


namespace NUMINAMATH_GPT_total_seashells_l1267_126758

def joans_seashells : Nat := 6
def jessicas_seashells : Nat := 8

theorem total_seashells : joans_seashells + jessicas_seashells = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_seashells_l1267_126758


namespace NUMINAMATH_GPT_semesters_per_year_l1267_126778

-- Definitions of conditions
def cost_per_semester : ℕ := 20000
def total_cost_13_years : ℕ := 520000
def years : ℕ := 13

-- Main theorem to prove
theorem semesters_per_year (S : ℕ) (h1 : total_cost_13_years = years * (S * cost_per_semester)) : S = 2 := by
  sorry

end NUMINAMATH_GPT_semesters_per_year_l1267_126778


namespace NUMINAMATH_GPT_portion_of_work_done_l1267_126777

variable (P W : ℕ)

-- Given conditions
def work_rate_P (P W : ℕ) : ℕ := W / 16
def work_rate_2P (P W : ℕ) : ℕ := 2 * (work_rate_P P W)

-- Lean theorem
theorem portion_of_work_done (h : work_rate_2P P W * 4 = W / 2) : 
    work_rate_2P P W * 4 = W / 2 := 
by 
  sorry

end NUMINAMATH_GPT_portion_of_work_done_l1267_126777


namespace NUMINAMATH_GPT_ab_non_positive_l1267_126769

-- Define the conditions as a structure if necessary.
variables {a b : ℝ}

-- State the theorem.
theorem ab_non_positive (h : 3 * a + 8 * b = 0) : a * b ≤ 0 :=
sorry

end NUMINAMATH_GPT_ab_non_positive_l1267_126769


namespace NUMINAMATH_GPT_bowling_ball_weight_l1267_126741

theorem bowling_ball_weight (b c : ℝ) (h1 : 5 * b = 3 * c) (h2 : 2 * c = 56) : b = 16.8 := by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l1267_126741


namespace NUMINAMATH_GPT_geometric_proportion_l1267_126789

theorem geometric_proportion (a b c d : ℝ) (h1 : a / b = c / d) (h2 : a / b = d / c) :
  (a = b ∧ b = c ∧ c = d) ∨ (|a| = |b| ∧ |b| = |c| ∧ |c| = |d| ∧ (a * b * c * d < 0)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_proportion_l1267_126789


namespace NUMINAMATH_GPT_ratio_pat_mark_l1267_126773

theorem ratio_pat_mark (P K M : ℕ) (h1 : P + K + M = 180) 
  (h2 : P = 2 * K) (h3 : M = K + 100) : P / gcd P M = 1 ∧ M / gcd P M = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_pat_mark_l1267_126773


namespace NUMINAMATH_GPT_verify_equation_holds_l1267_126760

noncomputable def verify_equation (m n : ℝ) : Prop :=
  1.55 * Real.sqrt (6 * m + 2 * Real.sqrt (9 * m^2 - n^2)) 
  - Real.sqrt (6 * m - 2 * Real.sqrt (9 * m^2 - n^2)) 
  = 2 * Real.sqrt (3 * m - n)

theorem verify_equation_holds (m n : ℝ) (h : 9 * m^2 - n^2 ≥ 0) : verify_equation m n :=
by
  -- Proof goes here. 
  -- Implement the proof as per the solution steps sketched in the problem statement.
  sorry

end NUMINAMATH_GPT_verify_equation_holds_l1267_126760


namespace NUMINAMATH_GPT_abs_ineq_range_m_l1267_126716

theorem abs_ineq_range_m :
  ∀ m : ℝ, (∀ x : ℝ, |x - 1| + |x + 2| ≥ m) ↔ m ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_range_m_l1267_126716


namespace NUMINAMATH_GPT_josh_money_left_l1267_126786

def initial_amount : ℝ := 9
def spent_on_drink : ℝ := 1.75
def spent_on_item : ℝ := 1.25

theorem josh_money_left : initial_amount - (spent_on_drink + spent_on_item) = 6 := by
  sorry

end NUMINAMATH_GPT_josh_money_left_l1267_126786


namespace NUMINAMATH_GPT_range_of_m_l1267_126727

theorem range_of_m (m : ℝ) : ((m + 3 > 0) ∧ (m - 1 < 0)) ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1267_126727


namespace NUMINAMATH_GPT_sum_gcd_lcm_is_159_l1267_126762

-- Definitions for GCD and LCM for specific values
def gcd_45_75 := Int.gcd 45 75
def lcm_48_18 := Int.lcm 48 18

-- The proof problem statement
theorem sum_gcd_lcm_is_159 : gcd_45_75 + lcm_48_18 = 159 := by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_is_159_l1267_126762


namespace NUMINAMATH_GPT_probability_losing_ticket_l1267_126767

theorem probability_losing_ticket (winning : ℕ) (losing : ℕ)
  (h_odds : winning = 5 ∧ losing = 8) :
  (losing : ℚ) / (winning + losing : ℚ) = 8 / 13 := by
  sorry

end NUMINAMATH_GPT_probability_losing_ticket_l1267_126767


namespace NUMINAMATH_GPT_students_and_swimmers_l1267_126719

theorem students_and_swimmers (N : ℕ) (x : ℕ) 
  (h1 : x = N / 4) 
  (h2 : x / 2 = 4) : 
  N = 32 ∧ N - x = 24 := 
by 
  sorry

end NUMINAMATH_GPT_students_and_swimmers_l1267_126719


namespace NUMINAMATH_GPT_total_blood_cells_correct_l1267_126746

-- Define the number of blood cells in the first and second samples.
def sample_1_blood_cells : ℕ := 4221
def sample_2_blood_cells : ℕ := 3120

-- Define the total number of blood cells.
def total_blood_cells : ℕ := sample_1_blood_cells + sample_2_blood_cells

-- Theorem stating the total number of blood cells based on the conditions.
theorem total_blood_cells_correct : total_blood_cells = 7341 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_blood_cells_correct_l1267_126746


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_sum_l1267_126742

theorem arithmetic_sequence_terms_sum
  (a : ℕ → ℝ)
  (h₁ : ∀ n, a (n+1) = a n + d)
  (h₂ : a 2 = 1 - a 1)
  (h₃ : a 4 = 9 - a 3)
  (h₄ : ∀ n, a n > 0):
  a 4 + a 5 = 27 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_sum_l1267_126742


namespace NUMINAMATH_GPT_find_age_of_b_l1267_126750

-- Definitions for the conditions
def is_two_years_older (a b : ℕ) : Prop := a = b + 2
def is_twice_as_old (b c : ℕ) : Prop := b = 2 * c
def total_age (a b c : ℕ) : Prop := a + b + c = 12

-- Proof statement
theorem find_age_of_b (a b c : ℕ) 
  (h1 : is_two_years_older a b) 
  (h2 : is_twice_as_old b c) 
  (h3 : total_age a b c) : 
  b = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_age_of_b_l1267_126750


namespace NUMINAMATH_GPT_min_value_l1267_126730

theorem min_value : ∀ (a b : ℝ), a + b^2 = 2 → (∀ x y : ℝ, x = a^2 + 6 * y^2 → y = b) → (∃ c : ℝ, c = 3) :=
by
  intros a b h₁ h₂
  sorry

end NUMINAMATH_GPT_min_value_l1267_126730


namespace NUMINAMATH_GPT_find_p_plus_q_l1267_126779

noncomputable def p (d e : ℝ) (x : ℝ) : ℝ := d * x + e
noncomputable def q (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_p_plus_q (d e a b c : ℝ)
  (h1 : p d e 0 / q a b c 0 = 4)
  (h2 : p d e (-1) = -1)
  (h3 : q a b c 1 = 3)
  (e_eq : e = 4 * c):
  (p d e x + q a b c x) = (3*x^2 + 26*x - 30) :=
by
  sorry

end NUMINAMATH_GPT_find_p_plus_q_l1267_126779


namespace NUMINAMATH_GPT_lcm_of_4_9_10_27_l1267_126775

theorem lcm_of_4_9_10_27 : Nat.lcm (Nat.lcm 4 9) (Nat.lcm 10 27) = 540 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_4_9_10_27_l1267_126775


namespace NUMINAMATH_GPT_range_of_m_l1267_126766

theorem range_of_m (m : ℝ) :
  (∀ x y : ℝ, (y = (m - 2) * x + m - 1 → (x ≥ 0 ∨ y ≥ 0))) ↔ (1 ≤ m ∧ m < 2) :=
by sorry

end NUMINAMATH_GPT_range_of_m_l1267_126766


namespace NUMINAMATH_GPT_number_of_substitution_ways_mod_1000_l1267_126780

theorem number_of_substitution_ways_mod_1000 :
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  total_ways % 1000 = 573 := by
  -- Definition
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_substitution_ways_mod_1000_l1267_126780


namespace NUMINAMATH_GPT_range_of_m_l1267_126756

noncomputable def abs_sum (x : ℝ) : ℝ := |x - 5| + |x - 3|

theorem range_of_m (m : ℝ) : (∃ x : ℝ, abs_sum x < m) ↔ m > 2 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1267_126756


namespace NUMINAMATH_GPT_solve_inequality_system_l1267_126798

theorem solve_inequality_system (x : ℝ) (h1 : x - 2 ≤ 0) (h2 : (x - 1) / 2 < x) : -1 < x ∧ x ≤ 2 := 
sorry

end NUMINAMATH_GPT_solve_inequality_system_l1267_126798


namespace NUMINAMATH_GPT_problem_l1267_126701

variable (a b : ℝ)

theorem problem (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 :=
by
  sorry

end NUMINAMATH_GPT_problem_l1267_126701


namespace NUMINAMATH_GPT_dan_destroyed_l1267_126710

def balloons_initial (fred: ℝ) (sam: ℝ) : ℝ := fred + sam

theorem dan_destroyed (fred: ℝ) (sam: ℝ) (final_balloons: ℝ) (destroyed_balloons: ℝ) :
  fred = 10.0 →
  sam = 46.0 →
  final_balloons = 40.0 →
  destroyed_balloons = (balloons_initial fred sam) - final_balloons →
  destroyed_balloons = 16.0 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_dan_destroyed_l1267_126710


namespace NUMINAMATH_GPT_mila_hours_to_match_agnes_monthly_earnings_l1267_126735

-- Definitions based on given conditions
def hourly_rate_mila : ℕ := 10
def hourly_rate_agnes : ℕ := 15
def weekly_hours_agnes : ℕ := 8
def weeks_in_month : ℕ := 4

-- Target statement to prove: Mila needs to work 48 hours to earn as much as Agnes in a month
theorem mila_hours_to_match_agnes_monthly_earnings :
  ∃ (h : ℕ), h = 48 ∧ (h * hourly_rate_mila) = (hourly_rate_agnes * weekly_hours_agnes * weeks_in_month) :=
by
  sorry

end NUMINAMATH_GPT_mila_hours_to_match_agnes_monthly_earnings_l1267_126735


namespace NUMINAMATH_GPT_find_cost_of_apple_l1267_126791

theorem find_cost_of_apple (A O : ℝ) 
  (h1 : 6 * A + 3 * O = 1.77) 
  (h2 : 2 * A + 5 * O = 1.27) : 
  A = 0.21 :=
by 
  sorry

end NUMINAMATH_GPT_find_cost_of_apple_l1267_126791


namespace NUMINAMATH_GPT_bonus_distribution_plans_l1267_126748

theorem bonus_distribution_plans (x y : ℕ) (A B : ℕ) 
  (h1 : x + y = 15)
  (h2 : x = 2 * y)
  (h3 : 10 * A + 5 * B = 20000)
  (hA : A ≥ B)
  (hB : B ≥ 800)
  (hAB_mult_100 : ∃ (k m : ℕ), A = k * 100 ∧ B = m * 100) :
  (x = 10 ∧ y = 5) ∧
  ((A = 1600 ∧ B = 800) ∨
   (A = 1500 ∧ B = 1000) ∨
   (A = 1400 ∧ B = 1200)) :=
by
  -- The proof should be provided here
  sorry

end NUMINAMATH_GPT_bonus_distribution_plans_l1267_126748


namespace NUMINAMATH_GPT_not_divisible_by_n_only_prime_3_l1267_126747

-- Problem 1: Prove that for any natural number \( n \) greater than 1, \( 2^n - 1 \) is not divisible by \( n \)
theorem not_divisible_by_n (n : ℕ) (h1 : 1 < n) : ¬ (n ∣ (2^n - 1)) :=
sorry

-- Problem 2: Prove that the only prime number \( n \) such that \( 2^n + 1 \) is divisible by \( n^2 \) is \( n = 3 \)
theorem only_prime_3 (n : ℕ) (hn : Nat.Prime n) (hdiv : n^2 ∣ (2^n + 1)) : n = 3 :=
sorry

end NUMINAMATH_GPT_not_divisible_by_n_only_prime_3_l1267_126747


namespace NUMINAMATH_GPT_even_function_solution_l1267_126723

theorem even_function_solution :
  ∀ (m : ℝ), (∀ x : ℝ, (m+1) * x^2 + (m-2) * x = (m+1) * x^2 - (m-2) * x) → (m = 2 ∧ ∀ x : ℝ, (2+1) * x^2 + (2-2) * x = 3 * x^2) :=
by
  sorry

end NUMINAMATH_GPT_even_function_solution_l1267_126723


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l1267_126737

theorem circumscribed_circle_radius (h8 h15 h17 : ℝ) (h_triangle : h8 = 8 ∧ h15 = 15 ∧ h17 = 17) : 
  ∃ R : ℝ, R = 17 := 
sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l1267_126737


namespace NUMINAMATH_GPT_find_m_value_l1267_126795

theorem find_m_value (m : ℚ) :
  (m - 10) / -10 = (5 - m) / -8 → m = 65 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l1267_126795


namespace NUMINAMATH_GPT_distance_between_points_l1267_126744

theorem distance_between_points:
  dist (0, 4) (3, 0) = 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1267_126744


namespace NUMINAMATH_GPT_angle_B_eq_18_l1267_126764

theorem angle_B_eq_18 
  (A B : ℝ) 
  (h1 : A = 4 * B) 
  (h2 : 90 - B = 4 * (90 - A)) : 
  B = 18 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_eq_18_l1267_126764


namespace NUMINAMATH_GPT_number_of_boundaries_l1267_126755

theorem number_of_boundaries 
  (total_runs : ℕ) 
  (number_of_sixes : ℕ) 
  (percentage_runs_by_running : ℝ) 
  (runs_per_six : ℕ) 
  (runs_per_boundary : ℕ)
  (h_total_runs : total_runs = 125)
  (h_number_of_sixes : number_of_sixes = 5)
  (h_percentage_runs_by_running : percentage_runs_by_running = 0.60)
  (h_runs_per_six : runs_per_six = 6)
  (h_runs_per_boundary : runs_per_boundary = 4) :
  (total_runs - percentage_runs_by_running * total_runs - number_of_sixes * runs_per_six) / runs_per_boundary = 5 := by 
  sorry

end NUMINAMATH_GPT_number_of_boundaries_l1267_126755


namespace NUMINAMATH_GPT_roots_quadratic_diff_by_12_l1267_126754

theorem roots_quadratic_diff_by_12 (P : ℝ) : 
  (∀ α β : ℝ, (α + β = 2) ∧ (α * β = -P) ∧ ((α - β) = 12)) → P = 35 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_roots_quadratic_diff_by_12_l1267_126754


namespace NUMINAMATH_GPT_range_of_a_l1267_126706

noncomputable def P (a : ℝ) : Prop :=
∀ x : ℝ, a * x^2 + a * x + 1 > 0

noncomputable def Q (a : ℝ) : Prop :=
(∃ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1)) ∧ ∀ (x y : ℝ), (x^2 / a + y^2 / (a - 3) = 1) → (a * (a - 3) < 0)

theorem range_of_a (a : ℝ) (h1 : P a ∨ Q a) (h2 : ¬ (P a ∧ Q a)) : a = 0 ∨ (3 ≤ a ∧ a < 4) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1267_126706


namespace NUMINAMATH_GPT_find_number_l1267_126770

-- Given conditions and declarations
variable (x : ℕ)
variable (h : x / 3 = x - 42)

-- Proof problem statement
theorem find_number : x = 63 := 
sorry

end NUMINAMATH_GPT_find_number_l1267_126770


namespace NUMINAMATH_GPT_junior_high_ten_total_games_l1267_126714

theorem junior_high_ten_total_games :
  let teams := 10
  let conference_games_per_team := 3
  let non_conference_games_per_team := 5
  let pairs_of_teams := Nat.choose teams 2
  let total_conference_games := pairs_of_teams * conference_games_per_team
  let total_non_conference_games := teams * non_conference_games_per_team
  let total_games := total_conference_games + total_non_conference_games
  total_games = 185 :=
by
  sorry

end NUMINAMATH_GPT_junior_high_ten_total_games_l1267_126714


namespace NUMINAMATH_GPT_sum_of_roots_l1267_126759

theorem sum_of_roots (x1 x2 : ℝ) (h1 : x1^2 + 5*x1 - 3 = 0) (h2 : x2^2 + 5*x2 - 3 = 0) (h3 : x1 ≠ x2) :
  x1 + x2 = -5 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1267_126759


namespace NUMINAMATH_GPT_parabola_point_comparison_l1267_126794

theorem parabola_point_comparison :
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  y1 < y2 :=
by
  let y1 := (1: ℝ)^2 - 2 * (1: ℝ) - 2
  let y2 := (3: ℝ)^2 - 2 * (3: ℝ) - 2
  have h : y1 < y2 := by sorry
  exact h

end NUMINAMATH_GPT_parabola_point_comparison_l1267_126794


namespace NUMINAMATH_GPT_candy_remaining_l1267_126713

theorem candy_remaining
  (initial_candies : ℕ)
  (talitha_took : ℕ)
  (solomon_took : ℕ)
  (h_initial : initial_candies = 349)
  (h_talitha : talitha_took = 108)
  (h_solomon : solomon_took = 153) :
  initial_candies - (talitha_took + solomon_took) = 88 :=
by
  sorry

end NUMINAMATH_GPT_candy_remaining_l1267_126713


namespace NUMINAMATH_GPT_sin_beta_acute_l1267_126751

theorem sin_beta_acute (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosα : Real.cos α = 4 / 5)
  (hcosαβ : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end NUMINAMATH_GPT_sin_beta_acute_l1267_126751


namespace NUMINAMATH_GPT_solve_fractions_l1267_126700

theorem solve_fractions : 
  ∃ (X Y : ℕ), 
    (5 + 1 / (X : ℝ)) * (Y + 1 / 2) = 43 ∧ X = 17 ∧ Y = 8 :=
by
  use 17, 8
  rw [←@Rat.cast_coe_nat ℝ _ 17, ←@Rat.cast_coe_nat ℝ _ 8]
  norm_num

end NUMINAMATH_GPT_solve_fractions_l1267_126700


namespace NUMINAMATH_GPT_max_board_size_l1267_126784

theorem max_board_size : ∀ (n : ℕ), 
  (∃ (board : Fin n → Fin n → Prop),
    ∀ i j k l : Fin n,
      (i ≠ k ∧ j ≠ l) → board i j ≠ board k l) ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_GPT_max_board_size_l1267_126784


namespace NUMINAMATH_GPT_g_at_5_l1267_126796

def g (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 28 * x^2 - 20 * x - 80

theorem g_at_5 : g 5 = -5 := 
  by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_g_at_5_l1267_126796


namespace NUMINAMATH_GPT_grace_earnings_l1267_126704

noncomputable def weekly_charge : ℕ := 300
noncomputable def payment_interval : ℕ := 2
noncomputable def target_weeks : ℕ := 6
noncomputable def target_amount : ℕ := 1800

theorem grace_earnings :
  (target_weeks * weekly_charge = target_amount) → 
  (target_weeks / payment_interval) * (payment_interval * weekly_charge) = target_amount :=
by
  sorry

end NUMINAMATH_GPT_grace_earnings_l1267_126704


namespace NUMINAMATH_GPT_probability_of_drawing_white_ball_l1267_126785

def total_balls (red white : ℕ) : ℕ := red + white

def number_of_white_balls : ℕ := 2

def number_of_red_balls : ℕ := 3

def probability_of_white_ball (white total : ℕ) : ℚ := white / total

-- Theorem statement
theorem probability_of_drawing_white_ball :
  probability_of_white_ball number_of_white_balls (total_balls number_of_red_balls number_of_white_balls) = 2 / 5 :=
sorry

end NUMINAMATH_GPT_probability_of_drawing_white_ball_l1267_126785


namespace NUMINAMATH_GPT_store_incur_loss_of_one_percent_l1267_126781

theorem store_incur_loss_of_one_percent
    (a b x : ℝ)
    (h1 : x = a * 1.1)
    (h2 : x = b * 0.9)
    : (2 * x - (a + b)) / (a + b) = -0.01 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_store_incur_loss_of_one_percent_l1267_126781


namespace NUMINAMATH_GPT_gcd_polynomial_multiple_of_345_l1267_126772

theorem gcd_polynomial_multiple_of_345 (b : ℕ) (h : ∃ k : ℕ, b = 345 * k) : 
  Nat.gcd (5 * b ^ 3 + 2 * b ^ 2 + 7 * b + 69) b = 69 := 
by
  sorry

end NUMINAMATH_GPT_gcd_polynomial_multiple_of_345_l1267_126772


namespace NUMINAMATH_GPT_weight_ratio_l1267_126708

-- Conditions
def initial_weight : ℕ := 99
def initial_loss : ℕ := 12
def weight_added_back (x : ℕ) : Prop := x = 81 + 30 - initial_weight
def times_lost : ℕ := 3 * initial_loss
def final_gain : ℕ := 6
def final_weight : ℕ := 81

-- Question
theorem weight_ratio (x : ℕ)
  (H1 : weight_added_back x)
  (H2 : initial_weight - initial_loss + x - times_lost + final_gain = final_weight) :
  x / initial_loss = 2 := by
  sorry

end NUMINAMATH_GPT_weight_ratio_l1267_126708


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1267_126722

theorem quadratic_inequality_solution (x : ℝ) : (2 * x^2 - 5 * x - 3 < 0) ↔ (-1/2 < x ∧ x < 3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1267_126722


namespace NUMINAMATH_GPT_X_is_N_l1267_126715

theorem X_is_N (X : Set ℕ) (h_nonempty : ∃ x, x ∈ X)
  (h_condition1 : ∀ x ∈ X, 4 * x ∈ X)
  (h_condition2 : ∀ x ∈ X, Nat.floor (Real.sqrt x) ∈ X) : 
  X = Set.univ := 
sorry

end NUMINAMATH_GPT_X_is_N_l1267_126715


namespace NUMINAMATH_GPT_only_n1_makes_n4_plus4_prime_l1267_126721

theorem only_n1_makes_n4_plus4_prime (n : ℕ) (h : n > 0) : (n = 1) ↔ Prime (n^4 + 4) :=
sorry

end NUMINAMATH_GPT_only_n1_makes_n4_plus4_prime_l1267_126721


namespace NUMINAMATH_GPT_ratio_proof_l1267_126788

variable {x y : ℝ}

theorem ratio_proof (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_proof_l1267_126788


namespace NUMINAMATH_GPT_natalie_needs_10_bushes_l1267_126725

-- Definitions based on the conditions
def bushes_to_containers (bushes : ℕ) := bushes * 10
def containers_to_zucchinis (containers : ℕ) := (containers * 3) / 4

-- The proof statement
theorem natalie_needs_10_bushes :
  ∃ bushes : ℕ, containers_to_zucchinis (bushes_to_containers bushes) ≥ 72 ∧ bushes = 10 :=
sorry

end NUMINAMATH_GPT_natalie_needs_10_bushes_l1267_126725


namespace NUMINAMATH_GPT_original_number_l1267_126792

-- Define the original statement and conditions
theorem original_number (x : ℝ) (h : 3 * (2 * x + 9) = 81) : x = 9 := by
  -- Sorry placeholder stands for the proof steps
  sorry

end NUMINAMATH_GPT_original_number_l1267_126792


namespace NUMINAMATH_GPT_first_number_percentage_of_second_l1267_126702

theorem first_number_percentage_of_second {X : ℝ} (H1 : ℝ) (H2 : ℝ) 
  (H1_def : H1 = 0.05 * X) (H2_def : H2 = 0.25 * X) : 
  (H1 / H2) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_first_number_percentage_of_second_l1267_126702


namespace NUMINAMATH_GPT_cos_neg245_l1267_126732

-- Define the given condition and declare the theorem to prove the required equality
variable (a : ℝ)
def cos_25_eq_a : Prop := (Real.cos 25 * Real.pi / 180 = a)

theorem cos_neg245 :
  cos_25_eq_a a → Real.cos (-245 * Real.pi / 180) = -Real.sqrt (1 - a^2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cos_neg245_l1267_126732


namespace NUMINAMATH_GPT_find_n_l1267_126717

theorem find_n (n k : ℕ) (a b : ℝ) (h_pos : k > 0) (h_n : n ≥ 2) (h_ab_neq : a ≠ 0 ∧ b ≠ 0) (h_a : a = (k + 1) * b) : n = 2 * k + 2 :=
by sorry

end NUMINAMATH_GPT_find_n_l1267_126717


namespace NUMINAMATH_GPT_min_value_expression_ge_072_l1267_126749

theorem min_value_expression_ge_072 (x y z : ℝ) 
  (hx : -0.5 ≤ x ∧ x ≤ 0.5) 
  (hy : |y| ≤ 0.5) 
  (hz : 0 ≤ z ∧ z < 1) :
  ((1 / ((1 - x) * (1 - y) * (1 - z))) - (1 / ((2 + x) * (2 + y) * (2 + z)))) ≥ 0.72 := sorry

end NUMINAMATH_GPT_min_value_expression_ge_072_l1267_126749


namespace NUMINAMATH_GPT_initial_pennies_l1267_126757

-- Defining the conditions
def pennies_spent : Nat := 93
def pennies_left : Nat := 5

-- Question: How many pennies did Sam have in his bank initially?
theorem initial_pennies : pennies_spent + pennies_left = 98 := by
  sorry

end NUMINAMATH_GPT_initial_pennies_l1267_126757


namespace NUMINAMATH_GPT_no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l1267_126739

-- Part (a): Prove that it is impossible to arrange five distinct-sized squares to form a rectangle.
theorem no_rectangle_with_five_distinct_squares (s1 s2 s3 s4 s5 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s4 ≠ s5) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5)) :=
by
  -- Proof placeholder
  sorry

-- Part (b): Prove that it is impossible to arrange six distinct-sized squares to form a rectangle.
theorem no_rectangle_with_six_distinct_squares (s1 s2 s3 s4 s5 s6 : ℕ) 
  (dist : s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧ s1 ≠ s6 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧ s2 ≠ s6 ∧ s3 ≠ s4 ∧ s3 ≠ s5 ∧ s3 ≠ s6 ∧ s4 ≠ s5 ∧ s4 ≠ s6 ∧ s5 ≠ s6) :
  ¬ ∃ (l w : ℕ), (s1 ≤ l ∧ s1 ≤ w) ∧ (s2 ≤ l ∧ s2 ≤ w) ∧ (s3 ≤ l ∧ s3 ≤ w) ∧ (s4 ≤ l ∧ s4 ≤ w) ∧ (s5 ≤ l ∧ s5 ≤ w) ∧ (s6 ≤ l ∧ s6 ≤ w) ∧
  (l * w = (s1 + s2 + s3 + s4 + s5 + s6)) :=
by
  -- Proof placeholder
  sorry

end NUMINAMATH_GPT_no_rectangle_with_five_distinct_squares_no_rectangle_with_six_distinct_squares_l1267_126739


namespace NUMINAMATH_GPT_sum_first_n_terms_of_arithmetic_sequence_l1267_126724

def arithmetic_sequence_sum (a1 d n: ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_n_terms_of_arithmetic_sequence :
  arithmetic_sequence_sum 2 2 n = n * (n + 1) / 2 :=
by sorry

end NUMINAMATH_GPT_sum_first_n_terms_of_arithmetic_sequence_l1267_126724


namespace NUMINAMATH_GPT_cos_sum_identity_l1267_126728

theorem cos_sum_identity :
  (Real.cos (75 * Real.pi / 180)) ^ 2 + (Real.cos (15 * Real.pi / 180)) ^ 2 + 
  (Real.cos (75 * Real.pi / 180)) * (Real.cos (15 * Real.pi / 180)) = 5 / 4 := 
by
  sorry

end NUMINAMATH_GPT_cos_sum_identity_l1267_126728


namespace NUMINAMATH_GPT_katie_total_earnings_l1267_126729

-- Define the conditions
def bead_necklaces := 4
def gem_necklaces := 3
def price_per_necklace := 3

-- The total money earned
def total_money_earned := bead_necklaces + gem_necklaces * price_per_necklace = 21

-- The statement to prove
theorem katie_total_earnings : total_money_earned :=
by
  sorry

end NUMINAMATH_GPT_katie_total_earnings_l1267_126729


namespace NUMINAMATH_GPT_min_value_a_plus_b_plus_c_l1267_126743

theorem min_value_a_plus_b_plus_c 
  (a b c : ℕ) 
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_c_pos : c > 0)
  (x1 x2 : ℝ)
  (hx1_neg : -1 < x1)
  (hx1_pos : x1 < 0)
  (hx2_neg : 0 < x2)
  (hx2_pos : x2 < 1)
  (h_distinct : x1 ≠ x2)
  (h_eqn_x1 : a * x1^2 + b * x1 + c = 0)
  (h_eqn_x2 : a * x2^2 + b * x2 + c = 0) :
  a + b + c = 11 :=
sorry

end NUMINAMATH_GPT_min_value_a_plus_b_plus_c_l1267_126743


namespace NUMINAMATH_GPT_log12_eq_abc_l1267_126793

theorem log12_eq_abc (a b : ℝ) (h1 : a = Real.log 7 / Real.log 6) (h2 : b = Real.log 4 / Real.log 3) : 
  Real.log 7 / Real.log 12 = (a * b + 2 * a) / (2 * b + 2) :=
by
  sorry

end NUMINAMATH_GPT_log12_eq_abc_l1267_126793


namespace NUMINAMATH_GPT_operation_equivalence_l1267_126763

theorem operation_equivalence :
  (∀ (x : ℝ), (x * (4 / 5) / (2 / 7)) = x * (7 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_operation_equivalence_l1267_126763


namespace NUMINAMATH_GPT_geometric_sequence_product_bound_l1267_126734

theorem geometric_sequence_product_bound {a1 a2 a3 m q : ℝ} (h_sum : a1 + a2 + a3 = 3 * m) (h_m_pos : 0 < m) (h_q_pos : 0 < q) (h_geom : a1 = a2 / q ∧ a3 = a2 * q) : 
  0 < a1 * a2 * a3 ∧ a1 * a2 * a3 ≤ m^3 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_product_bound_l1267_126734


namespace NUMINAMATH_GPT_length_of_EC_l1267_126711

variable (AC : ℝ) (AB : ℝ) (CD : ℝ) (EC : ℝ)

def is_trapezoid (AB CD : ℝ) : Prop := AB = 3 * CD
def perimeter (AB CD AC : ℝ) : Prop := AB + CD + AC + (AC / 3) = 36

theorem length_of_EC
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 18)
  (h3 : perimeter AB CD AC) :
  EC = 9 / 2 :=
  sorry

end NUMINAMATH_GPT_length_of_EC_l1267_126711


namespace NUMINAMATH_GPT_cody_increases_steps_by_1000_l1267_126783

theorem cody_increases_steps_by_1000 (x : ℕ) 
  (initial_steps : ℕ := 7000)
  (steps_logged_in_four_weeks : ℕ := 70000)
  (goal_steps : ℕ := 100000)
  (remaining_steps : ℕ := 30000)
  (condition : 1000 + 7 * (1 + 2 + 3) * x = 70000 → x = 1000) : x = 1000 :=
by
  sorry

end NUMINAMATH_GPT_cody_increases_steps_by_1000_l1267_126783


namespace NUMINAMATH_GPT_cone_volume_surface_area_sector_l1267_126790

theorem cone_volume_surface_area_sector (V : ℝ):
  (∃ (r l h : ℝ), (π * r * (r + l) = 15 * π) ∧ (l = 6 * r) ∧ (h = Real.sqrt (l^2 - r^2)) ∧ (V = (1/3) * π * r^2 * h)) →
  V = (25 * Real.sqrt 3 / 7) * π :=
by 
  sorry

end NUMINAMATH_GPT_cone_volume_surface_area_sector_l1267_126790


namespace NUMINAMATH_GPT_tom_total_payment_l1267_126740

def fruit_cost (lemons papayas mangos : ℕ) : ℕ :=
  2 * lemons + 1 * papayas + 4 * mangos

def discount (total_fruits : ℕ) : ℕ :=
  total_fruits / 4

def total_cost_with_discount (lemons papayas mangos : ℕ) : ℕ :=
  let total_fruits := lemons + papayas + mangos
  fruit_cost lemons papayas mangos - discount total_fruits

theorem tom_total_payment :
  total_cost_with_discount 6 4 2 = 21 :=
  by
    sorry

end NUMINAMATH_GPT_tom_total_payment_l1267_126740


namespace NUMINAMATH_GPT_find_ratio_l1267_126738

noncomputable def complex_numbers_are_non_zero (x y z : ℂ) : Prop :=
x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0

noncomputable def sum_is_30 (x y z : ℂ) : Prop :=
x + y + z = 30

noncomputable def expanded_equality (x y z : ℂ) : Prop :=
((x - y)^2 + (x - z)^2 + (y - z)^2) * (x + y + z) = x * y * z

theorem find_ratio (x y z : ℂ)
  (h1 : complex_numbers_are_non_zero x y z)
  (h2 : sum_is_30 x y z)
  (h3 : expanded_equality x y z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3.5 :=
sorry

end NUMINAMATH_GPT_find_ratio_l1267_126738


namespace NUMINAMATH_GPT_factor_difference_of_squares_l1267_126771

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end NUMINAMATH_GPT_factor_difference_of_squares_l1267_126771


namespace NUMINAMATH_GPT_slope_angle_at_point_l1267_126718

def f (x : ℝ) : ℝ := 2 * x^3 - 7 * x + 2

theorem slope_angle_at_point :
  let deriv_f := fun x : ℝ => 6 * x^2 - 7
  let slope := deriv_f 1
  let angle := Real.arctan slope
  angle = (3 * Real.pi) / 4 :=
by
  sorry

end NUMINAMATH_GPT_slope_angle_at_point_l1267_126718


namespace NUMINAMATH_GPT_min_value_expression_l1267_126731

open Real

theorem min_value_expression 
  (a : ℝ) 
  (b : ℝ) 
  (hb : 0 < b) 
  (e : ℝ) 
  (he : e = 2.718281828459045) :
  ∃ x : ℝ, 
  (x = 2 * (1 - log 2)^2) ∧
  ∀ a b, 
    0 < b → 
    ((1 / 2) * exp a - log (2 * b))^2 + (a - b)^2 ≥ x :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1267_126731


namespace NUMINAMATH_GPT_polynomial_solution_l1267_126782

noncomputable def p (x : ℝ) : ℝ := (7 / 4) * x^2 + 1

theorem polynomial_solution :
  (∀ x y : ℝ, p x * p y = p x + p y + p (x * y) - 2) ∧ p 2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_solution_l1267_126782


namespace NUMINAMATH_GPT_base_4_digits_l1267_126776

theorem base_4_digits (b : ℕ) (h1 : b^3 ≤ 216) (h2 : 216 < b^4) : b = 5 :=
sorry

end NUMINAMATH_GPT_base_4_digits_l1267_126776


namespace NUMINAMATH_GPT_unique_zero_of_f_l1267_126736

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (-x + 1))

theorem unique_zero_of_f (a : ℝ) : (∃! x, f x a = 0) ↔ a = 1 / 2 := sorry

end NUMINAMATH_GPT_unique_zero_of_f_l1267_126736
