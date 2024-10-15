import Mathlib

namespace NUMINAMATH_GPT_middle_value_bounds_l540_54073

theorem middle_value_bounds (a b c : ℝ) (h1 : a + b + c = 10)
  (h2 : a > b) (h3 : b > c) (h4 : a - c = 3) : 
  7 / 3 < b ∧ b < 13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_middle_value_bounds_l540_54073


namespace NUMINAMATH_GPT_base_8_sum_units_digit_l540_54022

section
  def digit_in_base (n : ℕ) (base : ℕ) (d : ℕ) : Prop :=
  ((n % base) = d)

theorem base_8_sum_units_digit :
  let n1 := 63
  let n2 := 74
  let base := 8
  (digit_in_base n1 base 3) →
  (digit_in_base n2 base 4) →
  digit_in_base (n1 + n2) base 7 :=
by
  intro h1 h2
  -- placeholder for the detailed proof
  sorry
end

end NUMINAMATH_GPT_base_8_sum_units_digit_l540_54022


namespace NUMINAMATH_GPT_sum_of_squares_l540_54020

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 10) (h2 : m * n = 24) : m^2 + n^2 = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l540_54020


namespace NUMINAMATH_GPT_find_x_l540_54045

noncomputable def x_half_y (x y : ℚ) : Prop := x = (1 / 2) * y
noncomputable def y_third_z (y z : ℚ) : Prop := y = (1 / 3) * z

theorem find_x (x y z : ℚ) (h₁ : x_half_y x y) (h₂ : y_third_z y z) (h₃ : z = 100) :
  x = 16 + (2 / 3 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l540_54045


namespace NUMINAMATH_GPT_at_least_one_did_not_land_stably_l540_54017

-- Define the propositions p and q
variables (p q : Prop)

-- Define the theorem to prove
theorem at_least_one_did_not_land_stably :
  (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_did_not_land_stably_l540_54017


namespace NUMINAMATH_GPT_circle_y_coords_sum_l540_54089

theorem circle_y_coords_sum (x y : ℝ) (hc : (x + 3)^2 + (y - 5)^2 = 64) (hx : x = 0) : y = 5 + Real.sqrt 55 ∨ y = 5 - Real.sqrt 55 → (5 + Real.sqrt 55) + (5 - Real.sqrt 55) = 10 := 
by
  intros
  sorry

end NUMINAMATH_GPT_circle_y_coords_sum_l540_54089


namespace NUMINAMATH_GPT_maximal_p_sum_consecutive_l540_54068

theorem maximal_p_sum_consecutive (k : ℕ) (h1 : k = 31250) : 
  ∃ p a : ℕ, p * (2 * a + p - 1) = k ∧ ∀ p' a', (p' * (2 * a' + p' - 1) = k) → p' ≤ p := by
  sorry

end NUMINAMATH_GPT_maximal_p_sum_consecutive_l540_54068


namespace NUMINAMATH_GPT_sum_not_prime_l540_54019

-- Definitions based on conditions:
variables {a b c d : ℕ}

-- Conditions:
axiom h_ab_eq_cd : a * b = c * d

-- Statement to prove:
theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) :=
sorry

end NUMINAMATH_GPT_sum_not_prime_l540_54019


namespace NUMINAMATH_GPT_tom_seashells_l540_54009

theorem tom_seashells 
  (days_at_beach : ℕ) (seashells_per_day : ℕ) (total_seashells : ℕ) 
  (h1 : days_at_beach = 5) (h2 : seashells_per_day = 7) (h3 : total_seashells = days_at_beach * seashells_per_day) : 
  total_seashells = 35 := 
by
  rw [h1, h2] at h3 
  exact h3

end NUMINAMATH_GPT_tom_seashells_l540_54009


namespace NUMINAMATH_GPT_total_fireworks_correct_l540_54061

variable (fireworks_num fireworks_reg)
variable (fireworks_H fireworks_E fireworks_L fireworks_O)
variable (fireworks_square fireworks_triangle fireworks_circle)
variable (boxes fireworks_per_box : ℕ)

-- Given Conditions
def fireworks_years_2021_2023 : ℕ := 6 * 4 * 3
def fireworks_HAPPY_NEW_YEAR : ℕ := 5 * 11 + 6
def fireworks_geometric_shapes : ℕ := 4 + 3 + 12
def fireworks_HELLO : ℕ := 8 + 7 + 6 * 2 + 9
def fireworks_additional_boxes : ℕ := 100 * 10

-- Total Fireworks
def total_fireworks : ℕ :=
  fireworks_years_2021_2023 + 
  fireworks_HAPPY_NEW_YEAR + 
  fireworks_geometric_shapes + 
  fireworks_HELLO + 
  fireworks_additional_boxes

theorem total_fireworks_correct : 
  total_fireworks = 1188 :=
  by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_total_fireworks_correct_l540_54061


namespace NUMINAMATH_GPT_specified_percentage_of_number_is_40_l540_54012

theorem specified_percentage_of_number_is_40 
  (N : ℝ) 
  (hN : (1 / 4) * (1 / 3) * (2 / 5) * N = 25) 
  (P : ℝ) 
  (hP : (P / 100) * N = 300) : 
  P = 40 := 
sorry

end NUMINAMATH_GPT_specified_percentage_of_number_is_40_l540_54012


namespace NUMINAMATH_GPT_bank_exceeds_50_dollars_l540_54008

theorem bank_exceeds_50_dollars (a : ℕ := 5) (r : ℕ := 2) :
  ∃ n : ℕ, 5 * (2 ^ n - 1) > 5000 ∧ (n ≡ 9 [MOD 7]) :=
by
  sorry

end NUMINAMATH_GPT_bank_exceeds_50_dollars_l540_54008


namespace NUMINAMATH_GPT_problem1_problem2_l540_54098

variable (a : ℝ) -- Declaring a as a real number

-- Proof statement for Problem 1
theorem problem1 : (a + 2) * (a - 2) = a^2 - 4 :=
sorry

-- Proof statement for Problem 2
theorem problem2 (h : a ≠ -2) : (a^2 - 4) / (a + 2) + 2 = a :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l540_54098


namespace NUMINAMATH_GPT_dan_took_pencils_l540_54087

theorem dan_took_pencils (initial_pencils remaining_pencils : ℕ) (h_initial : initial_pencils = 34) (h_remaining : remaining_pencils = 12) : (initial_pencils - remaining_pencils) = 22 := 
by
  sorry

end NUMINAMATH_GPT_dan_took_pencils_l540_54087


namespace NUMINAMATH_GPT_range_of_a_for_decreasing_function_l540_54015

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 4 else 3 * a / x

theorem range_of_a_for_decreasing_function :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≥ f a x2) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_decreasing_function_l540_54015


namespace NUMINAMATH_GPT_blood_pressure_systolic_diastolic_l540_54096

noncomputable def blood_pressure (t : ℝ) : ℝ :=
110 + 25 * Real.sin (160 * t)

theorem blood_pressure_systolic_diastolic :
  (∀ t : ℝ, blood_pressure t ≤ 135) ∧ (∀ t : ℝ, blood_pressure t ≥ 85) :=
by
  sorry

end NUMINAMATH_GPT_blood_pressure_systolic_diastolic_l540_54096


namespace NUMINAMATH_GPT_negation_of_p_l540_54029

-- Declare the proposition p as a condition
def p : Prop :=
  ∀ (x : ℝ), 0 ≤ x → x^2 + 4 * x + 3 > 0

-- State the problem
theorem negation_of_p : ¬ p ↔ ∃ (x : ℝ), 0 ≤ x ∧ x^2 + 4 * x + 3 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_p_l540_54029


namespace NUMINAMATH_GPT_prime_square_plus_eight_is_prime_l540_54050

theorem prime_square_plus_eight_is_prime (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 8)) : p = 3 :=
sorry

end NUMINAMATH_GPT_prime_square_plus_eight_is_prime_l540_54050


namespace NUMINAMATH_GPT_metro_earnings_in_6_minutes_l540_54057

theorem metro_earnings_in_6_minutes 
  (ticket_cost : ℕ) 
  (tickets_per_minute : ℕ) 
  (duration_minutes : ℕ) 
  (earnings_in_one_minute : ℕ) 
  (earnings_in_six_minutes : ℕ) 
  (h1 : ticket_cost = 3) 
  (h2 : tickets_per_minute = 5) 
  (h3 : duration_minutes = 6) 
  (h4 : earnings_in_one_minute = tickets_per_minute * ticket_cost) 
  (h5 : earnings_in_six_minutes = earnings_in_one_minute * duration_minutes) 
  : earnings_in_six_minutes = 90 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_metro_earnings_in_6_minutes_l540_54057


namespace NUMINAMATH_GPT_average_student_headcount_l540_54006

def student_headcount_fall_0203 : ℕ := 11700
def student_headcount_fall_0304 : ℕ := 11500
def student_headcount_fall_0405 : ℕ := 11600

theorem average_student_headcount : 
  (student_headcount_fall_0203 + student_headcount_fall_0304 + student_headcount_fall_0405) / 3 = 11600 := by
  sorry

end NUMINAMATH_GPT_average_student_headcount_l540_54006


namespace NUMINAMATH_GPT_chess_player_total_games_l540_54031

noncomputable def total_games_played (W L : ℕ) : ℕ :=
  W + L

theorem chess_player_total_games :
  ∃ (W L : ℕ), W = 16 ∧ (L : ℚ) / W = 7 / 4 ∧ total_games_played W L = 44 :=
by
  sorry

end NUMINAMATH_GPT_chess_player_total_games_l540_54031


namespace NUMINAMATH_GPT_remaining_average_l540_54085

-- Definitions
def original_average (n : ℕ) (avg : ℝ) := n = 50 ∧ avg = 38
def discarded_numbers (a b : ℝ) := a = 45 ∧ b = 55

-- Proof Statement
theorem remaining_average (n : ℕ) (avg : ℝ) (a b : ℝ) (s : ℝ) :
  original_average n avg →
  discarded_numbers a b →
  s = (n * avg - (a + b)) / (n - 2) →
  s = 37.5 :=
by
  intros h_avg h_discard h_s
  sorry

end NUMINAMATH_GPT_remaining_average_l540_54085


namespace NUMINAMATH_GPT_translate_quadratic_vertex_right_l540_54021

theorem translate_quadratic_vertex_right : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = 2 * (x - 4)^2 - 3) ∧ 
  (∃ (g : ℝ → ℝ), (∀ x, g x = 2 * ((x - 1) - 3)^2 - 3))) → 
  (∃ v : ℝ × ℝ, v = (4, -3)) :=
sorry

end NUMINAMATH_GPT_translate_quadratic_vertex_right_l540_54021


namespace NUMINAMATH_GPT_neg_exists_equiv_forall_l540_54091

theorem neg_exists_equiv_forall (p : Prop) :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := sorry

end NUMINAMATH_GPT_neg_exists_equiv_forall_l540_54091


namespace NUMINAMATH_GPT_eval_at_neg_five_l540_54038

def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem eval_at_neg_five : f (-5) = 12 :=
by
  sorry

end NUMINAMATH_GPT_eval_at_neg_five_l540_54038


namespace NUMINAMATH_GPT_quadratic_roots_relationship_l540_54092

theorem quadratic_roots_relationship 
  (a b c α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0) 
  (h_eq' : a * β^2 + b * β + c = 0)
  (h_roots : β = 3 * α) :
  3 * b^2 = 16 * a * c := 
sorry

end NUMINAMATH_GPT_quadratic_roots_relationship_l540_54092


namespace NUMINAMATH_GPT_fraction_of_power_l540_54082

noncomputable def m : ℕ := 32^500

theorem fraction_of_power (h : m = 2^2500) : m / 8 = 2^2497 :=
by
  have hm : m = 2^2500 := h
  sorry

end NUMINAMATH_GPT_fraction_of_power_l540_54082


namespace NUMINAMATH_GPT_fraction_product_equals_l540_54062

theorem fraction_product_equals :
  (7 / 4) * (14 / 49) * (10 / 15) * (12 / 36) * (21 / 14) * (40 / 80) * (33 / 22) * (16 / 64) = 1 / 12 := 
  sorry

end NUMINAMATH_GPT_fraction_product_equals_l540_54062


namespace NUMINAMATH_GPT_functional_eq_solve_l540_54088

theorem functional_eq_solve (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (2*x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
sorry

end NUMINAMATH_GPT_functional_eq_solve_l540_54088


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l540_54024

theorem sufficient_but_not_necessary (a : ℝ) :
  ((a + 2) * (3 * a - 4) - (a - 2) ^ 2 = 0 → a = 2 ∨ a = 1 / 2) →
  (a = 1 / 2 → ∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) →
  ( (∀ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2 → a = 1/2) ∧ 
  (∃ x y : ℝ, (a+2) * x + (a-2) * y = 1 ∧ (a-2) * x + (3*a-4) * y = 2) → a ≠ 1/2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l540_54024


namespace NUMINAMATH_GPT_maximum_value_expression_l540_54053

theorem maximum_value_expression (a b c : ℕ) (ha : 0 < a ∧ a ≤ 9) (hb : 0 < b ∧ b ≤ 9) (hc : 0 < c ∧ c ≤ 9) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  ∃ (v : ℚ), v = (1 / (a + 2010 / (b + 1 / c : ℚ))) ∧ v ≤ (1 / 203) :=
sorry

end NUMINAMATH_GPT_maximum_value_expression_l540_54053


namespace NUMINAMATH_GPT_exists_three_sticks_form_triangle_l540_54039

theorem exists_three_sticks_form_triangle 
  (l : Fin 5 → ℝ) 
  (h1 : ∀ i, 2 < l i) 
  (h2 : ∀ i, l i < 8) : 
  ∃ (i j k : Fin 5), i < j ∧ j < k ∧ 
    (l i + l j > l k) ∧ 
    (l j + l k > l i) ∧ 
    (l k + l i > l j) :=
sorry

end NUMINAMATH_GPT_exists_three_sticks_form_triangle_l540_54039


namespace NUMINAMATH_GPT_sandwich_cost_l540_54035

theorem sandwich_cost 
  (loaf_sandwiches : ℕ) (target_sandwiches : ℕ) 
  (bread_cost : ℝ) (meat_cost : ℝ) (cheese_cost : ℝ) 
  (cheese_coupon : ℝ) (meat_coupon : ℝ) (total_threshold : ℝ) 
  (discount_rate : ℝ)
  (h1 : loaf_sandwiches = 10) 
  (h2 : target_sandwiches = 50) 
  (h3 : bread_cost = 4) 
  (h4 : meat_cost = 5) 
  (h5 : cheese_cost = 4) 
  (h6 : cheese_coupon = 1) 
  (h7 : meat_coupon = 1) 
  (h8 : total_threshold = 60) 
  (h9 : discount_rate = 0.1) :
  ( ∃ cost_per_sandwich : ℝ, 
      cost_per_sandwich = 1.944 ) :=
  sorry

end NUMINAMATH_GPT_sandwich_cost_l540_54035


namespace NUMINAMATH_GPT_subcommittee_count_l540_54094

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose republicans subcommittee_republicans * choose democrats subcommittee_democrats = 11760 :=
by
  let republicans := 10
  let democrats := 8
  let subcommittee_republicans := 4
  let subcommittee_democrats := 3
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end NUMINAMATH_GPT_subcommittee_count_l540_54094


namespace NUMINAMATH_GPT_S_equals_2_l540_54056

noncomputable def problem_S := 
  1 / (2 - Real.sqrt 3) - 1 / (Real.sqrt 3 - Real.sqrt 2) + 
  1 / (Real.sqrt 2 - 1) - 1 / (1 - Real.sqrt 3 + Real.sqrt 2)

theorem S_equals_2 : problem_S = 2 := by
  sorry

end NUMINAMATH_GPT_S_equals_2_l540_54056


namespace NUMINAMATH_GPT_count_non_congruent_rectangles_l540_54046

-- Definitions based on conditions given in the problem
def is_rectangle (w h : ℕ) : Prop := 2 * (w + h) = 40 ∧ w % 2 = 0

-- Theorem that we need to prove based on the problem statement
theorem count_non_congruent_rectangles : 
  ∃ n : ℕ, n = 9 ∧ 
  (∀ p : ℕ × ℕ, p ∈ { p | is_rectangle p.1 p.2 } → ∀ q : ℕ × ℕ, q ∈ { q | is_rectangle q.1 q.2 } → p = q ∨ p ≠ q) := 
sorry

end NUMINAMATH_GPT_count_non_congruent_rectangles_l540_54046


namespace NUMINAMATH_GPT_floor_identity_l540_54051

theorem floor_identity (x : ℝ) : 
    (⌊(3 + x) / 6⌋ - ⌊(4 + x) / 6⌋ + ⌊(5 + x) / 6⌋ = ⌊(1 + x) / 2⌋ - ⌊(1 + x) / 3⌋) :=
by
  sorry

end NUMINAMATH_GPT_floor_identity_l540_54051


namespace NUMINAMATH_GPT_determine_marbles_l540_54076

noncomputable def marbles_total (x : ℚ) := (4 * x + 2) + (2 * x) + (3 * x - 1)

theorem determine_marbles (x : ℚ) (h1 : marbles_total x = 47) :
  (4 * x + 2 = 202 / 9) ∧ (2 * x = 92 / 9) ∧ (3 * x - 1 = 129 / 9) :=
by
  sorry

end NUMINAMATH_GPT_determine_marbles_l540_54076


namespace NUMINAMATH_GPT_sum_modulo_seven_l540_54080

theorem sum_modulo_seven (a b c : ℕ) (h1: a = 9^5) (h2: b = 8^6) (h3: c = 7^7) :
  (a + b + c) % 7 = 5 :=
by sorry

end NUMINAMATH_GPT_sum_modulo_seven_l540_54080


namespace NUMINAMATH_GPT_minimum_omega_l540_54034

noncomputable def f (omega phi x : ℝ) : ℝ := Real.sin (omega * x + phi)

theorem minimum_omega {omega : ℝ} (h_pos : omega > 0) (h_even : ∀ x : ℝ, f omega (Real.pi / 2) x = f omega (Real.pi / 2) (-x)) 
  (h_zero_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi ∧ f omega (Real.pi / 2) x = 0) :
  omega ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_omega_l540_54034


namespace NUMINAMATH_GPT_altitude_in_scientific_notation_l540_54090

theorem altitude_in_scientific_notation : 
  (389000 : ℝ) = 3.89 * (10 : ℝ) ^ 5 :=
by
  sorry

end NUMINAMATH_GPT_altitude_in_scientific_notation_l540_54090


namespace NUMINAMATH_GPT_sum_of_5_and_8_l540_54043

theorem sum_of_5_and_8 : 5 + 8 = 13 := by
  rfl

end NUMINAMATH_GPT_sum_of_5_and_8_l540_54043


namespace NUMINAMATH_GPT_visitors_yesterday_l540_54032

-- Definitions based on the given conditions
def visitors_today : ℕ := 583
def visitors_total : ℕ := 829

-- Theorem statement to prove the number of visitors the day before Rachel visited
theorem visitors_yesterday : ∃ v_yesterday: ℕ, v_yesterday = visitors_total - visitors_today ∧ v_yesterday = 246 :=
by
  sorry

end NUMINAMATH_GPT_visitors_yesterday_l540_54032


namespace NUMINAMATH_GPT_probability_two_yellow_apples_l540_54025

theorem probability_two_yellow_apples (total_apples : ℕ) (red_apples : ℕ) (green_apples : ℕ) (yellow_apples : ℕ) (choose : ℕ → ℕ → ℕ) (probability : ℕ → ℕ → ℝ) :
  total_apples = 10 →
  red_apples = 5 →
  green_apples = 3 →
  yellow_apples = 2 →
  choose total_apples 2 = 45 →
  choose yellow_apples 2 = 1 →
  probability (choose yellow_apples 2) (choose total_apples 2) = 1 / 45 := 
  by
  sorry

end NUMINAMATH_GPT_probability_two_yellow_apples_l540_54025


namespace NUMINAMATH_GPT_jar_water_transfer_l540_54065

theorem jar_water_transfer
  (C_x : ℝ) (C_y : ℝ)
  (h1 : C_y = 1/2 * C_x)
  (WaterInX : ℝ)
  (WaterInY : ℝ)
  (h2 : WaterInX = 1/2 * C_x)
  (h3 : WaterInY = 1/2 * C_y) :
  WaterInX + WaterInY = 3/4 * C_x :=
by
  sorry

end NUMINAMATH_GPT_jar_water_transfer_l540_54065


namespace NUMINAMATH_GPT_problem_I_II_l540_54010

noncomputable def f (x : ℝ) (φ : ℝ) := Real.sin (2 * x + φ)

theorem problem_I_II
  (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi / 2)
  (h3 : f 0 φ = 1 / 2) :
  ∃ T : ℝ, T = Real.pi ∧ φ = Real.pi / 6 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → (f x φ) ≥ -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_I_II_l540_54010


namespace NUMINAMATH_GPT_green_balloons_correct_l540_54000

-- Defining the quantities
def total_balloons : ℕ := 67
def red_balloons : ℕ := 29
def blue_balloons : ℕ := 21

-- Calculating the green balloons
def green_balloons : ℕ := total_balloons - red_balloons - blue_balloons

-- The theorem we want to prove
theorem green_balloons_correct : green_balloons = 17 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_green_balloons_correct_l540_54000


namespace NUMINAMATH_GPT_min_staff_members_l540_54023

theorem min_staff_members
  (num_male_students : ℕ)
  (num_benches_3_students : ℕ)
  (num_benches_4_students : ℕ)
  (num_female_students : ℕ)
  (total_students : ℕ)
  (total_seating_capacity : ℕ)
  (additional_seats_required : ℕ)
  (num_staff_members : ℕ)
  (h1 : num_female_students = 4 * num_male_students)
  (h2 : num_male_students = 29)
  (h3 : num_benches_3_students = 15)
  (h4 : num_benches_4_students = 14)
  (h5 : total_seating_capacity = 3 * num_benches_3_students + 4 * num_benches_4_students)
  (h6 : total_students = num_male_students + num_female_students)
  (h7 : additional_seats_required = total_students - total_seating_capacity)
  (h8 : num_staff_members = additional_seats_required)
  : num_staff_members = 44 := 
sorry

end NUMINAMATH_GPT_min_staff_members_l540_54023


namespace NUMINAMATH_GPT_gcd_75_225_l540_54069

theorem gcd_75_225 : Int.gcd 75 225 = 75 :=
by
  sorry

end NUMINAMATH_GPT_gcd_75_225_l540_54069


namespace NUMINAMATH_GPT_alcohol_percentage_l540_54064

theorem alcohol_percentage (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) 
(h3 : (0.6 + (x / 100) * 6 = 2.4)) : x = 30 :=
by sorry

end NUMINAMATH_GPT_alcohol_percentage_l540_54064


namespace NUMINAMATH_GPT_parabola_tangent_sequence_l540_54055

noncomputable def geom_seq_sum (a2 : ℕ) : ℕ :=
  a2 + a2 / 4 + a2 / 16

theorem parabola_tangent_sequence (a2 : ℕ) (h : a2 = 32) : geom_seq_sum a2 = 42 :=
by
  rw [h]
  norm_num
  sorry

end NUMINAMATH_GPT_parabola_tangent_sequence_l540_54055


namespace NUMINAMATH_GPT_find_brick_width_l540_54060

def SurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

theorem find_brick_width :
  ∃ width : ℝ, SurfaceArea 10 width 3 = 164 ∧ width = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_brick_width_l540_54060


namespace NUMINAMATH_GPT_luke_fish_fillets_l540_54018

def fish_per_day : ℕ := 2
def days : ℕ := 30
def fillets_per_fish : ℕ := 2

theorem luke_fish_fillets : fish_per_day * days * fillets_per_fish = 120 := 
by
  sorry

end NUMINAMATH_GPT_luke_fish_fillets_l540_54018


namespace NUMINAMATH_GPT_irrational_lattice_point_exists_l540_54036

theorem irrational_lattice_point_exists (k : ℝ) (h_irrational : ¬ ∃ q r : ℚ, q / r = k)
  (ε : ℝ) (h_pos : ε > 0) : ∃ m n : ℤ, |m * k - n| < ε :=
by
  sorry

end NUMINAMATH_GPT_irrational_lattice_point_exists_l540_54036


namespace NUMINAMATH_GPT_minimum_value_of_expression_l540_54049

noncomputable def min_value_expression (x y : ℝ) : ℝ :=
  x^2 + x * y + y^2 + 7

theorem minimum_value_of_expression :
  ∃ x y : ℝ, min_value_expression x y = 7 :=
by
  use 0, 0
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l540_54049


namespace NUMINAMATH_GPT_inequality_k_m_l540_54047

theorem inequality_k_m (k m : ℕ) (hk : 0 < k) (hm : 0 < m) (hkm : k > m) (hdiv : (k^3 - m^3) ∣ k * m * (k^2 - m^2)) :
  (k - m)^3 > 3 * k * m := 
by sorry

end NUMINAMATH_GPT_inequality_k_m_l540_54047


namespace NUMINAMATH_GPT_factorial_inequality_l540_54042

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n! ≤ ((n+1)/2)^n := 
by {
  sorry
}

end NUMINAMATH_GPT_factorial_inequality_l540_54042


namespace NUMINAMATH_GPT_find_other_number_l540_54028

open Nat

theorem find_other_number (A B lcm hcf : ℕ) (h_lcm : lcm = 2310) (h_hcf : hcf = 30) (h_A : A = 231) (h_eq : lcm * hcf = A * B) : 
  B = 300 :=
  sorry

end NUMINAMATH_GPT_find_other_number_l540_54028


namespace NUMINAMATH_GPT_sol_sells_more_candy_each_day_l540_54007

variable {x : ℕ}

-- Definition of the conditions
def sells_candy (first_day : ℕ) (rate : ℕ) (days : ℕ) : ℕ :=
  first_day + rate * (days - 1) * days / 2

def earns (bars_sold : ℕ) (price_cents : ℕ) : ℕ :=
  bars_sold * price_cents

-- Problem statement in Lean:
theorem sol_sells_more_candy_each_day
  (first_day_sales : ℕ := 10)
  (days : ℕ := 6)
  (price_cents : ℕ := 10)
  (total_earnings : ℕ := 1200) :
  earns (sells_candy first_day_sales x days) price_cents = total_earnings → x = 76 :=
sorry

end NUMINAMATH_GPT_sol_sells_more_candy_each_day_l540_54007


namespace NUMINAMATH_GPT_find_value_b_l540_54026

-- Define the problem-specific elements
noncomputable def is_line_eqn (y x : ℝ) : Prop := y = 4 - 2 * x

theorem find_value_b (b : ℝ) (h₀ : b > 0) (h₁ : b < 2)
  (hP : ∀ y, is_line_eqn y 0 → y = 4)
  (hS : ∀ y, is_line_eqn y 2 → y = 0)
  (h_ratio : ∀ Q R S O P,
    Q = (2, 0) ∧ R = (2, 0) ∧ S = (2, 0) ∧ P = (0, 4) ∧ O = (0, 0) →
    4 / 9 = 4 / ((Q.1 - O.1) * (Q.1 - O.1)) →
    (Q.1 - O.1) / (P.2 - O.2) = 2 / 3) :
  b = 2 :=
sorry

end NUMINAMATH_GPT_find_value_b_l540_54026


namespace NUMINAMATH_GPT_profit_per_cake_l540_54071

theorem profit_per_cake (ingredient_cost : ℝ) (packaging_cost : ℝ) (selling_price : ℝ) (cake_count : ℝ)
    (h1 : ingredient_cost = 12) (h2 : packaging_cost = 1) (h3 : selling_price = 15) (h4 : cake_count = 2) :
    selling_price - (ingredient_cost / cake_count + packaging_cost) = 8 := by
  sorry

end NUMINAMATH_GPT_profit_per_cake_l540_54071


namespace NUMINAMATH_GPT_sum_of_digits_b_n_l540_54084

def a_n (n : ℕ) : ℕ := 10^(2^n) - 1

def b_n (n : ℕ) : ℕ :=
  List.prod (List.map a_n (List.range (n + 1)))

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_b_n (n : ℕ) : sum_of_digits (b_n n) = 9 * 2^n :=
  sorry

end NUMINAMATH_GPT_sum_of_digits_b_n_l540_54084


namespace NUMINAMATH_GPT_trigonometry_identity_l540_54013

theorem trigonometry_identity (α : ℝ) (P : ℝ × ℝ) (h : P = (4, -3)) :
  let x := P.1
  let y := P.2
  let r := Real.sqrt (x^2 + y^2)
  x = 4 →
  y = -3 →
  r = 5 →
  Real.tan α = y / x := by
  intros x y r hx hy hr
  rw [hx, hy]
  simp [Real.tan, div_eq_mul_inv, mul_comm]
  sorry

end NUMINAMATH_GPT_trigonometry_identity_l540_54013


namespace NUMINAMATH_GPT_Mabel_marble_count_l540_54002

variable (K A M : ℕ)

axiom Amanda_condition : A + 12 = 2 * K
axiom Mabel_K_condition : M = 5 * K
axiom Mabel_A_condition : M = A + 63

theorem Mabel_marble_count : M = 85 := by
  sorry

end NUMINAMATH_GPT_Mabel_marble_count_l540_54002


namespace NUMINAMATH_GPT_fraction_white_surface_area_l540_54033

-- Definitions for conditions
def larger_cube_side : ℕ := 3
def smaller_cube_count : ℕ := 27
def white_cube_count : ℕ := 19
def black_cube_count : ℕ := 8
def black_corners : Nat := 8
def faces_per_cube : ℕ := 6
def exposed_faces_per_corner : ℕ := 3

-- Theorem statement for proving the fraction of the white surface area
theorem fraction_white_surface_area : (30 : ℚ) / 54 = 5 / 9 :=
by 
  -- Add the proof steps here if necessary
  sorry

end NUMINAMATH_GPT_fraction_white_surface_area_l540_54033


namespace NUMINAMATH_GPT_shirts_per_minute_l540_54079

/--
An industrial machine made 8 shirts today and worked for 4 minutes today. 
Prove that the machine can make 2 shirts per minute.
-/
theorem shirts_per_minute (shirts_today : ℕ) (minutes_today : ℕ)
  (h1 : shirts_today = 8) (h2 : minutes_today = 4) :
  (shirts_today / minutes_today) = 2 :=
by sorry

end NUMINAMATH_GPT_shirts_per_minute_l540_54079


namespace NUMINAMATH_GPT_candy_distribution_l540_54063

-- Define the problem conditions and theorem.

theorem candy_distribution (X : ℕ) (total_pieces : ℕ) (portions : ℕ) 
  (subsequent_more : ℕ) (h_total : total_pieces = 40) 
  (h_portions : portions = 4) 
  (h_subsequent : subsequent_more = 2) 
  (h_eq : X + (X + subsequent_more) + (X + subsequent_more * 2) + (X + subsequent_more * 3) = total_pieces) : 
  X = 7 := 
sorry

end NUMINAMATH_GPT_candy_distribution_l540_54063


namespace NUMINAMATH_GPT_one_interior_angle_of_polygon_with_five_diagonals_l540_54067

theorem one_interior_angle_of_polygon_with_five_diagonals (n : ℕ) (h : n - 3 = 5) :
  let interior_angle := 180 * (n - 2) / n
  interior_angle = 135 :=
by
  sorry

end NUMINAMATH_GPT_one_interior_angle_of_polygon_with_five_diagonals_l540_54067


namespace NUMINAMATH_GPT_polynomial_factorization_l540_54005

-- Define the polynomial and its factorized form
def polynomial (x : ℝ) : ℝ := x^2 - 4*x + 4
def factorized_form (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that the polynomial equals its factorized form
theorem polynomial_factorization (x : ℝ) : polynomial x = factorized_form x :=
by {
  sorry -- Proof skipped
}

end NUMINAMATH_GPT_polynomial_factorization_l540_54005


namespace NUMINAMATH_GPT_percentage_of_part_l540_54072

theorem percentage_of_part (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 50) : (Part / Whole) * 100 = 240 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_part_l540_54072


namespace NUMINAMATH_GPT_planes_parallel_if_any_line_parallel_l540_54001

axiom Plane : Type
axiom Line : Type
axiom contains : Plane → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_lines : Line → Plane → Prop

theorem planes_parallel_if_any_line_parallel (α β : Plane)
  (h₁ : ∀ l, contains α l → parallel_lines l β) :
  parallel α β :=
sorry

end NUMINAMATH_GPT_planes_parallel_if_any_line_parallel_l540_54001


namespace NUMINAMATH_GPT_fair_attendance_l540_54004

-- Define the variables x, y, and z
variables (x y z : ℕ)

-- Define the conditions given in the problem
def condition1 := z = 2 * y
def condition2 := x = z - 200
def condition3 := y = 600

-- State the main theorem proving the values of x, y, and z
theorem fair_attendance : condition1 y z → condition2 x z → condition3 y → (x = 1000 ∧ y = 600 ∧ z = 1200) := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_fair_attendance_l540_54004


namespace NUMINAMATH_GPT_probability_opposite_vertex_l540_54086

theorem probability_opposite_vertex (k : ℕ) (h : k > 0) : 
    P_k = (1 / 6 : ℝ) + (1 / (3 * (-2) ^ k) : ℝ) := 
sorry

end NUMINAMATH_GPT_probability_opposite_vertex_l540_54086


namespace NUMINAMATH_GPT_boat_speed_greater_than_stream_l540_54003

def boat_stream_speed_difference (S U V : ℝ) := 
  (S / (U - V)) - (S / (U + V)) + (S / (2 * V + 1)) = 1

theorem boat_speed_greater_than_stream 
  (S : ℝ) (U V : ℝ) 
  (h_dist : S = 1) 
  (h_time_diff : boat_stream_speed_difference S U V) :
  U - V = 1 :=
sorry

end NUMINAMATH_GPT_boat_speed_greater_than_stream_l540_54003


namespace NUMINAMATH_GPT_smallest_positive_debt_resolved_l540_54093

theorem smallest_positive_debt_resolved : ∃ (D : ℤ), D > 0 ∧ (∃ (p g : ℤ), D = 400 * p + 250 * g) ∧ D = 50 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_debt_resolved_l540_54093


namespace NUMINAMATH_GPT_train_speed_l540_54097

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 375.03) (time_eq : time = 5) :
  let speed_kmph := (length / 1000) / (time / 3600)
  speed_kmph = 270.02 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l540_54097


namespace NUMINAMATH_GPT_total_action_figures_l540_54078

theorem total_action_figures (figures_per_shelf : ℕ) (number_of_shelves : ℕ) (h1 : figures_per_shelf = 10) (h2 : number_of_shelves = 8) : figures_per_shelf * number_of_shelves = 80 := by
  sorry

end NUMINAMATH_GPT_total_action_figures_l540_54078


namespace NUMINAMATH_GPT_total_cost_is_correct_l540_54077

-- Define the number of total tickets and the number of children's tickets
def total_tickets : ℕ := 21
def children_tickets : ℕ := 16
def adult_tickets : ℕ := total_tickets - children_tickets

-- Define the cost of tickets for adults and children
def cost_per_adult_ticket : ℝ := 5.50
def cost_per_child_ticket : ℝ := 3.50

-- Define the total cost spent
def total_cost_spent : ℝ :=
  (adult_tickets * cost_per_adult_ticket) + (children_tickets * cost_per_child_ticket)

-- Prove that the total amount spent on tickets is $83.50
theorem total_cost_is_correct : total_cost_spent = 83.50 := by
  sorry

end NUMINAMATH_GPT_total_cost_is_correct_l540_54077


namespace NUMINAMATH_GPT_uncovered_side_length_l540_54041

theorem uncovered_side_length :
  ∃ (L : ℝ) (W : ℝ), L * W = 680 ∧ 2 * W + L = 146 ∧ L = 136 := by
  sorry

end NUMINAMATH_GPT_uncovered_side_length_l540_54041


namespace NUMINAMATH_GPT_number_of_boys_in_other_communities_l540_54059

-- Definitions from conditions
def total_boys : ℕ := 700
def percentage_muslims : ℕ := 44
def percentage_hindus : ℕ := 28
def percentage_sikhs : ℕ := 10

-- Proof statement
theorem number_of_boys_in_other_communities : 
  (700 * (100 - (44 + 28 + 10)) / 100) = 126 := 
by
  sorry

end NUMINAMATH_GPT_number_of_boys_in_other_communities_l540_54059


namespace NUMINAMATH_GPT_martins_travel_time_l540_54037

-- Declare the necessary conditions from the problem
variables (speed : ℝ) (distance : ℝ)
-- Define the conditions
def martin_speed := speed = 12 -- Martin's speed is 12 miles per hour
def martin_distance := distance = 72 -- Martin drove 72 miles

-- State the theorem to prove the time taken is 6 hours
theorem martins_travel_time (h1 : martin_speed speed) (h2 : martin_distance distance) : distance / speed = 6 :=
by
  -- To complete the problem statement, insert sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_martins_travel_time_l540_54037


namespace NUMINAMATH_GPT_child_grandmother_ratio_l540_54054

variable (G D C : ℕ)

axiom cond1 : G + D + C = 120
axiom cond2 : D + C = 60
axiom cond3 : D = 48

theorem child_grandmother_ratio : (C : ℚ) / G = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_child_grandmother_ratio_l540_54054


namespace NUMINAMATH_GPT_parabola_translation_l540_54044

-- Definitions based on the given conditions
def f (x : ℝ) : ℝ := (x - 1) ^ 2 + 5
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 3

-- Statement of the translation problem in Lean 4
theorem parabola_translation :
  ∀ x : ℝ, g x = f (x + 2) - 3 := 
sorry

end NUMINAMATH_GPT_parabola_translation_l540_54044


namespace NUMINAMATH_GPT_geometric_sequence_a6_l540_54030

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_a6 
  (a_1 q : ℝ) 
  (a2_eq : a_1 + a_1 * q = -1)
  (a3_eq : a_1 - a_1 * q ^ 2 = -3) : 
  a_n a_1 q 6 = -32 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l540_54030


namespace NUMINAMATH_GPT_pd_distance_l540_54027

theorem pd_distance (PA PB PC PD : ℝ) (hPA : PA = 17) (hPB : PB = 15) (hPC : PC = 6) :
  PA^2 + PC^2 = PB^2 + PD^2 → PD = 10 :=
by
  sorry

end NUMINAMATH_GPT_pd_distance_l540_54027


namespace NUMINAMATH_GPT_train_stop_time_l540_54070

theorem train_stop_time (speed_no_stops speed_with_stops : ℕ) (time_per_hour : ℕ) (stoppage_time_per_hour : ℕ) :
  speed_no_stops = 45 →
  speed_with_stops = 30 →
  time_per_hour = 60 →
  stoppage_time_per_hour = 20 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_train_stop_time_l540_54070


namespace NUMINAMATH_GPT_xyz_divides_xyz_squared_l540_54074

theorem xyz_divides_xyz_squared (x y z p : ℕ) (hxyz : x < y ∧ y < z ∧ z < p) (hp : Nat.Prime p) (hx3 : x^3 ≡ y^3 [MOD p])
    (hy3 : y^3 ≡ z^3 [MOD p]) (hz3 : z^3 ≡ x^3 [MOD p]) : (x + y + z) ∣ (x^2 + y^2 + z^2) :=
by
  sorry

end NUMINAMATH_GPT_xyz_divides_xyz_squared_l540_54074


namespace NUMINAMATH_GPT_scrooge_no_equal_coins_l540_54095

theorem scrooge_no_equal_coins (n : ℕ → ℕ)
  (initial_state : n 1 = 1 ∧ n 2 = 0 ∧ n 3 = 0 ∧ n 4 = 0 ∧ n 5 = 0 ∧ n 6 = 0)
  (operation : ∀ x i, 1 ≤ i ∧ i ≤ 6 → (n (i + 1) = n i - x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) + 6 * x) 
                      ∨ (n (i + 1) = n i + 6 * x ∧ n ((i % 6) + 2) = n ((i % 6) + 2) - x)) :
  ¬ ∃ k, n 1 = k ∧ n 2 = k ∧ n 3 = k ∧ n 4 = k ∧ n 5 = k ∧ n 6 = k :=
by {
  sorry
}

end NUMINAMATH_GPT_scrooge_no_equal_coins_l540_54095


namespace NUMINAMATH_GPT_six_digit_divisible_by_72_l540_54066

theorem six_digit_divisible_by_72 (n m : ℕ) (h1 : n = 920160 ∨ n = 120168) :
  (∃ (x y : ℕ), 10 * x + y = 2016 ∧ (10^5 * x + n * 10 + m) % 72 = 0) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_divisible_by_72_l540_54066


namespace NUMINAMATH_GPT_distance_between_points_eq_l540_54048

theorem distance_between_points_eq :
  let x1 := 2
  let y1 := -5
  let x2 := -8
  let y2 := 7
  let distance := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  distance = 2 * Real.sqrt 61 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_eq_l540_54048


namespace NUMINAMATH_GPT_bert_average_words_in_crossword_l540_54052

theorem bert_average_words_in_crossword :
  (10 * 35 + 4 * 65) / (10 + 4) = 43.57 :=
by
  sorry

end NUMINAMATH_GPT_bert_average_words_in_crossword_l540_54052


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l540_54011

open Real

theorem monotonic_increasing_interval (k : ℤ) : 
  ∀ x, -π / 12 + k * π ≤ x ∧ x ≤ 5 * π / 12 + k * π ↔ 
    ∀ t, -π / 2 + 2 * k * π ≤ 2 * t - π / 3 ∧ 2 * t - π / 3 ≤ π / 2 + 2 * k * π :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l540_54011


namespace NUMINAMATH_GPT_expression_equals_24_l540_54040

noncomputable def f : ℕ → ℝ := sorry

axiom f_add (m n : ℕ) : f (m + n) = f m * f n
axiom f_one : f 1 = 3

theorem expression_equals_24 :
  (f 1^2 + f 2) / f 1 + (f 2^2 + f 4) / f 3 + (f 3^2 + f 6) / f 5 + (f 4^2 + f 8) / f 7 = 24 :=
by sorry

end NUMINAMATH_GPT_expression_equals_24_l540_54040


namespace NUMINAMATH_GPT_find_fourth_number_in_sequence_l540_54058

-- Define the conditions of the sequence
def first_number : ℤ := 1370
def second_number : ℤ := 1310
def third_number : ℤ := 1070
def fifth_number : ℤ := -6430

-- Define the differences
def difference1 : ℤ := second_number - first_number
def difference2 : ℤ := third_number - second_number

-- Define the ratio of differences
def ratio : ℤ := 4
def next_difference : ℤ := difference2 * ratio

-- Define the fourth number
def fourth_number : ℤ := third_number - (-next_difference)

-- Theorem stating the proof problem
theorem find_fourth_number_in_sequence : fourth_number = 2030 :=
by sorry

end NUMINAMATH_GPT_find_fourth_number_in_sequence_l540_54058


namespace NUMINAMATH_GPT_right_triangle_acute_angles_l540_54083

variable (α β : ℝ)

noncomputable def prove_acute_angles (α β : ℝ) : Prop :=
  α + β = 90 ∧ 4 * α = 90

theorem right_triangle_acute_angles : 
  prove_acute_angles α β → α = 22.5 ∧ β = 67.5 := by
  sorry

end NUMINAMATH_GPT_right_triangle_acute_angles_l540_54083


namespace NUMINAMATH_GPT_part1_part2_l540_54016

theorem part1 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) :
  a*b ≤ 1 := sorry

theorem part2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + 4*b^2 = 1/(a*b) + 3) (hba : b > a) :
  1/a^3 - 1/b^3 ≥ 3 * (1/a - 1/b) := sorry

end NUMINAMATH_GPT_part1_part2_l540_54016


namespace NUMINAMATH_GPT_proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l540_54075

noncomputable def sin_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A + Real.sin B + Real.sin C) ≤ (3 / 2) * Real.sqrt 3

noncomputable def sin_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.sin A * Real.sin B * Real.sin C) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_sum_double_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (2 * A) + Real.cos (2 * B) + Real.cos (2 * C)) ≥ (-3 / 2)

noncomputable def cos_square_sum_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2) ≥ (3 / 4)

noncomputable def cos_half_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos (A / 2) * Real.cos (B / 2) * Real.cos (C / 2)) ≤ (3 / 8) * Real.sqrt 3

noncomputable def cos_product_ineq (A B C : ℝ) (hABC : A + B + C = π) : Prop := 
  (Real.cos A * Real.cos B * Real.cos C) ≤ (1 / 8)

theorem proof_sin_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_sum_ineq A B C hABC := sorry

theorem proof_sin_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : sin_product_ineq A B C hABC := sorry

theorem proof_cos_sum_double_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_sum_double_ineq A B C hABC := sorry

theorem proof_cos_square_sum_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_square_sum_ineq A B C hABC := sorry

theorem proof_cos_half_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_half_product_ineq A B C hABC := sorry

theorem proof_cos_product_ineq {A B C : ℝ} (hABC : A + B + C = π) : cos_product_ineq A B C hABC := sorry

end NUMINAMATH_GPT_proof_sin_sum_ineq_proof_sin_product_ineq_proof_cos_sum_double_ineq_proof_cos_square_sum_ineq_proof_cos_half_product_ineq_proof_cos_product_ineq_l540_54075


namespace NUMINAMATH_GPT_card_dealing_probability_l540_54081

noncomputable def probability_ace_then_ten_then_jack : ℚ :=
  let prob_ace := 4 / 52
  let prob_ten := 4 / 51
  let prob_jack := 4 / 50
  prob_ace * prob_ten * prob_jack

theorem card_dealing_probability :
  probability_ace_then_ten_then_jack = 16 / 33150 := by
  sorry

end NUMINAMATH_GPT_card_dealing_probability_l540_54081


namespace NUMINAMATH_GPT_sum_of_common_ratios_l540_54014

-- Definitions for the geometric sequence conditions
def geom_seq_a (m : ℝ) (s : ℝ) (n : ℕ) : ℝ := m * s^n
def geom_seq_b (m : ℝ) (t : ℝ) (n : ℕ) : ℝ := m * t^n

-- Theorem statement
theorem sum_of_common_ratios (m s t : ℝ) (h₀ : m ≠ 0) (h₁ : s ≠ t) 
    (h₂ : geom_seq_a m s 2 - geom_seq_b m t 2 = 3 * (geom_seq_a m s 1 - geom_seq_b m t 1)) :
    s + t = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l540_54014


namespace NUMINAMATH_GPT_carrots_per_bundle_l540_54099

theorem carrots_per_bundle (potatoes_total: ℕ) (potatoes_in_bundle: ℕ) (price_per_potato_bundle: ℝ) 
(carrot_total: ℕ) (price_per_carrot_bundle: ℝ) (total_revenue: ℝ) (carrots_per_bundle : ℕ) :
potatoes_total = 250 → potatoes_in_bundle = 25 → price_per_potato_bundle = 1.90 → 
carrot_total = 320 → price_per_carrot_bundle = 2 → total_revenue = 51 →
((carrots_per_bundle = carrot_total / ((total_revenue - (potatoes_total / potatoes_in_bundle) 
    * price_per_potato_bundle) / price_per_carrot_bundle))  ↔ carrots_per_bundle = 20) := by
  sorry

end NUMINAMATH_GPT_carrots_per_bundle_l540_54099
