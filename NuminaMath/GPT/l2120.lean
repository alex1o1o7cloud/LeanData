import Mathlib

namespace NUMINAMATH_GPT_red_button_probability_l2120_212056

theorem red_button_probability :
  let jarA_red := 6
  let jarA_blue := 9
  let jarA_total := jarA_red + jarA_blue
  let jarA_half := jarA_total / 2
  let removed_total := jarA_total - jarA_half
  let removed_red := removed_total / 2
  let removed_blue := removed_total / 2
  let jarA_red_remaining := jarA_red - removed_red
  let jarA_blue_remaining := jarA_blue - removed_blue
  let jarB_red := removed_red
  let jarB_blue := removed_blue
  let jarA_total_remaining := jarA_red_remaining + jarA_blue_remaining
  let jarB_total := jarB_red + jarB_blue
  (jarA_total = 15) →
  (jarA_red_remaining = 6 - removed_red) →
  (jarA_blue_remaining = 9 - removed_blue) →
  (jarB_red = removed_red) →
  (jarB_blue = removed_blue) →
  (jarA_red_remaining + jarA_blue_remaining = 9) →
  (jarB_red + jarB_blue = 6) →
  let prob_red_JarA := jarA_red_remaining / jarA_total_remaining
  let prob_red_JarB := jarB_red / jarB_total
  prob_red_JarA * prob_red_JarB = 1 / 6 := by sorry

end NUMINAMATH_GPT_red_button_probability_l2120_212056


namespace NUMINAMATH_GPT_greater_of_T_N_l2120_212006

/-- Define an 8x8 board and the number of valid domino placements. -/
def N : ℕ := 12988816

/-- A combinatorial number T representing the number of ways to place 24 dominoes on an 8x8 board. -/
axiom T : ℕ 

/-- We need to prove that T is greater than -N, where N is defined as 12988816. -/
theorem greater_of_T_N : T > - (N : ℤ) := sorry

end NUMINAMATH_GPT_greater_of_T_N_l2120_212006


namespace NUMINAMATH_GPT_fraction_increase_by_two_times_l2120_212036

theorem fraction_increase_by_two_times (x y : ℝ) : 
  let new_val := ((2 * x) * (2 * y)) / (2 * x + 2 * y)
  let original_val := (x * y) / (x + y)
  new_val = 2 * original_val := 
by
  sorry

end NUMINAMATH_GPT_fraction_increase_by_two_times_l2120_212036


namespace NUMINAMATH_GPT_mildred_total_oranges_l2120_212011

-- Conditions
def initial_oranges : ℕ := 77
def additional_oranges : ℕ := 2

-- Question/Goal
theorem mildred_total_oranges : initial_oranges + additional_oranges = 79 := by
  sorry

end NUMINAMATH_GPT_mildred_total_oranges_l2120_212011


namespace NUMINAMATH_GPT_crayons_left_l2120_212075

theorem crayons_left (start_crayons lost_crayons left_crayons : ℕ) 
  (h1 : start_crayons = 479) 
  (h2 : lost_crayons = 345) 
  (h3 : left_crayons = start_crayons - lost_crayons) : 
  left_crayons = 134 :=
sorry

end NUMINAMATH_GPT_crayons_left_l2120_212075


namespace NUMINAMATH_GPT_cube_surface_area_unchanged_l2120_212077

def cubeSurfaceAreaAfterCornersRemoved
  (original_side : ℕ)
  (corner_side : ℕ)
  (original_surface_area : ℕ)
  (number_of_corners : ℕ)
  (surface_reduction_per_corner : ℕ)
  (new_surface_addition_per_corner : ℕ) : Prop :=
  (original_side * original_side * 6 = original_surface_area) →
  (corner_side * corner_side * 3 = surface_reduction_per_corner) →
  (corner_side * corner_side * 3 = new_surface_addition_per_corner) →
  original_surface_area - (number_of_corners * surface_reduction_per_corner) + (number_of_corners * new_surface_addition_per_corner) = original_surface_area
  
theorem cube_surface_area_unchanged :
  cubeSurfaceAreaAfterCornersRemoved 4 1 96 8 3 3 :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_cube_surface_area_unchanged_l2120_212077


namespace NUMINAMATH_GPT_circle_equation_l2120_212020

open Real

theorem circle_equation (x y : ℝ) :
  let center := (2, -1)
  let line := (x + y = 7)
  (center.1 - 2)^2 + (center.2 + 1)^2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l2120_212020


namespace NUMINAMATH_GPT_total_dots_not_visible_l2120_212042

def total_dots_on_dice (n : ℕ): ℕ := n * 21
def visible_dots : ℕ := 1 + 1 + 2 + 3 + 4 + 4 + 5 + 6
def total_dice : ℕ := 4

theorem total_dots_not_visible :
  total_dots_on_dice total_dice - visible_dots = 58 := by
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_l2120_212042


namespace NUMINAMATH_GPT_average_age_without_teacher_l2120_212053

theorem average_age_without_teacher 
  (A : ℕ) 
  (h : 15 * A + 26 = 16 * (A + 1)) : 
  A = 10 :=
sorry

end NUMINAMATH_GPT_average_age_without_teacher_l2120_212053


namespace NUMINAMATH_GPT_find_base_l2120_212090

theorem find_base 
  (k : ℕ) 
  (h : 1 * k^2 + 3 * k^1 + 2 * k^0 = 30) : 
  k = 4 :=
  sorry

end NUMINAMATH_GPT_find_base_l2120_212090


namespace NUMINAMATH_GPT_find_original_price_l2120_212068

theorem find_original_price (a b x : ℝ) (h : x * (1 - 0.1) - a = b) : 
  x = (a + b) / (1 - 0.1) :=
sorry

end NUMINAMATH_GPT_find_original_price_l2120_212068


namespace NUMINAMATH_GPT_train_length_is_140_l2120_212069

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) (bridge_length_m : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let distance := speed_ms * time_s
  distance - bridge_length_m

theorem train_length_is_140 :
  train_length 45 30 235 = 140 := by
  sorry

end NUMINAMATH_GPT_train_length_is_140_l2120_212069


namespace NUMINAMATH_GPT_escalator_times_comparison_l2120_212044

variable (v v1 v2 l : ℝ)
variable (h_v_lt_v1 : v < v1)
variable (h_v1_lt_v2 : v1 < v2)

theorem escalator_times_comparison
  (h_cond : v < v1 ∧ v1 < v2) :
  (l / (v1 + v) + l / (v2 - v)) < (l / (v1 - v) + l / (v2 + v)) :=
  sorry

end NUMINAMATH_GPT_escalator_times_comparison_l2120_212044


namespace NUMINAMATH_GPT_extremum_point_is_three_l2120_212066

noncomputable def f (x : ℝ) : ℝ := (x - 2) / Real.exp x

theorem extremum_point_is_three {x₀ : ℝ} (h : ∀ x, f x₀ ≤ f x) : x₀ = 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_extremum_point_is_three_l2120_212066


namespace NUMINAMATH_GPT_sum_of_prime_factors_210630_l2120_212010

theorem sum_of_prime_factors_210630 : (2 + 3 + 5 + 7 + 17 + 59) = 93 := by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_210630_l2120_212010


namespace NUMINAMATH_GPT_can_determine_counterfeit_l2120_212040

-- Define the conditions of the problem
structure ProblemConditions where
  totalCoins : ℕ := 100
  exaggeration : ℕ

-- Define the problem statement
theorem can_determine_counterfeit (P : ProblemConditions) : 
  ∃ strategy : ℕ → Prop, 
    ∀ (k : ℕ), strategy P.exaggeration -> 
    (∀ i, i < 100 → (P.totalCoins = 100 ∧ ∃ n, n > 0 ∧ 
     ∀ j, j < P.totalCoins → (P.totalCoins = j + 1 ∨ P.totalCoins = 99 + j))) := 
sorry

end NUMINAMATH_GPT_can_determine_counterfeit_l2120_212040


namespace NUMINAMATH_GPT_calculate_f_g_of_1_l2120_212062

def f (x : ℝ) : ℝ := 4 * x + 3
def g (x : ℝ) : ℝ := (x + 2) ^ 2

theorem calculate_f_g_of_1 : f (g 1) = 39 :=
by
  -- Enable quick skippable proof with 'sorry'
  sorry

end NUMINAMATH_GPT_calculate_f_g_of_1_l2120_212062


namespace NUMINAMATH_GPT_volume_ratio_proof_l2120_212045

-- Definitions based on conditions
def edge_ratio (a b : ℝ) : Prop := a = 3 * b
def volume_ratio (V_large V_small : ℝ) : Prop := V_large = 27 * V_small

-- Problem statement
theorem volume_ratio_proof (e V_small V_large : ℝ) 
  (h1 : edge_ratio (3 * e) e)
  (h2 : volume_ratio V_large V_small) : 
  V_large / V_small = 27 := 
by sorry

end NUMINAMATH_GPT_volume_ratio_proof_l2120_212045


namespace NUMINAMATH_GPT_sum_abs_of_roots_l2120_212091

variables {p q r : ℤ}

theorem sum_abs_of_roots:
  p + q + r = 0 →
  p * q + q * r + r * p = -2023 →
  |p| + |q| + |r| = 94 := by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_sum_abs_of_roots_l2120_212091


namespace NUMINAMATH_GPT_evaluate_expression_l2120_212048

theorem evaluate_expression : 101^3 + 3 * 101^2 * 2 + 3 * 101 * 2^2 + 2^3 = 1092727 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2120_212048


namespace NUMINAMATH_GPT_smallest_distance_l2120_212016

open Complex

noncomputable def a := 2 + 4 * Complex.I
noncomputable def b := 8 + 6 * Complex.I

theorem smallest_distance (z w : ℂ)
    (hz : abs (z - a) = 2)
    (hw : abs (w - b) = 4) :
    abs (z - w) ≥ 2 * Real.sqrt 10 - 6 := by
  sorry

end NUMINAMATH_GPT_smallest_distance_l2120_212016


namespace NUMINAMATH_GPT_exists_b_for_a_ge_condition_l2120_212031

theorem exists_b_for_a_ge_condition (a : ℝ) (h : a ≥ -Real.sqrt 2 - 1 / 4) :
  ∃ b : ℝ, ∃ x y : ℝ, 
    y = x^2 - a ∧
    x^2 + y^2 + 8 * b^2 = 4 * b * (y - x) + 1 :=
sorry

end NUMINAMATH_GPT_exists_b_for_a_ge_condition_l2120_212031


namespace NUMINAMATH_GPT_num_zeros_in_decimal_representation_l2120_212085

theorem num_zeros_in_decimal_representation :
  let denom := 2^3 * 5^10
  let frac := (1 : ℚ) / denom
  ∃ n : ℕ, n = 7 ∧ (∃ (a : ℕ) (b : ℕ), frac = a / 10^b ∧ ∃ (k : ℕ), b = n + k + 3) :=
sorry

end NUMINAMATH_GPT_num_zeros_in_decimal_representation_l2120_212085


namespace NUMINAMATH_GPT_ruth_weekly_class_hours_l2120_212008

def hours_in_a_day : ℕ := 8
def days_in_a_week : ℕ := 5
def weekly_school_hours := hours_in_a_day * days_in_a_week

def math_class_percentage : ℚ := 0.25
def language_class_percentage : ℚ := 0.30
def science_class_percentage : ℚ := 0.20
def history_class_percentage : ℚ := 0.10

def math_hours := math_class_percentage * weekly_school_hours
def language_hours := language_class_percentage * weekly_school_hours
def science_hours := science_class_percentage * weekly_school_hours
def history_hours := history_class_percentage * weekly_school_hours

def total_class_hours := math_hours + language_hours + science_hours + history_hours

theorem ruth_weekly_class_hours : total_class_hours = 34 := by
  -- Calculation proof logic will go here
  sorry

end NUMINAMATH_GPT_ruth_weekly_class_hours_l2120_212008


namespace NUMINAMATH_GPT_M_necessary_for_N_l2120_212078

open Set

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem M_necessary_for_N : ∀ a : ℝ, a ∈ N → a ∈ M ∧ ¬(a ∈ M → a ∈ N) :=
by
  sorry

end NUMINAMATH_GPT_M_necessary_for_N_l2120_212078


namespace NUMINAMATH_GPT_polynomial_inequality_l2120_212046

theorem polynomial_inequality (x : ℝ) : -6 * x^2 + 2 * x - 8 < 0 :=
sorry

end NUMINAMATH_GPT_polynomial_inequality_l2120_212046


namespace NUMINAMATH_GPT_solve_weight_of_bowling_ball_l2120_212064

-- Conditions: Eight bowling balls equal the weight of five canoes
-- and four canoes weigh 120 pounds.

def weight_of_bowling_ball : Prop :=
  ∃ (b c : ℝ), (8 * b = 5 * c) ∧ (4 * c = 120) ∧ (b = 18.75)

theorem solve_weight_of_bowling_ball : weight_of_bowling_ball :=
  sorry

end NUMINAMATH_GPT_solve_weight_of_bowling_ball_l2120_212064


namespace NUMINAMATH_GPT_weight_feel_when_lowered_l2120_212043

-- Conditions from the problem
def num_plates : ℕ := 10
def weight_per_plate : ℝ := 30
def technology_increase : ℝ := 0.20
def incline_increase : ℝ := 0.15

-- Calculate the contributions
def total_weight_without_factors : ℝ := num_plates * weight_per_plate
def weight_with_technology : ℝ := total_weight_without_factors * (1 + technology_increase)
def weight_with_incline : ℝ := weight_with_technology * (1 + incline_increase)

-- Theorem statement we want to prove
theorem weight_feel_when_lowered : weight_with_incline = 414 := by
  sorry

end NUMINAMATH_GPT_weight_feel_when_lowered_l2120_212043


namespace NUMINAMATH_GPT_thabo_books_220_l2120_212041

def thabo_books_total (H PNF PF Total : ℕ) : Prop :=
  (H = 40) ∧
  (PNF = H + 20) ∧
  (PF = 2 * PNF) ∧
  (Total = H + PNF + PF)

theorem thabo_books_220 : ∃ H PNF PF Total : ℕ, thabo_books_total H PNF PF 220 :=
by {
  sorry
}

end NUMINAMATH_GPT_thabo_books_220_l2120_212041


namespace NUMINAMATH_GPT_example_number_is_not_octal_l2120_212067

-- Define a predicate that checks if a digit is valid in the octal system
def is_octal_digit (d : ℕ) : Prop :=
  d < 8

-- Define a predicate that checks if all digits in a number represented as list of ℕ are valid octal digits
def is_octal_number (n : List ℕ) : Prop :=
  ∀ d ∈ n, is_octal_digit d

-- Example number represented as a list of its digits
def example_number : List ℕ := [2, 8, 5, 3]

-- The statement we aim to prove
theorem example_number_is_not_octal : ¬ is_octal_number example_number := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_example_number_is_not_octal_l2120_212067


namespace NUMINAMATH_GPT_ratio_Polly_Willy_l2120_212065

theorem ratio_Polly_Willy (P S W : ℝ) (h1 : P / S = 4 / 5) (h2 : S / W = 5 / 2) :
  P / W = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_Polly_Willy_l2120_212065


namespace NUMINAMATH_GPT_cost_per_component_l2120_212059

theorem cost_per_component (C : ℝ) : 
  (150 * C + 150 * 4 + 16500 = 150 * 193.33) → 
  C = 79.33 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_per_component_l2120_212059


namespace NUMINAMATH_GPT_compare_powers_l2120_212086

theorem compare_powers : 2^24 < 10^8 ∧ 10^8 < 5^12 :=
by 
  -- proofs omitted
  sorry

end NUMINAMATH_GPT_compare_powers_l2120_212086


namespace NUMINAMATH_GPT_silver_status_families_l2120_212080

theorem silver_status_families 
  (goal : ℕ) 
  (remaining : ℕ) 
  (bronze_families : ℕ) 
  (bronze_donation : ℕ) 
  (gold_families : ℕ) 
  (gold_donation : ℕ) 
  (silver_donation : ℕ) 
  (total_raised_so_far : goal - remaining = 700)
  (amount_raised_by_bronze : bronze_families * bronze_donation = 250)
  (amount_raised_by_gold : gold_families * gold_donation = 100)
  (amount_raised_by_silver : 700 - 250 - 100 = 350) :
  ∃ (s : ℕ), s * silver_donation = 350 ∧ s = 7 :=
by
  sorry

end NUMINAMATH_GPT_silver_status_families_l2120_212080


namespace NUMINAMATH_GPT_expression_equals_one_l2120_212063

theorem expression_equals_one (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_sum : a + b + c = 1) :
  (a^3 * b^3 / ((a^3 - b * c) * (b^3 - a * c)) + a^3 * c^3 / ((a^3 - b * c) * (c^3 - a * b)) +
    b^3 * c^3 / ((b^3 - a * c) * (c^3 - a * b))) = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_one_l2120_212063


namespace NUMINAMATH_GPT_odd_function_equiv_l2120_212072

noncomputable def odd_function (f : ℝ → ℝ) :=
∀ x : ℝ, f (-x) = -f (x)

theorem odd_function_equiv (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f (x)) ↔ (∀ x : ℝ, f (-(-x)) = -f (-x)) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_equiv_l2120_212072


namespace NUMINAMATH_GPT_problem_l2120_212096

theorem problem (m : ℕ) (h : m = 16^2023) : m / 8 = 2^8089 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l2120_212096


namespace NUMINAMATH_GPT_symmetric_point_of_P_l2120_212025

-- Define a point in the Cartesian coordinate system
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define central symmetry with respect to the origin
def symmetric (p : Point) : Point :=
  { x := -p.x, y := -p.y }

-- Given point P with coordinates (1, -2)
def P : Point := { x := 1, y := -2 }

-- The theorem to be proved: the symmetric point of P is (-1, 2)
theorem symmetric_point_of_P :
  symmetric P = { x := -1, y := 2 } :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_symmetric_point_of_P_l2120_212025


namespace NUMINAMATH_GPT_average_of_remaining_primes_l2120_212026

theorem average_of_remaining_primes (avg30: ℕ) (avg15: ℕ) (h1 : avg30 = 110) (h2 : avg15 = 95) : 
  ((30 * avg30 - 15 * avg15) / 15) = 125 := 
by
  -- Proof
  sorry

end NUMINAMATH_GPT_average_of_remaining_primes_l2120_212026


namespace NUMINAMATH_GPT_carrots_picked_first_day_l2120_212024

theorem carrots_picked_first_day (X : ℕ) 
  (H1 : X - 10 + 47 = 60) : X = 23 :=
by 
  -- We state the proof steps here, completing the proof with sorry
  sorry

end NUMINAMATH_GPT_carrots_picked_first_day_l2120_212024


namespace NUMINAMATH_GPT_tan_theta_value_l2120_212083

theorem tan_theta_value (θ : ℝ) (h1 : Real.sin θ = 3/5) (h2 : Real.cos θ = -4/5) : 
  Real.tan θ = -3/4 :=
  sorry

end NUMINAMATH_GPT_tan_theta_value_l2120_212083


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l2120_212028

noncomputable def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_coordinates :
  spherical_to_rectangular 3 (3 * Real.pi / 2) (Real.pi / 3) = (0, -3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l2120_212028


namespace NUMINAMATH_GPT_minimum_students_exceeds_1000_l2120_212084

theorem minimum_students_exceeds_1000 (n : ℕ) :
  (∃ k : ℕ, k > 1000 ∧ k % 10 = 0 ∧ k % 14 = 0 ∧ k % 18 = 0 ∧ n = k) ↔ n = 1260 :=
sorry

end NUMINAMATH_GPT_minimum_students_exceeds_1000_l2120_212084


namespace NUMINAMATH_GPT_cos_alpha_value_l2120_212081

theorem cos_alpha_value (α : ℝ) (h₀ : 0 < α ∧ α < 90) (h₁ : Real.sin (α - 45) = - (Real.sqrt 2 / 10)) : 
  Real.cos α = 4 / 5 := 
sorry

end NUMINAMATH_GPT_cos_alpha_value_l2120_212081


namespace NUMINAMATH_GPT_complex_div_l2120_212061

open Complex

theorem complex_div (i : ℂ) (hi : i = Complex.I) : 
  (6 + 7 * i) / (1 + 2 * i) = 4 - i := 
by 
  sorry

end NUMINAMATH_GPT_complex_div_l2120_212061


namespace NUMINAMATH_GPT_simple_interest_years_l2120_212088

theorem simple_interest_years (SI P : ℝ) (R : ℝ) (T : ℝ) 
  (hSI : SI = 200) 
  (hP : P = 1600) 
  (hR : R = 3.125) : 
  T = 4 :=
by 
  sorry

end NUMINAMATH_GPT_simple_interest_years_l2120_212088


namespace NUMINAMATH_GPT_inequality_proof_l2120_212051

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2120_212051


namespace NUMINAMATH_GPT_rabbit_total_apples_90_l2120_212001

-- Define the number of apples each animal places in a basket
def rabbit_apple_per_basket : ℕ := 5
def deer_apple_per_basket : ℕ := 6

-- Define the number of baskets each animal uses
variable (h_r h_d : ℕ)

-- Define the total number of apples collected by both animals
def total_apples : ℕ := rabbit_apple_per_basket * h_r

-- Conditions
axiom deer_basket_count_eq_rabbit : h_d = h_r - 3
axiom same_total_apples : total_apples = deer_apple_per_basket * h_d

-- Goal: Prove that the total number of apples the rabbit collected is 90
theorem rabbit_total_apples_90 : total_apples = 90 := sorry

end NUMINAMATH_GPT_rabbit_total_apples_90_l2120_212001


namespace NUMINAMATH_GPT_roller_coaster_cars_l2120_212092

theorem roller_coaster_cars
  (people : ℕ)
  (runs : ℕ)
  (seats_per_car : ℕ)
  (people_per_run : ℕ)
  (h1 : people = 84)
  (h2 : runs = 6)
  (h3 : seats_per_car = 2)
  (h4 : people_per_run = people / runs) :
  (people_per_run / seats_per_car) = 7 :=
by
  sorry

end NUMINAMATH_GPT_roller_coaster_cars_l2120_212092


namespace NUMINAMATH_GPT_find_f_107_l2120_212012

-- Define the conditions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def periodic_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = -f x

def piecewise_function (f : ℝ → ℝ) : Prop :=
  ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x / 5

-- Main theorem to prove based on the conditions
theorem find_f_107 (f : ℝ → ℝ)
  (h_periodic : periodic_function f)
  (h_piece : piecewise_function f)
  (h_even : even_function f) : f 107 = 1 / 5 :=
sorry

end NUMINAMATH_GPT_find_f_107_l2120_212012


namespace NUMINAMATH_GPT_math_problem_l2120_212054

variable (a b c d : ℝ)
variable (h1 : a > b)
variable (h2 : c < d)

theorem math_problem : a - c > b - d :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_l2120_212054


namespace NUMINAMATH_GPT_max_complete_dresses_l2120_212050

namespace DressMaking

-- Define the initial quantities of fabric
def initial_silk : ℕ := 600
def initial_satin : ℕ := 400
def initial_chiffon : ℕ := 350

-- Define the quantities given to each of 8 friends
def silk_per_friend : ℕ := 15
def satin_per_friend : ℕ := 10
def chiffon_per_friend : ℕ := 5

-- Define the quantities required to make one dress
def silk_per_dress : ℕ := 5
def satin_per_dress : ℕ := 3
def chiffon_per_dress : ℕ := 2

-- Calculate the remaining quantities
def remaining_silk : ℕ := initial_silk - 8 * silk_per_friend
def remaining_satin : ℕ := initial_satin - 8 * satin_per_friend
def remaining_chiffon : ℕ := initial_chiffon - 8 * chiffon_per_friend

-- Calculate the maximum number of dresses that can be made
def max_dresses_silk : ℕ := remaining_silk / silk_per_dress
def max_dresses_satin : ℕ := remaining_satin / satin_per_dress
def max_dresses_chiffon : ℕ := remaining_chiffon / chiffon_per_dress

-- The main theorem indicating the number of complete dresses
theorem max_complete_dresses : max_dresses_silk = 96 ∧ max_dresses_silk ≤ max_dresses_satin ∧ max_dresses_silk ≤ max_dresses_chiffon := by
  sorry

end DressMaking

end NUMINAMATH_GPT_max_complete_dresses_l2120_212050


namespace NUMINAMATH_GPT_gcd_lcm_sum_eq_l2120_212095

-- Define the two numbers
def a : ℕ := 72
def b : ℕ := 8712

-- Define the GCD and LCM functions.
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Define the sum of the GCD and LCM.
def sum_gcd_lcm : ℕ := gcd_ab + lcm_ab

-- The theorem we want to prove
theorem gcd_lcm_sum_eq : sum_gcd_lcm = 26160 := by
  -- Details of the proof would go here
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_eq_l2120_212095


namespace NUMINAMATH_GPT_find_a_b_l2120_212099

noncomputable def curve (x a b : ℝ) : ℝ := x^2 + a * x + b

noncomputable def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem find_a_b (a b : ℝ) :
  (∃ (y : ℝ) (x : ℝ), (y = curve x a b) ∧ tangent_line 0 b ∧ (2 * 0 + a = -1) ∧ (0 - b + 1 = 0)) ->
  a = -1 ∧ b = 1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_b_l2120_212099


namespace NUMINAMATH_GPT_abc_sum_is_32_l2120_212023

theorem abc_sum_is_32 (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a * b + c = 31) (h5 : b * c + a = 31) (h6 : a * c + b = 31) : 
  a + b + c = 32 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_abc_sum_is_32_l2120_212023


namespace NUMINAMATH_GPT_handshaking_remainder_div_1000_l2120_212052

/-- Given eleven people where each person shakes hands with exactly three others, 
  let handshaking_count be the number of distinct handshaking arrangements.
  Find the remainder when handshaking_count is divided by 1000. -/
def handshaking_count (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) : Nat :=
  sorry

theorem handshaking_remainder_div_1000 (P : Type) [Fintype P] [DecidableEq P] (hP : Fintype.card P = 11)
  (handshakes : P → Finset P) (H : ∀ p : P, Fintype.card (handshakes p) = 3) :
  (handshaking_count P hP handshakes H) % 1000 = 800 :=
sorry

end NUMINAMATH_GPT_handshaking_remainder_div_1000_l2120_212052


namespace NUMINAMATH_GPT_carlos_blocks_l2120_212002

theorem carlos_blocks (initial_blocks : ℕ) (blocks_given : ℕ) (remaining_blocks : ℕ) 
  (h1 : initial_blocks = 58) (h2 : blocks_given = 21) : remaining_blocks = 37 :=
by
  sorry

end NUMINAMATH_GPT_carlos_blocks_l2120_212002


namespace NUMINAMATH_GPT_range_of_m_l2120_212009

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 -> (m^2 - m) * 2^x - (1/2)^x < 1) →
  -2 < m ∧ m < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2120_212009


namespace NUMINAMATH_GPT_minimum_value_fraction_l2120_212049

variable (a b c : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b + c ≥ a)

theorem minimum_value_fraction : (b / c + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end NUMINAMATH_GPT_minimum_value_fraction_l2120_212049


namespace NUMINAMATH_GPT_kittens_weight_problem_l2120_212071

theorem kittens_weight_problem
  (w_lightest : ℕ)
  (w_heaviest : ℕ)
  (w_total : ℕ)
  (total_lightest : w_lightest = 80)
  (total_heaviest : w_heaviest = 200)
  (total_weight : w_total = 500) :
  ∃ (n : ℕ), n = 11 :=
by sorry

end NUMINAMATH_GPT_kittens_weight_problem_l2120_212071


namespace NUMINAMATH_GPT_number_of_students_l2120_212094

-- Definitions based on the problem conditions
def mini_cupcakes := 14
def donut_holes := 12
def desserts_per_student := 2

-- Total desserts calculation
def total_desserts := mini_cupcakes + donut_holes

-- Prove the number of students
theorem number_of_students : total_desserts / desserts_per_student = 13 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_number_of_students_l2120_212094


namespace NUMINAMATH_GPT_necessarily_positive_expression_l2120_212013

theorem necessarily_positive_expression
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 2)
  (hb : -2 < b ∧ b < 0)
  (hc : 0 < c ∧ c < 3) :
  0 < b + 3 * b^2 := 
sorry

end NUMINAMATH_GPT_necessarily_positive_expression_l2120_212013


namespace NUMINAMATH_GPT_solve_for_x_l2120_212035

theorem solve_for_x (x : ℝ) (h : (10 - 6 * x)^ (1 / 3) = -2) : x = 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2120_212035


namespace NUMINAMATH_GPT_fruits_eaten_total_l2120_212039

variable (apples blueberries bonnies : ℕ)

noncomputable def total_fruits_eaten : ℕ :=
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 / 4 * third_dog_bonnies
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies

theorem fruits_eaten_total:
  let third_dog_bonnies := 60
  let second_dog_blueberries := 3 * third_dog_bonnies / 4
  let first_dog_apples := 3 * second_dog_blueberries
  first_dog_apples + second_dog_blueberries + third_dog_bonnies = 240 := by
  sorry

end NUMINAMATH_GPT_fruits_eaten_total_l2120_212039


namespace NUMINAMATH_GPT_scientific_notation_l2120_212093

theorem scientific_notation : (10374 * 10^9 : Real) = 1.037 * 10^13 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l2120_212093


namespace NUMINAMATH_GPT_sum_of_first_five_terms_l2120_212021

def a (n : ℕ) : ℚ := 1 / (n * (n + 1))

theorem sum_of_first_five_terms :
  (a 1 + a 2 + a 3 + a 4 + a 5) = 5 / 6 := 
by 
  unfold a
  -- sorry is used as a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_l2120_212021


namespace NUMINAMATH_GPT_fraction_received_l2120_212073

theorem fraction_received (total_money : ℝ) (spent_ratio : ℝ) (spent_amount : ℝ) (remaining_amount : ℝ) (fraction_received : ℝ) :
  total_money = 240 ∧ spent_ratio = 1/5 ∧ spent_amount = spent_ratio * total_money ∧ remaining_amount = 132 ∧ spent_amount + remaining_amount = fraction_received * total_money →
  fraction_received = 3 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_received_l2120_212073


namespace NUMINAMATH_GPT_people_visited_neither_l2120_212019

theorem people_visited_neither (total_people iceland_visitors norway_visitors both_visitors : ℕ)
  (h1 : total_people = 100)
  (h2 : iceland_visitors = 55)
  (h3 : norway_visitors = 43)
  (h4 : both_visitors = 61) :
  total_people - (iceland_visitors + norway_visitors - both_visitors) = 63 :=
by
  sorry

end NUMINAMATH_GPT_people_visited_neither_l2120_212019


namespace NUMINAMATH_GPT_log_division_simplification_l2120_212047

theorem log_division_simplification (log_base_half : ℝ → ℝ) (log_base_half_pow5 :  log_base_half (2 ^ 5) = 5 * log_base_half 2)
  (log_base_half_pow1 : log_base_half (2 ^ 1) = 1 * log_base_half 2) :
  (log_base_half 32) / (log_base_half 2) = 5 :=
sorry

end NUMINAMATH_GPT_log_division_simplification_l2120_212047


namespace NUMINAMATH_GPT_chi_squared_confidence_level_l2120_212018

theorem chi_squared_confidence_level 
  (chi_squared_value : ℝ)
  (p_value_3841 : ℝ)
  (p_value_5024 : ℝ)
  (h1 : chi_squared_value = 4.073)
  (h2 : p_value_3841 = 0.05)
  (h3 : p_value_5024 = 0.025)
  (h4 : 3.841 ≤ chi_squared_value ∧ chi_squared_value < 5.024) :
  ∃ confidence_level : ℝ, confidence_level = 0.95 :=
by 
  sorry

end NUMINAMATH_GPT_chi_squared_confidence_level_l2120_212018


namespace NUMINAMATH_GPT_sum_of_digits_decrease_by_10_percent_l2120_212055

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum -- Assuming this method computes the sum of the digits

theorem sum_of_digits_decrease_by_10_percent :
  ∃ (n m : ℕ), m = 11 * n / 10 ∧ sum_of_digits m = 9 * sum_of_digits n / 10 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_decrease_by_10_percent_l2120_212055


namespace NUMINAMATH_GPT_hilary_big_toenails_count_l2120_212037

def fit_toenails (total_capacity : ℕ) (big_toenail_space_ratio : ℕ) (current_regular : ℕ) (additional_regular : ℕ) : ℕ :=
  (total_capacity - (current_regular + additional_regular)) / big_toenail_space_ratio

theorem hilary_big_toenails_count :
  fit_toenails 100 2 40 20 = 10 :=
  by
    sorry

end NUMINAMATH_GPT_hilary_big_toenails_count_l2120_212037


namespace NUMINAMATH_GPT_admission_charge_for_adult_l2120_212082

theorem admission_charge_for_adult 
(admission_charge_per_child : ℝ)
(total_paid : ℝ)
(children_count : ℕ)
(admission_charge_for_adult : ℝ) :
admission_charge_per_child = 0.75 →
total_paid = 3.25 →
children_count = 3 →
admission_charge_for_adult + admission_charge_per_child * children_count = total_paid →
admission_charge_for_adult = 1.00 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_admission_charge_for_adult_l2120_212082


namespace NUMINAMATH_GPT_min_value_PA_minus_PF_l2120_212098

noncomputable def ellipse_condition : Prop :=
  ∃ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1)

noncomputable def focal_property (x y : ℝ) (P : ℝ × ℝ) : Prop :=
  dist P (2, 4) - dist P (1, 0) = 1

theorem min_value_PA_minus_PF :
  ∀ (P : ℝ × ℝ), 
    (∃ (x y : ℝ), x^2 / 4 + y^2 / 3 = 1) 
    → ∃ (a b : ℝ), a = 2 ∧ b = 4 ∧ focal_property x y P :=
  sorry

end NUMINAMATH_GPT_min_value_PA_minus_PF_l2120_212098


namespace NUMINAMATH_GPT_ana_wins_probability_l2120_212003

noncomputable def probability_ana_wins : ℚ :=
  (1 / 2) ^ 5 / (1 - (1 / 2) ^ 5)

theorem ana_wins_probability :
  probability_ana_wins = 1 / 31 :=
by
  sorry

end NUMINAMATH_GPT_ana_wins_probability_l2120_212003


namespace NUMINAMATH_GPT_inequality_abc_l2120_212087

theorem inequality_abc 
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) : 
  a^2 * b^2 + b^2 * c^2 + a^2 * c^2 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_abc_l2120_212087


namespace NUMINAMATH_GPT_find_cost_price_l2120_212017

theorem find_cost_price (C S : ℝ) (h1 : S = 1.35 * C) (h2 : S - 25 = 0.98 * C) : C = 25 / 0.37 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l2120_212017


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l2120_212033

noncomputable def P : Set ℝ := {x | 0 < Real.log x / Real.log 8 ∧ Real.log x / Real.log 8 < 2 * (Real.log 3 / Real.log 8)}
noncomputable def Q : Set ℝ := {x | 2 / (2 - x) > 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {x | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l2120_212033


namespace NUMINAMATH_GPT_correct_equation_D_l2120_212015

theorem correct_equation_D : (|5 - 3| = - (3 - 5)) :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_D_l2120_212015


namespace NUMINAMATH_GPT_count_two_digit_prime_with_digit_sum_10_l2120_212089

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_two_digit_prime_with_digit_sum_10 : 
  (∃ n1 n2 n3 : ℕ, 
    (sum_of_digits n1 = 10 ∧ is_prime n1 ∧ 10 ≤ n1 ∧ n1 < 100) ∧
    (sum_of_digits n2 = 10 ∧ is_prime n2 ∧ 10 ≤ n2 ∧ n2 < 100) ∧
    (sum_of_digits n3 = 10 ∧ is_prime n3 ∧ 10 ≤ n3 ∧ n3 < 100) ∧
    n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3 ) ∧
  ∀ n : ℕ, 
    (sum_of_digits n = 10 ∧ is_prime n ∧ 10 ≤ n ∧ n < 100)
    → (n = n1 ∨ n = n2 ∨ n = n3) :=
sorry

end NUMINAMATH_GPT_count_two_digit_prime_with_digit_sum_10_l2120_212089


namespace NUMINAMATH_GPT_profit_after_discount_l2120_212027

noncomputable def purchase_price : ℝ := 100
noncomputable def increase_rate : ℝ := 0.25
noncomputable def discount_rate : ℝ := 0.10

theorem profit_after_discount :
  let selling_price := purchase_price * (1 + increase_rate)
  let discounted_price := selling_price * (1 - discount_rate)
  let profit := discounted_price - purchase_price
  profit = 12.5 :=
by
  sorry 

end NUMINAMATH_GPT_profit_after_discount_l2120_212027


namespace NUMINAMATH_GPT_triplet_not_equal_to_one_l2120_212097

def A := (1/2, 1/3, 1/6)
def B := (2, -2, 1)
def C := (0.1, 0.3, 0.6)
def D := (1.1, -2.1, 1.0)
def E := (-3/2, -5/2, 5)

theorem triplet_not_equal_to_one (ha : A = (1/2, 1/3, 1/6))
                                (hb : B = (2, -2, 1))
                                (hc : C = (0.1, 0.3, 0.6))
                                (hd : D = (1.1, -2.1, 1.0))
                                (he : E = (-3/2, -5/2, 5)) :
  (1/2 + 1/3 + 1/6 = 1) ∧
  (2 + -2 + 1 = 1) ∧
  (0.1 + 0.3 + 0.6 = 1) ∧
  (1.1 + -2.1 + 1.0 ≠ 1) ∧
  (-3/2 + -5/2 + 5 = 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_triplet_not_equal_to_one_l2120_212097


namespace NUMINAMATH_GPT_solve_equation_l2120_212022

theorem solve_equation (x : ℝ) : 2 * x + 17 = 32 - 3 * x → x = 3 := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l2120_212022


namespace NUMINAMATH_GPT_talent_show_girls_count_l2120_212074

theorem talent_show_girls_count (B G : ℕ) (h1 : B + G = 34) (h2 : G = B + 22) : G = 28 :=
by
  sorry

end NUMINAMATH_GPT_talent_show_girls_count_l2120_212074


namespace NUMINAMATH_GPT_max_integer_k_l2120_212070

-- Definitions of the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * x^2 - 2 * x
noncomputable def g' (x : ℝ) : ℝ := x - 2

-- Definition of the inequality condition
theorem max_integer_k (k : ℝ) : 
  (∀ x : ℝ, x > 2 → k * (x - 2) < x * f x + 2 * g' x + 3) ↔
  k ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_integer_k_l2120_212070


namespace NUMINAMATH_GPT_one_bag_covers_250_sqfeet_l2120_212014

noncomputable def lawn_length : ℝ := 22
noncomputable def lawn_width : ℝ := 36
noncomputable def bags_count : ℝ := 4
noncomputable def extra_area : ℝ := 208

noncomputable def lawn_area : ℝ := lawn_length * lawn_width
noncomputable def total_covered_area : ℝ := lawn_area + extra_area
noncomputable def one_bag_area : ℝ := total_covered_area / bags_count

theorem one_bag_covers_250_sqfeet :
  one_bag_area = 250 := 
by
  sorry

end NUMINAMATH_GPT_one_bag_covers_250_sqfeet_l2120_212014


namespace NUMINAMATH_GPT_find_unknown_gift_l2120_212034

def money_from_aunt : ℝ := 9
def money_from_uncle : ℝ := 9
def money_from_bestfriend1 : ℝ := 22
def money_from_bestfriend2 : ℝ := 22
def money_from_bestfriend3 : ℝ := 22
def money_from_sister : ℝ := 7
def mean_money : ℝ := 16.3
def number_of_gifts : ℕ := 7

theorem find_unknown_gift (X : ℝ)
  (h1: money_from_aunt = 9)
  (h2: money_from_uncle = 9)
  (h3: money_from_bestfriend1 = 22)
  (h4: money_from_bestfriend2 = 22)
  (h5: money_from_bestfriend3 = 22)
  (h6: money_from_sister = 7)
  (h7: mean_money = 16.3)
  (h8: number_of_gifts = 7)
  : X = 23.1 := sorry

end NUMINAMATH_GPT_find_unknown_gift_l2120_212034


namespace NUMINAMATH_GPT_megan_eggs_per_meal_l2120_212007

-- Define the initial conditions
def initial_eggs_from_store : Nat := 12
def initial_eggs_from_neighbor : Nat := 12
def eggs_used_for_omelet : Nat := 2
def eggs_used_for_cake : Nat := 4
def meals_to_divide : Nat := 3

-- Calculate various steps
def total_initial_eggs : Nat := initial_eggs_from_store + initial_eggs_from_neighbor
def eggs_after_cooking : Nat := total_initial_eggs - eggs_used_for_omelet - eggs_used_for_cake
def eggs_after_giving_away : Nat := eggs_after_cooking / 2
def eggs_per_meal : Nat := eggs_after_giving_away / meals_to_divide

-- State the theorem to prove the value of eggs_per_meal
theorem megan_eggs_per_meal : eggs_per_meal = 3 := by
  sorry

end NUMINAMATH_GPT_megan_eggs_per_meal_l2120_212007


namespace NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_div_2_l2120_212057

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_600_eq_neg_sqrt_3_div_2_l2120_212057


namespace NUMINAMATH_GPT_sum_of_digits_l2120_212030

noncomputable def digits_divisibility (C F : ℕ) : Prop :=
  (C >= 0 ∧ C <= 9 ∧ F >= 0 ∧ F <= 9) ∧
  (C + 8 + 5 + 4 + F + 7 + 2) % 9 = 0 ∧
  (100 * 4 + 10 * F + 72) % 8 = 0

theorem sum_of_digits (C F : ℕ) (h : digits_divisibility C F) : C + F = 10 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l2120_212030


namespace NUMINAMATH_GPT_number_of_female_students_in_sample_l2120_212032

theorem number_of_female_students_in_sample (male_students female_students sample_size : ℕ)
  (h1 : male_students = 560)
  (h2 : female_students = 420)
  (h3 : sample_size = 280) :
  (female_students * sample_size) / (male_students + female_students) = 120 := 
sorry

end NUMINAMATH_GPT_number_of_female_students_in_sample_l2120_212032


namespace NUMINAMATH_GPT_necessary_sufficient_condition_l2120_212079

theorem necessary_sufficient_condition (a : ℝ) :
  (∃ x : ℝ, ax^2 + 2 * x + 1 = 0 ∧ x < 0) ↔ a ≤ 1 := sorry

end NUMINAMATH_GPT_necessary_sufficient_condition_l2120_212079


namespace NUMINAMATH_GPT_combined_area_correct_l2120_212000

noncomputable def breadth : ℝ := 20
noncomputable def length : ℝ := 1.15 * breadth
noncomputable def area_rectangle : ℝ := 460
noncomputable def radius_semicircle : ℝ := breadth / 2
noncomputable def area_semicircle : ℝ := (1/2) * Real.pi * radius_semicircle^2
noncomputable def combined_area : ℝ := area_rectangle + area_semicircle

theorem combined_area_correct : combined_area = 460 + 50 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_combined_area_correct_l2120_212000


namespace NUMINAMATH_GPT_symmetric_point_coordinates_l2120_212076

theorem symmetric_point_coordinates (M N : ℝ × ℝ) (x y : ℝ) 
  (hM : M = (-2, 1)) 
  (hN_symmetry : N = (M.1, -M.2)) : N = (-2, -1) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_coordinates_l2120_212076


namespace NUMINAMATH_GPT_brick_height_l2120_212038

/-- A certain number of bricks, each measuring 25 cm x 11.25 cm x some height, 
are needed to build a wall of 8 m x 6 m x 22.5 cm. 
If 6400 bricks are needed, prove that the height of each brick is 6 cm. -/
theorem brick_height (h : ℝ) : 
  6400 * (25 * 11.25 * h) = (800 * 600 * 22.5) → h = 6 :=
by
  sorry

end NUMINAMATH_GPT_brick_height_l2120_212038


namespace NUMINAMATH_GPT_area_of_triangle_DEF_l2120_212005

-- Definitions of the given conditions
def angle_D : ℝ := 45
def DF : ℝ := 4
def DE : ℝ := DF -- Because it's a 45-45-90 triangle

-- Leam statement proving the area of the triangle
theorem area_of_triangle_DEF : 
  (1 / 2) * DE * DF = 8 := by
  -- Since DE = DF = 4, the area of the triangle can be computed
  sorry

end NUMINAMATH_GPT_area_of_triangle_DEF_l2120_212005


namespace NUMINAMATH_GPT_find_common_students_l2120_212058

theorem find_common_students
  (total_english : ℕ)
  (total_math : ℕ)
  (difference_only_english_math : ℕ)
  (both_english_math : ℕ) :
  total_english = both_english_math + (both_english_math + 10) →
  total_math = both_english_math + both_english_math →
  difference_only_english_math = 10 →
  total_english = 30 →
  total_math = 20 →
  both_english_math = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_common_students_l2120_212058


namespace NUMINAMATH_GPT_knights_and_liars_l2120_212004

-- Define the conditions: 
variables (K L : ℕ) 

-- Total number of council members is 101
def total_members : Prop := K + L = 101

-- Inequality conditions
def knight_inequality : Prop := L > (K + L - 1) / 2
def liar_inequality : Prop := K <= (K + L - 1) / 2

-- The theorem we need to prove
theorem knights_and_liars (K L : ℕ) (h1 : total_members K L) (h2 : knight_inequality K L) (h3 : liar_inequality K L) : K = 50 ∧ L = 51 :=
by {
  sorry
}

end NUMINAMATH_GPT_knights_and_liars_l2120_212004


namespace NUMINAMATH_GPT_find_arithmetic_sequence_l2120_212060

theorem find_arithmetic_sequence (a d : ℝ) (h1 : (a - d) + a + (a + d) = 12) (h2 : (a - d) * a * (a + d) = 48) :
  (a = 4 ∧ d = 2) ∨ (a = 4 ∧ d = -2) :=
sorry

end NUMINAMATH_GPT_find_arithmetic_sequence_l2120_212060


namespace NUMINAMATH_GPT_Joan_orange_balloons_l2120_212029

theorem Joan_orange_balloons (originally_has : ℕ) (received : ℕ) (final_count : ℕ) 
  (h1 : originally_has = 8) (h2 : received = 2) : 
  final_count = 10 := by
  sorry

end NUMINAMATH_GPT_Joan_orange_balloons_l2120_212029
