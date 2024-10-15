import Mathlib

namespace NUMINAMATH_GPT_negation_of_p_is_neg_p_l420_42097

-- Define proposition p
def p : Prop := ∃ x : ℝ, x^2 + x - 1 ≥ 0

-- Define the negation of p
def neg_p : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

theorem negation_of_p_is_neg_p : ¬p = neg_p := by
  -- The proof is omitted as per the instruction
  sorry

end NUMINAMATH_GPT_negation_of_p_is_neg_p_l420_42097


namespace NUMINAMATH_GPT_find_a_from_roots_l420_42027

theorem find_a_from_roots (a : ℝ) :
  let A := {x | (x = a) ∨ (x = a - 1)}
  2 ∈ A → a = 2 ∨ a = 3 :=
by
  intros A h
  sorry

end NUMINAMATH_GPT_find_a_from_roots_l420_42027


namespace NUMINAMATH_GPT_correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l420_42026

theorem correct_operation_B (a : ℝ) : a^3 / a = a^2 := 
by sorry

theorem incorrect_operation_A (a : ℝ) : a^2 + a^5 ≠ a^7 := 
by sorry

theorem incorrect_operation_C (a : ℝ) : (3 * a^2)^2 ≠ 6 * a^4 := 
by sorry

theorem incorrect_operation_D (a b : ℝ) : (a - b)^2 ≠ a^2 - b^2 := 
by sorry

end NUMINAMATH_GPT_correct_operation_B_incorrect_operation_A_incorrect_operation_C_incorrect_operation_D_l420_42026


namespace NUMINAMATH_GPT_rows_of_pies_l420_42067

theorem rows_of_pies (baked_pecan_pies : ℕ) (baked_apple_pies : ℕ) (pies_per_row : ℕ) : 
  baked_pecan_pies = 16 ∧ baked_apple_pies = 14 ∧ pies_per_row = 5 → 
  (baked_pecan_pies + baked_apple_pies) / pies_per_row = 6 :=
by
  sorry

end NUMINAMATH_GPT_rows_of_pies_l420_42067


namespace NUMINAMATH_GPT_non_congruent_right_triangles_unique_l420_42039

theorem non_congruent_right_triangles_unique :
  ∃! (a: ℝ) (b: ℝ) (c: ℝ), a > 0 ∧ b = 2 * a ∧ c = a * Real.sqrt 5 ∧
  (3 * a + a * Real.sqrt 5 - a^2 = a * Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_non_congruent_right_triangles_unique_l420_42039


namespace NUMINAMATH_GPT_total_cost_over_8_weeks_l420_42086

def cost_per_weekday_edition : ℝ := 0.50
def cost_per_sunday_edition : ℝ := 2.00
def num_weekday_editions_per_week : ℕ := 3
def duration_in_weeks : ℕ := 8

theorem total_cost_over_8_weeks :
  (num_weekday_editions_per_week * cost_per_weekday_edition + cost_per_sunday_edition) * duration_in_weeks = 28.00 := by
  sorry

end NUMINAMATH_GPT_total_cost_over_8_weeks_l420_42086


namespace NUMINAMATH_GPT_solve_ineq_for_a_eq_0_values_of_a_l420_42004

theorem solve_ineq_for_a_eq_0 :
  ∀ x : ℝ, (|x + 2| - 3 * |x|) ≥ 0 ↔ (-1/2 <= x ∧ x <= 1) := 
by
  sorry

theorem values_of_a :
  ∀ x a : ℝ, (|x + 2| - 3 * |x|) ≥ a → (a ≤ 2) := 
by
  sorry

end NUMINAMATH_GPT_solve_ineq_for_a_eq_0_values_of_a_l420_42004


namespace NUMINAMATH_GPT_happy_numbers_l420_42033

theorem happy_numbers (n : ℕ) (h1 : n < 1000) 
(h2 : 7 ∣ n^2) (h3 : 8 ∣ n^2) (h4 : 9 ∣ n^2) (h5 : 10 ∣ n^2) : 
n = 420 ∨ n = 840 :=
sorry

end NUMINAMATH_GPT_happy_numbers_l420_42033


namespace NUMINAMATH_GPT_simplify_expression_l420_42056

theorem simplify_expression:
  (a = 2) ∧ (b = 1) →
  - (1 / 3 : ℚ) * (a^3 * b - a * b) 
  + a * b^3 
  - (a * b - b) / 2 
  - b / 2 
  + (1 / 3 : ℚ) * (a^3 * b) 
  = (5 / 3 : ℚ) := by 
  intros h
  simp [h.1, h.2]
  sorry

end NUMINAMATH_GPT_simplify_expression_l420_42056


namespace NUMINAMATH_GPT_largest_integer_x_l420_42089

theorem largest_integer_x (x : ℤ) (h : 3 - 5 * x > 22) : x ≤ -4 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_x_l420_42089


namespace NUMINAMATH_GPT_non_subset_condition_l420_42063

theorem non_subset_condition (M P : Set α) (non_empty : M ≠ ∅) : 
  ¬(M ⊆ P) ↔ ∃ x ∈ M, x ∉ P := 
sorry

end NUMINAMATH_GPT_non_subset_condition_l420_42063


namespace NUMINAMATH_GPT_cos_540_eq_neg_one_l420_42007

theorem cos_540_eq_neg_one : Real.cos (540 : ℝ) = -1 := by
  sorry

end NUMINAMATH_GPT_cos_540_eq_neg_one_l420_42007


namespace NUMINAMATH_GPT_find_weight_of_sausages_l420_42094

variable (packages : ℕ) (cost_per_pound : ℕ) (total_cost : ℕ) (total_weight : ℕ) (weight_per_package : ℕ)

-- Defining the given conditions
def jake_buys_packages (packages : ℕ) : Prop := packages = 3
def cost_of_sausages (cost_per_pound : ℕ) : Prop := cost_per_pound = 4
def amount_paid (total_cost : ℕ) : Prop := total_cost = 24

-- Derived condition to find total weight
def total_weight_of_sausages (total_cost : ℕ) (cost_per_pound : ℕ) : ℕ := total_cost / cost_per_pound

-- Derived condition to find weight per package
def weight_of_each_package (total_weight : ℕ) (packages : ℕ) : ℕ := total_weight / packages

-- The theorem statement
theorem find_weight_of_sausages
  (h1 : jake_buys_packages packages)
  (h2 : cost_of_sausages cost_per_pound)
  (h3 : amount_paid total_cost) :
  weight_of_each_package (total_weight_of_sausages total_cost cost_per_pound) packages = 2 :=
by
  sorry  -- Proof placeholder

end NUMINAMATH_GPT_find_weight_of_sausages_l420_42094


namespace NUMINAMATH_GPT_subway_length_in_meters_l420_42084

noncomputable def subway_speed : ℝ := 1.6 -- km per minute
noncomputable def crossing_time : ℝ := 3 + 15 / 60 -- minutes
noncomputable def bridge_length : ℝ := 4.85 -- km

theorem subway_length_in_meters :
  let total_distance_traveled := subway_speed * crossing_time
  let subway_length_km := total_distance_traveled - bridge_length
  let subway_length_m := subway_length_km * 1000
  subway_length_m = 350 :=
by
  sorry

end NUMINAMATH_GPT_subway_length_in_meters_l420_42084


namespace NUMINAMATH_GPT_boys_chairs_problem_l420_42043

theorem boys_chairs_problem :
  ∃ (n k : ℕ), n * k = 123 ∧ (∀ p q : ℕ, p * q = 123 → p = n ∧ q = k ∨ p = k ∧ q = n) :=
by
  sorry

end NUMINAMATH_GPT_boys_chairs_problem_l420_42043


namespace NUMINAMATH_GPT_radius_of_tangent_circle_l420_42037

-- Define the conditions
def is_45_45_90_triangle (A B C : ℝ × ℝ) (AB BC AC : ℝ) : Prop :=
  (AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2) ∧
  (A = (0, 0) ∧ B = (2, 0) ∧ C = (2, 2))

def is_tangent_to_axes (O : ℝ × ℝ) (r : ℝ) : Prop :=
  O = (r, r)

def is_tangent_to_hypotenuse (O : ℝ × ℝ) (r : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - O.1) = Real.sqrt 2 * r ∧ (C.2 - O.2) = Real.sqrt 2 * r

-- Main theorem
theorem radius_of_tangent_circle :
  ∃ r : ℝ, ∀ (A B C O : ℝ × ℝ),
    is_45_45_90_triangle A B C (2) (2) (2 * Real.sqrt 2) →
    is_tangent_to_axes O r →
    is_tangent_to_hypotenuse O r C →
    r = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_tangent_circle_l420_42037


namespace NUMINAMATH_GPT_symmetric_sum_l420_42016

theorem symmetric_sum (m n : ℤ) (hA : n = 3) (hB : m = -2) : m + n = 1 :=
by
  rw [hA, hB]
  exact rfl

end NUMINAMATH_GPT_symmetric_sum_l420_42016


namespace NUMINAMATH_GPT_minimum_cost_of_candies_l420_42018

variable (Orange Apple Grape Strawberry : ℕ)

-- Conditions
def CandyRelation1 := Apple = 2 * Orange
def CandyRelation2 := Strawberry = 2 * Grape
def CandyRelation3 := Apple = 2 * Strawberry
def TotalCandies := Orange + Apple + Grape + Strawberry = 90
def CandyCost := 0.1

-- Question
theorem minimum_cost_of_candies :
  CandyRelation1 Orange Apple → 
  CandyRelation2 Grape Strawberry → 
  CandyRelation3 Apple Strawberry → 
  TotalCandies Orange Apple Grape Strawberry → 
  Orange ≥ 3 ∧ Apple ≥ 3 ∧ Grape ≥ 3 ∧ Strawberry ≥ 3 →
  (5 * CandyCost + 3 * CandyCost + 3 * CandyCost + 3 * CandyCost = 1.4) :=
sorry

end NUMINAMATH_GPT_minimum_cost_of_candies_l420_42018


namespace NUMINAMATH_GPT_new_person_weight_l420_42012

noncomputable def weight_of_new_person (weight_of_replaced : ℕ) (number_of_persons : ℕ) (increase_in_average : ℕ) := 
  weight_of_replaced + number_of_persons * increase_in_average

theorem new_person_weight:
  weight_of_new_person 70 8 3 = 94 :=
  by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_new_person_weight_l420_42012


namespace NUMINAMATH_GPT_largest_two_digit_number_l420_42074

-- Define the conditions and the theorem to be proven
theorem largest_two_digit_number (n : ℕ) : 
  (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 4) ∧ (10 ≤ n) ∧ (n < 100) → n = 84 := by
  sorry

end NUMINAMATH_GPT_largest_two_digit_number_l420_42074


namespace NUMINAMATH_GPT_jewel_price_reduction_l420_42061

theorem jewel_price_reduction (P x : ℝ) (P1 : ℝ) (hx : x ≠ 0) 
  (hP1 : P1 = P * (1 - (x / 100) ^ 2))
  (h_final : P1 * (1 - (x / 100) ^ 2) = 2304) : 
  P1 = 2304 / (1 - (x / 100) ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_jewel_price_reduction_l420_42061


namespace NUMINAMATH_GPT_part1_part2_l420_42065

open Nat

variable {a : ℕ → ℝ} -- Defining the arithmetic sequence
variable {S : ℕ → ℝ} -- Defining the sum of the first n terms
variable {m n p q : ℕ} -- Defining the positive integers m, n, p, q
variable {d : ℝ} -- The common difference

-- Conditions
axiom arithmetic_sequence_pos_terms : (∀ k, a k = a 1 + (k - 1) * d) ∧ ∀ k, a k > 0
axiom sum_of_first_n_terms : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2
axiom positive_common_difference : d > 0
axiom constraints_on_mnpq : n < p ∧ q < m ∧ m + n = p + q

-- Parts to prove
theorem part1 : a m * a n < a p * a q :=
by sorry

theorem part2 : S m + S n > S p + S q :=
by sorry

end NUMINAMATH_GPT_part1_part2_l420_42065


namespace NUMINAMATH_GPT_point_in_third_quadrant_l420_42072

theorem point_in_third_quadrant :
  let sin2018 := Real.sin (2018 * Real.pi / 180)
  let tan117 := Real.tan (117 * Real.pi / 180)
  sin2018 < 0 ∧ tan117 < 0 → 
  (sin2018 < 0 ∧ tan117 < 0) :=
by
  intros
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l420_42072


namespace NUMINAMATH_GPT_infinite_series_eq_5_over_16_l420_42093

noncomputable def infinite_series_sum : ℝ :=
  ∑' (n : ℕ), (n + 1 : ℝ) / (5 ^ (n + 1))

theorem infinite_series_eq_5_over_16 :
  infinite_series_sum = 5 / 16 :=
sorry

end NUMINAMATH_GPT_infinite_series_eq_5_over_16_l420_42093


namespace NUMINAMATH_GPT_smallest_multiple_of_6_and_15_l420_42076

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ b = 30 := 
by 
  use 30 
  sorry

end NUMINAMATH_GPT_smallest_multiple_of_6_and_15_l420_42076


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l420_42095

noncomputable def average_gas_mileage
  (d1 d2 : ℕ) (m1 m2 : ℕ) : ℚ :=
  let total_distance := d1 + d2
  let total_fuel := (d1 / m1) + (d2 / m2)
  total_distance / total_fuel

theorem average_gas_mileage_round_trip :
  average_gas_mileage 150 180 25 15 = 18.3 := by
  sorry

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l420_42095


namespace NUMINAMATH_GPT_extra_yellow_balls_dispatched_eq_49_l420_42050

-- Define the given conditions
def ordered_balls : ℕ := 114
def white_balls : ℕ := ordered_balls / 2
def yellow_balls := ordered_balls / 2

-- Define the additional yellow balls dispatched and the ratio condition
def dispatch_error_ratio : ℚ := 8 / 15

-- The statement to prove the number of extra yellow balls dispatched
theorem extra_yellow_balls_dispatched_eq_49
  (ordered_balls_rounded : ordered_balls = 114)
  (white_balls_57 : white_balls = 57)
  (yellow_balls_57 : yellow_balls = 57)
  (ratio_condition : white_balls / (yellow_balls + x) = dispatch_error_ratio) :
  x = 49 :=
  sorry

end NUMINAMATH_GPT_extra_yellow_balls_dispatched_eq_49_l420_42050


namespace NUMINAMATH_GPT_distinct_zeros_arithmetic_geometric_sequence_l420_42057

theorem distinct_zeros_arithmetic_geometric_sequence 
  (a b p q : ℝ)
  (h1 : a ≠ b)
  (h2 : a + b = p)
  (h3 : ab = q)
  (h4 : p > 0)
  (h5 : q > 0)
  (h6 : (a = 4 ∧ b = 1) ∨ (a = 1 ∧ b = 4))
  : p + q = 9 := 
sorry

end NUMINAMATH_GPT_distinct_zeros_arithmetic_geometric_sequence_l420_42057


namespace NUMINAMATH_GPT_find_positive_number_l420_42029

noncomputable def solve_number (x : ℝ) : Prop :=
  (2/3 * x = 64/216 * (1/x)) ∧ (x > 0)

theorem find_positive_number (x : ℝ) : solve_number x → x = (2/9) * Real.sqrt 3 :=
  by
  sorry

end NUMINAMATH_GPT_find_positive_number_l420_42029


namespace NUMINAMATH_GPT_ratio_of_x_and_y_l420_42034

theorem ratio_of_x_and_y (x y : ℝ) (h : (x - y) / (x + y) = 4) : x / y = -5 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_x_and_y_l420_42034


namespace NUMINAMATH_GPT_square_area_and_diagonal_ratio_l420_42041

theorem square_area_and_diagonal_ratio
    (a b : ℕ)
    (h_perimeter : 4 * a = 16 * b) :
    (a = 4 * b) ∧ ((a^2) / (b^2) = 16) ∧ ((a * Real.sqrt 2) / (b * Real.sqrt 2) = 4) :=
  by
  sorry

end NUMINAMATH_GPT_square_area_and_diagonal_ratio_l420_42041


namespace NUMINAMATH_GPT_common_difference_divisible_by_6_l420_42049

theorem common_difference_divisible_by_6 (p q r d : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hp3 : p > 3) (hq3 : q > 3) (hr3 : r > 3) (h1 : q = p + d) (h2 : r = p + 2 * d) : d % 6 = 0 := 
sorry

end NUMINAMATH_GPT_common_difference_divisible_by_6_l420_42049


namespace NUMINAMATH_GPT_polynomial_factorization_l420_42077

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l420_42077


namespace NUMINAMATH_GPT_sum_of_y_for_f_l420_42088

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + 5

theorem sum_of_y_for_f (y1 y2 y3 : ℝ) :
  (∀ y, 64 * y^3 - 8 * y + 5 = 7) →
  y1 + y2 + y3 = 0 :=
by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_sum_of_y_for_f_l420_42088


namespace NUMINAMATH_GPT_midpoint_product_coordinates_l420_42068

theorem midpoint_product_coordinates :
  ∃ (x y : ℝ), (4 : ℝ) = (-2 + x) / 2 ∧ (-3 : ℝ) = (-7 + y) / 2 ∧ x * y = 10 := by
  sorry

end NUMINAMATH_GPT_midpoint_product_coordinates_l420_42068


namespace NUMINAMATH_GPT_like_monomials_are_same_l420_42010

theorem like_monomials_are_same (m n : ℤ) (h1 : 2 * m + 4 = 8) (h2 : 2 * n - 3 = 5) : m = 2 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_like_monomials_are_same_l420_42010


namespace NUMINAMATH_GPT_printing_machine_completion_time_l420_42098

-- Definitions of times in hours
def start_time : ℕ := 9 -- 9:00 AM
def half_job_time : ℕ := 12 -- 12:00 PM
def completion_time : ℕ := 15 -- 3:00 PM

-- Time taken to complete half the job
def half_job_duration : ℕ := half_job_time - start_time

-- Total time to complete the entire job
def total_job_duration : ℕ := 2 * half_job_duration

-- Proof that the machine will complete the job at 3:00 PM
theorem printing_machine_completion_time : 
    start_time + total_job_duration = completion_time :=
sorry

end NUMINAMATH_GPT_printing_machine_completion_time_l420_42098


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l420_42044

noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * a1 + (n * (n - 1) / 2) * d

theorem sum_of_arithmetic_sequence (a1 : ℤ) (d : ℤ)
  (h1 : a1 = -2010)
  (h2 : (S 2011 a1 d) / 2011 - (S 2009 a1 d) / 2009 = 2) :
  S 2010 a1 d = -2010 := 
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l420_42044


namespace NUMINAMATH_GPT_youngest_person_age_l420_42082

noncomputable def avg_age_seven_people := 30
noncomputable def avg_age_six_people_when_youngest_born := 25
noncomputable def num_people := 7
noncomputable def num_people_minus_one := 6

theorem youngest_person_age :
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  total_age_seven_people - total_age_six_people = 60 :=
by
  let total_age_seven_people := num_people * avg_age_seven_people
  let total_age_six_people := num_people_minus_one * avg_age_six_people_when_youngest_born
  sorry

end NUMINAMATH_GPT_youngest_person_age_l420_42082


namespace NUMINAMATH_GPT_hexagonal_prism_cross_section_l420_42025

theorem hexagonal_prism_cross_section (n : ℕ) (h₁: n ≥ 3) (h₂: n ≤ 8) : ¬ (n = 9):=
sorry

end NUMINAMATH_GPT_hexagonal_prism_cross_section_l420_42025


namespace NUMINAMATH_GPT_sum_even_integers_between_200_and_600_is_80200_l420_42071

noncomputable def sum_even_integers_between_200_and_600 (a d n : ℕ) : ℕ :=
  n / 2 * (a + (a + (n - 1) * d))

theorem sum_even_integers_between_200_and_600_is_80200 :
  sum_even_integers_between_200_and_600 202 2 200 = 80200 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_sum_even_integers_between_200_and_600_is_80200_l420_42071


namespace NUMINAMATH_GPT_mod_graph_sum_l420_42058

theorem mod_graph_sum (x₀ y₀ : ℕ) (h₁ : 2 * x₀ ≡ 1 [MOD 11]) (h₂ : 3 * y₀ ≡ 10 [MOD 11]) : x₀ + y₀ = 13 :=
by
  sorry

end NUMINAMATH_GPT_mod_graph_sum_l420_42058


namespace NUMINAMATH_GPT_quadratic_roots_eq_l420_42003

theorem quadratic_roots_eq (a : ℝ) (b : ℝ) :
  (∀ x, (2 * x^2 - 3 * x - 8 = 0) → 
         ((x + 3)^2 + a * (x + 3) + b = 0)) → 
  b = 9.5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_eq_l420_42003


namespace NUMINAMATH_GPT_ten_percent_markup_and_markdown_l420_42045

theorem ten_percent_markup_and_markdown (x : ℝ) (hx : x > 0) : 0.99 * x < x :=
by 
  sorry

end NUMINAMATH_GPT_ten_percent_markup_and_markdown_l420_42045


namespace NUMINAMATH_GPT_real_root_exists_l420_42046

theorem real_root_exists (p1 p2 q1 q2 : ℝ) 
(h : p1 * p2 = 2 * (q1 + q2)) : 
  (∃ x : ℝ, x^2 + p1 * x + q1 = 0) ∨ (∃ x : ℝ, x^2 + p2 * x + q2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_real_root_exists_l420_42046


namespace NUMINAMATH_GPT_greater_solution_of_quadratic_l420_42006

theorem greater_solution_of_quadratic :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 5 * x₁ - 84 = 0) ∧ (x₂^2 - 5 * x₂ - 84 = 0) ∧ (max x₁ x₂ = 12) :=
by
  sorry

end NUMINAMATH_GPT_greater_solution_of_quadratic_l420_42006


namespace NUMINAMATH_GPT_find_x_plus_2y_squared_l420_42055

theorem find_x_plus_2y_squared (x y : ℝ) (h1 : x * (x + 2 * y) = 48) (h2 : y * (x + 2 * y) = 72) :
  (x + 2 * y) ^ 2 = 96 := 
sorry

end NUMINAMATH_GPT_find_x_plus_2y_squared_l420_42055


namespace NUMINAMATH_GPT_jameson_badminton_medals_l420_42040

theorem jameson_badminton_medals (total_medals track_medals : ℕ) (swimming_medals : ℕ) :
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  total_medals - (track_medals + swimming_medals) = 5 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end NUMINAMATH_GPT_jameson_badminton_medals_l420_42040


namespace NUMINAMATH_GPT_book_arrangement_count_l420_42051

theorem book_arrangement_count :
  let n := 6
  let identical_pairs := 2
  let total_arrangements_if_unique := n.factorial
  let ident_pair_correction := (identical_pairs.factorial * identical_pairs.factorial)
  (total_arrangements_if_unique / ident_pair_correction) = 180 := by
  sorry

end NUMINAMATH_GPT_book_arrangement_count_l420_42051


namespace NUMINAMATH_GPT_divisibility_by_11_l420_42021

theorem divisibility_by_11
  (n : ℕ) (hn : n ≥ 2)
  (h : (n^2 + (4^n) + (7^n)) % n = 0) :
  (n^2 + 4^n + 7^n) % 11 = 0 := 
by
  sorry

end NUMINAMATH_GPT_divisibility_by_11_l420_42021


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_a_lt_neg_one_l420_42022

theorem necessary_but_not_sufficient_condition_for_a_lt_neg_one (a : ℝ) : 
  (1 / a > -1) ↔ (a < -1) :=
by sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_for_a_lt_neg_one_l420_42022


namespace NUMINAMATH_GPT_find_S_l420_42091

noncomputable def S : ℕ+ → ℝ := sorry
noncomputable def a : ℕ+ → ℝ := sorry

axiom h : ∀ n : ℕ+, 2 * S n = 3 * a n + 4

theorem find_S : ∀ n : ℕ+, S n = 2 - 2 * 3 ^ (n : ℕ) :=
  sorry

end NUMINAMATH_GPT_find_S_l420_42091


namespace NUMINAMATH_GPT_total_solutions_l420_42035

-- Definitions and conditions
def tetrahedron_solutions := 1
def cube_solutions := 1
def octahedron_solutions := 3
def dodecahedron_solutions := 2
def icosahedron_solutions := 3

-- Main theorem statement
theorem total_solutions : 
  tetrahedron_solutions + cube_solutions + octahedron_solutions + dodecahedron_solutions + icosahedron_solutions = 10 := by
  sorry

end NUMINAMATH_GPT_total_solutions_l420_42035


namespace NUMINAMATH_GPT_product_of_ages_l420_42096

theorem product_of_ages (O Y : ℕ) (h1 : O - Y = 12) (h2 : O + Y = (O - Y) + 40) : O * Y = 640 := by
  sorry

end NUMINAMATH_GPT_product_of_ages_l420_42096


namespace NUMINAMATH_GPT_cost_of_four_stamps_l420_42015

theorem cost_of_four_stamps (cost_one_stamp : ℝ) (h : cost_one_stamp = 0.34) : 4 * cost_one_stamp = 1.36 := 
by
  rw [h]
  norm_num

end NUMINAMATH_GPT_cost_of_four_stamps_l420_42015


namespace NUMINAMATH_GPT_probability_three_specific_cards_l420_42008

theorem probability_three_specific_cards :
  let total_deck := 52
  let total_spades := 13
  let total_tens := 4
  let total_queens := 4
  let p_case1 := ((12:ℚ) / total_deck) * (total_tens / (total_deck - 1)) * (total_queens / (total_deck - 2))
  let p_case2 := ((1:ℚ) / total_deck) * ((total_tens - 1) / (total_deck - 1)) * (total_queens / (total_deck - 2))
  p_case1 + p_case2 = (17:ℚ) / 11050 :=
by
  sorry

end NUMINAMATH_GPT_probability_three_specific_cards_l420_42008


namespace NUMINAMATH_GPT_no_real_solutions_l420_42085

-- Define the equation
def equation (x : ℝ) : Prop :=
  (2 * x ^ 2 - 6 * x + 5) ^ 2 + 1 = -|x|

-- Declare the theorem which states there are no real solutions to the given equation
theorem no_real_solutions : ∀ x : ℝ, ¬ equation x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_no_real_solutions_l420_42085


namespace NUMINAMATH_GPT_NewYearSeasonMarkup_theorem_l420_42083

def NewYearSeasonMarkup (C N : ℝ) : Prop :=
    (0.90 * (1.20 * C * (1 + N)) = 1.35 * C) -> N = 0.25

theorem NewYearSeasonMarkup_theorem (C : ℝ) (h₀ : C > 0) : ∃ (N : ℝ), NewYearSeasonMarkup C N :=
by
  use 0.25
  sorry

end NUMINAMATH_GPT_NewYearSeasonMarkup_theorem_l420_42083


namespace NUMINAMATH_GPT_proof_2_abs_a_plus_b_less_abs_4_plus_ab_l420_42013

theorem proof_2_abs_a_plus_b_less_abs_4_plus_ab (a b : ℝ) (h1 : abs a < 2) (h2 : abs b < 2) :
    2 * abs (a + b) < abs (4 + a * b) := 
by
  sorry

end NUMINAMATH_GPT_proof_2_abs_a_plus_b_less_abs_4_plus_ab_l420_42013


namespace NUMINAMATH_GPT_sum_of_roots_l420_42053

theorem sum_of_roots (f : ℝ → ℝ) (h_symmetric : ∀ x, f (3 + x) = f (3 - x)) (h_roots : ∃ (roots : Finset ℝ), roots.card = 6 ∧ ∀ r ∈ roots, f r = 0) : 
  ∃ S, S = 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l420_42053


namespace NUMINAMATH_GPT_prob_yellow_straight_l420_42005

variable {P : ℕ → ℕ → ℚ}
-- Defining the probabilities of the given events
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2
def prob_rose : ℚ := 1 / 4
def prob_daffodil : ℚ := 1 / 2
def prob_tulip : ℚ := 1 / 4
def prob_rose_straight : ℚ := 1 / 6
def prob_daffodil_curved : ℚ := 1 / 3
def prob_tulip_straight : ℚ := 1 / 8

/-- The probability of picking a yellow and straight-petaled flower is 1/6 -/
theorem prob_yellow_straight : P 1 1 = 1 / 6 := sorry

end NUMINAMATH_GPT_prob_yellow_straight_l420_42005


namespace NUMINAMATH_GPT_binary_operations_unique_l420_42000

def binary_operation (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → (f a (f b c) = (f a b) * c)
  ∧ ∀ a : ℝ, a > 0 → a ≥ 1 → f a a ≥ 1

theorem binary_operations_unique (f : ℝ → ℝ → ℝ) (h : binary_operation f) :
  (∀ a b, f a b = a * b) ∨ (∀ a b, f a b = a / b) :=
sorry

end NUMINAMATH_GPT_binary_operations_unique_l420_42000


namespace NUMINAMATH_GPT_part1_part2_l420_42052

-- Part 1: Prove that the range of values for k is k ≤ 1/4
theorem part1 (f : ℝ → ℝ) (k : ℝ) 
  (h1 : ∀ x0 : ℝ, f x0 ≥ |k+3| - |k-2|)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  k ≤ 1/4 := 
sorry

-- Part 2: Show that the minimum value of m+n is 8/3
theorem part2 (f : ℝ → ℝ) (m n : ℝ) 
  (h1 : ∀ x : ℝ, f x ≥ 1/m + 1/n)
  (h2 : ∀ x : ℝ, f x = |2*x - 1| + |x - 2| ) : 
  m + n ≥ 8/3 := 
sorry

end NUMINAMATH_GPT_part1_part2_l420_42052


namespace NUMINAMATH_GPT_find_circumcenter_l420_42036

-- Define a quadrilateral with vertices A, B, C, and D
structure Quadrilateral :=
  (A B C D : (ℝ × ℝ))

-- Define the coordinates of the circumcenter
def circumcenter (q : Quadrilateral) : ℝ × ℝ := (6, 1)

-- Given condition that A, B, C, and D are vertices of a quadrilateral
-- Prove that the circumcenter of the circumscribed circle is (6, 1)
theorem find_circumcenter (q : Quadrilateral) : 
  circumcenter q = (6, 1) :=
by sorry

end NUMINAMATH_GPT_find_circumcenter_l420_42036


namespace NUMINAMATH_GPT_balls_sold_l420_42048

theorem balls_sold (CP SP_total : ℕ) (loss : ℕ) (n : ℕ) :
  CP = 60 →
  SP_total = 720 →
  loss = 5 * CP →
  loss = n * CP - SP_total →
  n = 17 :=
by
  intros hCP hSP_total hloss htotal
  -- Your proof here
  sorry

end NUMINAMATH_GPT_balls_sold_l420_42048


namespace NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l420_42031

section LineEquation

variables (a : ℝ) (x y : ℝ)

def l1 := (a-2) * x + 3 * y + a = 0
def l2 := a * x + (a-2) * y - 1 = 0

theorem parallel_lines (a : ℝ) :
  ((a-2)/a = 3/(a-2)) ↔ (a = (7 + Real.sqrt 33) / 2 ∨ a = (7 - Real.sqrt 33) / 2) := sorry

theorem perpendicular_lines (a : ℝ) :
  (a = 2 ∨ ((2-a)/3 * (a/(2-a)) = -1)) ↔ (a = 2 ∨ a = -3) := sorry

end LineEquation

end NUMINAMATH_GPT_parallel_lines_perpendicular_lines_l420_42031


namespace NUMINAMATH_GPT_number_of_spotted_blue_fish_l420_42002

def total_fish := 60
def blue_fish := total_fish / 3
def spotted_blue_fish := blue_fish / 2

theorem number_of_spotted_blue_fish : spotted_blue_fish = 10 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_number_of_spotted_blue_fish_l420_42002


namespace NUMINAMATH_GPT_book_surface_area_l420_42087

variables (L : ℕ) (T : ℕ) (A1 : ℕ) (A2 : ℕ) (W : ℕ) (S : ℕ)

theorem book_surface_area (hL : L = 5) (hT : T = 2) 
                         (hA1 : A1 = L * W) (hA1_val : A1 = 50)
                         (hA2 : A2 = T * W) (hA2_val : A2 = 10) :
  S = 2 * A1 + A2 + 2 * (L * T) :=
sorry

end NUMINAMATH_GPT_book_surface_area_l420_42087


namespace NUMINAMATH_GPT_solve_system_eq_l420_42011

theorem solve_system_eq (x y : ℝ) (h1 : x - y = 1) (h2 : 2 * x + 3 * y = 7) :
  x = 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_GPT_solve_system_eq_l420_42011


namespace NUMINAMATH_GPT_incorrect_number_read_as_l420_42032

theorem incorrect_number_read_as (n a_incorrect a_correct correct_number incorrect_number : ℕ) 
(hn : n = 10) (h_inc_avg : a_incorrect = 18) (h_cor_avg : a_correct = 22) (h_cor_num : correct_number = 66) :
incorrect_number = 26 := by
  sorry

end NUMINAMATH_GPT_incorrect_number_read_as_l420_42032


namespace NUMINAMATH_GPT_shares_of_stocks_they_can_buy_l420_42042

def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def months_of_savings : ℕ := 4
def cost_per_share : ℕ := 50

theorem shares_of_stocks_they_can_buy :
  (((weekly_savings_wife * 4) + monthly_savings_husband) * months_of_savings / 2) / cost_per_share = 25 :=
by
  -- sorry for the implementation
  sorry

end NUMINAMATH_GPT_shares_of_stocks_they_can_buy_l420_42042


namespace NUMINAMATH_GPT_total_area_correct_l420_42001

noncomputable def total_area (r p q : ℝ) : ℝ :=
  r^2 + 4*p^2 + 12*q

theorem total_area_correct
  (r p q : ℝ)
  (h : 12 * q = r^2 + 4 * p^2 + 45)
  (r_val : r = 6)
  (p_val : p = 1.5)
  (q_val : q = 7.5) :
  total_area r p q = 135 := by
  sorry

end NUMINAMATH_GPT_total_area_correct_l420_42001


namespace NUMINAMATH_GPT_even_function_m_value_l420_42062

theorem even_function_m_value {m : ℤ} (h : ∀ (x : ℝ), (m^2 - m - 1) * (-x)^m = (m^2 - m - 1) * x^m) : m = 2 := 
by
  sorry

end NUMINAMATH_GPT_even_function_m_value_l420_42062


namespace NUMINAMATH_GPT_least_palindrome_divisible_by_25_l420_42020

theorem least_palindrome_divisible_by_25 : ∃ (n : ℕ), 
  (10^4 ≤ n ∧ n < 10^5) ∧
  (∀ (a b c : ℕ), n = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a) ∧
  n % 25 = 0 ∧
  n = 10201 :=
by
  sorry

end NUMINAMATH_GPT_least_palindrome_divisible_by_25_l420_42020


namespace NUMINAMATH_GPT_calculate_x_minus_y_l420_42090

theorem calculate_x_minus_y (x y z : ℝ) 
    (h1 : x - y + z = 23) 
    (h2 : x - y - z = 7) : 
    x - y = 15 :=
by
  sorry

end NUMINAMATH_GPT_calculate_x_minus_y_l420_42090


namespace NUMINAMATH_GPT_tan_30_eq_sqrt3_div3_l420_42028

theorem tan_30_eq_sqrt3_div3 (sin_30_cos_30 : ℝ → ℝ → Prop)
  (h1 : sin_30_cos_30 (1 / 2) (Real.sqrt 3 / 2)) :
  ∃ t, t = Real.tan (Real.pi / 6) ∧ t = Real.sqrt 3 / 3 :=
by
  existsi Real.tan (Real.pi / 6)
  sorry

end NUMINAMATH_GPT_tan_30_eq_sqrt3_div3_l420_42028


namespace NUMINAMATH_GPT_spent_on_music_l420_42078

variable (total_allowance : ℝ) (fraction_music : ℝ)

-- Assuming the conditions
def conditions : Prop :=
  total_allowance = 50 ∧ fraction_music = 3 / 10

-- The proof problem
theorem spent_on_music (h : conditions total_allowance fraction_music) : 
  total_allowance * fraction_music = 15 := by
  cases h with
  | intro h_total h_fraction =>
  sorry

end NUMINAMATH_GPT_spent_on_music_l420_42078


namespace NUMINAMATH_GPT_inequality_inequality_hold_l420_42030

theorem inequality_inequality_hold (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab_sum : a^2 + b^2 = 1/2) :
  (1 / (1 - a)) + (1 / (1 - b)) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_inequality_hold_l420_42030


namespace NUMINAMATH_GPT_solution_set_of_inequality_l420_42092

theorem solution_set_of_inequality :
  {x : ℝ | (x + 1) / (3 - x) < 0} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l420_42092


namespace NUMINAMATH_GPT_moles_NaCl_formed_in_reaction_l420_42075

noncomputable def moles_of_NaCl_formed (moles_NaOH moles_HCl : ℕ) : ℕ :=
  if moles_NaOH = 1 ∧ moles_HCl = 1 then 1 else 0

theorem moles_NaCl_formed_in_reaction : moles_of_NaCl_formed 1 1 = 1 := 
by
  sorry

end NUMINAMATH_GPT_moles_NaCl_formed_in_reaction_l420_42075


namespace NUMINAMATH_GPT_student_scores_marks_per_correct_answer_l420_42047

theorem student_scores_marks_per_correct_answer
  (total_questions : ℕ) (total_marks : ℤ) (correct_questions : ℕ)
  (wrong_questions : ℕ) (marks_wrong_answer : ℤ)
  (x : ℤ) (h1 : total_questions = 60) (h2 : total_marks = 110)
  (h3 : correct_questions = 34) (h4 : wrong_questions = total_questions - correct_questions)
  (h5 : marks_wrong_answer = -1) :
  34 * x - 26 = 110 → x = 4 := by
  sorry

end NUMINAMATH_GPT_student_scores_marks_per_correct_answer_l420_42047


namespace NUMINAMATH_GPT_length_after_5th_cut_l420_42073

theorem length_after_5th_cut (initial_length : ℝ) (n : ℕ) (h1 : initial_length = 1) (h2 : n = 5) :
  initial_length / 2^n = 1 / 2^5 := by
  sorry

end NUMINAMATH_GPT_length_after_5th_cut_l420_42073


namespace NUMINAMATH_GPT_solve_m_l420_42014

theorem solve_m (m : ℝ) : (m + 1) / 6 = m / 1 → m = 1 / 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_m_l420_42014


namespace NUMINAMATH_GPT_least_positive_integer_satisfying_congruences_l420_42060

theorem least_positive_integer_satisfying_congruences :
  ∃ b : ℕ, b > 0 ∧
    (b % 6 = 5) ∧
    (b % 7 = 6) ∧
    (b % 8 = 7) ∧
    (b % 9 = 8) ∧
    ∀ n : ℕ, (n > 0 → (n % 6 = 5) ∧ (n % 7 = 6) ∧ (n % 8 = 7) ∧ (n % 9 = 8) → n ≥ b) ∧
    b = 503 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_satisfying_congruences_l420_42060


namespace NUMINAMATH_GPT_fractional_sum_equals_015025_l420_42064

theorem fractional_sum_equals_015025 :
  (2 / 20) + (8 / 200) + (3 / 300) + (5 / 40000) * 2 = 0.15025 := 
by
  sorry

end NUMINAMATH_GPT_fractional_sum_equals_015025_l420_42064


namespace NUMINAMATH_GPT_gcd_max_value_l420_42069

theorem gcd_max_value : ∀ (n : ℕ), n > 0 → ∃ (d : ℕ), d = 9 ∧ d ∣ gcd (13 * n + 4) (8 * n + 3) :=
by
  sorry

end NUMINAMATH_GPT_gcd_max_value_l420_42069


namespace NUMINAMATH_GPT_tan_x_neg7_l420_42024

theorem tan_x_neg7 (x : ℝ) (h1 : Real.sin (x + π / 4) = 3 / 5) (h2 : Real.sin (x - π / 4) = 4 / 5) : 
  Real.tan x = -7 :=
sorry

end NUMINAMATH_GPT_tan_x_neg7_l420_42024


namespace NUMINAMATH_GPT_inequality_range_of_a_l420_42066

theorem inequality_range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ a ∈ Set.Icc (-2: ℝ) 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_range_of_a_l420_42066


namespace NUMINAMATH_GPT_question_1_question_2_question_3_question_4_l420_42079

-- Define each condition as a theorem
theorem question_1 (explanation: String) : explanation = "providing for the living" :=
  sorry

theorem question_2 (usage: String) : usage = "structural auxiliary word, placed between subject and predicate, negating sentence independence" :=
  sorry

theorem question_3 (explanation: String) : explanation = "The Shang dynasty called it 'Xu,' and the Zhou dynasty called it 'Xiang.'" :=
  sorry

theorem question_4 (analysis: String) : analysis = "The statement about the 'ultimate ideal' is incorrect; the original text states that 'enabling people to live and die without regret' is 'the beginning of the King's Way.'" :=
  sorry

end NUMINAMATH_GPT_question_1_question_2_question_3_question_4_l420_42079


namespace NUMINAMATH_GPT_exponent_equality_l420_42099

theorem exponent_equality (s m : ℕ) (h : (2^16) * (25^s) = 5 * (10^m)) : m = 16 :=
by sorry

end NUMINAMATH_GPT_exponent_equality_l420_42099


namespace NUMINAMATH_GPT_repeating_decimal_product_l420_42038

theorem repeating_decimal_product :
  (8 / 99) * (36 / 99) = 288 / 9801 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_product_l420_42038


namespace NUMINAMATH_GPT_train_length_l420_42080

theorem train_length (L V : ℝ) (h1 : L = V * 120) (h2 : L + 1000 = V * 220) : L = 1200 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l420_42080


namespace NUMINAMATH_GPT_neither_necessary_nor_sufficient_l420_42023

-- defining polynomial inequalities
def inequality_1 (a1 b1 c1 x : ℝ) : Prop := a1 * x^2 + b1 * x + c1 > 0
def inequality_2 (a2 b2 c2 x : ℝ) : Prop := a2 * x^2 + b2 * x + c2 > 0

-- defining proposition P and proposition Q
def P (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := ∀ x : ℝ, inequality_1 a1 b1 c1 x ↔ inequality_2 a2 b2 c2 x
def Q (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2

-- prove that Q is neither a necessary nor sufficient condition for P
theorem neither_necessary_nor_sufficient {a1 b1 c1 a2 b2 c2 : ℝ} : ¬(Q a1 b1 c1 a2 b2 c2 ↔ P a1 b1 c1 a2 b2 c2) := 
sorry

end NUMINAMATH_GPT_neither_necessary_nor_sufficient_l420_42023


namespace NUMINAMATH_GPT_iris_to_tulip_ratio_l420_42019

theorem iris_to_tulip_ratio (earnings_per_bulb : ℚ)
  (tulip_bulbs daffodil_bulbs crocus_ratio total_earnings : ℕ)
  (iris_bulbs : ℕ) (h0 : earnings_per_bulb = 0.50)
  (h1 : tulip_bulbs = 20) (h2 : daffodil_bulbs = 30)
  (h3 : crocus_ratio = 3) (h4 : total_earnings = 75)
  (h5 : total_earnings = earnings_per_bulb * (tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_ratio * daffodil_bulbs))
  : iris_bulbs = 10 → tulip_bulbs = 20 → (iris_bulbs : ℚ) / (tulip_bulbs : ℚ) = 1 / 2 :=
by {
  intros; sorry
}

end NUMINAMATH_GPT_iris_to_tulip_ratio_l420_42019


namespace NUMINAMATH_GPT_mary_remaining_cards_l420_42009

variable (initial_cards : ℝ) (bought_cards : ℝ) (promised_cards : ℝ)

def remaining_cards (initial : ℝ) (bought : ℝ) (promised : ℝ) : ℝ :=
  initial + bought - promised

theorem mary_remaining_cards :
  initial_cards = 18.0 →
  bought_cards = 40.0 →
  promised_cards = 26.0 →
  remaining_cards initial_cards bought_cards promised_cards = 32.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_mary_remaining_cards_l420_42009


namespace NUMINAMATH_GPT_factorization_correct_l420_42059

theorem factorization_correct (x : ℝ) : 
  (hxA : x^2 + 2*x + 1 ≠ x*(x + 2) + 1) → 
  (hxB : x^2 + 2*x + 1 ≠ (x + 1)*(x - 1)) → 
  (hxC : x^2 + x ≠ (x + 1/2)^2 - 1/4) →
  x^2 + x = x * (x + 1) := 
by sorry

end NUMINAMATH_GPT_factorization_correct_l420_42059


namespace NUMINAMATH_GPT_roots_inverse_sum_eq_two_thirds_l420_42054

theorem roots_inverse_sum_eq_two_thirds {x₁ x₂ : ℝ} (h1 : x₁ ^ 2 + 2 * x₁ - 3 = 0) (h2 : x₂ ^ 2 + 2 * x₂ - 3 = 0) : 
  (1 / x₁) + (1 / x₂) = 2 / 3 :=
sorry

end NUMINAMATH_GPT_roots_inverse_sum_eq_two_thirds_l420_42054


namespace NUMINAMATH_GPT_hyperbola_params_l420_42081

theorem hyperbola_params (a b h k : ℝ) (h_positivity : a > 0 ∧ b > 0)
  (asymptote_1 : ∀ x : ℝ, ∃ y : ℝ, y = (3/2) * x + 4)
  (asymptote_2 : ∀ x : ℝ, ∃ y : ℝ, y = -(3/2) * x + 2)
  (passes_through : ∃ x y : ℝ, x = 2 ∧ y = 8 ∧ (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1) 
  (standard_form : ∀ x y : ℝ, ((y - k)^2 / a^2 - (x - h)^2 / b^2 = 1)) : 
  a + h = 7/3 := sorry

end NUMINAMATH_GPT_hyperbola_params_l420_42081


namespace NUMINAMATH_GPT_problem_statement_l420_42017

theorem problem_statement (x y : ℕ) (hx : x = 7) (hy : y = 3) : (x - y)^2 * (x + y)^2 = 1600 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_problem_statement_l420_42017


namespace NUMINAMATH_GPT_max_unique_rankings_l420_42070

theorem max_unique_rankings (n : ℕ) : 
  ∃ (contestants : ℕ), 
    (∀ (scores : ℕ → ℕ), 
      (∀ i, 0 ≤ scores i ∧ scores i ≤ contestants) ∧
      (∀ i j, i ≠ j → scores i ≠ scores j)) 
    → contestants = 2^n := 
sorry

end NUMINAMATH_GPT_max_unique_rankings_l420_42070
