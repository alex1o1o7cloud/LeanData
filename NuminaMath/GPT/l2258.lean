import Mathlib

namespace NUMINAMATH_GPT_tank_depth_l2258_225830

theorem tank_depth (d : ℝ)
    (field_length : ℝ) (field_breadth : ℝ)
    (tank_length : ℝ) (tank_breadth : ℝ)
    (remaining_field_area : ℝ)
    (rise_in_field_level : ℝ)
    (field_area_eq : field_length * field_breadth = 4500)
    (tank_area_eq : tank_length * tank_breadth = 500)
    (remaining_field_area_eq : remaining_field_area = 4500 - 500)
    (earth_volume_spread_eq : remaining_field_area * rise_in_field_level = 2000)
    (volume_eq : tank_length * tank_breadth * d = 2000)
  : d = 4 := by
  sorry

end NUMINAMATH_GPT_tank_depth_l2258_225830


namespace NUMINAMATH_GPT_total_number_of_cookies_l2258_225899

open Nat -- Open the natural numbers namespace to work with natural number operations

def n_bags : Nat := 7
def cookies_per_bag : Nat := 2
def total_cookies : Nat := n_bags * cookies_per_bag

theorem total_number_of_cookies : total_cookies = 14 := by
  sorry

end NUMINAMATH_GPT_total_number_of_cookies_l2258_225899


namespace NUMINAMATH_GPT_digit_y_in_base_7_divisible_by_19_l2258_225876

def base7_to_decimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem digit_y_in_base_7_divisible_by_19 (y : ℕ) (hy : y < 7) :
  (∃ k : ℕ, base7_to_decimal 5 2 y 3 = 19 * k) ↔ y = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_digit_y_in_base_7_divisible_by_19_l2258_225876


namespace NUMINAMATH_GPT_larry_substitution_l2258_225835

theorem larry_substitution (a b c d e : ℤ)
  (h_a : a = 2)
  (h_b : b = 5)
  (h_c : c = 3)
  (h_d : d = 4)
  (h_expr1 : a + b - c - d * e = 4 - 4 * e)
  (h_expr2 : a + (b - (c - (d * e))) = 4 + 4 * e) :
  e = 0 :=
by
  sorry

end NUMINAMATH_GPT_larry_substitution_l2258_225835


namespace NUMINAMATH_GPT_find_x_l2258_225807

theorem find_x (number x : ℝ) (h1 : 24 * number = 173 * x) (h2 : 24 * number = 1730) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2258_225807


namespace NUMINAMATH_GPT_hypotenuse_length_l2258_225869

theorem hypotenuse_length (a b c : ℝ) (h : a^2 + b^2 + c^2 = 2500) (h_right : c^2 = a^2 + b^2) : c = 25 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l2258_225869


namespace NUMINAMATH_GPT_tom_has_18_apples_l2258_225840

-- Definitions based on conditions
def phillip_apples : ℕ := 40
def ben_apples : ℕ := phillip_apples + 8
def tom_apples : ℕ := (3 * ben_apples) / 8

-- Theorem stating Tom has 18 apples given the conditions
theorem tom_has_18_apples : tom_apples = 18 :=
sorry

end NUMINAMATH_GPT_tom_has_18_apples_l2258_225840


namespace NUMINAMATH_GPT_negation_of_p_l2258_225855

   -- Define the proposition p as an existential quantification
   def p : Prop := ∃ x₀ : ℝ, x₀^2 + 2 * x₀ + 3 > 0

   -- State the theorem that negation of p is a universal quantification
   theorem negation_of_p : ¬ p ↔ ∀ x : ℝ, x^2 + 2*x + 3 ≤ 0 :=
   by sorry
   
end NUMINAMATH_GPT_negation_of_p_l2258_225855


namespace NUMINAMATH_GPT_percentage_increase_from_second_to_third_building_l2258_225825

theorem percentage_increase_from_second_to_third_building :
  let first_building_units := 4000
  let second_building_units := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  (third_building_units - second_building_units) / second_building_units * 100 = 20 := by
  let first_building_units := 4000
  let second_building_units : ℝ := (2 / 5 : ℝ) * first_building_units
  let total_units := 7520
  let third_building_units := total_units - (first_building_units + second_building_units)
  have H : (third_building_units - second_building_units) / second_building_units * 100 = 20 := sorry
  exact H

end NUMINAMATH_GPT_percentage_increase_from_second_to_third_building_l2258_225825


namespace NUMINAMATH_GPT_compute_tensor_operation_l2258_225819

def tensor (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

theorem compute_tensor_operation :
  tensor (tensor 8 4) 2 = 202 / 9 :=
by
  sorry

end NUMINAMATH_GPT_compute_tensor_operation_l2258_225819


namespace NUMINAMATH_GPT_p_and_q_necessary_but_not_sufficient_l2258_225866

theorem p_and_q_necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := 
by 
  sorry

end NUMINAMATH_GPT_p_and_q_necessary_but_not_sufficient_l2258_225866


namespace NUMINAMATH_GPT_Emilee_earns_25_l2258_225894

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_Emilee_earns_25_l2258_225894


namespace NUMINAMATH_GPT_largest_prime_inequality_l2258_225897

def largest_prime_divisor (n : Nat) : Nat :=
  sorry  -- Placeholder to avoid distractions in problem statement

theorem largest_prime_inequality (q : Nat) (h_q_prime : Prime q) (hq_odd : q % 2 = 1) :
    ∃ k : Nat, k > 0 ∧ largest_prime_divisor (q^(2^k) - 1) < q ∧ q < largest_prime_divisor (q^(2^k) + 1) :=
sorry

end NUMINAMATH_GPT_largest_prime_inequality_l2258_225897


namespace NUMINAMATH_GPT_find_a1_l2258_225880

-- Define the sequence
def seq (a : ℕ → ℝ) := ∀ n : ℕ, 0 < n → a n = (1/2) * a (n + 1)

-- Given conditions
def a3_value (a : ℕ → ℝ) := a 3 = 12

-- Theorem statement
theorem find_a1 (a : ℕ → ℝ) (h_seq : seq a) (h_a3 : a3_value a) : a 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l2258_225880


namespace NUMINAMATH_GPT_base_length_of_isosceles_l2258_225847

-- Define the lengths of the sides and the perimeter of the triangle.
def side_length1 : ℝ := 10
def side_length2 : ℝ := 10
def perimeter : ℝ := 35

-- Define the problem statement to prove the length of the base.
theorem base_length_of_isosceles (b : ℝ) 
  (h1 : side_length1 = 10) 
  (h2 : side_length2 = 10) 
  (h3 : perimeter = 35) : b = 15 :=
by
  -- Skip the proof.
  sorry

end NUMINAMATH_GPT_base_length_of_isosceles_l2258_225847


namespace NUMINAMATH_GPT_minimal_face_sum_of_larger_cube_l2258_225890

-- Definitions
def num_small_cubes : ℕ := 27
def num_faces_per_cube : ℕ := 6

-- The goal: Prove the minimal sum of the integers shown on the faces of the larger cube
theorem minimal_face_sum_of_larger_cube (min_sum : ℤ) 
    (H : min_sum = 90) :
    min_sum = 90 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimal_face_sum_of_larger_cube_l2258_225890


namespace NUMINAMATH_GPT_plus_signs_count_l2258_225801

theorem plus_signs_count (num_symbols : ℕ) (at_least_one_plus_in_10 : ∀ s : Finset ℕ, s.card = 10 → (∃ i ∈ s, i < 14)) (at_least_one_minus_in_15 : ∀ s : Finset ℕ, s.card = 15 → (∃ i ∈ s, i ≥ 14)) : 
    ∃ (p m : ℕ), p + m = 23 ∧ p = 14 ∧ m = 9 := by
  sorry

end NUMINAMATH_GPT_plus_signs_count_l2258_225801


namespace NUMINAMATH_GPT_new_concentration_of_mixture_l2258_225838

theorem new_concentration_of_mixture
  (v1_cap : ℝ) (v1_alcohol_percent : ℝ)
  (v2_cap : ℝ) (v2_alcohol_percent : ℝ)
  (new_vessel_cap : ℝ) (poured_liquid : ℝ)
  (filled_water : ℝ) :
  v1_cap = 2 →
  v1_alcohol_percent = 0.25 →
  v2_cap = 6 →
  v2_alcohol_percent = 0.50 →
  new_vessel_cap = 10 →
  poured_liquid = 8 →
  filled_water = (new_vessel_cap - poured_liquid) →
  ((v1_cap * v1_alcohol_percent + v2_cap * v2_alcohol_percent) / new_vessel_cap) = 0.35 :=
by
  intros v1_h v1_per_h v2_h v2_per_h v_new_h poured_h filled_h
  sorry

end NUMINAMATH_GPT_new_concentration_of_mixture_l2258_225838


namespace NUMINAMATH_GPT_regular_ngon_on_parallel_lines_l2258_225891

theorem regular_ngon_on_parallel_lines (n : ℕ) : 
  (∃ f : ℝ → ℝ, (∀ m : ℕ, ∃ k : ℕ, f (m * (360 / n)) = k * (360 / n))) ↔
  n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end NUMINAMATH_GPT_regular_ngon_on_parallel_lines_l2258_225891


namespace NUMINAMATH_GPT_men_days_proof_l2258_225856

noncomputable def time_to_complete (m d e r : ℕ) : ℕ :=
  (m * d) / (e * (m + r))

theorem men_days_proof (m d e r t : ℕ) (h1 : d = (m * d) / (m * e))
  (h2 : t = (m * d) / (e * (m + r))) :
  t = (m * d) / (e * (m + r)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_men_days_proof_l2258_225856


namespace NUMINAMATH_GPT_jill_spent_on_other_items_l2258_225851

theorem jill_spent_on_other_items {T : ℝ} (h₁ : T > 0)
    (h₁ : 0.5 * T + 0.2 * T + O * T / 100 = T)
    (h₂ : 0.04 * 0.5 * T = 0.02 * T)
    (h₃ : 0 * 0.2 * T = 0)
    (h₄ : 0.08 * O * T / 100 = 0.0008 * O * T)
    (h₅ : 0.044 * T = 0.02 * T + 0 + 0.0008 * O * T) :
  O = 30 := 
sorry

end NUMINAMATH_GPT_jill_spent_on_other_items_l2258_225851


namespace NUMINAMATH_GPT_pumpkins_at_other_orchard_l2258_225874

-- Defining the initial conditions
def sunshine_pumpkins : ℕ := 54
def other_orchard_pumpkins : ℕ := 14

-- Equation provided in the problem
def condition_equation (P : ℕ) : Prop := 54 = 3 * P + 12

-- Proving the main statement using the conditions
theorem pumpkins_at_other_orchard : condition_equation other_orchard_pumpkins :=
by
  unfold condition_equation
  sorry -- To be completed with the proof

end NUMINAMATH_GPT_pumpkins_at_other_orchard_l2258_225874


namespace NUMINAMATH_GPT_appropriate_sampling_method_l2258_225877

theorem appropriate_sampling_method
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (survey_size : ℕ)
  (diff_interests : Prop)
  (h1 : total_students = 1000)
  (h2 : male_students = 500)
  (h3 : female_students = 500)
  (h4 : survey_size = 100)
  (h5 : diff_interests) : 
  sampling_method = "stratified sampling" :=
by
  sorry

end NUMINAMATH_GPT_appropriate_sampling_method_l2258_225877


namespace NUMINAMATH_GPT_no_integer_triplets_satisfying_eq_l2258_225854

theorem no_integer_triplets_satisfying_eq (x y z : ℤ) : 3 * x^2 + 7 * y^2 ≠ z^4 := 
by {
  sorry
}

end NUMINAMATH_GPT_no_integer_triplets_satisfying_eq_l2258_225854


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l2258_225850

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l2258_225850


namespace NUMINAMATH_GPT_denom_asymptotes_sum_l2258_225815

theorem denom_asymptotes_sum (A B C : ℤ)
  (h_denom : ∀ x, (x = -1 ∨ x = 3 ∨ x = 4) → x^3 + A * x^2 + B * x + C = 0) :
  A + B + C = 11 := 
sorry

end NUMINAMATH_GPT_denom_asymptotes_sum_l2258_225815


namespace NUMINAMATH_GPT_sqrt_10_integer_decimal_partition_l2258_225808

theorem sqrt_10_integer_decimal_partition:
  let a := Int.floor (Real.sqrt 10)
  let b := Real.sqrt 10 - a
  (Real.sqrt 10 + a) * b = 1 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_10_integer_decimal_partition_l2258_225808


namespace NUMINAMATH_GPT_standard_deviation_of_applicants_l2258_225822

theorem standard_deviation_of_applicants (σ : ℕ) 
  (h1 : ∃ avg : ℕ, avg = 30)
  (h2 : ∃ n : ℕ, n = 17)
  (h3 : ∃ range_count : ℕ, range_count = (30 + σ) - (30 - σ) + 1) :
  σ = 8 :=
by
  sorry

end NUMINAMATH_GPT_standard_deviation_of_applicants_l2258_225822


namespace NUMINAMATH_GPT_minimum_a_plus_2b_no_a_b_such_that_l2258_225860

noncomputable def minimum_value (a b : ℝ) :=
  a + 2 * b

theorem minimum_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  minimum_value a b ≥ 6 :=
sorry

theorem no_a_b_such_that (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  a^2 + 4 * b^2 ≠ 17 :=
sorry

end NUMINAMATH_GPT_minimum_a_plus_2b_no_a_b_such_that_l2258_225860


namespace NUMINAMATH_GPT_points_in_rectangle_distance_l2258_225832

/-- In a 3x4 rectangle, if 4 points are randomly located, 
    then the distance between at least two of them is at most 25/8. -/
theorem points_in_rectangle_distance (a b : ℝ) (h₁ : a = 3) (h₂ : b = 4)
  {points : Fin 4 → ℝ × ℝ}
  (h₃ : ∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ a)
  (h₄ : ∀ i, 0 ≤ (points i).2 ∧ (points i).2 ≤ b) :
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 25 / 8 := 
by
  sorry

end NUMINAMATH_GPT_points_in_rectangle_distance_l2258_225832


namespace NUMINAMATH_GPT_total_amount_earned_l2258_225824

theorem total_amount_earned (avg_price_per_pair : ℝ) (number_of_pairs : ℕ) (price : avg_price_per_pair = 9.8 ) (pairs : number_of_pairs = 50 ) : 
avg_price_per_pair * number_of_pairs = 490 := by
  -- Given conditions
  sorry

end NUMINAMATH_GPT_total_amount_earned_l2258_225824


namespace NUMINAMATH_GPT_derivative_of_even_function_is_odd_l2258_225849

variables {R : Type*}

-- Definitions and Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem derivative_of_even_function_is_odd (f g : ℝ → ℝ) (h1 : even_function f) (h2 : ∀ x, deriv f x = g x) : odd_function g :=
sorry

end NUMINAMATH_GPT_derivative_of_even_function_is_odd_l2258_225849


namespace NUMINAMATH_GPT_area_of_triangle_formed_by_medians_l2258_225848

variable {a b c m_a m_b m_c Δ Δ': ℝ}

-- Conditions from the problem
axiom rel_sum_of_squares : m_a^2 + m_b^2 + m_c^2 = (3 / 4) * (a^2 + b^2 + c^2)
axiom rel_fourth_powers : m_a^4 + m_b^4 + m_c^4 = (9 / 16) * (a^4 + b^4 + c^4)

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_medians :
  Δ' = (3 / 4) * Δ := sorry

end NUMINAMATH_GPT_area_of_triangle_formed_by_medians_l2258_225848


namespace NUMINAMATH_GPT_claire_speed_l2258_225893

def distance := 2067
def time := 39

def speed (d : ℕ) (t : ℕ) : ℕ := d / t

theorem claire_speed : speed distance time = 53 := by
  sorry

end NUMINAMATH_GPT_claire_speed_l2258_225893


namespace NUMINAMATH_GPT_fifth_number_in_pascal_row_l2258_225820

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end NUMINAMATH_GPT_fifth_number_in_pascal_row_l2258_225820


namespace NUMINAMATH_GPT_solution_set_f_lt_g_l2258_225814

noncomputable def f : ℝ → ℝ := sorry -- Assume f exists according to the given conditions

lemma f_at_one : f 1 = -2 := sorry

lemma f_derivative_neg (x : ℝ) : (deriv f x) < 0 := sorry

def g (x : ℝ) : ℝ := x - 3

lemma g_at_one : g 1 = -2 := sorry

theorem solution_set_f_lt_g :
  {x : ℝ | f x < g x} = {x : ℝ | 1 < x} :=
sorry

end NUMINAMATH_GPT_solution_set_f_lt_g_l2258_225814


namespace NUMINAMATH_GPT_fly_dist_ceiling_eq_sqrt255_l2258_225870

noncomputable def fly_distance_from_ceiling : ℝ :=
  let x := 3
  let y := 5
  let d := 17
  let z := Real.sqrt (d^2 - (x^2 + y^2))
  z

theorem fly_dist_ceiling_eq_sqrt255 :
  fly_distance_from_ceiling = Real.sqrt 255 :=
by
  sorry

end NUMINAMATH_GPT_fly_dist_ceiling_eq_sqrt255_l2258_225870


namespace NUMINAMATH_GPT_rth_term_arithmetic_progression_l2258_225872

-- Define the sum of the first n terms of the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^3

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating the r-th term of the arithmetic progression
theorem rth_term_arithmetic_progression (r : ℕ) : a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end NUMINAMATH_GPT_rth_term_arithmetic_progression_l2258_225872


namespace NUMINAMATH_GPT_distance_from_point_to_line_condition_l2258_225816

theorem distance_from_point_to_line_condition (a : ℝ) : (|a - 2| = 3) ↔ (a = 5 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_distance_from_point_to_line_condition_l2258_225816


namespace NUMINAMATH_GPT_charlie_first_week_usage_l2258_225861

noncomputable def data_used_week1 : ℕ :=
  let data_plan := 8
  let week2_usage := 3
  let week3_usage := 5
  let week4_usage := 10
  let total_extra_cost := 120
  let cost_per_gb_extra := 10
  let total_data_used := data_plan + (total_extra_cost / cost_per_gb_extra)
  let total_data_week_2_3_4 := week2_usage + week3_usage + week4_usage
  total_data_used - total_data_week_2_3_4

theorem charlie_first_week_usage : data_used_week1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_charlie_first_week_usage_l2258_225861


namespace NUMINAMATH_GPT_interest_rate_is_4_percent_l2258_225868

variable (P A n : ℝ)
variable (r : ℝ)
variable (n_pos : n ≠ 0)

-- Define the conditions
def principal : ℝ := P
def amount_after_n_years : ℝ := A
def years : ℝ := n
def interest_rate : ℝ := r

-- The compound interest formula
def compound_interest (P A r : ℝ) (n : ℝ) : Prop :=
  A = P * (1 + r) ^ n

-- The Lean theorem statement
theorem interest_rate_is_4_percent
  (P_val : principal = 7500)
  (A_val : amount_after_n_years = 8112)
  (n_val : years = 2)
  (h : compound_interest P A r n) :
  r = 0.04 :=
sorry

end NUMINAMATH_GPT_interest_rate_is_4_percent_l2258_225868


namespace NUMINAMATH_GPT_find_number_l2258_225862

-- Define the condition that k is a non-negative integer
def is_nonnegative_int (k : ℕ) : Prop := k ≥ 0

-- Define the condition that 18^k is a divisor of the number n
def is_divisor (n k : ℕ) : Prop := 18^k ∣ n

-- The main theorem statement
theorem find_number (n k : ℕ) (h_nonneg : is_nonnegative_int k) (h_eq : 6^k - k^6 = 1) (h_div : is_divisor n k) : n = 1 :=
  sorry

end NUMINAMATH_GPT_find_number_l2258_225862


namespace NUMINAMATH_GPT_lily_pad_half_coverage_l2258_225898

-- Define the conditions in Lean
def doubles_daily (size: ℕ → ℕ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def covers_entire_lake (size: ℕ → ℕ) (total_size: ℕ) : Prop :=
  size 34 = total_size

-- The main statement to prove
theorem lily_pad_half_coverage (size : ℕ → ℕ) (total_size : ℕ) 
  (h1 : doubles_daily size) 
  (h2 : covers_entire_lake size total_size) : 
  size 33 = total_size / 2 :=
sorry

end NUMINAMATH_GPT_lily_pad_half_coverage_l2258_225898


namespace NUMINAMATH_GPT_percentage_increase_l2258_225809

theorem percentage_increase (original_value : ℕ) (percentage_increase : ℚ) :  
  original_value = 1200 → 
  percentage_increase = 0.40 →
  original_value * (1 + percentage_increase) = 1680 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_percentage_increase_l2258_225809


namespace NUMINAMATH_GPT_no_nat_num_divisible_l2258_225881

open Nat

theorem no_nat_num_divisible : ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 := sorry

end NUMINAMATH_GPT_no_nat_num_divisible_l2258_225881


namespace NUMINAMATH_GPT_total_fires_l2258_225873

theorem total_fires (Doug_fires Kai_fires Eli_fires : ℕ)
  (h1 : Doug_fires = 20)
  (h2 : Kai_fires = 3 * Doug_fires)
  (h3 : Eli_fires = Kai_fires / 2) :
  Doug_fires + Kai_fires + Eli_fires = 110 :=
by
  sorry

end NUMINAMATH_GPT_total_fires_l2258_225873


namespace NUMINAMATH_GPT_determine_b2050_l2258_225878

theorem determine_b2050 (b : ℕ → ℝ)
  (h₀ : ∀ n ≥ 2, b n = b (n - 1) * b (n + 1))
  (h₁ : b 1 = 3 + Real.sqrt 2)
  (h₂ : b 2021 = 7 + 2 * Real.sqrt 2) :
  b 2050 = (7 - 2 * Real.sqrt 2) / 41 := 
sorry

end NUMINAMATH_GPT_determine_b2050_l2258_225878


namespace NUMINAMATH_GPT_minimum_value_proof_l2258_225887

noncomputable def minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem minimum_value_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  minimum_value x y z hx hy hz h ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_proof_l2258_225887


namespace NUMINAMATH_GPT_problem_l2258_225837

-- Step 1: Define the transformation functions
def rotate_90_counterclockwise (h k x y : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

def reflect_y_eq_x (x y : ℝ) : ℝ × ℝ :=
  (y, x)

-- Step 2: Define the given problem condition
theorem problem (a b : ℝ) :
  rotate_90_counterclockwise 2 3 (reflect_y_eq_x 5 1).fst (reflect_y_eq_x 5 1).snd = (a, b) →
  b - a = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_problem_l2258_225837


namespace NUMINAMATH_GPT_rectangle_area_perimeter_eq_l2258_225803

theorem rectangle_area_perimeter_eq (x : ℝ) (h : 4 * x * (x + 4) = 2 * 4 * x + 2 * (x + 4)) : x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_rectangle_area_perimeter_eq_l2258_225803


namespace NUMINAMATH_GPT_petya_max_margin_l2258_225892

def max_margin_votes (total_votes P1 P2 V1 V2: ℕ) : ℕ := P1 + P2 - (V1 + V2)

theorem petya_max_margin 
  (P1 P2 V1 V2: ℕ)
  (H1: P1 = V1 + 9) 
  (H2: V2 = P2 + 9) 
  (H3: P1 + P2 + V1 + V2 = 27) 
  (H_win: P1 + P2 > V1 + V2) : 
  max_margin_votes 27 P1 P2 V1 V2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_petya_max_margin_l2258_225892


namespace NUMINAMATH_GPT_travel_agency_choice_l2258_225883

noncomputable def y₁ (x : ℝ) : ℝ := 350 * x + 1000

noncomputable def y₂ (x : ℝ) : ℝ := 400 * x + 800

theorem travel_agency_choice (x : ℝ) (h : 0 < x) :
  (x < 4 → y₁ x > y₂ x) ∧ 
  (x = 4 → y₁ x = y₂ x) ∧ 
  (x > 4 → y₁ x < y₂ x) :=
by {
  sorry
}

end NUMINAMATH_GPT_travel_agency_choice_l2258_225883


namespace NUMINAMATH_GPT_area_of_Q1Q3Q5Q7_l2258_225800

def regular_octagon_apothem : ℝ := 3

def area_of_quadrilateral (a : ℝ) : Prop :=
  let s := 6 * (1 - Real.sqrt 2)
  let side_length := s * Real.sqrt 2
  let area := side_length ^ 2
  area = 72 * (3 - 2 * Real.sqrt 2)

theorem area_of_Q1Q3Q5Q7 : area_of_quadrilateral regular_octagon_apothem :=
  sorry

end NUMINAMATH_GPT_area_of_Q1Q3Q5Q7_l2258_225800


namespace NUMINAMATH_GPT_count_three_digit_numbers_increased_by_99_when_reversed_l2258_225812

def countValidNumbers : Nat := 80

theorem count_three_digit_numbers_increased_by_99_when_reversed :
  ∃ (a b c : Nat), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧
   (100 * a + 10 * b + c + 99 = 100 * c + 10 * b + a) ∧
  (countValidNumbers = 80) :=
sorry

end NUMINAMATH_GPT_count_three_digit_numbers_increased_by_99_when_reversed_l2258_225812


namespace NUMINAMATH_GPT_eight_pow_three_eq_two_pow_nine_l2258_225823

theorem eight_pow_three_eq_two_pow_nine : 8^3 = 2^9 := by
  sorry -- Proof is skipped

end NUMINAMATH_GPT_eight_pow_three_eq_two_pow_nine_l2258_225823


namespace NUMINAMATH_GPT_intersection_A_B_l2258_225802

-- Define the set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, x + 1) }

-- Define the set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, -2*x + 4) }

-- State the theorem to prove A ∩ B = {(1, 2)}
theorem intersection_A_B : A ∩ B = { (1, 2) } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2258_225802


namespace NUMINAMATH_GPT_problem1_problem2_l2258_225836

-- First problem
theorem problem1 :
  2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) - (-1 / 3) ^ 0 + (-1) ^ 2023 = -2 :=
by
  sorry

-- Second problem
theorem problem2 :
  abs (1 - Real.sqrt 2) - Real.sqrt 12 + (1 / 3) ^ (-1 : ℤ) - 2 * Real.cos (Real.pi / 4) = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2258_225836


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l2258_225879

noncomputable def repeating_decimal_sum (x y z : ℚ) : ℚ := x + y + z

theorem repeating_decimal_to_fraction :
  let x := 4 / 33
  let y := 34 / 999
  let z := 567 / 99999
  repeating_decimal_sum x y z = 134255 / 32929667 := by
  -- proofs are omitted
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l2258_225879


namespace NUMINAMATH_GPT_find_f_11_5_l2258_225888

-- Definitions based on the conditions.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def periodic_with_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -1 / f x

def f_defined_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ -2 → f x = 2 * x

-- The main theorem to prove.
theorem find_f_11_5 (f : ℝ → ℝ) :
  is_even_function f →
  functional_eqn f →
  f_defined_on_interval f →
  periodic_with_period f 6 →
  f 11.5 = 1 / 5 :=
  by
    intros h_even h_fun_eqn h_interval h_periodic
    sorry  -- proof goes here

end NUMINAMATH_GPT_find_f_11_5_l2258_225888


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_l2258_225884

theorem quadratic_real_roots_iff (α : ℝ) : (∃ x : ℝ, x^2 - 2 * x + α = 0) ↔ α ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_l2258_225884


namespace NUMINAMATH_GPT_no_four_distinct_sum_mod_20_l2258_225804

theorem no_four_distinct_sum_mod_20 (R : Fin 9 → ℕ) (h : ∀ i, R i < 19) :
  ¬ ∃ (a b c d : Fin 9), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  (R a + R b) % 20 = (R c + R d) % 20 := sorry

end NUMINAMATH_GPT_no_four_distinct_sum_mod_20_l2258_225804


namespace NUMINAMATH_GPT_fraction_simplification_l2258_225811

variable {x y z : ℝ}
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z - z / x ≠ 0)

theorem fraction_simplification :
  (x^2 - 1 / y^2) / (z - z / x) = x / z :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l2258_225811


namespace NUMINAMATH_GPT_cube_root_sum_is_integer_iff_l2258_225886

theorem cube_root_sum_is_integer_iff (n m : ℤ) (hn : n = m * (m^2 + 3) / 2) :
  ∃ (k : ℤ), (n + Real.sqrt (n^2 + 1))^(1/3) + (n - Real.sqrt (n^2 + 1))^(1/3) = k :=
by
  sorry

end NUMINAMATH_GPT_cube_root_sum_is_integer_iff_l2258_225886


namespace NUMINAMATH_GPT_find_k_l2258_225871

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h2 : k ≠ 0) : k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2258_225871


namespace NUMINAMATH_GPT_max_min_value_x_eq_1_l2258_225889

noncomputable def f (x k : ℝ) : ℝ := x^2 - 2 * (2 * k - 1) * x + 3 * k^2 - 2 * k + 6

theorem max_min_value_x_eq_1 :
  ∀ (k : ℝ), (∀ x : ℝ, ∃ m : ℝ, f x k = m → k = 1 → m = 6) → (∃ x : ℝ, x = 1) :=
by
  sorry

end NUMINAMATH_GPT_max_min_value_x_eq_1_l2258_225889


namespace NUMINAMATH_GPT_express_as_scientific_notation_l2258_225841

-- Definitions
def billion : ℝ := 10^9
def amount : ℝ := 850 * billion

-- Statement
theorem express_as_scientific_notation : amount = 8.5 * 10^11 :=
by
  sorry

end NUMINAMATH_GPT_express_as_scientific_notation_l2258_225841


namespace NUMINAMATH_GPT_solve_trig_eq_l2258_225810

open Real

theorem solve_trig_eq (x a : ℝ) (hx1 : 0 < x) (hx2 : x < 2 * π) (ha : a > 0) :
    (sin (3 * x) + a * sin (2 * x) + 2 * sin x = 0) →
    (0 < a ∧ a < 2 → x = 0 ∨ x = π) ∧ 
    (a > 5 / 2 → ∃ α, (x = α ∨ x = 2 * π - α)) :=
by sorry

end NUMINAMATH_GPT_solve_trig_eq_l2258_225810


namespace NUMINAMATH_GPT_barkley_total_net_buried_bones_l2258_225827

def monthly_bones_received (months : ℕ) : (ℕ × ℕ × ℕ) := (10 * months, 6 * months, 4 * months)

def burying_pattern_A (months : ℕ) : ℕ := 6 * months
def eating_pattern_A (months : ℕ) : ℕ := if months > 2 then 3 else 1

def burying_pattern_B (months : ℕ) : ℕ := if months = 5 then 0 else 4 * (months - 1)
def eating_pattern_B (months : ℕ) : ℕ := 2

def burying_pattern_C (months : ℕ) : ℕ := 2 * months
def eating_pattern_C (months : ℕ) : ℕ := 2

def total_net_buried_bones (months : ℕ) : ℕ :=
  let (received_A, received_B, received_C) := monthly_bones_received months
  let net_A := burying_pattern_A months - eating_pattern_A months
  let net_B := burying_pattern_B months - eating_pattern_B months
  let net_C := burying_pattern_C months - eating_pattern_C months
  net_A + net_B + net_C

theorem barkley_total_net_buried_bones : total_net_buried_bones 5 = 49 := by
  sorry

end NUMINAMATH_GPT_barkley_total_net_buried_bones_l2258_225827


namespace NUMINAMATH_GPT_solve_money_conditions_l2258_225896

theorem solve_money_conditions 
  (a b : ℝ)
  (h1 : b - 4 * a < 78)
  (h2 : 6 * a - b = 36) :
  a < 57 ∧ b > -36 :=
sorry

end NUMINAMATH_GPT_solve_money_conditions_l2258_225896


namespace NUMINAMATH_GPT_same_bill_at_300_minutes_l2258_225833

def monthlyBillA (x : ℕ) : ℝ := 15 + 0.1 * x
def monthlyBillB (x : ℕ) : ℝ := 0.15 * x

theorem same_bill_at_300_minutes : monthlyBillA 300 = monthlyBillB 300 := 
by
  sorry

end NUMINAMATH_GPT_same_bill_at_300_minutes_l2258_225833


namespace NUMINAMATH_GPT_randy_quiz_score_l2258_225853

theorem randy_quiz_score (q1 q2 q3 q5 : ℕ) (q4 : ℕ) :
  q1 = 90 → q2 = 98 → q3 = 94 → q5 = 96 → (q1 + q2 + q3 + q4 + q5) / 5 = 94 → q4 = 92 :=
by
  intros h1 h2 h3 h5 h_avg
  sorry

end NUMINAMATH_GPT_randy_quiz_score_l2258_225853


namespace NUMINAMATH_GPT_expression_evaluation_l2258_225806

theorem expression_evaluation : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l2258_225806


namespace NUMINAMATH_GPT_counting_numbers_dividing_48_with_remainder_7_l2258_225885

theorem counting_numbers_dividing_48_with_remainder_7 :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, n > 7 ∧ 48 % n = 0 :=
by
  sorry

end NUMINAMATH_GPT_counting_numbers_dividing_48_with_remainder_7_l2258_225885


namespace NUMINAMATH_GPT_fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l2258_225842

theorem fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes :
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) -> n = 45 :=
by
  sorry

end NUMINAMATH_GPT_fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l2258_225842


namespace NUMINAMATH_GPT_intersection_non_empty_l2258_225865

open Set

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}

theorem intersection_non_empty (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := 
sorry

end NUMINAMATH_GPT_intersection_non_empty_l2258_225865


namespace NUMINAMATH_GPT_restaurant_total_glasses_l2258_225831

theorem restaurant_total_glasses (x y t : ℕ) 
  (h1 : y = x + 16)
  (h2 : (12 * x + 16 * y) / (x + y) = 15)
  (h3 : t = 12 * x + 16 * y) : 
  t = 480 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_restaurant_total_glasses_l2258_225831


namespace NUMINAMATH_GPT_determine_subtracted_number_l2258_225867

theorem determine_subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 7 * x - y = 130) : y = 150 :=
by sorry

end NUMINAMATH_GPT_determine_subtracted_number_l2258_225867


namespace NUMINAMATH_GPT_fraction_of_project_completed_in_one_hour_l2258_225826

noncomputable def fraction_of_project_completed_together (a b : ℝ) : ℝ :=
  (1 / a) + (1 / b)

theorem fraction_of_project_completed_in_one_hour (a b : ℝ) :
  fraction_of_project_completed_together a b = (1 / a) + (1 / b) := by
  sorry

end NUMINAMATH_GPT_fraction_of_project_completed_in_one_hour_l2258_225826


namespace NUMINAMATH_GPT_amount_of_loan_l2258_225852

theorem amount_of_loan (P R T SI : ℝ) (hR : R = 6) (hT : T = 6) (hSI : SI = 432) :
  SI = (P * R * T) / 100 → P = 1200 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_amount_of_loan_l2258_225852


namespace NUMINAMATH_GPT_ramsey_example_l2258_225859

theorem ramsey_example (P : Fin 10 → Fin 10 → Prop) :
  (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(¬P i j ∧ ¬P j k ∧ ¬P k i))
  ∨ (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(P i j ∧ P j k ∧ P k i)) →
  (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (P i j ∧ P j k ∧ P k l ∧ P i k ∧ P j l ∧ P i l))
  ∨ (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (¬P i j ∧ ¬P j k ∧ ¬P k l ∧ ¬P i k ∧ ¬P j l ∧ ¬P i l)) :=
by
  sorry

end NUMINAMATH_GPT_ramsey_example_l2258_225859


namespace NUMINAMATH_GPT_cube_number_sum_is_102_l2258_225863

noncomputable def sum_of_cube_numbers (n1 n2 n3 n4 n5 n6 : ℕ) : ℕ := n1 + n2 + n3 + n4 + n5 + n6

theorem cube_number_sum_is_102 : 
  ∃ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 12 ∧ 
    n2 = n1 + 2 ∧ 
    n3 = n2 + 2 ∧ 
    n4 = n3 + 2 ∧ 
    n5 = n4 + 2 ∧ 
    n6 = n5 + 2 ∧ 
    ((n1 + n6 = n2 + n5) ∧ (n1 + n6 = n3 + n4)) ∧ 
    sum_of_cube_numbers n1 n2 n3 n4 n5 n6 = 102 :=
by
  sorry

end NUMINAMATH_GPT_cube_number_sum_is_102_l2258_225863


namespace NUMINAMATH_GPT_sqrt_expression_identity_l2258_225817

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 17 - 4

theorem sqrt_expression_identity : Real.sqrt ((-a)^3 + (b + 4)^2) = 4 :=
by
  -- Prove the statement

  sorry

end NUMINAMATH_GPT_sqrt_expression_identity_l2258_225817


namespace NUMINAMATH_GPT_probability_event_in_single_trial_l2258_225834

theorem probability_event_in_single_trial (p : ℝ) 
  (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (1 - p)^4 = 16 / 81) : 
  p = 1 / 3 :=
sorry

end NUMINAMATH_GPT_probability_event_in_single_trial_l2258_225834


namespace NUMINAMATH_GPT_lloyd_excess_rate_multiple_l2258_225844

theorem lloyd_excess_rate_multiple :
  let h_regular := 7.5
  let r := 4.00
  let h_total := 10.5
  let e_total := 48
  let e_regular := h_regular * r
  let excess_hours := h_total - h_regular
  let e_excess := e_total - e_regular
  let m := e_excess / (excess_hours * r)
  m = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_lloyd_excess_rate_multiple_l2258_225844


namespace NUMINAMATH_GPT_even_function_a_equals_one_l2258_225818

theorem even_function_a_equals_one (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x - a) = (1 - x) * (-x - a)) → a = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_even_function_a_equals_one_l2258_225818


namespace NUMINAMATH_GPT_factor_sum_l2258_225813

theorem factor_sum (P Q R : ℤ) (h : ∃ (b c : ℤ), (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + R*x + Q) : 
  P + Q + R = 11*P - 1 := 
sorry

end NUMINAMATH_GPT_factor_sum_l2258_225813


namespace NUMINAMATH_GPT_tangent_fraction_15_degrees_l2258_225857

theorem tangent_fraction_15_degrees : (1 + Real.tan (Real.pi / 12 )) / (1 - Real.tan (Real.pi / 12)) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_fraction_15_degrees_l2258_225857


namespace NUMINAMATH_GPT_correct_equation_by_moving_digit_l2258_225864

theorem correct_equation_by_moving_digit :
  (10^2 - 1 = 99) → (101 = 102 - 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_equation_by_moving_digit_l2258_225864


namespace NUMINAMATH_GPT_triangle_area_l2258_225846

theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) (h4 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 270 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l2258_225846


namespace NUMINAMATH_GPT_complex_number_second_quadrant_l2258_225858

theorem complex_number_second_quadrant 
  : (2 + 3 * Complex.I) / (1 - Complex.I) ∈ { z : Complex | z.re < 0 ∧ z.im > 0 } := 
by
  sorry

end NUMINAMATH_GPT_complex_number_second_quadrant_l2258_225858


namespace NUMINAMATH_GPT_value_of_T_l2258_225895

theorem value_of_T (T : ℝ) (h : (1 / 3) * (1 / 6) * T = (1 / 4) * (1 / 8) * 120) : T = 67.5 :=
sorry

end NUMINAMATH_GPT_value_of_T_l2258_225895


namespace NUMINAMATH_GPT_max_min_values_l2258_225843

theorem max_min_values (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  1 ≤ x^2 + 2*y^2 + 3*z^2 ∧ x^2 + 2*y^2 + 3*z^2 ≤ 3 := by
  sorry

end NUMINAMATH_GPT_max_min_values_l2258_225843


namespace NUMINAMATH_GPT_chocolate_eggs_total_weight_l2258_225821

def total_weight_after_discarding_box_b : ℕ :=
  let weight_large := 14
  let weight_medium := 10
  let weight_small := 6
  let box_A_weight := 4 * weight_large + 2 * weight_medium
  let box_B_weight := 6 * weight_small + 2 * weight_large
  let box_C_weight := 4 * weight_large + 3 * weight_medium
  let box_D_weight := 4 * weight_medium + 4 * weight_small
  let box_E_weight := 4 * weight_small + 2 * weight_medium
  box_A_weight + box_C_weight + box_D_weight + box_E_weight

theorem chocolate_eggs_total_weight : total_weight_after_discarding_box_b = 270 := by
  sorry

end NUMINAMATH_GPT_chocolate_eggs_total_weight_l2258_225821


namespace NUMINAMATH_GPT_greatest_number_of_bouquets_l2258_225829

def sara_red_flowers : ℕ := 16
def sara_yellow_flowers : ℕ := 24

theorem greatest_number_of_bouquets : Nat.gcd sara_red_flowers sara_yellow_flowers = 8 := by
  rfl

end NUMINAMATH_GPT_greatest_number_of_bouquets_l2258_225829


namespace NUMINAMATH_GPT_median_and_mode_of_successful_shots_l2258_225882

theorem median_and_mode_of_successful_shots :
  let shots := [3, 6, 4, 6, 4, 3, 6, 5, 7]
  let sorted_shots := [3, 3, 4, 4, 5, 6, 6, 6, 7]
  let median := sorted_shots[4]  -- 4 is the index for the 5th element (0-based indexing)
  let mode := 6  -- determined by the number that appears most frequently
  median = 5 ∧ mode = 6 :=
by
  sorry

end NUMINAMATH_GPT_median_and_mode_of_successful_shots_l2258_225882


namespace NUMINAMATH_GPT_price_increase_decrease_l2258_225828

theorem price_increase_decrease (P : ℝ) (h : 0.84 * P = P * (1 - (x / 100)^2)) : x = 40 := by
  sorry

end NUMINAMATH_GPT_price_increase_decrease_l2258_225828


namespace NUMINAMATH_GPT_range_of_m_l2258_225839

noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

theorem range_of_m :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → 2^x - Real.log x / Real.log (1/2) + m ≤ 0) →
  m ≤ -5 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2258_225839


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2258_225875

/-- Given an isosceles triangle with one side length of 3 cm and another side length of 5 cm,
    its perimeter is either 11 cm or 13 cm. -/
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (∃ c : ℝ, (c = 3 ∨ c = 5) ∧ (2 * a + b = 11 ∨ 2 * b + a = 13)) :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2258_225875


namespace NUMINAMATH_GPT_aaron_ate_more_apples_l2258_225845

-- Define the number of apples eaten by Aaron and Zeb
def apples_eaten_by_aaron : ℕ := 6
def apples_eaten_by_zeb : ℕ := 1

-- Theorem to prove the difference in apples eaten
theorem aaron_ate_more_apples :
  apples_eaten_by_aaron - apples_eaten_by_zeb = 5 :=
by
  sorry

end NUMINAMATH_GPT_aaron_ate_more_apples_l2258_225845


namespace NUMINAMATH_GPT_solve_for_x_l2258_225805

theorem solve_for_x :
  ∃ x : ℝ, (2015 + x)^2 = x^2 ∧ x = -2015 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2258_225805
