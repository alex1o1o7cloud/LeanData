import Mathlib

namespace NUMINAMATH_GPT_flag_pole_height_eq_150_l2211_221145

-- Define the conditions
def tree_height : ℝ := 12
def tree_shadow_length : ℝ := 8
def flag_pole_shadow_length : ℝ := 100

-- Problem statement: prove the height of the flag pole equals 150 meters
theorem flag_pole_height_eq_150 :
  ∃ (F : ℝ), (tree_height / tree_shadow_length) = (F / flag_pole_shadow_length) ∧ F = 150 :=
by
  -- Setup the proof scaffold
  have h : (tree_height / tree_shadow_length) = (150 / flag_pole_shadow_length) := by sorry
  exact ⟨150, h, rfl⟩

end NUMINAMATH_GPT_flag_pole_height_eq_150_l2211_221145


namespace NUMINAMATH_GPT_find_sister_candy_initially_l2211_221153

-- Defining the initial pieces of candy Katie had.
def katie_candy : ℕ := 8

-- Defining the pieces of candy Katie's sister had initially.
def sister_candy_initially : ℕ := sorry -- To be determined

-- The total number of candy pieces they had after eating 8 pieces.
def total_remaining_candy : ℕ := 23

theorem find_sister_candy_initially : 
  (katie_candy + sister_candy_initially - 8 = total_remaining_candy) → (sister_candy_initially = 23) :=
by
  sorry

end NUMINAMATH_GPT_find_sister_candy_initially_l2211_221153


namespace NUMINAMATH_GPT_find_base_c_l2211_221163

theorem find_base_c (c : ℕ) : (c^3 - 7*c^2 - 18*c - 8 = 0) → c = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_base_c_l2211_221163


namespace NUMINAMATH_GPT_solve_for_x_l2211_221112

-- Definitions for the problem conditions
def perimeter_triangle := 14 + 12 + 12
def perimeter_rectangle (x : ℝ) := 2 * x + 16

-- Lean 4 statement for the proof problem 
theorem solve_for_x (x : ℝ) : 
  perimeter_triangle = perimeter_rectangle x → 
  x = 11 := 
by 
  -- standard placeholders
  sorry

end NUMINAMATH_GPT_solve_for_x_l2211_221112


namespace NUMINAMATH_GPT_volume_pyramid_l2211_221175

theorem volume_pyramid (V : ℝ) : 
  ∃ V_P : ℝ, V_P = V / 6 :=
by
  sorry

end NUMINAMATH_GPT_volume_pyramid_l2211_221175


namespace NUMINAMATH_GPT_find_johns_allowance_l2211_221125

variable (A : ℝ)  -- John's weekly allowance

noncomputable def johns_allowance : Prop :=
  let arcade_spent := (3 / 5) * A
  let remaining_after_arcade := (2 / 5) * A
  let toy_store_spent := (1 / 3) * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spent
  let final_spent := 0.88
  final_spent = remaining_after_toy_store → A = 3.30

theorem find_johns_allowance : johns_allowance A := by
  sorry

end NUMINAMATH_GPT_find_johns_allowance_l2211_221125


namespace NUMINAMATH_GPT_proof_standard_deviation_l2211_221190

noncomputable def standard_deviation (average_age : ℝ) (max_diff_ages : ℕ) : ℝ := sorry

theorem proof_standard_deviation :
  let average_age := 31
  let max_diff_ages := 19
  standard_deviation average_age max_diff_ages = 9 := 
by
  sorry

end NUMINAMATH_GPT_proof_standard_deviation_l2211_221190


namespace NUMINAMATH_GPT_ratio_josh_to_doug_l2211_221183

theorem ratio_josh_to_doug (J D B : ℕ) (h1 : J + D + B = 68) (h2 : J = 2 * B) (h3 : D = 32) : J / D = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_ratio_josh_to_doug_l2211_221183


namespace NUMINAMATH_GPT_compute_result_l2211_221118

-- Define the operations a # b and b # c
def operation (a b : ℤ) : ℤ := a * b - b + b^2

-- Define the expression for (3 # 8) # z given the operations
def evaluate (z : ℤ) : ℤ := operation (operation 3 8) z

-- Prove that (3 # 8) # z = 79z + z^2
theorem compute_result (z : ℤ) : evaluate z = 79 * z + z^2 := 
by
  sorry

end NUMINAMATH_GPT_compute_result_l2211_221118


namespace NUMINAMATH_GPT_no_positive_integer_solution_l2211_221199

theorem no_positive_integer_solution (a b c d : ℕ) (h1 : a^2 + b^2 = c^2 - d^2) (h2 : a * b = c * d) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : false := 
by 
  sorry

end NUMINAMATH_GPT_no_positive_integer_solution_l2211_221199


namespace NUMINAMATH_GPT_average_speed_rest_of_trip_l2211_221184

variable (v : ℝ) -- The average speed for the rest of the trip
variable (d1 : ℝ := 30 * 5) -- Distance for the first part of the trip
variable (t1 : ℝ := 5) -- Time for the first part of the trip
variable (t_total : ℝ := 7.5) -- Total time for the trip
variable (avg_total : ℝ := 34) -- Average speed for the entire trip

def total_distance := avg_total * t_total
def d2 := total_distance - d1
def t2 := t_total - t1

theorem average_speed_rest_of_trip : 
  v = 42 :=
by
  let distance_rest := d2
  let time_rest := t2
  have v_def : v = distance_rest / time_rest := by sorry
  have v_value : v = 42 := by sorry
  exact v_value

end NUMINAMATH_GPT_average_speed_rest_of_trip_l2211_221184


namespace NUMINAMATH_GPT_largest_four_digit_sum_23_l2211_221191

theorem largest_four_digit_sum_23 : ∃ (n : ℕ), (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ a + b + c + d = 23 ∧ 1000 ≤ n ∧ n < 10000) ∧ n = 9950 :=
  sorry

end NUMINAMATH_GPT_largest_four_digit_sum_23_l2211_221191


namespace NUMINAMATH_GPT_triangle_area_l2211_221130

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (0, 0)) (hB : B = (0, 8)) (hC : C = (10, 15)) : 
  let base := 8
  let height := 10
  let area := 1 / 2 * base * height
  area = 40.0 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l2211_221130


namespace NUMINAMATH_GPT_last_digit_of_product_of_consecutive_numbers_l2211_221192

theorem last_digit_of_product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h1 : k > 5)
    (h2 : n = (k + 1) * (k + 2) * (k + 3) * (k + 4))
    (h3 : n % 10 ≠ 0) : n % 10 = 4 :=
sorry -- Proof not provided as per instructions.

end NUMINAMATH_GPT_last_digit_of_product_of_consecutive_numbers_l2211_221192


namespace NUMINAMATH_GPT_mutually_coprime_divisors_l2211_221121

theorem mutually_coprime_divisors (a x y : ℕ) (h1 : a = 1944) 
  (h2 : ∃ d1 d2 d3, d1 * d2 * d3 = a ∧ gcd x y = 1 ∧ gcd x (x + y) = 1 ∧ gcd y (x + y) = 1) : 
  (x = 1 ∧ y = 2 ∧ x + y = 3) ∨ 
  (x = 1 ∧ y = 8 ∧ x + y = 9) ∨ 
  (x = 1 ∧ y = 3 ∧ x + y = 4) :=
sorry

end NUMINAMATH_GPT_mutually_coprime_divisors_l2211_221121


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2211_221105

theorem sufficient_but_not_necessary (a b c : ℝ) :
  (b^2 = a * c → (c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) ∨ (b = 0)) ∧ 
  ¬ ((c ≠ 0 ∧ a ≠ 0 ∧ b * b = a * c) → b^2 = a * c) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2211_221105


namespace NUMINAMATH_GPT_volume_of_right_triangle_pyramid_l2211_221173

noncomputable def pyramid_volume (H α β : ℝ) : ℝ :=
  (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2)

theorem volume_of_right_triangle_pyramid (H α β : ℝ) (alpha_acute : 0 < α ∧ α < π / 2) (H_pos : 0 < H) (beta_acute : 0 < β ∧ β < π / 2) :
  pyramid_volume H α β = (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2) := 
sorry

end NUMINAMATH_GPT_volume_of_right_triangle_pyramid_l2211_221173


namespace NUMINAMATH_GPT_min_value_of_inverse_sum_l2211_221161

theorem min_value_of_inverse_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : (1/x) + (1/y) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_inverse_sum_l2211_221161


namespace NUMINAMATH_GPT_quadratic_sum_of_coefficients_l2211_221110

theorem quadratic_sum_of_coefficients (x : ℝ) : 
  let a := 1
  let b := 1
  let c := -4
  a + b + c = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_sum_of_coefficients_l2211_221110


namespace NUMINAMATH_GPT_remainder_73_to_73_plus73_div137_l2211_221141

theorem remainder_73_to_73_plus73_div137 :
  ((73 ^ 73 + 73) % 137) = 9 := by
  sorry

end NUMINAMATH_GPT_remainder_73_to_73_plus73_div137_l2211_221141


namespace NUMINAMATH_GPT_manicure_cost_per_person_l2211_221172

-- Definitions based on given conditions
def fingers_per_person : ℕ := 10
def total_fingers : ℕ := 210
def total_revenue : ℕ := 200  -- in dollars
def non_clients : ℕ := 11

-- Statement we want to prove
theorem manicure_cost_per_person :
  (total_revenue : ℚ) / (total_fingers / fingers_per_person - non_clients) = 9.52 :=
by
  sorry

end NUMINAMATH_GPT_manicure_cost_per_person_l2211_221172


namespace NUMINAMATH_GPT_seating_arrangements_l2211_221155

theorem seating_arrangements :
  ∀ (chairs people : ℕ), 
  chairs = 8 → 
  people = 3 → 
  (∃ gaps : ℕ, gaps = 4) → 
  (∀ pos, pos = Nat.choose 3 4) → 
  pos = 24 :=
by
  intros chairs people h1 h2 h3 h4
  have gaps := 4
  have pos := Nat.choose 4 3
  sorry

end NUMINAMATH_GPT_seating_arrangements_l2211_221155


namespace NUMINAMATH_GPT_additional_trams_proof_l2211_221170

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end NUMINAMATH_GPT_additional_trams_proof_l2211_221170


namespace NUMINAMATH_GPT_students_passed_both_l2211_221168

noncomputable def F_H : ℝ := 32
noncomputable def F_E : ℝ := 56
noncomputable def F_HE : ℝ := 12
noncomputable def total_percentage : ℝ := 100

theorem students_passed_both : (total_percentage - (F_H + F_E - F_HE)) = 24 := by
  sorry

end NUMINAMATH_GPT_students_passed_both_l2211_221168


namespace NUMINAMATH_GPT_radius_of_sphere_l2211_221151

theorem radius_of_sphere 
  (shadow_length_sphere : ℝ)
  (stick_height : ℝ)
  (stick_shadow : ℝ)
  (parallel_sun_rays : Prop) 
  (tan_θ : ℝ) 
  (h1 : tan_θ = stick_height / stick_shadow)
  (h2 : tan_θ = shadow_length_sphere / 20) :
  shadow_length_sphere / 20 = 1/4 → shadow_length_sphere = 5 := by
  sorry

end NUMINAMATH_GPT_radius_of_sphere_l2211_221151


namespace NUMINAMATH_GPT_candy_ratio_l2211_221114

theorem candy_ratio (chocolate_bars M_and_Ms marshmallows total_candies : ℕ)
  (h1 : chocolate_bars = 5)
  (h2 : M_and_Ms = 7 * chocolate_bars)
  (h3 : total_candies = 25 * 10)
  (h4 : marshmallows = total_candies - chocolate_bars - M_and_Ms) :
  marshmallows / M_and_Ms = 6 :=
by
  sorry

end NUMINAMATH_GPT_candy_ratio_l2211_221114


namespace NUMINAMATH_GPT_sugar_more_than_flour_l2211_221178

def flour_needed : Nat := 9
def sugar_needed : Nat := 11
def flour_added : Nat := 4
def sugar_added : Nat := 0

def flour_remaining : Nat := flour_needed - flour_added
def sugar_remaining : Nat := sugar_needed - sugar_added

theorem sugar_more_than_flour : sugar_remaining - flour_remaining = 6 :=
by
  sorry

end NUMINAMATH_GPT_sugar_more_than_flour_l2211_221178


namespace NUMINAMATH_GPT_complement_inter_of_A_and_B_l2211_221122

open Set

variable (U A B : Set ℕ)

theorem complement_inter_of_A_and_B:
  U = {1, 2, 3, 4, 5}
  ∧ A = {1, 2, 3}
  ∧ B = {2, 3, 4} 
  → U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_inter_of_A_and_B_l2211_221122


namespace NUMINAMATH_GPT_evaluate_fraction_l2211_221149

theorem evaluate_fraction : 1 + 3 / (4 + 5 / (6 + 7 / 8)) = 85 / 52 :=
by sorry

end NUMINAMATH_GPT_evaluate_fraction_l2211_221149


namespace NUMINAMATH_GPT_right_angled_triangle_l2211_221142

-- Define the lengths of the sides of the triangle
def a : ℕ := 3
def b : ℕ := 4
def c : ℕ := 5

-- The theorem to prove that these lengths form a right-angled triangle
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_l2211_221142


namespace NUMINAMATH_GPT_pete_mileage_l2211_221138

def steps_per_flip : Nat := 100000
def flips : Nat := 50
def final_reading : Nat := 25000
def steps_per_mile : Nat := 2000

theorem pete_mileage :
  let total_steps := (steps_per_flip * flips) + final_reading
  let total_miles := total_steps.toFloat / steps_per_mile.toFloat
  total_miles = 2512.5 :=
by
  sorry

end NUMINAMATH_GPT_pete_mileage_l2211_221138


namespace NUMINAMATH_GPT_simplify_expression_l2211_221133

variable (a b : ℝ)

theorem simplify_expression (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (a ^ (7 / 3) - 2 * a ^ (5 / 3) * b ^ (2 / 3) + a * b ^ (4 / 3)) / 
  (a ^ (5 / 3) - a ^ (4 / 3) * b ^ (1 / 3) - a * b ^ (2 / 3) + a ^ (2 / 3) * b) / 
  a ^ (1 / 3) =
  a ^ (1 / 3) + b ^ (1 / 3) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l2211_221133


namespace NUMINAMATH_GPT_number_of_candidates_l2211_221160

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
sorry

end NUMINAMATH_GPT_number_of_candidates_l2211_221160


namespace NUMINAMATH_GPT_sqrt_equiv_1715_l2211_221154

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end NUMINAMATH_GPT_sqrt_equiv_1715_l2211_221154


namespace NUMINAMATH_GPT_find_t_l2211_221136

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def tangent_point (t : ℝ) : ℝ × ℝ := (t, 0)

theorem find_t :
  (∀ (A : ℝ × ℝ), ellipse_eq A.1 A.2 → 
    ∃ (C : ℝ × ℝ),
      tangent_point 2 = C ∧
      -- C is tangent to the extended line of F1A
      -- C is tangent to the extended line of F1F2
      -- C is tangent to segment AF2
      true
  ) :=
sorry

end NUMINAMATH_GPT_find_t_l2211_221136


namespace NUMINAMATH_GPT_factorization_correct_l2211_221182

theorem factorization_correct (c d : ℤ) (h : 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) : c + 2 * d = -2 := 
sorry

end NUMINAMATH_GPT_factorization_correct_l2211_221182


namespace NUMINAMATH_GPT_train_crossing_time_l2211_221148

theorem train_crossing_time :
  ∀ (length_train1 length_train2 : ℕ) 
    (speed_train1_kmph speed_train2_kmph : ℝ), 
  length_train1 = 420 →
  speed_train1_kmph = 72 →
  length_train2 = 640 →
  speed_train2_kmph = 36 →
  (length_train1 + length_train2) / ((speed_train1_kmph - speed_train2_kmph) * (1000 / 3600)) = 106 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_crossing_time_l2211_221148


namespace NUMINAMATH_GPT_dig_second_hole_l2211_221126

theorem dig_second_hole (w1 h1 d1 w2 d2 : ℕ) (extra_workers : ℕ) (h2 : ℕ) :
  w1 = 45 ∧ h1 = 8 ∧ d1 = 30 ∧ extra_workers = 65 ∧
  w2 = w1 + extra_workers ∧ d2 = 55 →
  360 * d2 / d1 = w2 * h2 →
  h2 = 6 :=
by
  intros h cond
  sorry

end NUMINAMATH_GPT_dig_second_hole_l2211_221126


namespace NUMINAMATH_GPT_part_A_part_B_part_D_l2211_221106

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < 1)
variable (hβ : 0 < β ∧ β < 1)

-- Part A: single transmission probability
theorem part_A (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  (1 - β) * (1 - α) * (1 - β) = (1 - α) * (1 - β)^2 :=
by sorry

-- Part B: triple transmission probability
theorem part_B (α β : ℝ) (hα : 0 < α ∧ α < 1) (hβ : 0 < β ∧ β < 1) :
  β * (1 - β)^2 = β * (1 - β)^2 :=
by sorry

-- Part D: comparing single and triple transmission
theorem part_D (α β : ℝ) (hα : 0 < α ∧ α < 0.5) (hβ : 0 < β ∧ β < 1) :
  (1 - α) < (1 - α)^3 + 3 * α * (1 - α)^2 :=
by sorry

end NUMINAMATH_GPT_part_A_part_B_part_D_l2211_221106


namespace NUMINAMATH_GPT_sample_size_is_correct_l2211_221174

-- Define the conditions
def num_classes := 40
def students_per_class := 50
def selected_students := 150

-- Define the statement to prove the sample size
theorem sample_size_is_correct : selected_students = 150 := by 
  -- Proof is skipped with sorry
  sorry

end NUMINAMATH_GPT_sample_size_is_correct_l2211_221174


namespace NUMINAMATH_GPT_sin_neg_three_halves_pi_l2211_221157

theorem sin_neg_three_halves_pi : Real.sin (-3 * Real.pi / 2) = 1 := sorry

end NUMINAMATH_GPT_sin_neg_three_halves_pi_l2211_221157


namespace NUMINAMATH_GPT_initial_number_of_persons_l2211_221139

-- Define the given conditions
def initial_weights (N : ℕ) : ℝ := 65 * N
def new_person_weight : ℝ := 80
def increased_average_weight : ℝ := 2.5
def weight_increase (N : ℕ) : ℝ := increased_average_weight * N

-- Mathematically equivalent proof problem
theorem initial_number_of_persons 
    (N : ℕ)
    (h : weight_increase N = new_person_weight - 65) : N = 6 :=
by
  -- Place proof here when necessary
  sorry

end NUMINAMATH_GPT_initial_number_of_persons_l2211_221139


namespace NUMINAMATH_GPT_reservoir_water_l2211_221127

-- Conditions definitions
def total_capacity (C : ℝ) : Prop :=
  ∃ (x : ℝ), x = C

def normal_level (C : ℝ) : ℝ :=
  C - 20

def water_end_of_month (C : ℝ) : ℝ :=
  0.75 * C

def condition_equation (C : ℝ) : Prop :=
  water_end_of_month C = 2 * normal_level C

-- The theorem proving the amount of water at the end of the month is 24 million gallons given the conditions
theorem reservoir_water (C : ℝ) (hC : total_capacity C) (h_condition : condition_equation C) : water_end_of_month C = 24 :=
by
  sorry

end NUMINAMATH_GPT_reservoir_water_l2211_221127


namespace NUMINAMATH_GPT_linear_dependency_k_l2211_221152

theorem linear_dependency_k (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧
    (c1 * 1 + c2 * 4 = 0) ∧
    (c1 * 2 + c2 * k = 0) ∧
    (c1 * 3 + c2 * 6 = 0)) ↔ k = 8 :=
by
  sorry

end NUMINAMATH_GPT_linear_dependency_k_l2211_221152


namespace NUMINAMATH_GPT_coeff_x5_of_expansion_l2211_221186

theorem coeff_x5_of_expansion : 
  (Polynomial.coeff ((Polynomial.C (1 : ℤ)) * (Polynomial.X ^ 2 - Polynomial.X - Polynomial.C 2) ^ 3) 5) = -3 := 
by sorry

end NUMINAMATH_GPT_coeff_x5_of_expansion_l2211_221186


namespace NUMINAMATH_GPT_number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l2211_221102

-- Define the statement about the total number of 5-letter words.
theorem number_of_5_letter_words : 26^5 = 26^5 := by
  sorry

-- Define the statement about the total number of 5-letter words with all different letters.
theorem number_of_5_letter_words_with_all_different_letters : 
  26 * 25 * 24 * 23 * 22 = 26 * 25 * 24 * 23 * 22 := by
  sorry

-- Define the statement about the total number of 5-letter words with no consecutive letters being the same.
theorem number_of_5_letter_words_with_no_consecutive_repeating_letters : 
  26 * 25 * 25 * 25 * 25 = 26 * 25 * 25 * 25 * 25 := by
  sorry

end NUMINAMATH_GPT_number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l2211_221102


namespace NUMINAMATH_GPT_amy_total_spending_l2211_221156

def initial_tickets : ℕ := 33
def cost_per_ticket : ℝ := 1.50
def additional_tickets : ℕ := 21
def total_cost : ℝ := 81.00

theorem amy_total_spending :
  (initial_tickets * cost_per_ticket + additional_tickets * cost_per_ticket) = total_cost := 
sorry

end NUMINAMATH_GPT_amy_total_spending_l2211_221156


namespace NUMINAMATH_GPT_distinct_arrangements_TOOL_l2211_221132

/-- The word "TOOL" consists of four letters where "O" is repeated twice. 
Prove that the number of distinct arrangements of the letters in the word is 12. -/
theorem distinct_arrangements_TOOL : 
  let total_letters := 4
  let repeated_O := 2
  (Nat.factorial total_letters / Nat.factorial repeated_O) = 12 := 
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_TOOL_l2211_221132


namespace NUMINAMATH_GPT_quadratic_transform_l2211_221176

theorem quadratic_transform (x : ℝ) : x^2 - 6 * x - 5 = 0 → (x - 3)^2 = 14 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_transform_l2211_221176


namespace NUMINAMATH_GPT_cookies_in_jar_l2211_221187

-- Let C be the total number of cookies in the jar.
def C : ℕ := sorry

-- Conditions
def adults_eat_one_third (C : ℕ) : ℕ := C / 3
def children_get_each (C : ℕ) : ℕ := 20
def num_children : ℕ := 4

-- Proof statement
theorem cookies_in_jar (C : ℕ) (h1 : C / 3 = adults_eat_one_third C)
  (h2 : children_get_each C * num_children = 80)
  (h3 : 2 * (C / 3) = 80) :
  C = 120 :=
sorry

end NUMINAMATH_GPT_cookies_in_jar_l2211_221187


namespace NUMINAMATH_GPT_john_new_cards_l2211_221103

def cards_per_page : ℕ := 3
def old_cards : ℕ := 16
def pages_used : ℕ := 8

theorem john_new_cards : (pages_used * cards_per_page) - old_cards = 8 := by
  sorry

end NUMINAMATH_GPT_john_new_cards_l2211_221103


namespace NUMINAMATH_GPT_selling_price_of_cycle_l2211_221189

theorem selling_price_of_cycle (cp : ℝ) (loss_percentage : ℝ) (sp : ℝ) : 
  cp = 1400 → loss_percentage = 20 → sp = cp - (loss_percentage / 100) * cp → sp = 1120 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_selling_price_of_cycle_l2211_221189


namespace NUMINAMATH_GPT_values_of_m_and_n_l2211_221164

theorem values_of_m_and_n (m n : ℕ) (h_cond1 : 2 * m + 3 = 5 * n - 2) (h_cond2 : 5 * n - 2 < 15) : m = 5 ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_values_of_m_and_n_l2211_221164


namespace NUMINAMATH_GPT_tiles_needed_l2211_221195

def tile_area : ℕ := 3 * 4
def floor_area : ℕ := 36 * 60

theorem tiles_needed : floor_area / tile_area = 180 := by
  sorry

end NUMINAMATH_GPT_tiles_needed_l2211_221195


namespace NUMINAMATH_GPT_calculate_expression_l2211_221100

theorem calculate_expression :
  ((1 / 3 : ℝ) ^ (-2 : ℝ)) + Real.tan (Real.pi / 4) - Real.sqrt ((-10 : ℝ) ^ 2) = 0 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2211_221100


namespace NUMINAMATH_GPT_twice_x_plus_one_third_y_l2211_221109

theorem twice_x_plus_one_third_y (x y : ℝ) : 2 * x + (1 / 3) * y = 2 * x + (1 / 3) * y := 
by 
  sorry

end NUMINAMATH_GPT_twice_x_plus_one_third_y_l2211_221109


namespace NUMINAMATH_GPT_correct_operation_l2211_221124

variable (m n : ℝ)

-- Define the statement to be proved
theorem correct_operation : (-2 * m * n) ^ 2 = 4 * m ^ 2 * n ^ 2 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l2211_221124


namespace NUMINAMATH_GPT_hexagon_perimeter_l2211_221113

theorem hexagon_perimeter
  (A B C D E F : Type)  -- vertices of the hexagon
  (angle_A : ℝ) (angle_C : ℝ) (angle_E : ℝ)  -- nonadjacent angles
  (angle_B : ℝ) (angle_D : ℝ) (angle_F : ℝ)  -- adjacent angles
  (area_hexagon : ℝ)
  (side_length : ℝ)
  (h1 : angle_A = 120) (h2 : angle_C = 120) (h3 : angle_E = 120)
  (h4 : angle_B = 60) (h5 : angle_D = 60) (h6 : angle_F = 60)
  (h7 : area_hexagon = 24)
  (h8 : ∃ s, ∀ (u v : Type), side_length = s) :
  6 * side_length = 24 / (Real.sqrt 3 ^ (1/4)) :=
by
  sorry

end NUMINAMATH_GPT_hexagon_perimeter_l2211_221113


namespace NUMINAMATH_GPT_find_A_minus_C_l2211_221197

theorem find_A_minus_C (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 :=
by
  sorry

end NUMINAMATH_GPT_find_A_minus_C_l2211_221197


namespace NUMINAMATH_GPT_problem_solution_l2211_221115

theorem problem_solution (a b : ℝ) (h : (a + 1)^2 + |b - 2| = 0) : a + b = 1 :=
sorry

end NUMINAMATH_GPT_problem_solution_l2211_221115


namespace NUMINAMATH_GPT_total_wolves_l2211_221177

theorem total_wolves (x y : ℕ) :
  (x + 2 * y = 20) →
  (4 * x + 3 * y = 55) →
  (x + y = 15) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_total_wolves_l2211_221177


namespace NUMINAMATH_GPT_gcd_204_85_l2211_221129

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_204_85_l2211_221129


namespace NUMINAMATH_GPT_find_c_l2211_221180

noncomputable def f (c x : ℝ) : ℝ :=
  c * x^3 + 17 * x^2 - 4 * c * x + 45

theorem find_c (h : f c (-5) = 0) : c = 94 / 21 :=
by sorry

end NUMINAMATH_GPT_find_c_l2211_221180


namespace NUMINAMATH_GPT_find_original_number_l2211_221131

theorem find_original_number (x : ℝ) : ((x - 3) / 6) * 12 = 8 → x = 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_original_number_l2211_221131


namespace NUMINAMATH_GPT_technicians_count_l2211_221123

theorem technicians_count 
  (T R : ℕ) 
  (h1 : T + R = 14) 
  (h2 : 12000 * T + 6000 * R = 9000 * 14) : 
  T = 7 :=
by
  sorry

end NUMINAMATH_GPT_technicians_count_l2211_221123


namespace NUMINAMATH_GPT_jose_to_haylee_ratio_l2211_221143

variable (J : ℕ)

def haylee_guppies := 36
def charliz_guppies := J / 3
def nicolai_guppies := 4 * (J / 3)
def total_guppies := haylee_guppies + J + charliz_guppies + nicolai_guppies

theorem jose_to_haylee_ratio :
  haylee_guppies = 36 ∧ total_guppies = 84 →
  J / haylee_guppies = 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_jose_to_haylee_ratio_l2211_221143


namespace NUMINAMATH_GPT_men_in_business_class_l2211_221158

theorem men_in_business_class (total_passengers : ℕ) (percentage_men : ℝ)
  (fraction_business_class : ℝ) (num_men_in_business_class : ℕ) 
  (h1 : total_passengers = 160) 
  (h2 : percentage_men = 0.75) 
  (h3 : fraction_business_class = 1 / 4) 
  (h4 : num_men_in_business_class = total_passengers * percentage_men * fraction_business_class) : 
  num_men_in_business_class = 30 := 
  sorry

end NUMINAMATH_GPT_men_in_business_class_l2211_221158


namespace NUMINAMATH_GPT_find_max_value_l2211_221120

-- We define the conditions as Lean definitions and hypotheses
def is_distinct_digits (A B C D E F : ℕ) : Prop :=
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧
  (D ≠ E) ∧ (D ≠ F) ∧
  (E ≠ F)

def all_digits_in_range (A B C D E F : ℕ) : Prop :=
  (1 ≤ A) ∧ (A ≤ 8) ∧
  (1 ≤ B) ∧ (B ≤ 8) ∧
  (1 ≤ C) ∧ (C ≤ 8) ∧
  (1 ≤ D) ∧ (D ≤ 8) ∧
  (1 ≤ E) ∧ (E ≤ 8) ∧
  (1 ≤ F) ∧ (F ≤ 8)

def divisible_by_99 (n : ℕ) : Prop :=
  (n % 99 = 0)

theorem find_max_value (A B C D E F : ℕ) :
  is_distinct_digits A B C D E F →
  all_digits_in_range A B C D E F →
  divisible_by_99 (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F) →
  100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F = 87653412 :=
sorry

end NUMINAMATH_GPT_find_max_value_l2211_221120


namespace NUMINAMATH_GPT_cost_per_serving_is_3_62_l2211_221185

noncomputable def cost_per_serving : ℝ :=
  let beef_cost := 4 * 6
  let chicken_cost := (2.2 * 5) * 0.85
  let carrots_cost := 2 * 1.50
  let potatoes_cost := (1.5 * 1.80) * 0.85
  let onions_cost := 1 * 3
  let discounted_carrots := carrots_cost * 0.80
  let discounted_potatoes := potatoes_cost * 0.80
  let total_cost_before_tax := beef_cost + chicken_cost + discounted_carrots + discounted_potatoes + onions_cost
  let sales_tax := total_cost_before_tax * 0.07
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax / 12

theorem cost_per_serving_is_3_62 : cost_per_serving = 3.62 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_serving_is_3_62_l2211_221185


namespace NUMINAMATH_GPT_tile_coverage_fraction_l2211_221171

structure Room where
  rect_length : ℝ
  rect_width : ℝ
  tri_base : ℝ
  tri_height : ℝ
  
structure Tiles where
  square_tiles : ℕ
  triangular_tiles : ℕ
  triangle_base : ℝ
  triangle_height : ℝ
  tile_area : ℝ
  triangular_tile_area : ℝ
  
noncomputable def fractionalTileCoverage (room : Room) (tiles : Tiles) : ℝ :=
  let rect_area := room.rect_length * room.rect_width
  let tri_area := (room.tri_base * room.tri_height) / 2
  let total_room_area := rect_area + tri_area
  let total_tile_area := (tiles.square_tiles * tiles.tile_area) + (tiles.triangular_tiles * tiles.triangular_tile_area)
  total_tile_area / total_room_area

theorem tile_coverage_fraction
  (room : Room) (tiles : Tiles)
  (h1 : room.rect_length = 12)
  (h2 : room.rect_width = 20)
  (h3 : room.tri_base = 10)
  (h4 : room.tri_height = 8)
  (h5 : tiles.square_tiles = 40)
  (h6 : tiles.triangular_tiles = 4)
  (h7 : tiles.tile_area = 1)
  (h8 : tiles.triangular_tile_area = (1 * 1) / 2) :
  fractionalTileCoverage room tiles = 3 / 20 :=
by 
  sorry

end NUMINAMATH_GPT_tile_coverage_fraction_l2211_221171


namespace NUMINAMATH_GPT_ellipse_equation_l2211_221108

theorem ellipse_equation (a : ℝ) (x y : ℝ) (h : (x, y) = (-3, 2)) :
  (∃ a : ℝ, ∀ x y : ℝ, x^2 / 15 + y^2 / 10 = 1) ↔ (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 15 + p.2^2 / 10 = 1 } :=
by
  have h1 : 15 = a^2 := by
    sorry
  have h2 : 10 = a^2 - 5 := by
    sorry
  sorry

end NUMINAMATH_GPT_ellipse_equation_l2211_221108


namespace NUMINAMATH_GPT_range_of_a_l2211_221117

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) : a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2211_221117


namespace NUMINAMATH_GPT_find_abs_diff_of_average_and_variance_l2211_221165

noncomputable def absolute_difference (x y : ℝ) (a1 a2 a3 a4 a5 : ℝ) : ℝ :=
  |x - y|

theorem find_abs_diff_of_average_and_variance (x y : ℝ) (h1 : (x + y + 30 + 29 + 31) / 5 = 30)
  (h2 : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) :
  absolute_difference x y 30 30 29 31 = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_abs_diff_of_average_and_variance_l2211_221165


namespace NUMINAMATH_GPT_least_k_for_divisibility_l2211_221181

theorem least_k_for_divisibility (k : ℕ) : (k ^ 4) % 1260 = 0 ↔ k ≥ 210 :=
sorry

end NUMINAMATH_GPT_least_k_for_divisibility_l2211_221181


namespace NUMINAMATH_GPT_like_terms_implies_a_plus_2b_eq_3_l2211_221146

theorem like_terms_implies_a_plus_2b_eq_3 (a b : ℤ) (h1 : 2 * a + b = 6) (h2 : a - b = 3) : a + 2 * b = 3 :=
sorry

end NUMINAMATH_GPT_like_terms_implies_a_plus_2b_eq_3_l2211_221146


namespace NUMINAMATH_GPT_sofia_running_time_l2211_221104

theorem sofia_running_time :
  ∃ t : ℤ, t = 8 * 60 + 20 ∧ 
  (∀ (laps : ℕ) (d1 d2 v1 v2 : ℤ),
    laps = 5 →
    d1 = 200 →
    v1 = 4 →
    d2 = 300 →
    v2 = 6 →
    t = laps * ((d1 / v1 + d2 / v2))) :=
by
  sorry

end NUMINAMATH_GPT_sofia_running_time_l2211_221104


namespace NUMINAMATH_GPT_frog_reaches_vertical_side_l2211_221135

def P (x y : ℕ) : ℝ := 
  if (x = 3 ∧ y = 3) then 0 -- blocked cell
  else if (x = 0 ∨ x = 5) then 1 -- vertical boundary
  else if (y = 0 ∨ y = 5) then 0 -- horizontal boundary
  else sorry -- inner probabilities to be calculated

theorem frog_reaches_vertical_side : P 2 2 = 5 / 8 :=
by sorry

end NUMINAMATH_GPT_frog_reaches_vertical_side_l2211_221135


namespace NUMINAMATH_GPT_first_year_students_sampled_equals_40_l2211_221137

-- Defining the conditions
def num_first_year_students := 800
def num_second_year_students := 600
def num_third_year_students := 500
def num_sampled_third_year_students := 25
def total_students := num_first_year_students + num_second_year_students + num_third_year_students

-- Proving the number of first-year students sampled
theorem first_year_students_sampled_equals_40 :
  (num_first_year_students * num_sampled_third_year_students) / num_third_year_students = 40 := by
  sorry

end NUMINAMATH_GPT_first_year_students_sampled_equals_40_l2211_221137


namespace NUMINAMATH_GPT_negation_true_l2211_221144

theorem negation_true (a : ℝ) : ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) :=
sorry

end NUMINAMATH_GPT_negation_true_l2211_221144


namespace NUMINAMATH_GPT_correct_option_D_l2211_221167

theorem correct_option_D (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2 * x :=
by sorry

end NUMINAMATH_GPT_correct_option_D_l2211_221167


namespace NUMINAMATH_GPT_chewbacca_gum_packs_l2211_221128

theorem chewbacca_gum_packs (x : ℕ) :
  (30 - 2 * x) * (40 + 4 * x) = 1200 → x = 5 :=
by
  -- This is where the proof would go. We'll leave it as sorry for now.
  sorry

end NUMINAMATH_GPT_chewbacca_gum_packs_l2211_221128


namespace NUMINAMATH_GPT_reflect_y_axis_correct_l2211_221166

-- Define the initial coordinates of the point M
def M_orig : ℝ × ℝ := (3, 2)

-- Define the reflection function across the y-axis
def reflect_y_axis (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, M.2)

-- Prove that reflecting M_orig across the y-axis results in the coordinates (-3, 2)
theorem reflect_y_axis_correct : reflect_y_axis M_orig = (-3, 2) :=
  by
    -- Provide the missing steps of the proof
    sorry

end NUMINAMATH_GPT_reflect_y_axis_correct_l2211_221166


namespace NUMINAMATH_GPT_quick_calc_formula_l2211_221193

variables (a b A B C : ℤ)

theorem quick_calc_formula (h1 : (100 - a) * (100 - b) = (A + B - 100) * 100 + C)
                           (h2 : (100 + a) * (100 + b) = (A + B - 100) * 100 + C) :
  A = 100 ∨ A = 100 ∧ B = 100 ∨ B = 100 ∧ C = a * b :=
sorry

end NUMINAMATH_GPT_quick_calc_formula_l2211_221193


namespace NUMINAMATH_GPT_sequence_properties_l2211_221162

-- Define the arithmetic-geometric sequence and its sum
def a_n (n : ℕ) : ℕ := 2^(n-1)
def S_n (n : ℕ) : ℕ := 2^n - 1
def T_n (n : ℕ) : ℕ := 2^(n+1) - n - 2

theorem sequence_properties : 
(S_n 3 = 7) ∧ (S_n 6 = 63) → 
(∀ n: ℕ, a_n n = 2^(n-1)) ∧ 
(∀ n: ℕ, S_n n = 2^n - 1) ∧ 
(∀ n: ℕ, T_n n = 2^(n+1) - n - 2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_properties_l2211_221162


namespace NUMINAMATH_GPT_isosceles_triangle_properties_l2211_221116

noncomputable def isosceles_triangle_sides (a : ℝ) : ℝ × ℝ × ℝ :=
  let x := a * Real.sqrt 3
  let y := 2 * x / 3
  let z := (x + y) / 2
  (x, z, z)

theorem isosceles_triangle_properties (a x y z : ℝ) 
  (h1 : x * y = 2 * a ^ 2) 
  (h2 : x + y = 2 * z) 
  (h3 : y ^ 2 + (x / 2) ^ 2 = z ^ 2) : 
  x = a * Real.sqrt 3 ∧ 
  z = 5 * a * Real.sqrt 3 / 6 :=
by
-- Proof goes here
sorry

end NUMINAMATH_GPT_isosceles_triangle_properties_l2211_221116


namespace NUMINAMATH_GPT_prob_same_gender_eq_two_fifths_l2211_221196

-- Define the number of male and female students
def num_male_students : ℕ := 3
def num_female_students : ℕ := 2

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability calculation
def probability_same_gender := (num_male_students * (num_male_students - 1) / 2 + num_female_students * (num_female_students - 1) / 2) / (total_students * (total_students - 1) / 2)

theorem prob_same_gender_eq_two_fifths :
  probability_same_gender = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_prob_same_gender_eq_two_fifths_l2211_221196


namespace NUMINAMATH_GPT_factorize_expression_l2211_221150

theorem factorize_expression (a b : ℝ) : 3 * a ^ 2 - 3 * b ^ 2 = 3 * (a + b) * (a - b) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2211_221150


namespace NUMINAMATH_GPT_at_least_one_no_less_than_two_l2211_221101

variable (a b c : ℝ)
variable (ha : 0 < a)
variable (hb : 0 < b)
variable (hc : 0 < c)

theorem at_least_one_no_less_than_two :
  ∃ x ∈ ({a + 1/b, b + 1/c, c + 1/a} : Set ℝ), 2 ≤ x := by
  sorry

end NUMINAMATH_GPT_at_least_one_no_less_than_two_l2211_221101


namespace NUMINAMATH_GPT_find_pairs_l2211_221107

noncomputable def possibleValues (α β : ℝ) : Prop :=
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = -(Real.pi/3) + 2*l*Real.pi) ∨
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = (Real.pi/3) + 2*l*Real.pi)

theorem find_pairs (α β : ℝ) (h1 : Real.sin (α - β) = Real.sin α - Real.sin β)
  (h2 : Real.cos (α - β) = Real.cos α - Real.cos β) :
  possibleValues α β :=
sorry

end NUMINAMATH_GPT_find_pairs_l2211_221107


namespace NUMINAMATH_GPT_division_multiplication_result_l2211_221119

theorem division_multiplication_result : (180 / 6) * 3 = 90 := by
  sorry

end NUMINAMATH_GPT_division_multiplication_result_l2211_221119


namespace NUMINAMATH_GPT_div_poly_odd_power_l2211_221179

theorem div_poly_odd_power (a b : ℤ) (n : ℕ) : (a + b) ∣ (a^(2*n+1) + b^(2*n+1)) :=
sorry

end NUMINAMATH_GPT_div_poly_odd_power_l2211_221179


namespace NUMINAMATH_GPT_blueberry_pies_count_l2211_221147

-- Definitions and conditions
def total_pies := 30
def ratio_parts := 10
def pies_per_part := total_pies / ratio_parts
def blueberry_ratio := 3

-- Problem statement
theorem blueberry_pies_count :
  blueberry_ratio * pies_per_part = 9 := by
  -- The solution step that leads to the proof
  sorry

end NUMINAMATH_GPT_blueberry_pies_count_l2211_221147


namespace NUMINAMATH_GPT_circle_tangent_to_ellipse_l2211_221134

theorem circle_tangent_to_ellipse {r : ℝ} 
  (h1: ∀ p: ℝ × ℝ, p ≠ (0, 0) → ((p.1 - r)^2 + p.2^2 = r^2 → p.1^2 + 4 * p.2^2 = 8))
  (h2: ∃ p: ℝ × ℝ, p ≠ (0, 0) ∧ ((p.1 - r)^2 + p.2^2 = r^2 ∧ p.1^2 + 4 * p.2^2 = 8)):
  r = Real.sqrt (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_tangent_to_ellipse_l2211_221134


namespace NUMINAMATH_GPT_num_men_scenario1_is_15_l2211_221159

-- Definitions based on the conditions
def hours_per_day_scenario1 : ℕ := 9
def days_scenario1 : ℕ := 16
def men_scenario2 : ℕ := 18
def hours_per_day_scenario2 : ℕ := 8
def days_scenario2 : ℕ := 15
def total_work_done : ℕ := men_scenario2 * hours_per_day_scenario2 * days_scenario2

-- Definition of the number of men M in the first scenario
noncomputable def men_scenario1 : ℕ := total_work_done / (hours_per_day_scenario1 * days_scenario1)

-- Statement of desired proof: prove that the number of men in the first scenario is 15
theorem num_men_scenario1_is_15 :
  men_scenario1 = 15 := by
  sorry

end NUMINAMATH_GPT_num_men_scenario1_is_15_l2211_221159


namespace NUMINAMATH_GPT_cuboid_edge_length_l2211_221169

theorem cuboid_edge_length (x : ℝ) (h1 : (2 * (x * 5 + x * 6 + 5 * 6)) = 148) : x = 4 :=
by 
  sorry

end NUMINAMATH_GPT_cuboid_edge_length_l2211_221169


namespace NUMINAMATH_GPT_grace_age_is_60_l2211_221188

def Grace : ℕ := 60
def motherAge : ℕ := 80
def grandmotherAge : ℕ := 2 * motherAge
def graceAge : ℕ := (3 / 8) * grandmotherAge

theorem grace_age_is_60 : graceAge = Grace := by
  sorry

end NUMINAMATH_GPT_grace_age_is_60_l2211_221188


namespace NUMINAMATH_GPT_vector_addition_l2211_221140

-- Let vectors a and b be defined as
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -3)

-- Theorem statement to prove
theorem vector_addition : a + 2 • b = (4, -5) :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_l2211_221140


namespace NUMINAMATH_GPT_smallest_four_consecutive_numbers_l2211_221111

theorem smallest_four_consecutive_numbers (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 4574880) : n = 43 :=
sorry

end NUMINAMATH_GPT_smallest_four_consecutive_numbers_l2211_221111


namespace NUMINAMATH_GPT_find_a8_l2211_221194

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^(n-1)

noncomputable def sum_geom (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

theorem find_a8 (a_1 q a_2 a_5 a_8 : ℝ) (S : ℕ → ℝ) 
  (Hsum : ∀ n, S n = sum_geom a_1 q n)
  (H1 : 2 * S 9 = S 3 + S 6)
  (H2 : a_2 = geometric_sequence a_1 q 2)
  (H3 : a_5 = geometric_sequence a_1 q 5)
  (H4 : a_2 + a_5 = 4)
  (H5 : a_8 = geometric_sequence a_1 q 8) :
  a_8 = 2 :=
sorry

end NUMINAMATH_GPT_find_a8_l2211_221194


namespace NUMINAMATH_GPT_false_implies_exists_nonpositive_l2211_221198

variable (f : ℝ → ℝ)

theorem false_implies_exists_nonpositive (h : ¬ ∀ x > 0, f x > 0) : ∃ x > 0, f x ≤ 0 :=
by sorry

end NUMINAMATH_GPT_false_implies_exists_nonpositive_l2211_221198
