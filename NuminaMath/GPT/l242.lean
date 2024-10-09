import Mathlib

namespace triangle_inequality_l242_24254

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a + b - c) * (a - b + c) * (-a + b + c) ≤ a * b * c := 
sorry

end triangle_inequality_l242_24254


namespace system1_solution_l242_24269

theorem system1_solution (x y : ℝ) (h₁ : x = 2 * y) (h₂ : 3 * x - 2 * y = 8) : x = 4 ∧ y = 2 := 
by admit

end system1_solution_l242_24269


namespace solution_set_f_x_le_5_l242_24210

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 3 + Real.log x / Real.log 2 else x^2 - x - 1

theorem solution_set_f_x_le_5 : {x : ℝ | f x ≤ 5} = Set.Icc (-2 : ℝ) 4 := by
  sorry

end solution_set_f_x_le_5_l242_24210


namespace find_square_side_length_l242_24264

/-- Define the side lengths of the rectangle and the square --/
def rectangle_side_lengths (k : ℕ) (n : ℕ) : Prop := 
  k ≥ 7 ∧ n = 12 ∧ k * (k - 7) = n * n

theorem find_square_side_length (k n : ℕ) : rectangle_side_lengths k n → n = 12 :=
by
  intros
  sorry

end find_square_side_length_l242_24264


namespace maximum_value_N_27_l242_24281

variable (N : Nat)
variable (long_ears : Nat)
variable (jump_far : Nat)
variable (both_traits : Nat)

theorem maximum_value_N_27 (hN : N = 27) 
  (h_long_ears : long_ears = 13) 
  (h_jump_far : jump_far = 17) 
  (h_both_traits : both_traits >= 3) : 
    N <= 27 := 
sorry

end maximum_value_N_27_l242_24281


namespace melanie_has_4_plums_l242_24289

theorem melanie_has_4_plums (initial_plums : ℕ) (given_plums : ℕ) :
  initial_plums = 7 ∧ given_plums = 3 → initial_plums - given_plums = 4 :=
by
  sorry

end melanie_has_4_plums_l242_24289


namespace union_A_B_equals_C_l242_24212

-- Define Set A
def A : Set ℝ := {x : ℝ | 3 - 2 * x > 0}

-- Define Set B
def B : Set ℝ := {x : ℝ | x^2 ≤ 4}

-- Define the target set C which is supposed to be A ∪ B
def C : Set ℝ := {x : ℝ | x ≤ 2}

theorem union_A_B_equals_C : A ∪ B = C := by 
  -- Proof is omitted here
  sorry

end union_A_B_equals_C_l242_24212


namespace percentage_of_50_of_125_l242_24272

theorem percentage_of_50_of_125 : (50 / 125) * 100 = 40 :=
by
  sorry

end percentage_of_50_of_125_l242_24272


namespace ages_of_sons_l242_24209

variable (x y z : ℕ)

def father_age_current : ℕ := 33

def youngest_age_current : ℕ := 2

def father_age_in_12_years : ℕ := father_age_current + 12

def sum_of_ages_in_12_years : ℕ := youngest_age_current + 12 + y + 12 + z + 12

theorem ages_of_sons (x y z : ℕ) 
  (h1 : x = 2)
  (h2 : father_age_current = 33)
  (h3 : father_age_in_12_years = 45)
  (h4 : sum_of_ages_in_12_years = 45) :
  x = 2 ∧ y + z = 7 ∧ ((y = 3 ∧ z = 4) ∨ (y = 4 ∧ z = 3)) :=
by
  sorry

end ages_of_sons_l242_24209


namespace max_intersections_three_circles_two_lines_l242_24287

noncomputable def max_intersections_3_circles_2_lines : ℕ :=
  3 * 2 * 1 + 2 * 3 * 2 + 1

theorem max_intersections_three_circles_two_lines :
  max_intersections_3_circles_2_lines = 19 :=
by
  sorry

end max_intersections_three_circles_two_lines_l242_24287


namespace triangle_angle_contradiction_l242_24242

theorem triangle_angle_contradiction (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) (h4 : A + B + C = 180) :
  A > 60 → B > 60 → C > 60 → false :=
by
  sorry

end triangle_angle_contradiction_l242_24242


namespace calculate_expression_l242_24262

theorem calculate_expression (x : ℤ) (h : x^2 = 1681) : (x + 3) * (x - 3) = 1672 :=
by
  sorry

end calculate_expression_l242_24262


namespace find_k_value_l242_24294

theorem find_k_value (k : ℝ) : (∃ k, ∀ x y, y = k * x + 3 ∧ (x, y) = (1, 2)) → k = -1 :=
by
  sorry

end find_k_value_l242_24294


namespace min_value_gx2_plus_fx_l242_24200

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_gx2_plus_fx (a b c : ℝ) (h_a : a ≠ 0)
    (h_min_fx_gx : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ -6) :
    ∃ x : ℝ, (g a c x)^2 + f a b x = 11/2 := sorry

end min_value_gx2_plus_fx_l242_24200


namespace total_distance_journey_l242_24222

theorem total_distance_journey :
  let south := 40
  let east := south + 20
  let north := 2 * east
  (south + east + north) = 220 :=
by
  sorry

end total_distance_journey_l242_24222


namespace two_pow_a_add_three_pow_b_eq_square_l242_24238

theorem two_pow_a_add_three_pow_b_eq_square (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) 
(h : 2 ^ a + 3 ^ b = n ^ 2) : (a = 4 ∧ b = 2) :=
sorry

end two_pow_a_add_three_pow_b_eq_square_l242_24238


namespace probability_factor_24_l242_24292

theorem probability_factor_24 : 
  (∃ (k : ℚ), k = 1 / 3 ∧ 
  ∀ (n : ℕ), n ≤ 24 ∧ n > 0 → 
  (∃ (m : ℕ), 24 = m * n)) := sorry

end probability_factor_24_l242_24292


namespace determine_m_value_l242_24276

theorem determine_m_value 
  (a b m : ℝ)
  (h1 : 2^a = m)
  (h2 : 5^b = m)
  (h3 : 1/a + 1/b = 2) : 
  m = Real.sqrt 10 := 
sorry

end determine_m_value_l242_24276


namespace positive_function_characterization_l242_24252

theorem positive_function_characterization (f : ℝ → ℝ) (h₁ : ∀ x, x > 0 → f x > 0) (h₂ : ∀ a b : ℝ, a > 0 → b > 0 → a * b ≤ 0.5 * (a * f a + b * (f b)⁻¹)) :
  ∃ C > 0, ∀ x > 0, f x = C * x :=
sorry

end positive_function_characterization_l242_24252


namespace find_other_number_l242_24273

theorem find_other_number (HCF LCM num1 num2 : ℕ) (h1 : HCF = 16) (h2 : LCM = 396) (h3 : num1 = 36) (h4 : HCF * LCM = num1 * num2) : num2 = 176 :=
sorry

end find_other_number_l242_24273


namespace product_remainder_mod_7_l242_24247

theorem product_remainder_mod_7 {a b c : ℕ}
    (h1 : a % 7 = 2)
    (h2 : b % 7 = 3)
    (h3 : c % 7 = 5)
    : (a * b * c) % 7 = 2 :=
by
    sorry

end product_remainder_mod_7_l242_24247


namespace arith_seq_general_formula_l242_24239

noncomputable def increasing_arith_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

theorem arith_seq_general_formula (a : ℕ → ℤ) (d : ℤ)
  (h_arith : increasing_arith_sequence a)
  (h_a1 : a 1 = 1)
  (h_a3 : a 3 = (a 2)^2 - 4) :
  ∀ n, a n = 3 * n - 2 :=
sorry

end arith_seq_general_formula_l242_24239


namespace arithmetic_series_sum_l242_24231

def a := 5
def l := 20
def n := 16
def S := (n / 2) * (a + l)

theorem arithmetic_series_sum :
  S = 200 :=
by
  sorry

end arithmetic_series_sum_l242_24231


namespace max_xy_min_x2y2_l242_24298

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x * y ≤ 1 / 8) :=
sorry

theorem min_x2y2 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x ^ 2 + y ^ 2 ≥ 1 / 5) :=
sorry


end max_xy_min_x2y2_l242_24298


namespace sausages_placement_and_path_length_l242_24241

variables {a b x y : ℝ} (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
variables (h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y)

theorem sausages_placement_and_path_length (h1 : 2 * x = y) (h2 : b = 1 / (x + y))
(h3 : (x + y) = 8 * a) (h4 : x = 1.4 * y) : 
  x < y ∧ (x / y) = 1.4 :=
by {
  sorry
}

end sausages_placement_and_path_length_l242_24241


namespace rows_count_mod_pascals_triangle_l242_24253

-- Define the modified Pascal's triangle function that counts the required rows.
def modified_pascals_triangle_satisfying_rows (n : ℕ) : ℕ := sorry

-- Statement of the problem
theorem rows_count_mod_pascals_triangle :
  modified_pascals_triangle_satisfying_rows 30 = 4 :=
sorry

end rows_count_mod_pascals_triangle_l242_24253


namespace sqrt_neg4_squared_l242_24250

theorem sqrt_neg4_squared : Real.sqrt ((-4 : ℝ) ^ 2) = 4 := 
by 
-- add proof here
sorry

end sqrt_neg4_squared_l242_24250


namespace strawberries_count_l242_24202

def strawberries_total (J M Z : ℕ) : ℕ :=
  J + M + Z

theorem strawberries_count (J M Z : ℕ) (h1 : J + M = 350) (h2 : M + Z = 250) (h3 : Z = 200) : 
  strawberries_total J M Z = 550 :=
by
  sorry

end strawberries_count_l242_24202


namespace largest_divisible_n_l242_24267

theorem largest_divisible_n (n : ℕ) :
  (n^3 + 2006) % (n + 26) = 0 → n = 15544 :=
sorry

end largest_divisible_n_l242_24267


namespace cost_of_pencils_l242_24206

open Nat

theorem cost_of_pencils (P : ℕ) : 
  (H : 20 * P + 80 * 3 = 360) → 
  P = 6 :=
by 
  sorry

end cost_of_pencils_l242_24206


namespace total_students_l242_24274

theorem total_students (T : ℕ)
  (A_cond : (2/9 : ℚ) * T = (a_real : ℚ))
  (B_cond : (1/3 : ℚ) * T = (b_real : ℚ))
  (C_cond : (2/9 : ℚ) * T = (c_real : ℚ))
  (D_cond : (1/9 : ℚ) * T = (d_real : ℚ))
  (E_cond : 15 = e_real) :
  (2/9 : ℚ) * T + (1/3 : ℚ) * T + (2/9 : ℚ) * T + (1/9 : ℚ) * T + 15 = T → T = 135 :=
by
  sorry

end total_students_l242_24274


namespace hacker_cannot_change_grades_l242_24218

theorem hacker_cannot_change_grades :
  ¬ ∃ n1 n2 n3 n4 : ℤ,
    2 * n1 + n2 - 2 * n3 - n4 = 27 ∧
    -n1 + 2 * n2 + n3 - 2 * n4 = -27 := by
  sorry

end hacker_cannot_change_grades_l242_24218


namespace seventyFifthTermInSequence_l242_24207

/-- Given a sequence that starts at 2 and increases by 4 each term, 
prove that the 75th term in this sequence is 298. -/
theorem seventyFifthTermInSequence : 
  (∃ a : ℕ → ℤ, (∀ n : ℕ, a n = 2 + 4 * n) ∧ a 74 = 298) :=
by
  sorry

end seventyFifthTermInSequence_l242_24207


namespace simplify_sqrt_expression_l242_24203

theorem simplify_sqrt_expression :
  (Real.sqrt (3 * 5) * Real.sqrt (3^3 * 5^3)) = 225 := 
by 
  sorry

end simplify_sqrt_expression_l242_24203


namespace annual_growth_rate_equation_l242_24208

theorem annual_growth_rate_equation
  (initial_capital : ℝ)
  (final_capital : ℝ)
  (n : ℕ)
  (x : ℝ)
  (h1 : initial_capital = 10)
  (h2 : final_capital = 14.4)
  (h3 : n = 2) :
  1000 * (1 + x)^2 = 1440 :=
by
  sorry

end annual_growth_rate_equation_l242_24208


namespace find_k_value_l242_24233

theorem find_k_value (k : ℝ) (h : (7 * (-1)^3 - 3 * (-1)^2 + k * -1 + 5 = 0)) :
  k^3 + 2 * k^2 - 11 * k - 85 = -105 :=
by {
  sorry
}

end find_k_value_l242_24233


namespace max_marks_mike_l242_24224

theorem max_marks_mike (pass_percentage : ℝ) (scored_marks : ℝ) (shortfall : ℝ) : 
  pass_percentage = 0.30 → 
  scored_marks = 212 → 
  shortfall = 28 → 
  (scored_marks + shortfall) = 240 → 
  (scored_marks + shortfall) = pass_percentage * (max_marks : ℝ) → 
  max_marks = 800 := 
by 
  intros hp hs hsh hps heq 
  sorry

end max_marks_mike_l242_24224


namespace martin_and_martina_ages_l242_24296

-- Conditions
def martin_statement (x y : ℕ) : Prop := x = 3 * (2 * y - x)
def martina_statement (x y : ℕ) : Prop := 3 * x - y = 77

-- Proof problem
theorem martin_and_martina_ages :
  ∃ (x y : ℕ), martin_statement x y ∧ martina_statement x y ∧ x = 33 ∧ y = 22 :=
by {
  -- No proof required, just the statement
  sorry
}

end martin_and_martina_ages_l242_24296


namespace matrix_multiplication_comm_l242_24243

theorem matrix_multiplication_comm {C D : Matrix (Fin 2) (Fin 2) ℝ}
    (h₁ : C + D = C * D)
    (h₂ : C * D = !![5, 1; -2, 4]) :
    (D * C = !![5, 1; -2, 4]) :=
by
  sorry

end matrix_multiplication_comm_l242_24243


namespace percentage_short_l242_24211

def cost_of_goldfish : ℝ := 0.25
def sale_price_of_goldfish : ℝ := 0.75
def tank_price : ℝ := 100
def goldfish_sold : ℕ := 110

theorem percentage_short : ((tank_price - (sale_price_of_goldfish - cost_of_goldfish) * goldfish_sold) / tank_price) * 100 = 45 := 
by
  sorry

end percentage_short_l242_24211


namespace tangent_parallel_to_given_line_l242_24214

theorem tangent_parallel_to_given_line (a : ℝ) : 
  let y := λ x : ℝ => x^2 + a / x
  let y' := λ x : ℝ => (deriv y) x
  y' 1 = 2 
  → a = 0 := by
  -- y'(1) is the derivative of y at x=1
  sorry

end tangent_parallel_to_given_line_l242_24214


namespace range_of_a_l242_24205

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → (f a x) * (f a (1 - x)) ≥ 1) ↔ (1 ≤ a) ∨ (a ≤ - (1/4)) := 
by
  sorry

end range_of_a_l242_24205


namespace remainder_product_mod_5_l242_24258

theorem remainder_product_mod_5 
  (a b c : ℕ) 
  (ha : a % 5 = 1) 
  (hb : b % 5 = 2) 
  (hc : c % 5 = 3) : 
  (a * b * c) % 5 = 1 :=
by
  sorry

end remainder_product_mod_5_l242_24258


namespace find_number_l242_24291

-- Define the condition
def exceeds_by_30 (x : ℝ) : Prop :=
  x = (3/8) * x + 30

-- Prove the main statement
theorem find_number : ∃ x : ℝ, exceeds_by_30 x ∧ x = 48 := by
  sorry

end find_number_l242_24291


namespace greatest_possible_NPMPP_l242_24249

theorem greatest_possible_NPMPP :
  ∃ (M N P PP : ℕ),
    0 ≤ M ∧ M ≤ 9 ∧
    M^2 % 10 = M ∧
    NPMPP = M * (1111 * M) ∧
    NPMPP = 89991 := by
  sorry

end greatest_possible_NPMPP_l242_24249


namespace a_put_his_oxen_for_grazing_for_7_months_l242_24219

theorem a_put_his_oxen_for_grazing_for_7_months
  (x : ℕ)
  (a_oxen : ℕ := 10)
  (b_oxen : ℕ := 12)
  (b_months : ℕ := 5)
  (c_oxen : ℕ := 15)
  (c_months : ℕ := 3)
  (total_rent : ℝ := 105)
  (c_share : ℝ := 27) :
  (c_share / total_rent = (c_oxen * c_months) / ((a_oxen * x) + (b_oxen * b_months) + (c_oxen * c_months))) → (x = 7) :=
by
  sorry

end a_put_his_oxen_for_grazing_for_7_months_l242_24219


namespace largest_of_eight_consecutive_l242_24299

theorem largest_of_eight_consecutive (n : ℕ) (h : 8 * n + 28 = 2024) : n + 7 = 256 := by
  -- This means you need to solve for n first, then add 7 to get the largest number
  sorry

end largest_of_eight_consecutive_l242_24299


namespace fill_tank_with_leak_l242_24227

theorem fill_tank_with_leak (R L T: ℝ)
(h1: R = 1 / 7) (h2: L = 1 / 56) (h3: R - L = 1 / T) : T = 8 := by
  sorry

end fill_tank_with_leak_l242_24227


namespace max_distance_to_pole_l242_24230

noncomputable def max_distance_to_origin (r1 r2 : ℝ) (c1 c2 : ℝ) : ℝ :=
  r1 + r2

theorem max_distance_to_pole (r : ℝ) (c : ℝ) : max_distance_to_origin 2 1 0 0 = 3 := by
  sorry

end max_distance_to_pole_l242_24230


namespace xyz_div_by_27_l242_24275

theorem xyz_div_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  27 ∣ (x + y + z) :=
sorry

end xyz_div_by_27_l242_24275


namespace acute_angled_triangle_count_l242_24217

def num_vertices := 8

def total_triangles := Nat.choose num_vertices 3

def right_angled_triangles := 8 * 6

def acute_angled_triangles := total_triangles - right_angled_triangles

theorem acute_angled_triangle_count : acute_angled_triangles = 8 :=
by
  sorry

end acute_angled_triangle_count_l242_24217


namespace base_digits_equality_l242_24293

theorem base_digits_equality (b : ℕ) (h_condition : b^5 ≤ 200 ∧ 200 < b^6) : b = 2 := 
by {
  sorry -- proof not required as per the instructions
}

end base_digits_equality_l242_24293


namespace average_age_after_swap_l242_24282

theorem average_age_after_swap :
  let initial_average_age := 28
  let num_people_initial := 8
  let person_leaving_age := 20
  let person_entering_age := 25
  let initial_total_age := initial_average_age * num_people_initial
  let total_age_after_leaving := initial_total_age - person_leaving_age
  let total_age_final := total_age_after_leaving + person_entering_age
  let num_people_final := 8
  initial_average_age / num_people_initial = 28 ->
  total_age_final / num_people_final = 28.625 :=
by
  intros
  sorry

end average_age_after_swap_l242_24282


namespace pat_interest_rate_l242_24234

noncomputable def interest_rate (t : ℝ) : ℝ := 70 / t

theorem pat_interest_rate (r : ℝ) (t : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (years : ℝ) : 
  initial_amount * 2^((years / t)) = final_amount ∧ 
  years = 18 ∧ 
  final_amount = 28000 ∧ 
  initial_amount = 7000 →    
  r = interest_rate 9 := 
by
  sorry

end pat_interest_rate_l242_24234


namespace inv_g_inv_5_l242_24204

noncomputable def g (x : ℝ) : ℝ := 25 / (2 + 5 * x)
noncomputable def g_inv (y : ℝ) : ℝ := (15 - 10) / 25  -- g^{-1}(5) as shown in the derivation above

theorem inv_g_inv_5 : (g_inv 5)⁻¹ = 5 / 3 := by
  have h_g_inv_5 : g_inv 5 = 3 / 5 := by sorry
  rw [h_g_inv_5]
  exact inv_div 3 5

end inv_g_inv_5_l242_24204


namespace line_equation_final_equation_l242_24285

theorem line_equation (k : ℝ) : 
  (∀ x y, y = k * (x - 1) + 1 ↔ 
  ∀ x y, y = k * ((x + 2) - 1) + 1 - 1) → 
  k = 1 / 2 :=
by
  sorry

theorem final_equation : 
  ∃ k : ℝ, k = 1 / 2 ∧ (∀ x y, y = k * (x - 1) + 1) → 
  ∀ x y, x - 2 * y + 1 = 0 :=
by
  sorry

end line_equation_final_equation_l242_24285


namespace mean_difference_l242_24277

variable (a1 a2 a3 a4 a5 a6 A : ℝ)

-- Arithmetic mean of six numbers is A
axiom mean_six_numbers : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

-- Arithmetic mean of the first four numbers is A + 10
axiom mean_first_four : (a1 + a2 + a3 + a4) / 4 = A + 10

-- Arithmetic mean of the last four numbers is A - 7
axiom mean_last_four : (a3 + a4 + a5 + a6) / 4 = A - 7

-- Prove the arithmetic mean of the first, second, fifth, and sixth numbers differs from A by 3
theorem mean_difference :
  (a1 + a2 + a5 + a6) / 4 = A - 3 := 
sorry

end mean_difference_l242_24277


namespace evaluate_F_2_f_3_l242_24266

def f (a : ℤ) : ℤ := a^2 - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 341 := by
  sorry

end evaluate_F_2_f_3_l242_24266


namespace equal_candy_distribution_l242_24256

theorem equal_candy_distribution :
  ∀ (candies friends : ℕ), candies = 30 → friends = 4 → candies % friends = 2 :=
by
  sorry

end equal_candy_distribution_l242_24256


namespace draw_balls_equiv_l242_24278

noncomputable def number_of_ways_to_draw_balls (n : ℕ) (k : ℕ) (ball1 : ℕ) (ball2 : ℕ) : ℕ :=
  if n = 15 ∧ k = 4 ∧ ball1 = 1 ∧ ball2 = 15 then
    4 * (Nat.choose 14 3 * Nat.factorial 3) * 2
  else
    0

theorem draw_balls_equiv : number_of_ways_to_draw_balls 15 4 1 15 = 17472 :=
by
  dsimp [number_of_ways_to_draw_balls]
  rw [Nat.choose, Nat.factorial]
  norm_num
  sorry

end draw_balls_equiv_l242_24278


namespace max_m_sufficient_min_m_necessary_l242_24235

-- Define variables and conditions
variables (x m : ℝ) (p : Prop := abs x ≤ m) (q : Prop := -1 ≤ x ∧ x ≤ 4) 

-- Problem 1: Maximum value of m for sufficient condition
theorem max_m_sufficient : (∀ x, abs x ≤ m → (-1 ≤ x ∧ x ≤ 4)) → m = 4 := sorry

-- Problem 2: Minimum value of m for necessary condition
theorem min_m_necessary : (∀ x, (-1 ≤ x ∧ x ≤ 4) → abs x ≤ m) → m = 4 := sorry

end max_m_sufficient_min_m_necessary_l242_24235


namespace min_possible_frac_l242_24261

theorem min_possible_frac (x A C : ℝ) (hx : x ≠ 0) (hC_pos : 0 < C) (hA_pos : 0 < A)
  (h1 : x^2 + (1/x)^2 = A)
  (h2 : x - 1/x = C)
  (hC : C = Real.sqrt 3):
  A / C = (5 * Real.sqrt 3) / 3 := by
  sorry

end min_possible_frac_l242_24261


namespace most_accurate_reading_l242_24263

def temperature_reading (temp: ℝ) : Prop := 
  98.6 ≤ temp ∧ temp ≤ 99.1 ∧ temp ≠ 98.85 ∧ temp > 98.85

theorem most_accurate_reading (temp: ℝ) : temperature_reading temp → temp = 99.1 :=
by
  intros h
  sorry 

end most_accurate_reading_l242_24263


namespace factorial_fraction_eq_seven_l242_24259

theorem factorial_fraction_eq_seven : 
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / (Nat.factorial 8) = 7 := 
by 
  sorry

end factorial_fraction_eq_seven_l242_24259


namespace circle_area_percentage_increase_l242_24215

theorem circle_area_percentage_increase (r : ℝ) (h : r > 0) :
  let original_area := (Real.pi * r^2)
  let new_radius := (2.5 * r)
  let new_area := (Real.pi * new_radius^2)
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 525 := by
  let original_area := Real.pi * r^2
  let new_radius := 2.5 * r
  let new_area := Real.pi * new_radius^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  sorry

end circle_area_percentage_increase_l242_24215


namespace files_remaining_l242_24286

theorem files_remaining 
(h_music_files : ℕ := 16) 
(h_video_files : ℕ := 48) 
(h_files_deleted : ℕ := 30) :
(h_music_files + h_video_files - h_files_deleted = 34) := 
by sorry

end files_remaining_l242_24286


namespace proof_tan_alpha_proof_exp_l242_24283

-- Given conditions
variables (α : ℝ) (h_condition1 : Real.tan (α + Real.pi / 4) = - 1 / 2) (h_condition2 : Real.pi / 2 < α ∧ α < Real.pi)

-- To prove
theorem proof_tan_alpha :
  Real.tan α = -3 :=
sorry -- proof goes here

theorem proof_exp :
  (Real.sin (2 * α) - 2 * Real.cos α ^ 2) / Real.sin (α - Real.pi / 4) = - 2 * Real.sqrt 5 / 5 :=
sorry -- proof goes here

end proof_tan_alpha_proof_exp_l242_24283


namespace original_deck_card_count_l242_24288

theorem original_deck_card_count (r b u : ℕ)
  (h1 : r / (r + b + u) = 1 / 5)
  (h2 : r / (r + b + u + 3) = 1 / 6) :
  r + b + u = 15 := by
  sorry

end original_deck_card_count_l242_24288


namespace largest_of_three_consecutive_odds_l242_24279

theorem largest_of_three_consecutive_odds (n : ℤ) (h_sum : n + (n + 2) + (n + 4) = -147) : n + 4 = -47 :=
by {
  -- Proof steps here, but we're skipping for this exercise
  sorry
}

end largest_of_three_consecutive_odds_l242_24279


namespace number_of_sequences_l242_24251

-- Define the number of targets and their columns
def targetSequence := ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']

-- Define our problem statement
theorem number_of_sequences :
  (List.permutations targetSequence).length = 4200 := by
  sorry

end number_of_sequences_l242_24251


namespace total_fruits_sum_l242_24248

theorem total_fruits_sum (Mike_oranges Matt_apples Mark_bananas Mary_grapes : ℕ)
  (hMike : Mike_oranges = 3)
  (hMatt : Matt_apples = 2 * Mike_oranges)
  (hMark : Mark_bananas = Mike_oranges + Matt_apples)
  (hMary : Mary_grapes = Mike_oranges + Matt_apples + Mark_bananas + 5) :
  Mike_oranges + Matt_apples + Mark_bananas + Mary_grapes = 41 :=
by
  sorry

end total_fruits_sum_l242_24248


namespace probability_same_color_correct_l242_24297

-- conditions
def sides := ["maroon", "teal", "cyan", "sparkly"]
def die : Type := {v // v ∈ sides}
def maroon_count := 6
def teal_count := 9
def cyan_count := 10
def sparkly_count := 5
def total_sides := 30

-- calculate probabilities
def prob (count : ℕ) : ℚ := (count ^ 2) / (total_sides ^ 2)
def prob_same_color : ℚ :=
  prob maroon_count +
  prob teal_count +
  prob cyan_count +
  prob sparkly_count

-- statement
theorem probability_same_color_correct :
  prob_same_color = 121 / 450 :=
sorry

end probability_same_color_correct_l242_24297


namespace change_calculation_l242_24280

def cost_of_apple : ℝ := 0.75
def amount_paid : ℝ := 5.00

theorem change_calculation : (amount_paid - cost_of_apple) = 4.25 := by
  sorry

end change_calculation_l242_24280


namespace total_calculators_sold_l242_24237

theorem total_calculators_sold 
    (x y : ℕ)
    (h₁ : y = 35)
    (h₂ : 15 * x + 67 * y = 3875) :
    x + y = 137 :=
by 
  -- We will insert the proof here
  sorry

end total_calculators_sold_l242_24237


namespace females_dont_listen_correct_l242_24240

/-- Number of males who listen to the station -/
def males_listen : ℕ := 45

/-- Number of females who don't listen to the station -/
def females_dont_listen : ℕ := 87

/-- Total number of people who listen to the station -/
def total_listen : ℕ := 120

/-- Total number of people who don't listen to the station -/
def total_dont_listen : ℕ := 135

/-- Number of females surveyed based on the problem description -/
def total_females_surveyed (total_peoples_total : ℕ) (males_dont_listen : ℕ) : ℕ := 
  total_peoples_total - (males_listen + males_dont_listen)

/-- Number of females who listen to the station -/
def females_listen (total_females : ℕ) : ℕ := total_females - females_dont_listen

/-- Proof that the number of females who do not listen to the station is 87 -/
theorem females_dont_listen_correct 
  (total_peoples_total : ℕ)
  (males_dont_listen : ℕ)
  (total_females := total_females_surveyed total_peoples_total males_dont_listen)
  (females_listen := females_listen total_females) :
  females_dont_listen = 87 :=
sorry

end females_dont_listen_correct_l242_24240


namespace esperanza_savings_l242_24201

-- Define the conditions as constants
def rent := 600
def gross_salary := 4840
def food_cost := (3 / 5) * rent
def mortgage_bill := 3 * food_cost
def total_expenses := rent + food_cost + mortgage_bill
def savings := gross_salary - total_expenses
def taxes := (2 / 5) * savings
def actual_savings := savings - taxes

theorem esperanza_savings : actual_savings = 1680 := by
  sorry

end esperanza_savings_l242_24201


namespace third_bowler_points_162_l242_24244

variable (x : ℕ)

def total_score (x : ℕ) : Prop :=
  let first_bowler_points := x
  let second_bowler_points := 3 * x
  let third_bowler_points := x
  first_bowler_points + second_bowler_points + third_bowler_points = 810

theorem third_bowler_points_162 (x : ℕ) (h : total_score x) : x = 162 := by
  sorry

end third_bowler_points_162_l242_24244


namespace total_cost_of_vacation_l242_24255

noncomputable def total_cost (C : ℝ) : Prop :=
  let cost_per_person_three := C / 3
  let cost_per_person_four := C / 4
  cost_per_person_three - cost_per_person_four = 60

theorem total_cost_of_vacation (C : ℝ) (h : total_cost C) : C = 720 :=
  sorry

end total_cost_of_vacation_l242_24255


namespace minimum_parents_needed_l242_24213

/-- 
Given conditions:
1. There are 30 students going on the excursion.
2. Each car can accommodate 5 people, including the driver.
Prove that the minimum number of parents needed to be invited on the excursion is 8.
-/
theorem minimum_parents_needed (students : ℕ) (car_capacity : ℕ) (drivers_needed : ℕ) 
  (h1 : students = 30) (h2 : car_capacity = 5) (h3 : drivers_needed = 1) 
  : ∃ (parents : ℕ), parents = 8 :=
by
  existsi 8
  sorry

end minimum_parents_needed_l242_24213


namespace trees_in_park_l242_24220

variable (W O T : Nat)

theorem trees_in_park (h1 : W = 36) (h2 : O = W + 11) (h3 : T = W + O) : T = 83 := by
  sorry

end trees_in_park_l242_24220


namespace distance_to_school_l242_24223

theorem distance_to_school : 
  ∀ (d v : ℝ), (d = v * (1 / 3)) → (d = (v + 20) * (1 / 4)) → d = 20 :=
by
  intros d v h1 h2
  sorry

end distance_to_school_l242_24223


namespace lives_lost_l242_24228

-- Conditions given in the problem
def initial_lives : ℕ := 83
def current_lives : ℕ := 70

-- Prove the number of lives lost
theorem lives_lost : initial_lives - current_lives = 13 :=
by
  sorry

end lives_lost_l242_24228


namespace remainder_modulo_9_l242_24271

noncomputable def power10 := 10^15
noncomputable def power3  := 3^15

theorem remainder_modulo_9 : (7 * power10 + power3) % 9 = 7 := by
  -- Define the conditions given in the problem
  have h1 : (10 % 9 = 1) := by 
    norm_num
  have h2 : (3^2 % 9 = 0) := by 
    norm_num
  
  -- Utilize these conditions to prove the statement
  sorry

end remainder_modulo_9_l242_24271


namespace toothpicks_pattern_100th_stage_l242_24246

theorem toothpicks_pattern_100th_stage :
  let a_1 := 5
  let d := 4
  let n := 100
  (a_1 + (n - 1) * d) = 401 := by
  sorry

end toothpicks_pattern_100th_stage_l242_24246


namespace sara_museum_visit_l242_24226

theorem sara_museum_visit (S : Finset ℕ) (hS : S.card = 6) :
  ∃ count : ℕ, count = 720 ∧ 
  (∀ M A : Finset ℕ, M.card = 3 → A.card = 3 → M ∪ A = S → 
    count = (S.card.choose M.card) * M.card.factorial * A.card.factorial) :=
by
  sorry

end sara_museum_visit_l242_24226


namespace system_of_linear_equations_l242_24245

-- Define the system of linear equations and a lemma stating the given conditions and the proof goals.
theorem system_of_linear_equations (x y m : ℚ) :
  (x + 3 * y = 7) ∧ (2 * x - 3 * y = 2) ∧ (x - 3 * y + m * x + 3 = 0) ↔ 
  (x = 4 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ m = -2 / 3 :=
by
  sorry

end system_of_linear_equations_l242_24245


namespace generalized_schur_inequality_l242_24236

theorem generalized_schur_inequality (t : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^t * (a - b) * (a - c) + b^t * (b - c) * (b - a) + c^t * (c - a) * (c - b) ≥ 0 :=
sorry

end generalized_schur_inequality_l242_24236


namespace gear_teeth_count_l242_24270

theorem gear_teeth_count 
  (x y z: ℕ) 
  (h1: x + y + z = 60) 
  (h2: 4 * x - 20 = 5 * y) 
  (h3: 5 * y = 10 * z):
  x = 30 ∧ y = 20 ∧ z = 10 :=
by
  sorry

end gear_teeth_count_l242_24270


namespace difference_of_squirrels_and_nuts_l242_24260

-- Definitions
def number_of_squirrels : ℕ := 4
def number_of_nuts : ℕ := 2

-- Theorem statement with conditions and conclusion
theorem difference_of_squirrels_and_nuts : number_of_squirrels - number_of_nuts = 2 := by
  sorry

end difference_of_squirrels_and_nuts_l242_24260


namespace polygons_sides_l242_24257

theorem polygons_sides 
  (n1 n2 : ℕ)
  (h1 : n1 * (n1 - 3) / 2 + n2 * (n2 - 3) / 2 = 158)
  (h2 : 180 * (n1 + n2 - 4) = 4320) :
  (n1 = 16 ∧ n2 = 12) ∨ (n1 = 12 ∧ n2 = 16) :=
sorry

end polygons_sides_l242_24257


namespace polynomial_example_properties_l242_24229

open Polynomial

noncomputable def polynomial_example : Polynomial ℚ :=
- (1 / 2) * (X^2 + X - 1) * (X^2 + 1)

theorem polynomial_example_properties :
  ∃ P : Polynomial ℚ, (X^2 + 1) ∣ P ∧ (X^3 + 1) ∣ (P - 1) :=
by
  use polynomial_example
  -- To complete the proof, one would typically verify the divisibility properties here.
  sorry

end polynomial_example_properties_l242_24229


namespace rhombus_diagonal_length_l242_24221

-- Definitions of given conditions
def d1 : ℝ := 10
def Area : ℝ := 60

-- Proof of desired condition
theorem rhombus_diagonal_length (d2 : ℝ) : 
  (Area = d1 * d2 / 2) → d2 = 12 :=
by
  sorry

end rhombus_diagonal_length_l242_24221


namespace probability_three_green_is_14_over_99_l242_24290

noncomputable def probability_three_green :=
  let total_combinations := Nat.choose 12 4
  let successful_outcomes := (Nat.choose 5 3) * (Nat.choose 7 1)
  (successful_outcomes : ℚ) / total_combinations

theorem probability_three_green_is_14_over_99 :
  probability_three_green = 14 / 99 :=
by
  sorry

end probability_three_green_is_14_over_99_l242_24290


namespace original_price_of_car_l242_24216

theorem original_price_of_car (spent price_percent original_price : ℝ) (h1 : spent = 15000) (h2 : price_percent = 0.40) (h3 : spent = price_percent * original_price) : original_price = 37500 :=
by
  sorry

end original_price_of_car_l242_24216


namespace digit_equation_l242_24225

-- Define the digits for the letters L, O, V, E, and S in base 10.
def digit_L := 4
def digit_O := 3
def digit_V := 7
def digit_E := 8
def digit_S := 6

-- Define the numeral representations.
def LOVE := digit_L * 1000 + digit_O * 100 + digit_V * 10 + digit_E
def EVOL := digit_E * 1000 + digit_V * 100 + digit_O * 10 + digit_L
def SOLVES := digit_S * 100000 + digit_O * 10000 + digit_L * 1000 + digit_V * 100 + digit_E * 10 + digit_S

-- Prove that LOVE + EVOL + LOVE = SOLVES in base 10.
theorem digit_equation :
  LOVE + EVOL + LOVE = SOLVES :=
by
  -- Proof is omitted; include a proper proof in your verification process.
  sorry

end digit_equation_l242_24225


namespace algebraic_expression_value_l242_24284

theorem algebraic_expression_value (a b : ℝ) (h : a = b + 1) : a^2 - 2 * a * b + b^2 + 2 = 3 :=
by
  sorry

end algebraic_expression_value_l242_24284


namespace find_a_l242_24295

open Set

def set_A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def set_B : Set ℝ := {2, 3}
def set_C : Set ℝ := {2, -4}

theorem find_a (a : ℝ) (haB : (set_A a) ∩ set_B ≠ ∅) (haC : (set_A a) ∩ set_C = ∅) : a = -2 :=
sorry

end find_a_l242_24295


namespace symmetry_of_F_l242_24268

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ :=
|f x| + f (|x|)

theorem symmetry_of_F (f : ℝ → ℝ) (h : is_odd_function f) :
    ∀ x : ℝ, F f x = F f (-x) :=
by
  sorry

end symmetry_of_F_l242_24268


namespace calculate_expression_l242_24232

theorem calculate_expression :
  (2 ^ (1/3) * 8 ^ (1/3) + 18 / (3 * 3) - 8 ^ (5/3)) = 2 ^ (4/3) - 30 :=
by
  sorry

end calculate_expression_l242_24232


namespace calc_g_3_l242_24265

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 3 * x - 1

theorem calc_g_3 : g (g (g (g 3))) = 1 := by
  sorry

end calc_g_3_l242_24265
