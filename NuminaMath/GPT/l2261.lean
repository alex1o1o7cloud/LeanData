import Mathlib

namespace NUMINAMATH_GPT_complex_number_quadrant_l2261_226120

open Complex

theorem complex_number_quadrant 
  (a b c d : ℤ) : 
  (a + b * Complex.I) * (c - d * Complex.I) = (a*c + b*d) + (a*d + b*c) * Complex.I → 
  (0 < (a*c + b*d) ∧ 0 < (a*d + b*c)) → 
  True := 
by
  intro h_mul h_coord
  sorry


end NUMINAMATH_GPT_complex_number_quadrant_l2261_226120


namespace NUMINAMATH_GPT_correct_calculation_l2261_226170

-- Define the base type for exponents
variables (a : ℝ)

theorem correct_calculation :
  (a^3 * a^5 = a^8) ∧ 
  ¬((a^3)^2 = a^5) ∧ 
  ¬(a^5 + a^2 = a^7) ∧ 
  ¬(a^6 / a^2 = a^3) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2261_226170


namespace NUMINAMATH_GPT_quad_area_FDBG_l2261_226124

open Real

noncomputable def area_quad_FDBG (AB AC area_ABC : ℝ) : ℝ :=
  let AD := AB / 2
  let AE := AC / 2
  let area_ADE := area_ABC / 4
  let x := 2 * area_ABC / (AB * AC)
  let sin_A := x
  let hyp_ratio := sin_A / (area_ABC / AC)
  let factor := hyp_ratio / 2
  let area_AFG := factor * area_ADE
  area_ABC - area_ADE - 2 * area_AFG

theorem quad_area_FDBG (AB AC area_ABC : ℝ) (hAB : AB = 60) (hAC : AC = 15) (harea : area_ABC = 180) :
  area_quad_FDBG AB AC area_ABC = 117 := by
  sorry

end NUMINAMATH_GPT_quad_area_FDBG_l2261_226124


namespace NUMINAMATH_GPT_cost_price_of_book_l2261_226126

theorem cost_price_of_book
  (C : ℝ)
  (h : 1.09 * C - 0.91 * C = 9) :
  C = 50 :=
sorry

end NUMINAMATH_GPT_cost_price_of_book_l2261_226126


namespace NUMINAMATH_GPT_initial_average_weight_l2261_226138

theorem initial_average_weight
  (A : ℚ) -- Define A as a rational number since we are dealing with division 
  (h1 : 6 * A + 133 = 7 * 151) : -- Condition from the problem translated into an equation
  A = 154 := -- Statement we need to prove
by
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_initial_average_weight_l2261_226138


namespace NUMINAMATH_GPT_integer_solutions_count_l2261_226188

theorem integer_solutions_count : ∃ (s : Finset ℤ), (∀ x ∈ s, x^2 - x - 2 ≤ 0) ∧ (Finset.card s = 4) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l2261_226188


namespace NUMINAMATH_GPT_total_food_per_day_l2261_226127

theorem total_food_per_day :
  let num_puppies := 4
  let num_dogs := 3
  let dog_meal_weight := 4
  let dog_meals_per_day := 3
  let dog_food_per_day := dog_meal_weight * dog_meals_per_day
  let total_dog_food_per_day := dog_food_per_day * num_dogs
  let puppy_meal_weight := dog_meal_weight / 2
  let puppy_meals_per_day := dog_meals_per_day * 3
  let puppy_food_per_day := puppy_meal_weight * puppy_meals_per_day
  let total_puppy_food_per_day := puppy_food_per_day * num_puppies
  total_dog_food_per_day + total_puppy_food_per_day = 108 :=
by
  sorry

end NUMINAMATH_GPT_total_food_per_day_l2261_226127


namespace NUMINAMATH_GPT_parabola_equation_1_parabola_equation_2_l2261_226152

noncomputable def parabola_vertex_focus (vertex focus : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, (focus.1 = p / 2 ∧ focus.2 = 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 24 * x)

noncomputable def standard_parabola_through_point (point : ℝ × ℝ) : Prop :=
  ∃ p : ℝ, ( ( point.1^2 = 2 * p * point.2 ∧ point.2 ≠ 0 ∧ point.1 ≠ 0) ∧ (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = y / 2) ) ∨
           ( ( point.2^2 = 2 * p * point.1 ∧ point.1 ≠ 0 ∧ point.2 ≠ 0) ∧ (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) )

theorem parabola_equation_1 : parabola_vertex_focus (0, 0) (6, 0) := 
  sorry

theorem parabola_equation_2 : standard_parabola_through_point (1, 2) := 
  sorry

end NUMINAMATH_GPT_parabola_equation_1_parabola_equation_2_l2261_226152


namespace NUMINAMATH_GPT_symmetric_line_equation_l2261_226141

theorem symmetric_line_equation (x y : ℝ) :
  (2 : ℝ) * (2 - x) + (3 : ℝ) * (-2 - y) - 6 = 0 → 2 * x + 3 * y + 8 = 0 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l2261_226141


namespace NUMINAMATH_GPT_find_a_l2261_226182

theorem find_a (a : ℝ) (h_pos : 0 < a) (h_neq_one : a ≠ 1)
  (h_diff : |a^2 - a| = 6) : a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_l2261_226182


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l2261_226107

theorem hyperbola_eccentricity 
  (a b : ℝ) (h1 : 2 * (1 : ℝ) + 1 = 0) (h2 : 0 < a) (h3 : 0 < b) 
  (h4 : b = 2 * a) : 
  (∃ e : ℝ, e = (Real.sqrt 5)) 
:= 
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l2261_226107


namespace NUMINAMATH_GPT_units_digit_quotient_l2261_226171

theorem units_digit_quotient (n : ℕ) :
  (2^1993 + 3^1993) % 5 = 0 →
  ((2^1993 + 3^1993) / 5) % 10 = 3 := by
  sorry

end NUMINAMATH_GPT_units_digit_quotient_l2261_226171


namespace NUMINAMATH_GPT_algebraic_notation_3m_minus_n_squared_l2261_226166

theorem algebraic_notation_3m_minus_n_squared (m n : ℝ) : 
  (3 * m - n)^2 = (3 * m - n) ^ 2 :=
by sorry

end NUMINAMATH_GPT_algebraic_notation_3m_minus_n_squared_l2261_226166


namespace NUMINAMATH_GPT_find_f_pi_six_value_l2261_226137

noncomputable def f (x : ℝ) (f'₀ : ℝ) : ℝ := f'₀ * Real.sin x + Real.cos x

theorem find_f_pi_six_value (f'₀ : ℝ) (h : f'₀ = 2 + Real.sqrt 3) : f (π / 6) f'₀ = 1 + Real.sqrt 3 := 
by
  -- condition from the problem
  let f₀ := f (π / 6) f'₀
  -- final goal to prove
  sorry

end NUMINAMATH_GPT_find_f_pi_six_value_l2261_226137


namespace NUMINAMATH_GPT_probability_N_14_mod_5_is_1_l2261_226178

theorem probability_N_14_mod_5_is_1 :
  let total := 1950
  let favorable := 2
  let outcomes := 5
  (favorable / outcomes) = (2 / 5) := by
  sorry

end NUMINAMATH_GPT_probability_N_14_mod_5_is_1_l2261_226178


namespace NUMINAMATH_GPT_cost_of_candy_l2261_226114

theorem cost_of_candy (initial_amount pencil_cost remaining_after_candy : ℕ) 
  (h1 : initial_amount = 43) 
  (h2 : pencil_cost = 20) 
  (h3 : remaining_after_candy = 18) :
  ∃ candy_cost : ℕ, candy_cost = initial_amount - pencil_cost - remaining_after_candy :=
by
  sorry

end NUMINAMATH_GPT_cost_of_candy_l2261_226114


namespace NUMINAMATH_GPT_power_function_is_odd_l2261_226177

theorem power_function_is_odd (m : ℝ) (x : ℝ) (h : (m^2 - m - 1) * (-x)^m = -(m^2 - m - 1) * x^m) : m = -1 :=
sorry

end NUMINAMATH_GPT_power_function_is_odd_l2261_226177


namespace NUMINAMATH_GPT_angle_x_in_triangle_l2261_226139

theorem angle_x_in_triangle :
  ∀ (x : ℝ), x + 2 * x + 50 = 180 → x = 130 / 3 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_angle_x_in_triangle_l2261_226139


namespace NUMINAMATH_GPT_sum_of_money_l2261_226123

theorem sum_of_money (A B C : ℝ) (hB : B = 0.65 * A) (hC : C = 0.40 * A) (hC_val : C = 56) :
  A + B + C = 287 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_money_l2261_226123


namespace NUMINAMATH_GPT_domain_of_f_l2261_226183

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.log (x + 1)) + Real.sqrt (4 - x^2)

theorem domain_of_f : 
  {x : ℝ | x > -1 ∧ x ≤ 2 ∧ x ≠ 0 ∧ 4 - x^2 ≥ 0} = {x : ℝ | (-1 < x ∧ x < 0) ∨ (0 < x ∧ x ≤ 2)} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2261_226183


namespace NUMINAMATH_GPT_greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l2261_226184

theorem greatest_divisor_of_product_of_5_consecutive_multiples_of_4 :
  let n1 := 4
  let n2 := 8
  let n3 := 12
  let n4 := 16
  let n5 := 20
  let spf1 := 2 -- smallest prime factor of 4
  let spf2 := 2 -- smallest prime factor of 8
  let spf3 := 2 -- smallest prime factor of 12
  let spf4 := 2 -- smallest prime factor of 16
  let spf5 := 2 -- smallest prime factor of 20
  let p1 := n1^spf1
  let p2 := n2^spf2
  let p3 := n3^spf3
  let p4 := n4^spf4
  let p5 := n5^spf5
  let product := p1 * p2 * p3 * p4 * p5
  product % (2^24) = 0 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_product_of_5_consecutive_multiples_of_4_l2261_226184


namespace NUMINAMATH_GPT_width_of_rectangular_prism_l2261_226105

theorem width_of_rectangular_prism (l h d : ℕ) (w : ℤ) 
  (hl : l = 3) (hh : h = 12) (hd : d = 13) 
  (diag_eq : d = Int.sqrt (l^2 + w^2 + h^2)) : 
  w = 4 := by
  sorry

end NUMINAMATH_GPT_width_of_rectangular_prism_l2261_226105


namespace NUMINAMATH_GPT_trains_cross_time_l2261_226108

theorem trains_cross_time
  (length_each_train : ℝ)
  (speed_each_train_kmh : ℝ)
  (relative_speed_m_s : ℝ)
  (total_distance : ℝ)
  (conversion_factor : ℝ) :
  length_each_train = 120 →
  speed_each_train_kmh = 27 →
  conversion_factor = 1000 / 3600 →
  relative_speed_m_s = speed_each_train_kmh * conversion_factor →
  total_distance = 2 * length_each_train →
  total_distance / relative_speed_m_s = 16 :=
by
  sorry

end NUMINAMATH_GPT_trains_cross_time_l2261_226108


namespace NUMINAMATH_GPT_largest_five_digit_congruent_to_18_mod_25_l2261_226161

theorem largest_five_digit_congruent_to_18_mod_25 : 
  ∃ (x : ℕ), x < 100000 ∧ 10000 ≤ x ∧ x % 25 = 18 ∧ x = 99993 :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_congruent_to_18_mod_25_l2261_226161


namespace NUMINAMATH_GPT_inequality_proof_l2261_226148

variable (x y z : ℝ)

theorem inequality_proof (h : x + y + z = x * y + y * z + z * x) :
  x / (x^2 + 1) + y / (y^2 + 1) + z / (z^2 + 1) ≥ -1/2 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2261_226148


namespace NUMINAMATH_GPT_max_sum_of_ten_consecutive_in_hundred_l2261_226125

theorem max_sum_of_ten_consecutive_in_hundred :
  ∀ (s : Fin 100 → ℕ), (∀ i : Fin 100, 1 ≤ s i ∧ s i ≤ 100) → 
  (∃ i : Fin 91, (s i + s (i + 1) + s (i + 2) + s (i + 3) +
  s (i + 4) + s (i + 5) + s (i + 6) + s (i + 7) + s (i + 8) + s (i + 9)) ≥ 505) :=
by
  intro s hs
  sorry

end NUMINAMATH_GPT_max_sum_of_ten_consecutive_in_hundred_l2261_226125


namespace NUMINAMATH_GPT_speed_of_student_B_l2261_226146

open Function
open Real

theorem speed_of_student_B (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) (b_speed : ℝ) :
  distance = 12 ∧ speed_ratio = 1.2 ∧ time_difference = 1 / 6 → b_speed = 12 :=
by
  intro h
  have h1 := h.1
  have h2 := (h.2).1
  have h3 := (h.2).2
  sorry

end NUMINAMATH_GPT_speed_of_student_B_l2261_226146


namespace NUMINAMATH_GPT_find_n_for_primes_l2261_226160

def A_n (n : ℕ) : ℕ := 1 + 7 * (10^n - 1) / 9
def B_n (n : ℕ) : ℕ := 3 + 7 * (10^n - 1) / 9

theorem find_n_for_primes (n : ℕ) :
  (∀ n, n > 0 → (Nat.Prime (A_n n) ∧ Nat.Prime (B_n n)) ↔ n = 1) :=
sorry

end NUMINAMATH_GPT_find_n_for_primes_l2261_226160


namespace NUMINAMATH_GPT_problem1_problem2_l2261_226169

-- Problem 1
theorem problem1 (a b c d : ℝ) (hab : a * b > 0) (hbc_ad : b * c - a * d > 0) : (c / a) - (d / b) > 0 := sorry

-- Problem 2
theorem problem2 (a b c d : ℝ) (ha_gt_b : a > b) (hc_gt_d : c > d) : a - d > b - c := sorry

end NUMINAMATH_GPT_problem1_problem2_l2261_226169


namespace NUMINAMATH_GPT_monica_book_ratio_theorem_l2261_226175

/-
Given:
1. Monica read 16 books last year.
2. This year, she read some multiple of the number of books she read last year.
3. Next year, she will read 69 books.
4. Next year, she wants to read 5 more than twice the number of books she read this year.

Prove:
The ratio of the number of books she read this year to the number of books she read last year is 2.
-/

noncomputable def monica_book_ratio_proof : Prop :=
  let last_year_books := 16
  let next_year_books := 69
  ∃ (x : ℕ), (∃ (n : ℕ), x = last_year_books * n) ∧ (2 * x + 5 = next_year_books) ∧ (x / last_year_books = 2)

theorem monica_book_ratio_theorem : monica_book_ratio_proof :=
  by
    sorry

end NUMINAMATH_GPT_monica_book_ratio_theorem_l2261_226175


namespace NUMINAMATH_GPT_solve_quadratic_equation1_solve_quadratic_equation2_l2261_226187

theorem solve_quadratic_equation1 (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5 := 
by sorry

theorem solve_quadratic_equation2 (x : ℝ) : (x + 3) * (x - 3) = 3 * (x + 3) ↔ x = -3 ∨ x = 6 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_equation1_solve_quadratic_equation2_l2261_226187


namespace NUMINAMATH_GPT_combined_age_of_four_siblings_l2261_226192

theorem combined_age_of_four_siblings :
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  aaron_age + sister_age + henry_age + alice_age = 253 :=
by
  let aaron_age := 15
  let sister_age := 3 * aaron_age
  let henry_age := 4 * sister_age
  let alice_age := aaron_age - 2
  have h1 : aaron_age + sister_age + henry_age + alice_age = 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) := by sorry
  have h2 : 15 + 3 * 15 + 4 * (3 * 15) + (15 - 2) = 253 := by sorry
  exact h1.trans h2

end NUMINAMATH_GPT_combined_age_of_four_siblings_l2261_226192


namespace NUMINAMATH_GPT_certain_amount_of_seconds_l2261_226191

theorem certain_amount_of_seconds (X : ℕ)
    (cond1 : 12 / X = 16 / 480) :
    X = 360 :=
by
  sorry

end NUMINAMATH_GPT_certain_amount_of_seconds_l2261_226191


namespace NUMINAMATH_GPT_compute_fraction_l2261_226180

def x : ℚ := 2 / 3
def y : ℚ := 3 / 2
def z : ℚ := 1 / 3

theorem compute_fraction :
  (1 / 3) * x^7 * y^5 * z^4 = 11 / 600 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l2261_226180


namespace NUMINAMATH_GPT_find_m_values_l2261_226193

noncomputable def lines_cannot_form_triangle (m : ℝ) : Prop :=
  (4 * m - 1 = 0) ∨ (6 * m + 1 = 0) ∨ (m^2 + m / 3 - 2 / 3 = 0)

theorem find_m_values :
  { m : ℝ | lines_cannot_form_triangle m } = {4, -1 / 6, -1, 2 / 3} :=
by
  sorry

end NUMINAMATH_GPT_find_m_values_l2261_226193


namespace NUMINAMATH_GPT_minimum_value_expression_l2261_226112

theorem minimum_value_expression 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ 0 < x5) 
  (h_cond : x1^3 + x2^3 + x3^3 + x4^3 + x5^3 = 1) : 
  ∃ y, y = (3 * Real.sqrt 3) / 2 ∧ 
  (y = (x1 / (1 - x1^2) + x2 / (1 - x2^2) + x3 / (1 - x3^2) + x4 / (1 - x4^2) + x5 / (1 - x5^2))) :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l2261_226112


namespace NUMINAMATH_GPT_simplify_polynomial_l2261_226176

theorem simplify_polynomial :
  (3 * y - 2) * (6 * y ^ 12 + 3 * y ^ 11 + 6 * y ^ 10 + 3 * y ^ 9) =
  18 * y ^ 13 - 3 * y ^ 12 + 12 * y ^ 11 - 3 * y ^ 10 - 6 * y ^ 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l2261_226176


namespace NUMINAMATH_GPT_dust_storm_acres_l2261_226194

def total_acres : ℕ := 64013
def untouched_acres : ℕ := 522
def dust_storm_covered : ℕ := total_acres - untouched_acres

theorem dust_storm_acres :
  dust_storm_covered = 63491 := by
  sorry

end NUMINAMATH_GPT_dust_storm_acres_l2261_226194


namespace NUMINAMATH_GPT_hyperbola_center_l2261_226196

theorem hyperbola_center (x y : ℝ) :
    9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0 → (x, y) = (3, 5) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_center_l2261_226196


namespace NUMINAMATH_GPT_division_remainder_l2261_226164

-- Define the conditions
def dividend : ℝ := 9087.42
def divisor : ℝ := 417.35
def quotient : ℝ := 21

-- Define the expected remainder
def expected_remainder : ℝ := 323.07

-- Statement of the problem
theorem division_remainder : dividend - divisor * quotient = expected_remainder :=
by
  sorry

end NUMINAMATH_GPT_division_remainder_l2261_226164


namespace NUMINAMATH_GPT_cos_third_quadrant_l2261_226132

theorem cos_third_quadrant (B : ℝ) (hB : -π < B ∧ B < -π / 2) (sin_B : Real.sin B = 5 / 13) :
  Real.cos B = -12 / 13 :=
by
  sorry

end NUMINAMATH_GPT_cos_third_quadrant_l2261_226132


namespace NUMINAMATH_GPT_diff_of_squares_value_l2261_226117

theorem diff_of_squares_value :
  535^2 - 465^2 = 70000 :=
by sorry

end NUMINAMATH_GPT_diff_of_squares_value_l2261_226117


namespace NUMINAMATH_GPT_adjacent_irreducible_rationals_condition_l2261_226116

theorem adjacent_irreducible_rationals_condition 
  (a b c d : ℕ) 
  (hab_cop : Nat.gcd a b = 1) (hcd_cop : Nat.gcd c d = 1) 
  (h_ab_prod : a * b < 1988) (h_cd_prod : c * d < 1988) 
  (adj : ∀ p q r s, (Nat.gcd p q = 1) → (Nat.gcd r s = 1) → 
                  (p * q < 1988) → (r * s < 1988) →
                  (p / q < r / s) → (p * s - q * r = 1)) : 
  b * c - a * d = 1 :=
sorry

end NUMINAMATH_GPT_adjacent_irreducible_rationals_condition_l2261_226116


namespace NUMINAMATH_GPT_problem_statement_l2261_226128

variable {R : Type*} [LinearOrderedField R]

def is_even_function (f : R → R) : Prop := ∀ x : R, f x = f (-x)

theorem problem_statement (f : R → R)
  (h1 : is_even_function f)
  (h2 : ∀ x1 x2 : R, x1 ≤ -1 → x2 ≤ -1 → (x2 - x1) * (f x2 - f x1) < 0) :
  f (-1) < f (-3 / 2) ∧ f (-3 / 2) < f 2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l2261_226128


namespace NUMINAMATH_GPT_adam_action_figures_per_shelf_l2261_226121

-- Define the number of shelves and the total number of action figures
def shelves : ℕ := 4
def total_action_figures : ℕ := 44

-- Define the number of action figures per shelf
def action_figures_per_shelf : ℕ := total_action_figures / shelves

-- State the theorem to be proven
theorem adam_action_figures_per_shelf : action_figures_per_shelf = 11 :=
by sorry

end NUMINAMATH_GPT_adam_action_figures_per_shelf_l2261_226121


namespace NUMINAMATH_GPT_other_factor_of_product_l2261_226110

def product_has_factors (n : ℕ) : Prop :=
  ∃ a b c d e f : ℕ, n = (2^a) * (3^b) * (5^c) * (7^d) * (11^e) * (13^f) ∧ a ≥ 4 ∧ b ≥ 3

def smallest_w (x : ℕ) : ℕ :=
  if h : x = 1452 then 468 else 1

theorem other_factor_of_product (w : ℕ) : 
  (product_has_factors (1452 * w)) → (w = 468) :=
by
  sorry

end NUMINAMATH_GPT_other_factor_of_product_l2261_226110


namespace NUMINAMATH_GPT_find_x_l2261_226130

theorem find_x (x : ℝ) (h : ∑' n : ℕ, (n + 1) * x ^ n = 9) : x = 2 / 3 :=
sorry

end NUMINAMATH_GPT_find_x_l2261_226130


namespace NUMINAMATH_GPT_min_value_of_polynomial_l2261_226129

theorem min_value_of_polynomial : ∃ x : ℝ, (x^2 + x + 1) = 3 / 4 :=
by {
  -- Solution steps are omitted
  sorry
}

end NUMINAMATH_GPT_min_value_of_polynomial_l2261_226129


namespace NUMINAMATH_GPT_recycling_drive_l2261_226147

theorem recycling_drive (S : ℕ) 
  (h1 : ∀ (n : ℕ), n = 280 * S) -- Each section collected 280 kilos in two weeks
  (h2 : ∀ (t : ℕ), t = 2000 - 320) -- After the third week, they needed 320 kilos more to reach their target of 2000 kilos
  : S = 3 :=
by
  sorry

end NUMINAMATH_GPT_recycling_drive_l2261_226147


namespace NUMINAMATH_GPT_correct_propositions_l2261_226186

def Line := Type
def Plane := Type

variables (m n: Line) (α β γ: Plane)

-- Conditions from the problem statement
axiom perp (x: Line) (y: Plane): Prop -- x ⊥ y
axiom parallel (x: Line) (y: Plane): Prop -- x ∥ y
axiom perp_planes (x: Plane) (y: Plane): Prop -- x ⊥ y
axiom parallel_planes (x: Plane) (y: Plane): Prop -- x ∥ y

-- Given the conditions
axiom h1: perp m α
axiom h2: parallel n α
axiom h3: perp_planes α γ
axiom h4: perp_planes β γ
axiom h5: parallel_planes α β
axiom h6: parallel_planes β γ
axiom h7: parallel m α
axiom h8: parallel n α
axiom h9: perp m n
axiom h10: perp m γ

-- Lean statement for the problem: Prove that Propositions ① and ④ are correct.
theorem correct_propositions : (perp m n) ∧ (perp m γ) :=
by sorry -- Proof steps are not required.

end NUMINAMATH_GPT_correct_propositions_l2261_226186


namespace NUMINAMATH_GPT_unique_not_in_range_l2261_226106

open Real

noncomputable def f (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_not_in_range (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0)
  (h₅ : f a b c d 10 = 10) (h₆ : f a b c d 50 = 50) 
  (h₇ : ∀ x, x ≠ -d / c → f a b c d (f a b c d x) = x) :
  ∃! x, ¬ ∃ y, f a b c d y = x :=
  sorry

end NUMINAMATH_GPT_unique_not_in_range_l2261_226106


namespace NUMINAMATH_GPT_Joe_first_lift_weight_l2261_226162

variable (F S : ℝ)

theorem Joe_first_lift_weight (h1 : F + S = 600) (h2 : 2 * F = S + 300) : F = 300 := 
sorry

end NUMINAMATH_GPT_Joe_first_lift_weight_l2261_226162


namespace NUMINAMATH_GPT_range_of_a_bisection_method_solution_l2261_226149

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f a x = 0) :
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a) ∧ (a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

theorem bisection_method_solution (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f (32 / 17) x = 0) :
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ (|f (32 / 17) x| < 0.1) :=
sorry

end NUMINAMATH_GPT_range_of_a_bisection_method_solution_l2261_226149


namespace NUMINAMATH_GPT_gcd_1734_816_1343_l2261_226122

theorem gcd_1734_816_1343 : Int.gcd (Int.gcd 1734 816) 1343 = 17 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1734_816_1343_l2261_226122


namespace NUMINAMATH_GPT_andy_coats_l2261_226153

-- Define the initial number of minks Andy buys
def initial_minks : ℕ := 30

-- Define the number of babies each mink has
def babies_per_mink : ℕ := 6

-- Define the total initial minks including babies
def total_initial_minks : ℕ := initial_minks * babies_per_mink + initial_minks

-- Define the number of minks set free by activists
def minks_set_free : ℕ := total_initial_minks / 2

-- Define the number of minks remaining after half are set free
def remaining_minks : ℕ := total_initial_minks - minks_set_free

-- Define the number of mink skins needed for one coat
def mink_skins_per_coat : ℕ := 15

-- Define the number of coats Andy can make
def coats_andy_can_make : ℕ := remaining_minks / mink_skins_per_coat

-- The theorem to prove the number of coats Andy can make
theorem andy_coats : coats_andy_can_make = 7 := by
  sorry

end NUMINAMATH_GPT_andy_coats_l2261_226153


namespace NUMINAMATH_GPT_factorize_expression_l2261_226133

theorem factorize_expression (x y : ℝ) :
  9 * x^2 - y^2 - 4 * y - 4 = (3 * x + y + 2) * (3 * x - y - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2261_226133


namespace NUMINAMATH_GPT_percentage_reduction_in_price_l2261_226113

noncomputable def original_price_per_mango : ℝ := 416.67 / 125

noncomputable def original_num_mangoes : ℝ := 360 / original_price_per_mango

def additional_mangoes : ℝ := 12

noncomputable def new_num_mangoes : ℝ := original_num_mangoes + additional_mangoes

noncomputable def new_price_per_mango : ℝ := 360 / new_num_mangoes

noncomputable def percentage_reduction : ℝ := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100

theorem percentage_reduction_in_price : percentage_reduction = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_reduction_in_price_l2261_226113


namespace NUMINAMATH_GPT_books_still_to_read_l2261_226100

-- Define the given conditions
def total_books : ℕ := 22
def books_read : ℕ := 12

-- State the theorem to be proven
theorem books_still_to_read : total_books - books_read = 10 := 
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_books_still_to_read_l2261_226100


namespace NUMINAMATH_GPT_Kyle_papers_delivered_each_week_proof_l2261_226134

-- Definitions based on identified conditions
def k_m := 100        -- Number of papers delivered from Monday to Saturday
def d_m := 6          -- Number of days from Monday to Saturday
def k_s1 := 90        -- Number of regular customers on Sunday
def k_s2 := 30        -- Number of Sunday-only customers

-- Total number of papers delivered in a week
def total_papers_week := (k_m * d_m) + (k_s1 + k_s2)

theorem Kyle_papers_delivered_each_week_proof :
  total_papers_week = 720 :=
by
  sorry

end NUMINAMATH_GPT_Kyle_papers_delivered_each_week_proof_l2261_226134


namespace NUMINAMATH_GPT_smallest_n_l2261_226145

theorem smallest_n (n : ℕ) (h1 : 1826 % 26 = 6) (h2 : 5 * n % 26 = 6) : n = 20 :=
sorry

end NUMINAMATH_GPT_smallest_n_l2261_226145


namespace NUMINAMATH_GPT_ratio_of_full_boxes_l2261_226151

theorem ratio_of_full_boxes 
  (F H : ℕ)
  (boxes_count_eq : F + H = 20)
  (parsnips_count_eq : 20 * F + 10 * H = 350) :
  F / (F + H) = 3 / 4 := 
by
  -- proof will be placed here
  sorry

end NUMINAMATH_GPT_ratio_of_full_boxes_l2261_226151


namespace NUMINAMATH_GPT_melissa_games_l2261_226167

noncomputable def total_points_scored := 91
noncomputable def points_per_game := 7
noncomputable def number_of_games_played := total_points_scored / points_per_game

theorem melissa_games : number_of_games_played = 13 :=
by 
  sorry

end NUMINAMATH_GPT_melissa_games_l2261_226167


namespace NUMINAMATH_GPT_solve_for_A_l2261_226144

theorem solve_for_A (A B : ℕ) (h1 : 4 * 10 + A + 10 * B + 3 = 68) (h2 : 10 ≤ 4 * 10 + A) (h3 : 4 * 10 + A < 100) (h4 : 10 ≤ 10 * B + 3) (h5 : 10 * B + 3 < 100) (h6 : A < 10) (h7 : B < 10) : A = 5 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_A_l2261_226144


namespace NUMINAMATH_GPT_sequence_proofs_l2261_226143

theorem sequence_proofs (a b : ℕ → ℝ) :
  a 1 = 1 ∧ b 1 = 0 ∧ 
  (∀ n, 4 * a (n + 1) = 3 * a n - b n + 4) ∧ 
  (∀ n, 4 * b (n + 1) = 3 * b n - a n - 4) → 
  (∀ n, a n + b n = (1 / 2) ^ (n - 1)) ∧ 
  (∀ n, a n - b n = 2 * n - 1) ∧ 
  (∀ n, a n = (1 / 2) ^ n + n - 1 / 2 ∧ b n = (1 / 2) ^ n - n + 1 / 2) :=
sorry

end NUMINAMATH_GPT_sequence_proofs_l2261_226143


namespace NUMINAMATH_GPT_opposite_of_2023_l2261_226131

def opposite (n : ℤ) := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_2023_l2261_226131


namespace NUMINAMATH_GPT_exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l2261_226165

noncomputable def f (x a : ℝ) : ℝ :=
if x > a then (x - 1)^3 else abs (x - 1)

theorem exists_no_minimum_value :
  ∃ a : ℝ, ¬ ∃ m : ℝ, ∀ x : ℝ, f x a ≥ m :=
sorry

theorem has_zeros_for_any_a (a : ℝ) : ∃ x : ℝ, f x a = 0 :=
sorry

theorem not_monotonically_increasing_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  ¬ ∀ x y : ℝ, 1 < x → x < y → y < a → f x a ≤ f y a :=
sorry

theorem exists_m_for_3_distinct_roots (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ m : ℝ, ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 a = m ∧ f x2 a = m ∧ f x3 a = m :=
sorry

end NUMINAMATH_GPT_exists_no_minimum_value_has_zeros_for_any_a_not_monotonically_increasing_when_a_ge_1_exists_m_for_3_distinct_roots_l2261_226165


namespace NUMINAMATH_GPT_log_expression_value_l2261_226158

theorem log_expression_value
  (h₁ : x + (Real.log 32 / Real.log 8) = 1.6666666666666667)
  (h₂ : Real.log 32 / Real.log 8 = 1.6666666666666667) :
  x = 0 :=
by
  sorry

end NUMINAMATH_GPT_log_expression_value_l2261_226158


namespace NUMINAMATH_GPT_find_b_l2261_226190

theorem find_b (a b : ℝ) (h1 : (-6) * a^2 = 3 * (4 * a + b))
  (h2 : a = 1) : b = -6 :=
by 
  sorry

end NUMINAMATH_GPT_find_b_l2261_226190


namespace NUMINAMATH_GPT_jake_third_test_marks_l2261_226185

theorem jake_third_test_marks 
  (avg_marks : ℕ)
  (marks_test1 : ℕ)
  (marks_test2 : ℕ)
  (marks_test3 : ℕ)
  (marks_test4 : ℕ)
  (h_avg : avg_marks = 75)
  (h_test1 : marks_test1 = 80)
  (h_test2 : marks_test2 = marks_test1 + 10)
  (h_test3_eq_test4 : marks_test3 = marks_test4)
  (h_total : avg_marks * 4 = marks_test1 + marks_test2 + marks_test3 + marks_test4) : 
  marks_test3 = 65 :=
sorry

end NUMINAMATH_GPT_jake_third_test_marks_l2261_226185


namespace NUMINAMATH_GPT_final_center_coordinates_l2261_226150

-- Definition of the initial condition: the center of Circle U
def center_initial : ℝ × ℝ := (3, -4)

-- Definition of the reflection function across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

-- Definition of the translation function to translate a point 5 units up
def translate_up_5 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 5)

-- Defining the final coordinates after reflection and translation
def center_final : ℝ × ℝ :=
  translate_up_5 (reflect_y_axis center_initial)

-- Problem statement: Prove that the final center coordinates are (-3, 1)
theorem final_center_coordinates :
  center_final = (-3, 1) :=
by {
  -- Skipping the proof itself, but the theorem statement should be equivalent
  sorry
}

end NUMINAMATH_GPT_final_center_coordinates_l2261_226150


namespace NUMINAMATH_GPT_find_y_when_x_is_1_l2261_226198

theorem find_y_when_x_is_1 (t : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 6) 
  (h3 : x = 1) : 
  y = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_y_when_x_is_1_l2261_226198


namespace NUMINAMATH_GPT_tabby_average_speed_l2261_226174

noncomputable def overall_average_speed : ℝ := 
  let swimming_speed : ℝ := 1
  let cycling_speed : ℝ := 18
  let running_speed : ℝ := 6
  let time_swimming : ℝ := 2
  let time_cycling : ℝ := 3
  let time_running : ℝ := 2
  let distance_swimming := swimming_speed * time_swimming
  let distance_cycling := cycling_speed * time_cycling
  let distance_running := running_speed * time_running
  let total_distance := distance_swimming + distance_cycling + distance_running
  let total_time := time_swimming + time_cycling + time_running
  total_distance / total_time

theorem tabby_average_speed : overall_average_speed = 9.71 := sorry

end NUMINAMATH_GPT_tabby_average_speed_l2261_226174


namespace NUMINAMATH_GPT_intersection_A_B_l2261_226115

def setA : Set ℤ := { x | x < -3 }
def setB : Set ℤ := {-5, -4, -3, 1}

theorem intersection_A_B : setA ∩ setB = {-5, -4} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2261_226115


namespace NUMINAMATH_GPT_parabola_focus_l2261_226101

theorem parabola_focus (x y : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end NUMINAMATH_GPT_parabola_focus_l2261_226101


namespace NUMINAMATH_GPT_sean_divided_by_julie_is_2_l2261_226173

-- Define the sum of the first n natural numbers
def sum_natural (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum as twice the sum of the first 300 natural numbers
def sean_sum : ℕ := 2 * sum_natural 300

-- Define Julie's sum as the sum of the first 300 natural numbers
def julie_sum : ℕ := sum_natural 300

-- Prove that Sean's sum divided by Julie's sum is 2
theorem sean_divided_by_julie_is_2 : sean_sum / julie_sum = 2 := by
  sorry

end NUMINAMATH_GPT_sean_divided_by_julie_is_2_l2261_226173


namespace NUMINAMATH_GPT_arithmetic_mean_geometric_mean_l2261_226109

theorem arithmetic_mean_geometric_mean (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : 
  (a + b) / 2 ≥ Real.sqrt (a * b) :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_geometric_mean_l2261_226109


namespace NUMINAMATH_GPT_total_volume_of_four_cubes_is_500_l2261_226104

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_four_cubes_is_500_l2261_226104


namespace NUMINAMATH_GPT_diego_payment_l2261_226102

theorem diego_payment (d : ℤ) (celina : ℤ) (total : ℤ) (h₁ : celina = 1000 + 4 * d) (h₂ : total = celina + d) (h₃ : total = 50000) : d = 9800 :=
sorry

end NUMINAMATH_GPT_diego_payment_l2261_226102


namespace NUMINAMATH_GPT_calculate_heartsuit_ratio_l2261_226157

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem calculate_heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by sorry

end NUMINAMATH_GPT_calculate_heartsuit_ratio_l2261_226157


namespace NUMINAMATH_GPT_complex_division_l2261_226154

def imaginary_unit := Complex.I

theorem complex_division :
  (1 - 3 * imaginary_unit) / (2 + imaginary_unit) = -1 / 5 - 7 / 5 * imaginary_unit := by
  sorry

end NUMINAMATH_GPT_complex_division_l2261_226154


namespace NUMINAMATH_GPT_parallelogram_properties_l2261_226181

noncomputable def perimeter (x y : ℤ) : ℝ :=
  2 * (5 + Real.sqrt ((x - 7) ^ 2 + (y - 3) ^ 2))

noncomputable def area (x y : ℤ) : ℝ :=
  5 * abs (y - 3)

theorem parallelogram_properties (x y : ℤ) (hx : x = 7) (hy : y = 7) :
  (perimeter x y + area x y) = 38 :=
by
  simp [perimeter, area, hx, hy]
  sorry

end NUMINAMATH_GPT_parallelogram_properties_l2261_226181


namespace NUMINAMATH_GPT_y_coordinate_of_point_on_line_l2261_226103

theorem y_coordinate_of_point_on_line (x y : ℝ) (h1 : -4 = x) (h2 : ∃ m b : ℝ, y = m * x + b ∧ y = 3 ∧ x = 10 ∧ m * 4 + b = 0) : y = -4 :=
sorry

end NUMINAMATH_GPT_y_coordinate_of_point_on_line_l2261_226103


namespace NUMINAMATH_GPT_geometric_ratio_l2261_226111

noncomputable def S (n : ℕ) : ℝ := sorry  -- Let's assume S is a function that returns the sum of the first n terms of the geometric sequence.

-- Conditions
axiom S_10_eq_S_5 : S 10 = 2 * S 5

-- Definition to be proved
theorem geometric_ratio :
  (S 5 + S 10 + S 15) / (S 10 - S 5) = -9 / 2 :=
sorry

end NUMINAMATH_GPT_geometric_ratio_l2261_226111


namespace NUMINAMATH_GPT_polynomial_expansion_l2261_226118

theorem polynomial_expansion (x : ℝ) :
  (5 * x^2 + 3 * x - 7) * (4 * x^3) = 20 * x^5 + 12 * x^4 - 28 * x^3 :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_expansion_l2261_226118


namespace NUMINAMATH_GPT_F_double_prime_coordinates_correct_l2261_226189

structure Point where
  x : Int
  y : Int

def reflect_over_y_axis (p : Point) : Point :=
  { x := -p.x, y := p.y }

def reflect_over_x_axis (p : Point) : Point :=
  { x := p.x, y := -p.y }

def F : Point := { x := 6, y := -4 }

def F' : Point := reflect_over_y_axis F

def F'' : Point := reflect_over_x_axis F'

theorem F_double_prime_coordinates_correct : F'' = { x := -6, y := 4 } :=
  sorry

end NUMINAMATH_GPT_F_double_prime_coordinates_correct_l2261_226189


namespace NUMINAMATH_GPT_min_f_when_a_neg3_range_of_a_l2261_226142

open Real

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- First statement: Minimum value of f(x) when a = -3
theorem min_f_when_a_neg3 : (∀ x : ℝ, f x (-3) ≥ 4) ∧ (∃ x : ℝ,  f x (-3) = 4) := by
  sorry

-- Second statement: Range of a given the condition
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≤ 2 * a + 2 * abs (x - 1)) ↔ a ≥ 1/3 := by
  sorry

end NUMINAMATH_GPT_min_f_when_a_neg3_range_of_a_l2261_226142


namespace NUMINAMATH_GPT_stock_price_no_return_l2261_226163

/-- Define the increase and decrease factors. --/
def increase_factor := 117 / 100
def decrease_factor := 83 / 100

/-- Define the proof that the stock price cannot return to its initial value after any number of 
    increases and decreases. --/
theorem stock_price_no_return 
  (P0 : ℝ) (k l : ℕ) : 
  P0 * (increase_factor ^ k) * (decrease_factor ^ l) ≠ P0 :=
by
  sorry

end NUMINAMATH_GPT_stock_price_no_return_l2261_226163


namespace NUMINAMATH_GPT_least_positive_integer_l2261_226155

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 1) ∧ (a % 3 = 2) ∧ (a % 4 = 3) ∧ (a % 5 = 4) → a = 59 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_l2261_226155


namespace NUMINAMATH_GPT_abs_tan_45_eq_sqrt3_factor_4x2_36_l2261_226156

theorem abs_tan_45_eq_sqrt3 : abs (1 - Real.sqrt 3) + Real.tan (Real.pi / 4) = Real.sqrt 3 := 
by 
  sorry

theorem factor_4x2_36 (x : ℝ) : 4 * x ^ 2 - 36 = 4 * (x + 3) * (x - 3) := 
by 
  sorry

end NUMINAMATH_GPT_abs_tan_45_eq_sqrt3_factor_4x2_36_l2261_226156


namespace NUMINAMATH_GPT_equal_sharing_of_chicken_wings_l2261_226168

theorem equal_sharing_of_chicken_wings 
  (initial_wings : ℕ) (additional_wings : ℕ) (number_of_friends : ℕ)
  (total_wings : ℕ) (wings_per_person : ℕ)
  (h_initial : initial_wings = 8)
  (h_additional : additional_wings = 10)
  (h_number : number_of_friends = 3)
  (h_total : total_wings = initial_wings + additional_wings)
  (h_division : wings_per_person = total_wings / number_of_friends) :
  wings_per_person = 6 := 
  by
  sorry

end NUMINAMATH_GPT_equal_sharing_of_chicken_wings_l2261_226168


namespace NUMINAMATH_GPT_inequality_solution_l2261_226135

theorem inequality_solution :
  { x : ℝ | 0 < x ∧ x ≤ 7/3 ∨ 3 ≤ x } = { x : ℝ | (0 < x ∧ x ≤ 7/3) ∨ 3 ≤ x } :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2261_226135


namespace NUMINAMATH_GPT_expression_equals_one_l2261_226172

noncomputable def compute_expression : ℝ :=
  (1 - 1 / Real.cos (30 * Real.pi / 180)) *
  (1 + 1 / Real.sin (60 * Real.pi / 180)) *
  (1 - 1 / Real.sin (30 * Real.pi / 180)) *
  (1 + 1 / Real.cos (60 * Real.pi / 180))

theorem expression_equals_one : compute_expression = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_one_l2261_226172


namespace NUMINAMATH_GPT_factorize_expression_l2261_226136

theorem factorize_expression (x : ℝ) : 2 * x ^ 2 - 50 = 2 * (x + 5) * (x - 5) := 
  sorry

end NUMINAMATH_GPT_factorize_expression_l2261_226136


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l2261_226179

-- Definitions for the hyperbola and the asymptotes
def hyperbola_equation (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1
def asymptote_equation (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x ∨ y = - (Real.sqrt 2 / 2) * x

-- The theorem to prove
theorem asymptotes_of_hyperbola (x y : ℝ) (h : hyperbola_equation x y) :
  asymptote_equation x y :=
sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l2261_226179


namespace NUMINAMATH_GPT_general_term_of_sequence_l2261_226119

def A := {n : ℕ | ∃ k : ℕ, k + 1 = n }
def B := {m : ℕ | ∃ k : ℕ, 3 * k - 1 = m }

theorem general_term_of_sequence (k : ℕ) : 
  ∃ a_k : ℕ, a_k ∈ A ∩ B ∧ a_k = 9 * k^2 - 9 * k + 2 :=
sorry

end NUMINAMATH_GPT_general_term_of_sequence_l2261_226119


namespace NUMINAMATH_GPT_abc_order_l2261_226195

noncomputable def a : ℝ := Real.log (3 / 2) - 3 / 2
noncomputable def b : ℝ := Real.log Real.pi - Real.pi
noncomputable def c : ℝ := Real.log 3 - 3

theorem abc_order : a > c ∧ c > b := by
  have h₁: a = Real.log (3 / 2) - 3 / 2 := rfl
  have h₂: b = Real.log Real.pi - Real.pi := rfl
  have h₃: c = Real.log 3 - 3 := rfl
  sorry

end NUMINAMATH_GPT_abc_order_l2261_226195


namespace NUMINAMATH_GPT_curved_surface_area_cone_l2261_226197

variable (a α β : ℝ) (l := a * Real.sin α) (r := a * Real.cos β)

theorem curved_surface_area_cone :
  π * r * l = π * a^2 * Real.sin α * Real.cos β := by
  sorry

end NUMINAMATH_GPT_curved_surface_area_cone_l2261_226197


namespace NUMINAMATH_GPT_grade_assignment_ways_l2261_226140

/-- Define the number of students and the number of grade choices -/
def num_students : ℕ := 15
def num_grades : ℕ := 4

/-- Define the total number of ways to assign grades -/
def total_ways : ℕ := num_grades ^ num_students

/-- Prove that the total number of ways to assign grades is 4^15 -/
theorem grade_assignment_ways : total_ways = 1073741824 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_grade_assignment_ways_l2261_226140


namespace NUMINAMATH_GPT_quadratic_function_min_value_l2261_226159

theorem quadratic_function_min_value :
  ∃ x, ∀ y, 5 * x^2 - 15 * x + 2 ≤ 5 * y^2 - 15 * y + 2 ∧ (5 * x^2 - 15 * x + 2 = -9.25) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_min_value_l2261_226159


namespace NUMINAMATH_GPT_find_a_l2261_226199

-- Definition of the curve y = x^3 + ax + 1
def curve (x a : ℝ) : ℝ := x^3 + a * x + 1

-- Definition of the tangent line y = 2x + 1
def tangent_line (x : ℝ) : ℝ := 2 * x + 1

-- The slope of the tangent line is 2
def slope_of_tangent_line (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + a

theorem find_a (a : ℝ) : 
  (∃ x₀, curve x₀ a = tangent_line x₀) ∧ (∃ x₀, slope_of_tangent_line x₀ a = 2) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2261_226199
