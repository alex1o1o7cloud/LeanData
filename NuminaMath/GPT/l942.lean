import Mathlib

namespace NUMINAMATH_GPT_parabola_focus_l942_94220

theorem parabola_focus (p : ℝ) (hp : p > 0) :
    ∀ (x y : ℝ), (x = 2 * p * y^2) ↔ (x, y) = (1 / (8 * p), 0) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_focus_l942_94220


namespace NUMINAMATH_GPT_polynomial_integer_roots_l942_94233

theorem polynomial_integer_roots (b1 b2 : ℤ) (x : ℤ) (h : x^3 + b2 * x^2 + b1 * x + 18 = 0) :
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
sorry

end NUMINAMATH_GPT_polynomial_integer_roots_l942_94233


namespace NUMINAMATH_GPT_count_valid_triples_l942_94248

theorem count_valid_triples :
  ∃! (a c : ℕ), a ≤ 101 ∧ 101 ≤ c ∧ a * c = 101^2 :=
sorry

end NUMINAMATH_GPT_count_valid_triples_l942_94248


namespace NUMINAMATH_GPT_find_trousers_l942_94214

variables (S T Ti : ℝ) -- Prices of shirt, trousers, and tie respectively
variables (x : ℝ)      -- The number of trousers in the first scenario

-- Conditions given in the problem
def condition1 : Prop := 6 * S + x * T + 2 * Ti = 80
def condition2 : Prop := 4 * S + 2 * T + 2 * Ti = 140
def condition3 : Prop := 5 * S + 3 * T + 2 * Ti = 110

-- Theorem to prove
theorem find_trousers : condition1 S T Ti x ∧ condition2 S T Ti ∧ condition3 S T Ti → x = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_trousers_l942_94214


namespace NUMINAMATH_GPT_initial_average_l942_94282

theorem initial_average (A : ℝ) (h : (15 * A + 14 * 15) / 15 = 54) : A = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_average_l942_94282


namespace NUMINAMATH_GPT_line_equations_through_point_with_intercepts_l942_94268

theorem line_equations_through_point_with_intercepts (x y : ℝ) :
  (x = -10 ∧ y = 10) ∧ (∃ a : ℝ, 4 * a = intercept_x ∧ a = intercept_y) →
  (x + y = 0 ∨ x + 4 * y - 30 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equations_through_point_with_intercepts_l942_94268


namespace NUMINAMATH_GPT_exterior_angle_BAC_l942_94246

-- Definitions for the problem conditions
def regular_nonagon_interior_angle :=
  140

def square_interior_angle :=
  90

-- The proof statement
theorem exterior_angle_BAC (regular_nonagon_interior_angle square_interior_angle : ℝ) : 
  regular_nonagon_interior_angle = 140 ∧ square_interior_angle = 90 -> 
  ∃ (BAC : ℝ), BAC = 130 :=
by
  sorry

end NUMINAMATH_GPT_exterior_angle_BAC_l942_94246


namespace NUMINAMATH_GPT_solution_exists_solution_unique_l942_94243

noncomputable def abc_solutions : Finset (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (2, 2, 4), (2, 4, 8), (3, 5, 15), 
   (2, 4, 2), (4, 2, 2), (4, 2, 8), (8, 4, 2), 
   (2, 8, 4), (8, 2, 4), (5, 3, 15), (15, 3, 5), (3, 15, 5),
   (2, 2, 4), (4, 2, 2), (4, 8, 2)}

theorem solution_exists (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a * b * c - 1 = (a - 1) * (b - 1) * (c - 1)) ↔ (a, b, c) ∈ abc_solutions := 
by
  sorry

theorem solution_unique (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a, b, c) ∈ abc_solutions → a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) :=
by
  sorry

end NUMINAMATH_GPT_solution_exists_solution_unique_l942_94243


namespace NUMINAMATH_GPT_days_provisions_initially_meant_l942_94215

theorem days_provisions_initially_meant (x : ℕ) (h1 : 250 * x = 200 * 50) : x = 40 :=
by sorry

end NUMINAMATH_GPT_days_provisions_initially_meant_l942_94215


namespace NUMINAMATH_GPT_find_f_of_neg_2_l942_94217

theorem find_f_of_neg_2
  (f : ℚ → ℚ)
  (h : ∀ (x : ℚ), x ≠ 0 → 3 * f (1/x) + 2 * f x / x = x^2)
  : f (-2) = 13/5 :=
sorry

end NUMINAMATH_GPT_find_f_of_neg_2_l942_94217


namespace NUMINAMATH_GPT_maximum_expr_value_l942_94226

theorem maximum_expr_value :
  ∃ (x y e f : ℕ), (e = 4 ∧ x = 3 ∧ y = 2 ∧ f = 0) ∧
  (e = 1 ∨ e = 2 ∨ e = 3 ∨ e = 4) ∧
  (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) ∧
  (y = 1 ∨ y = 2 ∨ y = 3 ∨ y = 4) ∧
  (f = 1 ∨ f = 2 ∨ f = 3 ∨ f = 4) ∧
  (e ≠ x ∧ e ≠ y ∧ e ≠ f ∧ x ≠ y ∧ x ≠ f ∧ y ≠ f) ∧
  (e * x^y - f = 36) :=
by
  sorry

end NUMINAMATH_GPT_maximum_expr_value_l942_94226


namespace NUMINAMATH_GPT_problem_l942_94299

variable {a b c x y z : ℝ}

theorem problem 
  (h1 : 5 * x + b * y + c * z = 0)
  (h2 : a * x + 7 * y + c * z = 0)
  (h3 : a * x + b * y + 9 * z = 0)
  (h4 : a ≠ 5)
  (h5 : x ≠ 0) :
  (a / (a - 5)) + (b / (b - 7)) + (c / (c - 9)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_l942_94299


namespace NUMINAMATH_GPT_lcm_is_perfect_square_l942_94265

theorem lcm_is_perfect_square (a b : ℕ) (h : (a^3 + b^3 + a * b) % (a * b * (a - b)) = 0) : ∃ k : ℕ, k^2 = Nat.lcm a b :=
by
  sorry

end NUMINAMATH_GPT_lcm_is_perfect_square_l942_94265


namespace NUMINAMATH_GPT_melted_ice_cream_depth_l942_94276

theorem melted_ice_cream_depth :
  ∀ (r_sphere r_cylinder : ℝ) (h_cylinder : ℝ),
    r_sphere = 3 ∧ r_cylinder = 12 ∧
    (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder →
    h_cylinder = 1 / 4 :=
by
  intros r_sphere r_cylinder h_cylinder h
  have r_sphere_eq : r_sphere = 3 := h.1
  have r_cylinder_eq : r_cylinder = 12 := h.2.1
  have volume_eq : (4 / 3) * Real.pi * r_sphere^3 = Real.pi * r_cylinder^2 * h_cylinder := h.2.2
  sorry

end NUMINAMATH_GPT_melted_ice_cream_depth_l942_94276


namespace NUMINAMATH_GPT_polynomial_has_real_root_l942_94213

noncomputable def P : Polynomial ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ) (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
variables (h_eq : ∀ x : ℝ, P.eval (a1 * x + b1) + P.eval (a2 * x + b2) = P.eval (a3 * x + b3))

theorem polynomial_has_real_root : ∃ x : ℝ, P.eval x = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_has_real_root_l942_94213


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l942_94230

def is_real (m : ℝ) : Prop := (m^2 - 3 * m) = 0
def is_complex (m : ℝ) : Prop := (m^2 - 3 * m) ≠ 0
def is_pure_imaginary (m : ℝ) : Prop := (m^2 - 5 * m + 6) = 0 ∧ (m^2 - 3 * m) ≠ 0

theorem problem1 (m : ℝ) : is_real m ↔ (m = 0 ∨ m = 3) :=
sorry

theorem problem2 (m : ℝ) : is_complex m ↔ (m ≠ 0 ∧ m ≠ 3) :=
sorry

theorem problem3 (m : ℝ) : is_pure_imaginary m ↔ (m = 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l942_94230


namespace NUMINAMATH_GPT_students_taking_neither_l942_94270

theorem students_taking_neither (total biology chemistry both : ℕ)
  (h1 : total = 60)
  (h2 : biology = 40)
  (h3 : chemistry = 35)
  (h4 : both = 25) :
  (total - (biology + chemistry - both)) = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_students_taking_neither_l942_94270


namespace NUMINAMATH_GPT_carrot_servings_l942_94222

theorem carrot_servings (C : ℕ) 
  (H1 : ∀ (corn_servings : ℕ), corn_servings = 5 * C)
  (H2 : ∀ (green_bean_servings : ℕ) (corn_servings : ℕ), green_bean_servings = corn_servings / 2)
  (H3 : ∀ (plot_plants : ℕ), plot_plants = 9)
  (H4 : ∀ (total_servings : ℕ) 
         (carrot_servings : ℕ)
         (corn_servings : ℕ)
         (green_bean_servings : ℕ), 
         total_servings = carrot_servings + corn_servings + green_bean_servings ∧
         total_servings = 306) : 
  C = 4 := 
    sorry

end NUMINAMATH_GPT_carrot_servings_l942_94222


namespace NUMINAMATH_GPT_find_c_l942_94277

theorem find_c (c : ℝ) (h1 : 0 < c) (h2 : c < 6) (h3 : ((6 - c) / c) = 4 / 9) : c = 54 / 13 :=
sorry

end NUMINAMATH_GPT_find_c_l942_94277


namespace NUMINAMATH_GPT_simplify_expression_l942_94289

theorem simplify_expression (x : ℝ) (h : x ≤ 2) : 
  (Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9)) = -1 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l942_94289


namespace NUMINAMATH_GPT_frustum_lateral_area_l942_94283

def frustum_upper_base_radius : ℝ := 3
def frustum_lower_base_radius : ℝ := 4
def frustum_slant_height : ℝ := 6

theorem frustum_lateral_area : 
  (1 / 2) * (frustum_upper_base_radius + frustum_lower_base_radius) * 2 * Real.pi * frustum_slant_height = 42 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_frustum_lateral_area_l942_94283


namespace NUMINAMATH_GPT_find_c_l942_94267

-- Definition of the function f
def f (x a b c : ℤ) : ℤ := x^3 + a * x^2 + b * x + c

-- Theorem statement
theorem find_c (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : f a a b c = a^3)
  (h4 : f b a b c = b^3) : c = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l942_94267


namespace NUMINAMATH_GPT_find_middle_number_l942_94224

theorem find_middle_number
  (S1 S2 M : ℤ)
  (h1 : S1 = 6 * 5)
  (h2 : S2 = 6 * 7)
  (h3 : 13 * 9 = S1 + M + S2) :
  M = 45 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_find_middle_number_l942_94224


namespace NUMINAMATH_GPT_taco_beef_per_taco_l942_94255

open Real

theorem taco_beef_per_taco
  (total_beef : ℝ)
  (sell_price : ℝ)
  (cost_per_taco : ℝ)
  (profit : ℝ)
  (h1 : total_beef = 100)
  (h2 : sell_price = 2)
  (h3 : cost_per_taco = 1.5)
  (h4 : profit = 200) :
  ∃ (x : ℝ), x = 1/4 := 
by
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_taco_beef_per_taco_l942_94255


namespace NUMINAMATH_GPT_sum_of_areas_is_858_l942_94205

def first_six_odd_squares : List ℕ := [1^2, 3^2, 5^2, 7^2, 9^2, 11^2]

def rectangle_area (width length : ℕ) : ℕ := width * length

def sum_of_areas : ℕ := (first_six_odd_squares.map (rectangle_area 3)).sum

theorem sum_of_areas_is_858 : sum_of_areas = 858 := 
by
  -- Our aim is to show that sum_of_areas is 858
  -- The proof will be developed here
  sorry

end NUMINAMATH_GPT_sum_of_areas_is_858_l942_94205


namespace NUMINAMATH_GPT_area_BCD_sixteen_area_BCD_with_new_ABD_l942_94271

-- Define the conditions and parameters of the problem.
variables (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Given conditions from part (a)
variable (AB_length : Real) (BC_length : Real) (area_ABD : Real)

-- Define the lengths and areas in our problem.
axiom AB_eq_five : AB_length = 5
axiom BC_eq_eight : BC_length = 8
axiom area_ABD_eq_ten : area_ABD = 10

-- Part (a) problem statement
theorem area_BCD_sixteen (AB_length BC_length area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → area_ABD = 10 → (∃ area_BCD : Real, area_BCD = 16) :=
by
  sorry

-- Given conditions from part (b)
variable (new_area_ABD : Real)

-- Define the new area.
axiom new_area_ABD_eq_hundred : new_area_ABD = 100

-- Part (b) problem statement
theorem area_BCD_with_new_ABD (AB_length BC_length new_area_ABD : Real) :
  AB_length = 5 → BC_length = 8 → new_area_ABD = 100 → (∃ area_BCD : Real, area_BCD = 160) :=
by
  sorry

end NUMINAMATH_GPT_area_BCD_sixteen_area_BCD_with_new_ABD_l942_94271


namespace NUMINAMATH_GPT_binom_12_11_l942_94291

theorem binom_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end NUMINAMATH_GPT_binom_12_11_l942_94291


namespace NUMINAMATH_GPT_statement_B_statement_C_statement_D_l942_94206

variables (a b : ℝ)

-- Condition: a > 0
axiom a_pos : a > 0

-- Condition: e^a + ln b = 1
axiom eq1 : Real.exp a + Real.log b = 1

-- Statement B: a + ln b < 0
theorem statement_B : a + Real.log b < 0 :=
  sorry

-- Statement C: e^a + b > 2
theorem statement_C : Real.exp a + b > 2 :=
  sorry

-- Statement D: a + b > 1
theorem statement_D : a + b > 1 :=
  sorry

end NUMINAMATH_GPT_statement_B_statement_C_statement_D_l942_94206


namespace NUMINAMATH_GPT_max_students_received_less_than_given_l942_94256

def max_students_received_less := 27
def max_possible_n := 13

theorem max_students_received_less_than_given (n : ℕ) :
  n <= max_students_received_less -> n = max_possible_n :=
sorry
 
end NUMINAMATH_GPT_max_students_received_less_than_given_l942_94256


namespace NUMINAMATH_GPT_solve_for_x_l942_94235

noncomputable def is_satisfied (x : ℝ) : Prop :=
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 2

theorem solve_for_x :
  ∀ x : ℝ, 0 < x → x ≠ 1 ↔ is_satisfied x := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l942_94235


namespace NUMINAMATH_GPT_sum_infinite_series_l942_94238

theorem sum_infinite_series :
  (∑' n : ℕ, (4 * (n + 1) + 1) / (3^(n + 1))) = 7 / 2 :=
sorry

end NUMINAMATH_GPT_sum_infinite_series_l942_94238


namespace NUMINAMATH_GPT_probability_no_3x3_red_square_l942_94260

theorem probability_no_3x3_red_square (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_prob : 65152 / 65536 = m / n) :
  m + n = 1021 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_3x3_red_square_l942_94260


namespace NUMINAMATH_GPT_purchasing_plans_count_l942_94221

theorem purchasing_plans_count :
  ∃ (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0) ∧ (∃! (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0)) := sorry

end NUMINAMATH_GPT_purchasing_plans_count_l942_94221


namespace NUMINAMATH_GPT_monitor_height_l942_94239

theorem monitor_height (width_in_inches : ℕ) (pixels_per_inch : ℕ) (total_pixels : ℕ) 
  (h1 : width_in_inches = 21) (h2 : pixels_per_inch = 100) (h3 : total_pixels = 2520000) : 
  total_pixels / (width_in_inches * pixels_per_inch) / pixels_per_inch = 12 :=
by
  sorry

end NUMINAMATH_GPT_monitor_height_l942_94239


namespace NUMINAMATH_GPT_no_rational_roots_l942_94261

theorem no_rational_roots (x : ℚ) : ¬(3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1 = 0) :=
by sorry

end NUMINAMATH_GPT_no_rational_roots_l942_94261


namespace NUMINAMATH_GPT_simplify_expression_l942_94228

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^3 - b^3) / (a * b^2) - (ab^2 - b^3) / (ab^2 - a^3) = (a^3 - ab^2 + b^4) / (a * b^2) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l942_94228


namespace NUMINAMATH_GPT_ceil_floor_sum_l942_94286

theorem ceil_floor_sum :
  (Int.ceil (7 / 3 : ℚ)) + (Int.floor (-7 / 3 : ℚ)) = 0 := 
sorry

end NUMINAMATH_GPT_ceil_floor_sum_l942_94286


namespace NUMINAMATH_GPT_standard_equation_of_ellipse_l942_94207

-- Define the conditions of the ellipse
def ellipse_condition_A (m n : ℝ) : Prop := n * (5 / 3) ^ 2 = 1
def ellipse_condition_B (m n : ℝ) : Prop := m + n = 1

-- The theorem to prove the standard equation of the ellipse
theorem standard_equation_of_ellipse (m n : ℝ) (hA : ellipse_condition_A m n) (hB : ellipse_condition_B m n) :
  m = 16 / 25 ∧ n = 9 / 25 :=
sorry

end NUMINAMATH_GPT_standard_equation_of_ellipse_l942_94207


namespace NUMINAMATH_GPT_det_new_matrix_l942_94208

variables {a b c d : ℝ}

theorem det_new_matrix (h : a * d - b * c = 5) : (a - c) * d - (b - d) * c = 5 :=
by sorry

end NUMINAMATH_GPT_det_new_matrix_l942_94208


namespace NUMINAMATH_GPT_tic_tac_toe_board_configurations_l942_94258

theorem tic_tac_toe_board_configurations :
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  total_configurations = 592 :=
by 
  let sections := 4
  let horizontal_vertical_configurations := 6 * 18
  let diagonal_configurations := 2 * 20
  let configurations_per_section := horizontal_vertical_configurations + diagonal_configurations
  let total_configurations := sections * configurations_per_section
  sorry

end NUMINAMATH_GPT_tic_tac_toe_board_configurations_l942_94258


namespace NUMINAMATH_GPT_ratio_of_inscribed_to_circumscribed_l942_94241

theorem ratio_of_inscribed_to_circumscribed (a : ℝ) :
  let r' := a * Real.sqrt 6 / 12
  let R' := a * Real.sqrt 6 / 4
  r' / R' = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_inscribed_to_circumscribed_l942_94241


namespace NUMINAMATH_GPT_solution_set_for_fractional_inequality_l942_94266

theorem solution_set_for_fractional_inequality :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_for_fractional_inequality_l942_94266


namespace NUMINAMATH_GPT_pencil_cost_l942_94218

-- Definitions of given conditions
def has_amount : ℝ := 5.00  -- Elizabeth has 5 dollars
def borrowed_amount : ℝ := 0.53  -- She borrowed 53 cents
def needed_amount : ℝ := 0.47  -- She needs 47 cents more

-- Theorem to prove the cost of the pencil
theorem pencil_cost : has_amount + borrowed_amount + needed_amount = 6.00 := by 
  sorry

end NUMINAMATH_GPT_pencil_cost_l942_94218


namespace NUMINAMATH_GPT_min_value_expression_l942_94212

theorem min_value_expression (a d b c : ℝ) (habd : a ≥ 0 ∧ d ≥ 0) (hbc : b > 0 ∧ c > 0) (h_cond : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l942_94212


namespace NUMINAMATH_GPT_max_value_of_expression_l942_94202

theorem max_value_of_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) :
  (∃ (x : ℝ), x = (ab/(a + b)) + (ac/(a + c)) + (bc/(b + c)) ∧ x = 9/4) :=
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l942_94202


namespace NUMINAMATH_GPT_hexagon_coloring_l942_94236

def hex_colorings : ℕ := 2

theorem hexagon_coloring :
  ∃ c : ℕ, c = hex_colorings := by
  sorry

end NUMINAMATH_GPT_hexagon_coloring_l942_94236


namespace NUMINAMATH_GPT_DanGreenMarbles_l942_94262

theorem DanGreenMarbles : 
  ∀ (initial_green marbles_taken remaining_green : ℕ), 
  initial_green = 32 →
  marbles_taken = 23 →
  remaining_green = initial_green - marbles_taken →
  remaining_green = 9 :=
by sorry

end NUMINAMATH_GPT_DanGreenMarbles_l942_94262


namespace NUMINAMATH_GPT_class_average_gpa_l942_94290

theorem class_average_gpa (n : ℕ) (hn : 0 < n) :
  ((1/3 * n) * 45 + (2/3 * n) * 60) / n = 55 :=
by
  sorry

end NUMINAMATH_GPT_class_average_gpa_l942_94290


namespace NUMINAMATH_GPT_arcsin_one_half_l942_94278

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_one_half_l942_94278


namespace NUMINAMATH_GPT_merchant_profit_percentage_l942_94297

noncomputable def cost_price : ℝ := 100
noncomputable def marked_up_price : ℝ := cost_price + (0.75 * cost_price)
noncomputable def discount : ℝ := 0.30 * marked_up_price
noncomputable def selling_price : ℝ := marked_up_price - discount
noncomputable def profit : ℝ := selling_price - cost_price
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem merchant_profit_percentage :
  profit_percentage = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_merchant_profit_percentage_l942_94297


namespace NUMINAMATH_GPT_line_perpendicular_to_two_planes_parallel_l942_94210

-- Declare lines and planes
variables {Line Plane : Type}

-- Define the perpendicular and parallel relationships
variables (perpendicular : Line → Plane → Prop)
variables (parallel : Plane → Plane → Prop)

-- Given conditions
variables (m n : Line) (α β : Plane)
-- The known conditions are:
-- 1. m is perpendicular to α
-- 2. m is perpendicular to β
-- We want to prove:
-- 3. α is parallel to β

theorem line_perpendicular_to_two_planes_parallel (h1 : perpendicular m α) (h2 : perpendicular m β) : parallel α β :=
sorry

end NUMINAMATH_GPT_line_perpendicular_to_two_planes_parallel_l942_94210


namespace NUMINAMATH_GPT_evening_water_usage_is_6_l942_94216

-- Define the conditions: daily water usage and total water usage over 5 days.
def daily_water_usage (E : ℕ) : ℕ := 4 + E
def total_water_usage (E : ℕ) (days : ℕ) : ℕ := days * daily_water_usage E

-- Define the condition that over 5 days the total water usage is 50 liters.
axiom water_usage_condition : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6

-- Conjecture stating the amount of water used in the evening.
theorem evening_water_usage_is_6 : ∀ (E : ℕ), total_water_usage E 5 = 50 → E = 6 :=
by
  intro E
  intro h
  exact water_usage_condition E h

end NUMINAMATH_GPT_evening_water_usage_is_6_l942_94216


namespace NUMINAMATH_GPT_Sophie_donuts_problem_l942_94237

noncomputable def total_cost_before_discount (cost_per_box : ℝ) (num_boxes : ℕ) : ℝ :=
  cost_per_box * num_boxes

noncomputable def discount_amount (total_cost : ℝ) (discount_rate : ℝ) : ℝ :=
  total_cost * discount_rate

noncomputable def total_cost_after_discount (total_cost : ℝ) (discount : ℝ) : ℝ :=
  total_cost - discount

noncomputable def total_donuts (donuts_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  donuts_per_box * num_boxes

noncomputable def donuts_left (total_donuts : ℕ) (donuts_given_away : ℕ) : ℕ :=
  total_donuts - donuts_given_away

theorem Sophie_donuts_problem
  (budget : ℝ)
  (cost_per_box : ℝ)
  (discount_rate : ℝ)
  (num_boxes : ℕ)
  (donuts_per_box : ℕ)
  (donuts_given_to_mom : ℕ)
  (donuts_given_to_sister : ℕ)
  (half_dozen : ℕ) :
  budget = 50 →
  cost_per_box = 12 →
  discount_rate = 0.10 →
  num_boxes = 4 →
  donuts_per_box = 12 →
  donuts_given_to_mom = 12 →
  donuts_given_to_sister = 6 →
  half_dozen = 6 →
  total_cost_after_discount (total_cost_before_discount cost_per_box num_boxes) (discount_amount (total_cost_before_discount cost_per_box num_boxes) discount_rate) = 43.2 ∧
  donuts_left (total_donuts donuts_per_box num_boxes) (donuts_given_to_mom + donuts_given_to_sister) = 30 :=
by
  sorry

end NUMINAMATH_GPT_Sophie_donuts_problem_l942_94237


namespace NUMINAMATH_GPT_length_of_AB_l942_94292

theorem length_of_AB 
  (P Q A B : ℝ)
  (h_P_on_AB : P > 0 ∧ P < B)
  (h_Q_on_AB : Q > P ∧ Q < B)
  (h_ratio_P : P = 3 / 7 * B)
  (h_ratio_Q : Q = 4 / 9 * B)
  (h_PQ : Q - P = 3) 
: B = 189 := 
sorry

end NUMINAMATH_GPT_length_of_AB_l942_94292


namespace NUMINAMATH_GPT_value_of_k_l942_94284

theorem value_of_k (k : ℝ) : (2 - k * 2 = -4 * (-1)) → k = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_value_of_k_l942_94284


namespace NUMINAMATH_GPT_jessica_speed_last_40_l942_94225

theorem jessica_speed_last_40 
  (total_distance : ℕ)
  (total_time_min : ℕ)
  (first_segment_avg_speed : ℕ)
  (second_segment_avg_speed : ℕ)
  (last_segment_avg_speed : ℕ) :
  total_distance = 120 →
  total_time_min = 120 →
  first_segment_avg_speed = 50 →
  second_segment_avg_speed = 60 →
  last_segment_avg_speed = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_jessica_speed_last_40_l942_94225


namespace NUMINAMATH_GPT_number_of_ways_to_form_committee_with_president_l942_94203

open Nat

def number_of_ways_to_choose_members (total_members : ℕ) (committee_size : ℕ) (president_required : Bool) : ℕ :=
  if president_required then choose (total_members - 1) (committee_size - 1) else choose total_members committee_size

theorem number_of_ways_to_form_committee_with_president :
  number_of_ways_to_choose_members 30 5 true = 23741 :=
by
  -- Given that total_members = 30, committee_size = 5, and president_required = true,
  -- we need to show that the number of ways to choose the remaining members is 23741.
  sorry

end NUMINAMATH_GPT_number_of_ways_to_form_committee_with_president_l942_94203


namespace NUMINAMATH_GPT_central_vs_northern_chess_match_l942_94200

noncomputable def schedule_chess_match : Nat :=
  let players_team1 := ["A", "B", "C"];
  let players_team2 := ["X", "Y", "Z"];
  let total_games := 3 * 3 * 3;
  let games_per_round := 4;
  let total_rounds := 7;
  Nat.factorial total_rounds

theorem central_vs_northern_chess_match :
    schedule_chess_match = 5040 :=
by
  sorry

end NUMINAMATH_GPT_central_vs_northern_chess_match_l942_94200


namespace NUMINAMATH_GPT_discount_difference_l942_94263

theorem discount_difference (bill_amt : ℝ) (d1 : ℝ) (d2 : ℝ) (d3 : ℝ) :
  bill_amt = 12000 → d1 = 0.42 → d2 = 0.35 → d3 = 0.05 →
  (bill_amt * (1 - d2) * (1 - d3) - bill_amt * (1 - d1) = 450) :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_discount_difference_l942_94263


namespace NUMINAMATH_GPT_value_of_expression_l942_94279

variable (a b c : ℝ)

theorem value_of_expression (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1)
                            (h2 : abc = 1)
                            (h3 : a^2 + b^2 + c^2 - ((1 / (a^2)) + (1 / (b^2)) + (1 / (c^2))) = 8 * (a + b + c) - 8 * (ab + bc + ca)) :
                            (1 / (a - 1)) + (1 / (b - 1)) + (1 / (c - 1)) = -3/2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l942_94279


namespace NUMINAMATH_GPT_sparkling_water_cost_l942_94273

theorem sparkling_water_cost
  (drinks_per_day : ℚ := 1 / 5)
  (bottle_cost : ℝ := 2.00)
  (days_in_year : ℤ := 365) :
  (drinks_per_day * days_in_year) * bottle_cost = 146 :=
by
  sorry

end NUMINAMATH_GPT_sparkling_water_cost_l942_94273


namespace NUMINAMATH_GPT_total_money_from_tshirts_l942_94294

def num_tshirts_sold := 20
def money_per_tshirt := 215

theorem total_money_from_tshirts :
  num_tshirts_sold * money_per_tshirt = 4300 :=
by
  sorry

end NUMINAMATH_GPT_total_money_from_tshirts_l942_94294


namespace NUMINAMATH_GPT_solve_for_z_l942_94249

theorem solve_for_z (z : ℂ) : ((1 - I) ^ 2) * z = 3 + 2 * I → z = -1 + (3 / 2) * I :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_z_l942_94249


namespace NUMINAMATH_GPT_solve_for_x_l942_94281

theorem solve_for_x : ∀ (x : ℕ), (1000 = 10^3) → (40 = 2^3 * 5) → 1000^5 = 40^x → x = 15 :=
by
  intros x h1 h2 h3
  sorry

end NUMINAMATH_GPT_solve_for_x_l942_94281


namespace NUMINAMATH_GPT_real_roots_for_all_K_l942_94247

theorem real_roots_for_all_K (K : ℝ) : 
  ∃ x : ℝ, x = K^2 * (x-1) * (x-2) + 2 * x :=
sorry

end NUMINAMATH_GPT_real_roots_for_all_K_l942_94247


namespace NUMINAMATH_GPT_sum_of_areas_of_circles_l942_94257

-- Definitions and given conditions
variables (r s t : ℝ)
variables (h1 : r + s = 5)
variables (h2 : r + t = 12)
variables (h3 : s + t = 13)

-- The sum of the areas
theorem sum_of_areas_of_circles : 
  π * r^2 + π * s^2 + π * t^2 = 113 * π :=
  by
    sorry

end NUMINAMATH_GPT_sum_of_areas_of_circles_l942_94257


namespace NUMINAMATH_GPT_vanya_faster_speed_l942_94280

theorem vanya_faster_speed (v : ℝ) (h : v + 2 = 2.5 * v) : (v + 4) / v = 4 := by
  sorry

end NUMINAMATH_GPT_vanya_faster_speed_l942_94280


namespace NUMINAMATH_GPT_min_value_of_n_l942_94288

theorem min_value_of_n :
  ∀ (h : ℝ), ∃ n : ℝ, (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → -x^2 + 2 * h * x - h ≤ n) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ -x^2 + 2 * h * x - h = n) ∧
  n = -1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_n_l942_94288


namespace NUMINAMATH_GPT_smallest_c_in_progressions_l942_94223

def is_arithmetic_progression (a b c : ℤ) : Prop := b - a = c - b

def is_geometric_progression (b c a : ℤ) : Prop := c^2 = a*b

theorem smallest_c_in_progressions :
  ∃ (a b c : ℤ), is_arithmetic_progression a b c ∧ is_geometric_progression b c a ∧ 
  (∀ (a' b' c' : ℤ), is_arithmetic_progression a' b' c' ∧ is_geometric_progression b' c' a' → c ≤ c') ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_c_in_progressions_l942_94223


namespace NUMINAMATH_GPT_thousandths_place_digit_of_7_div_32_l942_94252

noncomputable def decimal_thousandths_digit : ℚ := 7 / 32

theorem thousandths_place_digit_of_7_div_32 :
  (decimal_thousandths_digit * 1000) % 10 = 8 :=
sorry

end NUMINAMATH_GPT_thousandths_place_digit_of_7_div_32_l942_94252


namespace NUMINAMATH_GPT_camilla_blueberry_jelly_beans_l942_94219

theorem camilla_blueberry_jelly_beans (b c : ℕ) (h1 : b = 2 * c) (h2 : b - 10 = 3 * (c - 10)) : b = 40 := 
sorry

end NUMINAMATH_GPT_camilla_blueberry_jelly_beans_l942_94219


namespace NUMINAMATH_GPT_total_points_l942_94245

noncomputable def Noa_score : ℕ := 30
noncomputable def Phillip_score : ℕ := 2 * Noa_score
noncomputable def Lucy_score : ℕ := (3 / 2) * Phillip_score

theorem total_points : 
  Noa_score + Phillip_score + Lucy_score = 180 := 
by
  sorry

end NUMINAMATH_GPT_total_points_l942_94245


namespace NUMINAMATH_GPT_initial_worth_of_wears_l942_94295

theorem initial_worth_of_wears (W : ℝ) 
  (h1 : W + 2/5 * W = 1.4 * W)
  (h2 : 0.85 * (W + 2/5 * W) = W + 95) : 
  W = 500 := 
by 
  sorry

end NUMINAMATH_GPT_initial_worth_of_wears_l942_94295


namespace NUMINAMATH_GPT_cycling_problem_l942_94269

theorem cycling_problem (x : ℝ) (h₀ : x > 0) :
  30 / x - 30 / (x + 3) = 2 / 3 :=
sorry

end NUMINAMATH_GPT_cycling_problem_l942_94269


namespace NUMINAMATH_GPT_at_least_half_team_B_can_serve_l942_94211

theorem at_least_half_team_B_can_serve (height_limit : ℕ)
    (avg_A : ℕ) (median_B : ℕ) (tallest_C : ℕ) (mode_D : ℕ)
    (H1 : height_limit = 168)
    (H2 : avg_A = 166)
    (H3 : median_B = 167)
    (H4 : tallest_C = 169)
    (H5 : mode_D = 167) :
    ∃ (eligible_B : ℕ → Prop), (∀ n, eligible_B n → n ≤ height_limit) ∧ (∃ k, k ≤ median_B ∧ eligible_B k ∧ k ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_at_least_half_team_B_can_serve_l942_94211


namespace NUMINAMATH_GPT_lollipops_remainder_l942_94250

theorem lollipops_remainder :
  let total_lollipops := 8362
  let lollipops_per_package := 12
  total_lollipops % lollipops_per_package = 10 :=
by
  let total_lollipops := 8362
  let lollipops_per_package := 12
  sorry

end NUMINAMATH_GPT_lollipops_remainder_l942_94250


namespace NUMINAMATH_GPT_combined_value_l942_94259

theorem combined_value (a b : ℝ) (h1 : 0.005 * a = 95 / 100) (h2 : b = 3 * a - 50) : a + b = 710 := by
  sorry

end NUMINAMATH_GPT_combined_value_l942_94259


namespace NUMINAMATH_GPT_least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l942_94253

noncomputable def sum_of_cubes (n : ℕ) : ℕ :=
  (n * (n + 1)/2)^2

noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem least_m_for_sum_of_cubes_is_perfect_cube 
  (h : ∃ m : ℕ, ∀ (a : ℕ), (sum_of_cubes (2*m+1) = a^3) → a = 6):
  m = 1 := sorry

theorem least_k_for_sum_of_squares_is_perfect_square 
  (h : ∃ k : ℕ, ∀ (b : ℕ), (sum_of_squares (2*k+1) = b^2) → b = 77):
  k = 5 := sorry

end NUMINAMATH_GPT_least_m_for_sum_of_cubes_is_perfect_cube_least_k_for_sum_of_squares_is_perfect_square_l942_94253


namespace NUMINAMATH_GPT_find_right_triangle_conditions_l942_94293

def is_right_triangle (A B C : ℝ) : Prop := 
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem find_right_triangle_conditions (A B C : ℝ):
  (A + B = C ∧ is_right_triangle A B C) ∨ 
  (A = B ∧ B = 2 * C ∧ is_right_triangle A B C) ∨ 
  (A / 30 = 1 ∧ B / 30 = 2 ∧ C / 30 = 3 ∧ is_right_triangle A B C) :=
sorry

end NUMINAMATH_GPT_find_right_triangle_conditions_l942_94293


namespace NUMINAMATH_GPT_rank_friends_l942_94251

variable (Amy Bill Celine : Prop)

-- Statement definitions
def statement_I := Bill
def statement_II := ¬Amy
def statement_III := ¬Celine

-- Exactly one of the statements is true
def exactly_one_true (s1 s2 s3 : Prop) :=
  (s1 ∧ ¬s2 ∧ ¬s3) ∨ (¬s1 ∧ s2 ∧ ¬s3) ∨ (¬s1 ∧ ¬s2 ∧ s3)

theorem rank_friends (h : exactly_one_true (statement_I Bill) (statement_II Amy) (statement_III Celine)) :
  (Amy ∧ ¬Bill ∧ Celine) :=
sorry

end NUMINAMATH_GPT_rank_friends_l942_94251


namespace NUMINAMATH_GPT_mystery_book_shelves_l942_94274

-- Define the conditions from the problem
def total_books : ℕ := 72
def picture_book_shelves : ℕ := 2
def books_per_shelf : ℕ := 9

-- Determine the number of mystery book shelves
theorem mystery_book_shelves : 
  let books_on_picture_shelves := picture_book_shelves * books_per_shelf
  let mystery_books := total_books - books_on_picture_shelves
  let mystery_shelves := mystery_books / books_per_shelf
  mystery_shelves = 6 :=
by {
  -- This space is intentionally left incomplete, as the proof itself is not required.
  sorry
}

end NUMINAMATH_GPT_mystery_book_shelves_l942_94274


namespace NUMINAMATH_GPT_twenty_is_80_percent_of_what_number_l942_94254

theorem twenty_is_80_percent_of_what_number : ∃ y : ℕ, (20 : ℚ) / y = 4 / 5 ∧ y = 25 := by
  sorry

end NUMINAMATH_GPT_twenty_is_80_percent_of_what_number_l942_94254


namespace NUMINAMATH_GPT_value_of_f2_l942_94204

noncomputable def f : ℕ → ℕ :=
  sorry

axiom f_condition : ∀ x : ℕ, f (x + 1) = 2 * x + 3

theorem value_of_f2 : f 2 = 5 :=
by sorry

end NUMINAMATH_GPT_value_of_f2_l942_94204


namespace NUMINAMATH_GPT_petya_wins_if_and_only_if_m_ne_n_l942_94232

theorem petya_wins_if_and_only_if_m_ne_n 
  (m n : ℕ) 
  (game : ∀ m n : ℕ, Prop)
  (win_condition : (game m n ↔ m ≠ n)) : 
  Prop := 
by 
  sorry

end NUMINAMATH_GPT_petya_wins_if_and_only_if_m_ne_n_l942_94232


namespace NUMINAMATH_GPT_total_balls_l942_94275

def num_white : ℕ := 50
def num_green : ℕ := 30
def num_yellow : ℕ := 10
def num_red : ℕ := 7
def num_purple : ℕ := 3

def prob_neither_red_nor_purple : ℝ := 0.9

theorem total_balls (T : ℕ) 
  (h : prob_red_purple = 1 - prob_neither_red_nor_purple) 
  (h_prob : prob_red_purple = (num_red + num_purple : ℝ) / (T : ℝ)) :
  T = 100 :=
by sorry

end NUMINAMATH_GPT_total_balls_l942_94275


namespace NUMINAMATH_GPT_total_odd_green_red_marbles_l942_94231

def Sara_green : ℕ := 3
def Sara_red : ℕ := 5
def Tom_green : ℕ := 4
def Tom_red : ℕ := 7
def Lisa_green : ℕ := 5
def Lisa_red : ℕ := 3

theorem total_odd_green_red_marbles : 
  (if Sara_green % 2 = 1 then Sara_green else 0) +
  (if Sara_red % 2 = 1 then Sara_red else 0) +
  (if Tom_green % 2 = 1 then Tom_green else 0) +
  (if Tom_red % 2 = 1 then Tom_red else 0) +
  (if Lisa_green % 2 = 1 then Lisa_green else 0) +
  (if Lisa_red % 2 = 1 then Lisa_red else 0) = 23 := by
  sorry

end NUMINAMATH_GPT_total_odd_green_red_marbles_l942_94231


namespace NUMINAMATH_GPT_number_of_students_absent_l942_94298

def classes := 18
def students_per_class := 28
def students_present := 496
def students_absent := (classes * students_per_class) - students_present

theorem number_of_students_absent : students_absent = 8 := 
by
  sorry

end NUMINAMATH_GPT_number_of_students_absent_l942_94298


namespace NUMINAMATH_GPT_find_x_value_l942_94296

theorem find_x_value (a b c x y z : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : xy / (x + y) = a) (h5 : xz / (x + z) = b) (h6 : yz / (y + z) = c)
  (h7 : x + y + z = abc) : 
  x = (2 * a * b * c) / (a * b + b * c + a * c) :=
sorry

end NUMINAMATH_GPT_find_x_value_l942_94296


namespace NUMINAMATH_GPT_smallest_n_property_l942_94229

noncomputable def smallest_n : ℕ := 13

theorem smallest_n_property :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (x * y * z ∣ (x + y + z) ^ smallest_n) :=
by
  intros x y z hx hy hz hxy hyz hzx
  use smallest_n
  sorry

end NUMINAMATH_GPT_smallest_n_property_l942_94229


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_l942_94264

def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else if n = 2 then 4 else 2 + (n - 1) * 2

theorem arithmetic_sequence_a4 :
  a 4 = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_a4_l942_94264


namespace NUMINAMATH_GPT_value_of_a_minus_b_l942_94287

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 2) (h3 : |a + b| = a + b) :
  a - b = 2 ∨ a - b = 6 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l942_94287


namespace NUMINAMATH_GPT_inverse_of_97_mod_98_l942_94240

theorem inverse_of_97_mod_98 : 97 * 97 ≡ 1 [MOD 98] :=
by
  sorry

end NUMINAMATH_GPT_inverse_of_97_mod_98_l942_94240


namespace NUMINAMATH_GPT_perpendicular_lines_sum_is_minus_four_l942_94201

theorem perpendicular_lines_sum_is_minus_four 
  (a b c : ℝ) 
  (h1 : (a * 2) / (4 * 5) = 1)
  (h2 : 10 * 1 + 4 * c - 2 = 0)
  (h3 : 2 * 1 - 5 * (-2) + b = 0) : 
  a + b + c = -4 := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_sum_is_minus_four_l942_94201


namespace NUMINAMATH_GPT_Frank_can_buy_7_candies_l942_94234

def tickets_whack_a_mole := 33
def tickets_skee_ball := 9
def cost_per_candy := 6

def total_tickets := tickets_whack_a_mole + tickets_skee_ball

theorem Frank_can_buy_7_candies : total_tickets / cost_per_candy = 7 := by
  sorry

end NUMINAMATH_GPT_Frank_can_buy_7_candies_l942_94234


namespace NUMINAMATH_GPT_total_video_hours_in_june_l942_94227

-- Definitions for conditions
def upload_rate_first_half : ℕ := 10 -- one-hour videos per day
def upload_rate_second_half : ℕ := 20 -- doubled one-hour videos per day
def days_in_half_month : ℕ := 15
def total_days_in_june : ℕ := 30

-- Number of video hours uploaded in the first half of the month
def video_hours_first_half : ℕ := upload_rate_first_half * days_in_half_month

-- Number of video hours uploaded in the second half of the month
def video_hours_second_half : ℕ := upload_rate_second_half * days_in_half_month

-- Total number of video hours in June
theorem total_video_hours_in_june : video_hours_first_half + video_hours_second_half = 450 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_video_hours_in_june_l942_94227


namespace NUMINAMATH_GPT_skateboard_price_after_discounts_l942_94272

-- Defining all necessary conditions based on the given problem.
def original_price : ℝ := 150
def discount1 : ℝ := 0.40 * original_price
def price_after_discount1 : ℝ := original_price - discount1
def discount2 : ℝ := 0.25 * price_after_discount1
def final_price : ℝ := price_after_discount1 - discount2

-- Goal: Prove that the final price after both discounts is $67.50.
theorem skateboard_price_after_discounts : final_price = 67.50 := by
  sorry

end NUMINAMATH_GPT_skateboard_price_after_discounts_l942_94272


namespace NUMINAMATH_GPT_bob_repay_l942_94285

theorem bob_repay {x : ℕ} (h : 50 + 10 * x >= 150) : x >= 10 :=
by
  sorry

end NUMINAMATH_GPT_bob_repay_l942_94285


namespace NUMINAMATH_GPT_simplify_sqrt_neg_five_squared_l942_94242

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_neg_five_squared_l942_94242


namespace NUMINAMATH_GPT_num_integers_satisfying_inequality_l942_94244

theorem num_integers_satisfying_inequality (n : ℤ) (h : n ≠ 0) : (1 / |(n:ℤ)| ≥ 1 / 5) → (number_of_satisfying_integers = 10) :=
by
  sorry

end NUMINAMATH_GPT_num_integers_satisfying_inequality_l942_94244


namespace NUMINAMATH_GPT_minimum_number_of_girls_l942_94209

theorem minimum_number_of_girls (total_students : ℕ) (d : ℕ) 
  (h_students : total_students = 20) 
  (h_unique_lists : ∀ n : ℕ, ∃! k : ℕ, k ≤ 20 - d ∧ n = 2 * k) :
  d ≥ 6 :=
sorry

end NUMINAMATH_GPT_minimum_number_of_girls_l942_94209
