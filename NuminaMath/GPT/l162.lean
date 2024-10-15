import Mathlib

namespace NUMINAMATH_GPT_tiles_needed_l162_16292

theorem tiles_needed (S : ℕ) (n : ℕ) (k : ℕ) (N : ℕ) (H1 : S = 18144) 
  (H2 : n * k^2 = S) (H3 : n = (N * (N + 1)) / 2) : n = 2016 := 
sorry

end NUMINAMATH_GPT_tiles_needed_l162_16292


namespace NUMINAMATH_GPT_sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l162_16228

theorem sqrt_of_16_eq_4 : Real.sqrt 16 = 4 := 
by sorry

theorem sqrt_of_364_eq_pm19 : Real.sqrt 364 = 19 ∨ Real.sqrt 364 = -19 := 
by sorry

theorem opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2 : -(2 - Real.sqrt 6) = Real.sqrt 6 - 2 := 
by sorry

end NUMINAMATH_GPT_sqrt_of_16_eq_4_sqrt_of_364_eq_pm19_opposite_of_2_sub_sqrt6_eq_sqrt6_sub_2_l162_16228


namespace NUMINAMATH_GPT_max_mn_l162_16259

theorem max_mn (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (m n : ℝ)
  (h₂ : 2 * m + n = 2) : m * n ≤ 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_max_mn_l162_16259


namespace NUMINAMATH_GPT_teamA_fraction_and_sum_l162_16238

def time_to_minutes (t : ℝ) : ℝ := t * 60

def fraction_teamA_worked (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) : Prop :=
  (90 - 60) / 150 = m / n

theorem teamA_fraction_and_sum (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_fraction : m = 1 ∧ n = 5) :
  90 / 150 = 1 / 5 → m + n = 6 :=
by
  sorry

end NUMINAMATH_GPT_teamA_fraction_and_sum_l162_16238


namespace NUMINAMATH_GPT_solution_set_f_neg_x_l162_16268

noncomputable def f (a b x : Real) : Real := (a * x - 1) * (x - b)

theorem solution_set_f_neg_x (a b : Real) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) : 
  ∀ x, f a b (-x) < 0 ↔ x < -3 ∨ x > 1 := 
by
  sorry

end NUMINAMATH_GPT_solution_set_f_neg_x_l162_16268


namespace NUMINAMATH_GPT_function_value_corresponds_to_multiple_independent_variables_l162_16298

theorem function_value_corresponds_to_multiple_independent_variables
  {α β : Type*} (f : α → β) :
  ∃ (b : β), ∃ (a1 a2 : α), a1 ≠ a2 ∧ f a1 = b ∧ f a2 = b :=
sorry

end NUMINAMATH_GPT_function_value_corresponds_to_multiple_independent_variables_l162_16298


namespace NUMINAMATH_GPT_inequality_abc_l162_16207

theorem inequality_abc (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + b*c + c*a :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l162_16207


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l162_16241

theorem common_difference_arithmetic_sequence
  (a : ℕ) (d : ℚ) (n : ℕ) (a_n : ℕ) (S_n : ℕ)
  (h1 : a = 2)
  (h2 : a_n = 20)
  (h3 : S_n = 132)
  (h4 : a_n = a + (n - 1) * d)
  (h5 : S_n = n * (a + a_n) / 2) :
  d = 18 / 11 := sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l162_16241


namespace NUMINAMATH_GPT_trapezoid_shorter_base_l162_16289

theorem trapezoid_shorter_base (a b : ℕ) (mid_segment : ℕ) (longer_base : ℕ) 
    (h1 : mid_segment = 5) (h2 : longer_base = 105) 
    (h3 : mid_segment = (longer_base - a) / 2) : 
  a = 95 := 
by
  sorry

end NUMINAMATH_GPT_trapezoid_shorter_base_l162_16289


namespace NUMINAMATH_GPT_distance_from_star_l162_16251

def speed_of_light : ℝ := 3 * 10^5 -- km/s
def time_years : ℝ := 4 -- years
def seconds_per_year : ℝ := 3 * 10^7 -- s

theorem distance_from_star :
  let distance := speed_of_light * (time_years * seconds_per_year)
  distance = 3.6 * 10^13 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_star_l162_16251


namespace NUMINAMATH_GPT_evaluate_expression_l162_16249

-- Definitions for a and b
def a : Int := 1
def b : Int := -1

theorem evaluate_expression : 
  5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b) + 1 = -17 := by
  -- Simplification steps skipped
  sorry

end NUMINAMATH_GPT_evaluate_expression_l162_16249


namespace NUMINAMATH_GPT_no_polynomial_exists_l162_16264

open Polynomial

theorem no_polynomial_exists (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) :
  ¬ ∃ (P : ℤ[X]), P.eval a = b ∧ P.eval b = c ∧ P.eval c = a :=
sorry

end NUMINAMATH_GPT_no_polynomial_exists_l162_16264


namespace NUMINAMATH_GPT_exists_rectangle_with_perimeter_divisible_by_4_l162_16297

-- Define the problem conditions in Lean
def square_length : ℕ := 2015

-- Define what it means to cut the square into rectangles with integer sides
def is_rectangle (a b : ℕ) := 1 ≤ a ∧ a ≤ square_length ∧ 1 ≤ b ∧ b ≤ square_length

-- Define the perimeter condition
def perimeter_divisible_by_4 (a b : ℕ) := (2 * a + 2 * b) % 4 = 0

-- Final theorem statement
theorem exists_rectangle_with_perimeter_divisible_by_4 :
  ∃ (a b : ℕ), is_rectangle a b ∧ perimeter_divisible_by_4 a b :=
by {
  sorry -- The proof itself will be filled in to establish the theorem
}

end NUMINAMATH_GPT_exists_rectangle_with_perimeter_divisible_by_4_l162_16297


namespace NUMINAMATH_GPT_alyssa_bought_224_new_cards_l162_16256

theorem alyssa_bought_224_new_cards
  (initial_cards : ℕ)
  (after_purchase_cards : ℕ)
  (h1 : initial_cards = 676)
  (h2 : after_purchase_cards = 900) :
  after_purchase_cards - initial_cards = 224 :=
by
  -- Placeholder to avoid proof since it's explicitly not required 
  sorry

end NUMINAMATH_GPT_alyssa_bought_224_new_cards_l162_16256


namespace NUMINAMATH_GPT_math_problem_proof_l162_16260

noncomputable def ellipse_equation : Prop := 
  let e := (Real.sqrt 2) / 2
  ∃ (a b : ℝ), 0 < a ∧ a > b ∧ e = (Real.sqrt 2) / 2 ∧ 
    (∀ x y, (x^2) / (a^2) + (y^2) / (b^2) = 1 ↔ x^2 / 2 + y^2 = 1)

noncomputable def fixed_point_exist : Prop :=
  let S := (0, 1/3) 
  ∀ k : ℝ, ∃ A B : ℝ × ℝ, 
    let M := (0, 1)
    ( 
        (A.1, A.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (B.1, B.2) ∈ {P : ℝ × ℝ | (P.1^2) / 2 + P.2^2 = 1} ∧ 
        (S.2 = k * S.1 - 1 / 3) ∧ 
        ((A.1 - M.1)^2 + (A.2 - M.2)^2) + ((B.1 - M.1)^2 + (B.2 - M.2)^2) = ((A.1 - B.1)^2 + (A.2 - M.2)^2) / 2)

theorem math_problem_proof : ellipse_equation ∧ fixed_point_exist := sorry

end NUMINAMATH_GPT_math_problem_proof_l162_16260


namespace NUMINAMATH_GPT_find_values_f_l162_16234

open Real

noncomputable def f (ω A x : ℝ) : ℝ := 2 * sin (ω * x) * cos (ω * x) + 2 * A * (cos (ω * x))^2 - A

theorem find_values_f (θ : ℝ) (A : ℝ) (ω : ℝ) (hA : A > 0) (hω : ω = 1)
  (h1 : π / 6 < θ) (h2 : θ < π / 3) (h3 : f ω A θ = 2 / 3) :
  f ω A (π / 3 - θ) = (1 + 2 * sqrt 6) / 3 :=
  sorry

end NUMINAMATH_GPT_find_values_f_l162_16234


namespace NUMINAMATH_GPT_no_such_functions_l162_16232

open Real

theorem no_such_functions : ¬ ∃ (f g : ℝ → ℝ), (∀ x y : ℝ, f (x^2 + g y) - f (x^2) + g (y) - g (x) ≤ 2 * y) ∧ (∀ x : ℝ, f (x) ≥ x^2) := by
  sorry

end NUMINAMATH_GPT_no_such_functions_l162_16232


namespace NUMINAMATH_GPT_probability_product_positive_is_5_div_9_l162_16261

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_product_positive_is_5_div_9_l162_16261


namespace NUMINAMATH_GPT_inequality_holds_l162_16294

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
    a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l162_16294


namespace NUMINAMATH_GPT_appropriate_sampling_method_is_stratified_l162_16208

-- Definition of the problem conditions
def total_students := 500 + 500
def male_students := 500
def female_students := 500
def survey_sample_size := 100

-- The goal is to show that given these conditions, the appropriate sampling method is Stratified sampling method.
theorem appropriate_sampling_method_is_stratified :
  total_students = 1000 ∧
  male_students = 500 ∧
  female_students = 500 ∧
  survey_sample_size = 100 →
  sampling_method = "Stratified" :=
by
  intros h
  sorry

end NUMINAMATH_GPT_appropriate_sampling_method_is_stratified_l162_16208


namespace NUMINAMATH_GPT_solve_for_x_l162_16221

theorem solve_for_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 := by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l162_16221


namespace NUMINAMATH_GPT_infinite_solutions_to_congruence_l162_16285

theorem infinite_solutions_to_congruence :
  ∃ᶠ n in atTop, 3^((n-2)^(n-1)-1) ≡ 1 [MOD 17 * n^2] :=
by
  sorry

end NUMINAMATH_GPT_infinite_solutions_to_congruence_l162_16285


namespace NUMINAMATH_GPT_area_ratio_of_octagon_l162_16273

theorem area_ratio_of_octagon (A : ℝ) (hA : 0 < A) :
  let triangle_ABJ_area := A / 8
  let triangle_ACE_area := A / 2
  triangle_ABJ_area / triangle_ACE_area = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_area_ratio_of_octagon_l162_16273


namespace NUMINAMATH_GPT_cost_per_use_correct_l162_16224

-- Definitions based on conditions in the problem
def total_cost : ℕ := 30
def uses_per_week : ℕ := 3
def number_of_weeks : ℕ := 2
def total_uses : ℕ := uses_per_week * number_of_weeks

-- Statement based on the question and correct answer
theorem cost_per_use_correct : (total_cost / total_uses) = 5 := sorry

end NUMINAMATH_GPT_cost_per_use_correct_l162_16224


namespace NUMINAMATH_GPT_reduce_entanglement_l162_16272

/- 
Define a graph structure and required operations as per the given conditions. 
-/
structure Graph (V : Type) :=
  (E : V -> V -> Prop)

def remove_odd_degree_verts (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph reduction logic

def duplicate_graph (G : Graph V) : Graph V :=
  sorry -- Placeholder for graph duplication logic

/--
  Prove that any graph where each vertex can be part of multiple entanglements 
  can be reduced to a state where no two vertices are connected using the given operations.
-/
theorem reduce_entanglement (G : Graph V) : ∃ G', 
  G' = remove_odd_degree_verts (duplicate_graph G) ∧
  (∀ (v1 v2 : V), ¬ G'.E v1 v2) :=
  by
  sorry

end NUMINAMATH_GPT_reduce_entanglement_l162_16272


namespace NUMINAMATH_GPT_number_of_white_balls_l162_16226

theorem number_of_white_balls (x : ℕ) (h1 : 3 + x ≠ 0) (h2 : (3 : ℚ) / (3 + x) = 1 / 5) : x = 12 :=
sorry

end NUMINAMATH_GPT_number_of_white_balls_l162_16226


namespace NUMINAMATH_GPT_no_solution_for_conditions_l162_16299

theorem no_solution_for_conditions :
  ∀ (x y : ℝ), 0 < x → 0 < y → x * y = 2^15 → (Real.log x / Real.log 2) * (Real.log y / Real.log 2) = 60 → False :=
by
  intro x y x_pos y_pos h1 h2
  sorry

end NUMINAMATH_GPT_no_solution_for_conditions_l162_16299


namespace NUMINAMATH_GPT_simplify_expr_l162_16253

def expr (y : ℝ) := y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8)

theorem simplify_expr (y : ℝ) : expr y = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l162_16253


namespace NUMINAMATH_GPT_bricks_per_course_l162_16203

theorem bricks_per_course : 
  ∃ B : ℕ, (let initial_courses := 3
            let additional_courses := 2
            let total_courses := initial_courses + additional_courses
            let last_course_half_removed := B / 2
            let total_bricks := B * total_courses - last_course_half_removed
            total_bricks = 1800) ↔ B = 400 :=
by {sorry}

end NUMINAMATH_GPT_bricks_per_course_l162_16203


namespace NUMINAMATH_GPT_total_hooligans_l162_16265

def hooligans_problem (X Y : ℕ) : Prop :=
  (X * Y = 365) ∧ (X + Y = 78 ∨ X + Y = 366)

theorem total_hooligans (X Y : ℕ) (h : hooligans_problem X Y) : X + Y = 78 ∨ X + Y = 366 :=
  sorry

end NUMINAMATH_GPT_total_hooligans_l162_16265


namespace NUMINAMATH_GPT_complex_magnitude_comparison_l162_16212

open Complex

theorem complex_magnitude_comparison :
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  abs z1 < abs z2 :=
by 
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  sorry

end NUMINAMATH_GPT_complex_magnitude_comparison_l162_16212


namespace NUMINAMATH_GPT_a_less_than_2_l162_16250

-- Define the quadratic function f(x)
noncomputable def f (x : ℝ) : ℝ := x^2 - 6 * x + 2

-- Define the condition that the inequality f(x) - a > 0 has solutions in the interval [0,5]
def inequality_holds (a : ℝ) : Prop := ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 5 → f x - a > 0

-- Theorem stating that a must be less than 2 to satisfy the above condition
theorem a_less_than_2 : ∀ (a : ℝ), (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 5 ∧ f x - a > 0) → a < 2 := 
sorry

end NUMINAMATH_GPT_a_less_than_2_l162_16250


namespace NUMINAMATH_GPT_otimes_calculation_l162_16216

def otimes (x y : ℝ) : ℝ := x^2 - 2*y

theorem otimes_calculation (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k :=
by
  sorry

end NUMINAMATH_GPT_otimes_calculation_l162_16216


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_l162_16233

variables {x y z t : ℝ}

-- Variables are positive
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom pos_z : 0 < z
axiom pos_t : 0 < t

-- Problem a)
theorem problem_a : x^4 * y^2 * z + y^4 * x^2 * z + y^4 * z^2 * x + z^4 * y^2 * x + x^4 * z^2 * y + z^4 * x^2 * y
  ≥ 2 * (x^3 * y^2 * z^2 + x^2 * y^3 * z^2 + x^2 * y^2 * z^3) :=
sorry

-- Problem b)
theorem problem_b : x^5 + y^5 + z^5 ≥ x^2 * y^2 * z + x^2 * y * z^2 + x * y^2 * z^2 :=
sorry

-- Problem c)
theorem problem_c : x^3 + y^3 + z^3 + t^3 ≥ x * y * z + x * y * t + x * z * t + y * z * t :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_l162_16233


namespace NUMINAMATH_GPT_product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l162_16205

section TriangularNumbers

-- Define triangular numbers
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

-- Statement 1: The product of two consecutive triangular numbers is not a perfect square
theorem product_of_consecutive_triangular_not_square (n : ℕ) (hn : n > 0) :
  ¬ ∃ m : ℕ, triangular (n - 1) * triangular n = m * m := by
  sorry

-- Statement 2: There exist infinitely many larger triangular numbers such that the product with t_n is a perfect square
theorem infinite_larger_triangular_numbers_square_product (n : ℕ) :
  ∃ᶠ m in at_top, ∃ k : ℕ, triangular n * triangular m = k * k := by
  sorry

end TriangularNumbers

end NUMINAMATH_GPT_product_of_consecutive_triangular_not_square_infinite_larger_triangular_numbers_square_product_l162_16205


namespace NUMINAMATH_GPT_commutative_binary_op_no_identity_element_associative_binary_op_l162_16288

def binary_op (x y : ℤ) : ℤ :=
  2 * (x + 2) * (y + 2) - 3

theorem commutative_binary_op (x y : ℤ) : binary_op x y = binary_op y x := by
  sorry

theorem no_identity_element (x e : ℤ) : ¬ (∀ x, binary_op x e = x) := by
  sorry

theorem associative_binary_op (x y z : ℤ) : (binary_op (binary_op x y) z = binary_op x (binary_op y z)) ∨ ¬ (binary_op (binary_op x y) z = binary_op x (binary_op y z)) := by
  sorry

end NUMINAMATH_GPT_commutative_binary_op_no_identity_element_associative_binary_op_l162_16288


namespace NUMINAMATH_GPT_midpoint_polar_coordinates_l162_16293

noncomputable def polar_midpoint :=
  let A := (10, 7 * Real.pi / 6)
  let B := (10, 11 * Real.pi / 6)
  let A_cartesian := (10 * Real.cos (7 * Real.pi / 6), 10 * Real.sin (7 * Real.pi / 6))
  let B_cartesian := (10 * Real.cos (11 * Real.pi / 6), 10 * Real.sin (11 * Real.pi / 6))
  let midpoint_cartesian := ((A_cartesian.1 + B_cartesian.1) / 2, (A_cartesian.2 + B_cartesian.2) / 2)
  let r := Real.sqrt (midpoint_cartesian.1 ^ 2 + midpoint_cartesian.2 ^ 2)
  let θ := if midpoint_cartesian.1 = 0 then 0 else Real.arctan (midpoint_cartesian.2 / midpoint_cartesian.1)
  (r, θ)

theorem midpoint_polar_coordinates :
  polar_midpoint = (5 * Real.sqrt 3, Real.pi) := by
  sorry

end NUMINAMATH_GPT_midpoint_polar_coordinates_l162_16293


namespace NUMINAMATH_GPT_wheel_stop_probability_l162_16286

theorem wheel_stop_probability 
  (pD pE pG pF : ℚ) 
  (h1 : pD = 1 / 4) 
  (h2 : pE = 1 / 3) 
  (h3 : pG = 1 / 6) 
  (h4 : pD + pE + pG + pF = 1) : 
  pF = 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_wheel_stop_probability_l162_16286


namespace NUMINAMATH_GPT_daily_profit_9080_l162_16231

theorem daily_profit_9080 (num_employees : Nat) (shirts_per_employee_per_day : Nat) (hours_per_shift : Nat) (wage_per_hour : Nat) (bonus_per_shirt : Nat) (shirt_sale_price : Nat) (nonemployee_expenses : Nat) :
  num_employees = 20 →
  shirts_per_employee_per_day = 20 →
  hours_per_shift = 8 →
  wage_per_hour = 12 →
  bonus_per_shirt = 5 →
  shirt_sale_price = 35 →
  nonemployee_expenses = 1000 →
  (num_employees * shirts_per_employee_per_day * shirt_sale_price) - ((num_employees * (hours_per_shift * wage_per_hour + shirts_per_employee_per_day * bonus_per_shirt)) + nonemployee_expenses) = 9080 := 
by
  intros
  sorry

end NUMINAMATH_GPT_daily_profit_9080_l162_16231


namespace NUMINAMATH_GPT_sum_of_coordinates_of_center_l162_16210

theorem sum_of_coordinates_of_center (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, -6)) (h2 : (x2, y2) = (-1, 4)) :
  let center_x := (x1 + x2) / 2
  let center_y := (y1 + y2) / 2
  center_x + center_y = 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_center_l162_16210


namespace NUMINAMATH_GPT_prime_1011_n_l162_16240

theorem prime_1011_n (n : ℕ) (h : n ≥ 2) : 
  n = 2 ∨ n = 3 ∨ (∀ m : ℕ, m ∣ (n^3 + n + 1) → m = 1 ∨ m = n^3 + n + 1) :=
by sorry

end NUMINAMATH_GPT_prime_1011_n_l162_16240


namespace NUMINAMATH_GPT_complement_union_correct_l162_16254

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4})
variable (hA : A = {0, 1, 2})
variable (hB : B = {2, 3})

theorem complement_union_correct :
  (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_correct_l162_16254


namespace NUMINAMATH_GPT_squares_in_rectangle_l162_16271

theorem squares_in_rectangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a ≤ 1) (h5 : b ≤ 1) (h6 : c ≤ 1) (h7 : a + b + c = 2)  : 
  a + b + c ≤ 2 := sorry

end NUMINAMATH_GPT_squares_in_rectangle_l162_16271


namespace NUMINAMATH_GPT_walking_distance_l162_16248

theorem walking_distance (a b : ℝ) (h1 : 10 * a + 45 * b = a * 15)
(h2 : x * (a + 9 * b) = 10 * a + 45 * b) : x = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_walking_distance_l162_16248


namespace NUMINAMATH_GPT_symmetric_point_about_origin_l162_16245

theorem symmetric_point_about_origin (P Q : ℤ × ℤ) (h : P = (-2, -3)) : Q = (2, 3) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_about_origin_l162_16245


namespace NUMINAMATH_GPT_naomi_regular_bikes_l162_16239
-- Import necessary libraries

-- Define the condition and the proof problem
theorem naomi_regular_bikes (R C : ℕ) (h1 : C = 11) 
  (h2 : 2 * R + 4 * C = 58) : R = 7 := 
  by 
  -- Include all necessary conditions as assumptions
  have hC : C = 11 := h1
  have htotal : 2 * R + 4 * C = 58 := h2
  -- Skip the proof itself
  sorry

end NUMINAMATH_GPT_naomi_regular_bikes_l162_16239


namespace NUMINAMATH_GPT_school_C_paintings_l162_16243

theorem school_C_paintings
  (A B C : ℕ)
  (h1 : B + C = 41)
  (h2 : A + C = 38)
  (h3 : A + B = 43) : 
  C = 18 :=
by
  sorry

end NUMINAMATH_GPT_school_C_paintings_l162_16243


namespace NUMINAMATH_GPT_sum_of_coeffs_l162_16230

theorem sum_of_coeffs 
  (a b c d e x : ℝ)
  (h : (729 * x ^ 3 + 8) = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 78 :=
sorry

end NUMINAMATH_GPT_sum_of_coeffs_l162_16230


namespace NUMINAMATH_GPT_land_tax_calculation_l162_16281

theorem land_tax_calculation
  (area : ℝ)
  (value_per_acre : ℝ)
  (tax_rate : ℝ)
  (total_cadastral_value : ℝ := area * value_per_acre)
  (land_tax : ℝ := total_cadastral_value * tax_rate) :
  area = 15 → value_per_acre = 100000 → tax_rate = 0.003 → land_tax = 4500 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_land_tax_calculation_l162_16281


namespace NUMINAMATH_GPT_bankers_discount_l162_16246

theorem bankers_discount {TD S BD : ℝ} (hTD : TD = 66) (hS : S = 429) :
  BD = (TD * S) / (S - TD) → BD = 78 :=
by
  intros h
  rw [hTD, hS] at h
  sorry

end NUMINAMATH_GPT_bankers_discount_l162_16246


namespace NUMINAMATH_GPT_volume_surface_ratio_l162_16242

-- Define the structure of the shape
structure Shape where
  center_cube : unit
  surrounding_cubes : Fin 6 -> unit
  top_cube : unit

-- Define the properties for the calculation
def volume (s : Shape) : ℕ := 8
def surface_area (s : Shape) : ℕ := 28
def ratio_volume_surface_area (s : Shape) : ℚ := volume s / surface_area s

-- Main theorem statement
theorem volume_surface_ratio (s : Shape) : ratio_volume_surface_area s = 2 / 7 := sorry

end NUMINAMATH_GPT_volume_surface_ratio_l162_16242


namespace NUMINAMATH_GPT_ratio_of_boys_l162_16296

theorem ratio_of_boys (p : ℚ) (h : p = (3/5) * (1 - p)) : p = 3 / 8 := by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_l162_16296


namespace NUMINAMATH_GPT_toaster_sales_promotion_l162_16283

theorem toaster_sales_promotion :
  ∀ (p : ℕ) (c₁ c₂ : ℕ) (k : ℕ), 
    (c₁ = 600 ∧ p = 15 ∧ k = p * c₁) ∧ 
    (c₂ = 450 ∧ (p * c₂ = k) ) ∧ 
    (p' = p * 11 / 10) →
    p' = 22 :=
by 
  sorry

end NUMINAMATH_GPT_toaster_sales_promotion_l162_16283


namespace NUMINAMATH_GPT_num_words_at_least_one_vowel_l162_16209

-- Definitions based on conditions.
def letters : List Char := ['A', 'B', 'E', 'G', 'H']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'G', 'H']

-- The main statement posing the question and answer.
theorem num_words_at_least_one_vowel :
  let total_words := (letters.length) ^ 5
  let consonant_words := (consonants.length) ^ 5
  let result := total_words - consonant_words
  result = 2882 :=
by {
  let total_words := 5 ^ 5
  let consonant_words := 3 ^ 5
  let result := total_words - consonant_words
  have : result = 2882 := by sorry
  exact this
}

end NUMINAMATH_GPT_num_words_at_least_one_vowel_l162_16209


namespace NUMINAMATH_GPT_fraction_value_l162_16287

variable (x y : ℝ)

theorem fraction_value (h : 1/x - 1/y = 3) : (2 * x + 3 * x * y - 2 * y) / (x - 2 * x * y - y) = 3 / 5 := 
by sorry

end NUMINAMATH_GPT_fraction_value_l162_16287


namespace NUMINAMATH_GPT_find_c_l162_16262

/-
Given:
1. c and d are integers.
2. x^2 - x - 1 is a factor of cx^{18} + dx^{17} + x^2 + 1.
Show that c = -1597 under these conditions.

Assume we have the following Fibonacci number definitions:
F_16 = 987,
F_17 = 1597,
F_18 = 2584,
then:
Proof that c = -1597.
-/

noncomputable def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

theorem find_c (c d : ℤ) (h1 : c * 2584 + d * 1597 + 1 = 0) (h2 : c * 1597 + d * 987 + 2 = 0) :
  c = -1597 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l162_16262


namespace NUMINAMATH_GPT_certain_number_value_l162_16275

theorem certain_number_value :
  ∃ n : ℚ, 9 - (4 / 6) = 7 + (n / 6) ∧ n = 8 := by
sorry

end NUMINAMATH_GPT_certain_number_value_l162_16275


namespace NUMINAMATH_GPT_trapezoid_ABCD_BCE_area_l162_16269

noncomputable def triangle_area (a b c : ℝ) (angle_abc : ℝ) : ℝ :=
  1 / 2 * a * b * Real.sin angle_abc

noncomputable def area_of_triangle_BCE (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ) : ℝ :=
  let ratio := AB / DC
  (ratio / (1 + ratio)) * area_triangle_DCB

theorem trapezoid_ABCD_BCE_area :
  ∀ (AB DC AD : ℝ) (angle_DAB : ℝ) (area_triangle_DCB : ℝ),
    AB = 30 →
    DC = 24 →
    AD = 3 →
    angle_DAB = Real.pi / 3 →
    area_triangle_DCB = 18 * Real.sqrt 3 →
    area_of_triangle_BCE AB DC AD angle_DAB area_triangle_DCB = 10 * Real.sqrt 3 := 
by
  intros
  sorry

end NUMINAMATH_GPT_trapezoid_ABCD_BCE_area_l162_16269


namespace NUMINAMATH_GPT_find_m_l162_16237

theorem find_m (S : ℕ → ℕ) (a : ℕ → ℕ) (m : ℕ) :
  (∀ n, S n = (n * (3 * n - 1)) / 2) →
  (a 1 = 1) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (a m = 3 * m - 2) →
  (a 4 * a 4 = a 1 * a m) →
  m = 34 :=
by
  intro hS h1 ha1 ha2 hgeom
  sorry

end NUMINAMATH_GPT_find_m_l162_16237


namespace NUMINAMATH_GPT_solve_abs_eq_l162_16263

theorem solve_abs_eq (x : ℝ) : (|x + 2| = 3*x - 6) → x = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l162_16263


namespace NUMINAMATH_GPT_banana_production_total_l162_16244

def banana_production (nearby_island_production : ℕ) (jakies_multiplier : ℕ) : ℕ :=
  nearby_island_production + (jakies_multiplier * nearby_island_production)

theorem banana_production_total
  (nearby_island_production : ℕ)
  (jakies_multiplier : ℕ)
  (h1 : nearby_island_production = 9000)
  (h2 : jakies_multiplier = 10)
  : banana_production nearby_island_production jakies_multiplier = 99000 :=
by
  sorry

end NUMINAMATH_GPT_banana_production_total_l162_16244


namespace NUMINAMATH_GPT_incorrect_statement_B_l162_16278

variable (a : Nat → Int) (S : Nat → Int)
variable (d : Int)

-- Given conditions
axiom S_5_lt_S_6 : S 5 < S 6
axiom S_6_eq_S_7 : S 6 = S 7
axiom S_7_gt_S_8 : S 7 > S 8
axiom S_n : ∀ n, S n = n * a n

-- Question to prove statement B is incorrect 
theorem incorrect_statement_B : ∃ (d : Int), (S 9 < S 5) :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_incorrect_statement_B_l162_16278


namespace NUMINAMATH_GPT_find_a1_for_geometric_sequence_l162_16236

noncomputable def geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : geometric_sequence) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

theorem find_a1_for_geometric_sequence (a : geometric_sequence)
  (h_geom : is_geometric_sequence a)
  (h1 : a 2 * a 5 = 2 * a 3)
  (h2 : (a 4 + a 6) / 2 = 5 / 4) :
  a 1 = 16 ∨ a 1 = -16 :=
sorry

end NUMINAMATH_GPT_find_a1_for_geometric_sequence_l162_16236


namespace NUMINAMATH_GPT_edge_length_of_divided_cube_l162_16255

theorem edge_length_of_divided_cube (volume_original_cube : ℕ) (num_divisions : ℕ) (volume_of_one_smaller_cube : ℕ) (edge_length : ℕ) :
  volume_original_cube = 1000 →
  num_divisions = 8 →
  volume_of_one_smaller_cube = volume_original_cube / num_divisions →
  volume_of_one_smaller_cube = edge_length ^ 3 →
  edge_length = 5 :=
by
  sorry

end NUMINAMATH_GPT_edge_length_of_divided_cube_l162_16255


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l162_16219

theorem trigonometric_identity_proof 
  (α : ℝ) 
  (h1 : Real.tan (2 * α) = 3 / 4) 
  (h2 : α ∈ Set.Ioo (-(Real.pi / 2)) (Real.pi / 2))
  (h3 : ∃ x : ℝ, (Real.sin (x + 2) + Real.sin (α - x) - 2 * Real.sin α) = 0) : 
  Real.cos (2 * α) = -4 / 5 ∧ Real.tan (α / 2) = (1 - Real.sqrt 10) / 3 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l162_16219


namespace NUMINAMATH_GPT_sam_and_erica_money_total_l162_16229

def sam_money : ℕ := 38
def erica_money : ℕ := 53

theorem sam_and_erica_money_total : sam_money + erica_money = 91 :=
by
  -- the proof is not required; hence we skip it
  sorry

end NUMINAMATH_GPT_sam_and_erica_money_total_l162_16229


namespace NUMINAMATH_GPT_reciprocals_of_each_other_l162_16258

theorem reciprocals_of_each_other (a b : ℝ) (h : (a + b)^2 - (a - b)^2 = 4) : a * b = 1 :=
by 
  sorry

end NUMINAMATH_GPT_reciprocals_of_each_other_l162_16258


namespace NUMINAMATH_GPT_sphere_volume_l162_16284

theorem sphere_volume (A : ℝ) (d : ℝ) (V : ℝ) : 
    (A = 2 * Real.pi) →  -- Cross-sectional area is 2π cm²
    (d = 1) →            -- Distance from center to cross-section is 1 cm
    (V = 4 * Real.sqrt 3 * Real.pi) :=  -- Volume of sphere is 4√3 π cm³
by 
  intros hA hd
  sorry

end NUMINAMATH_GPT_sphere_volume_l162_16284


namespace NUMINAMATH_GPT_tetrahedron_a_exists_tetrahedron_b_not_exists_l162_16267

/-- Part (a): There exists a tetrahedron with two edges shorter than 1 cm,
    and the other four edges longer than 1 km. -/
theorem tetrahedron_a_exists : 
  ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ 1000 < c ∧ 1000 < d ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) := 
sorry

/-- Part (b): There does not exist a tetrahedron with four edges shorter than 1 cm,
    and the other two edges longer than 1 km. -/
theorem tetrahedron_b_not_exists : 
  ¬ ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ) := 
sorry

end NUMINAMATH_GPT_tetrahedron_a_exists_tetrahedron_b_not_exists_l162_16267


namespace NUMINAMATH_GPT_positive_number_condition_l162_16291

theorem positive_number_condition (y : ℝ) (h: 0.04 * y = 16): y = 400 := 
by sorry

end NUMINAMATH_GPT_positive_number_condition_l162_16291


namespace NUMINAMATH_GPT_part1_part2_l162_16223

def f (x : ℝ) : ℝ := abs (x - 2)

theorem part1 (x : ℝ) : f x > 4 - abs (x + 1) ↔ x < -3 / 2 ∨ x > 5 / 2 := 
sorry

theorem part2 (a b : ℝ) (ha : 0 < a ∧ a < 1/2) (hb : 0 < b ∧ b < 1/2)
  (h : f (1 / a) + f (2 / b) = 10) : a + b / 2 ≥ 2 / 7 := 
sorry

end NUMINAMATH_GPT_part1_part2_l162_16223


namespace NUMINAMATH_GPT_volume_of_box_is_correct_l162_16220

def metallic_sheet_initial_length : ℕ := 48
def metallic_sheet_initial_width : ℕ := 36
def square_cut_side_length : ℕ := 8

def box_length : ℕ := metallic_sheet_initial_length - 2 * square_cut_side_length
def box_width : ℕ := metallic_sheet_initial_width - 2 * square_cut_side_length
def box_height : ℕ := square_cut_side_length

def box_volume : ℕ := box_length * box_width * box_height

theorem volume_of_box_is_correct : box_volume = 5120 := by
  sorry

end NUMINAMATH_GPT_volume_of_box_is_correct_l162_16220


namespace NUMINAMATH_GPT_magnitude_of_T_l162_16218

open Complex

noncomputable def i : ℂ := Complex.I

noncomputable def T : ℂ := (1 + i)^19 - (1 - i)^19

theorem magnitude_of_T : Complex.abs T = 1024 := by
  sorry

end NUMINAMATH_GPT_magnitude_of_T_l162_16218


namespace NUMINAMATH_GPT_fido_leash_problem_l162_16295

theorem fido_leash_problem
  (r : ℝ) 
  (octagon_area : ℝ := 2 * r^2 * Real.sqrt 2)
  (circle_area : ℝ := Real.pi * r^2)
  (explore_fraction : ℝ := circle_area / octagon_area)
  (a b : ℝ) 
  (h_simplest_form : explore_fraction = (Real.sqrt a) / b * Real.pi)
  (h_a : a = 2)
  (h_b : b = 2) : a * b = 4 :=
by sorry

end NUMINAMATH_GPT_fido_leash_problem_l162_16295


namespace NUMINAMATH_GPT_megawheel_seat_capacity_l162_16227

theorem megawheel_seat_capacity (seats people : ℕ) (h1 : seats = 15) (h2 : people = 75) : people / seats = 5 := by
  sorry

end NUMINAMATH_GPT_megawheel_seat_capacity_l162_16227


namespace NUMINAMATH_GPT_problem_proof_l162_16202

variable {a1 a2 b1 b2 b3 : ℝ}

theorem problem_proof 
  (h1 : ∃ d, -7 + d = a1 ∧ a1 + d = a2 ∧ a2 + d = -1)
  (h2 : ∃ r, -4 * r = b1 ∧ b1 * r = b2 ∧ b2 * r = b3 ∧ b3 * r = -1)
  (ha : a2 - a1 = 2)
  (hb : b2 = -2) :
  (a2 - a1) / b2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l162_16202


namespace NUMINAMATH_GPT_evaluate_expression_m_4_evaluate_expression_m_negative_4_l162_16225

variables (a b c d m : ℝ)

theorem evaluate_expression_m_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_4 : m = 4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = 35 :=
by sorry

theorem evaluate_expression_m_negative_4 (h1 : a + b = 0) (h2 : c * d = 1) (h3 : |m| = 4) (h_m_negative_4 : m = -4) :
  (a + b) / (3 * m) + m^2 - 5 * (c * d) + 6 * m = -13 :=
by sorry

end NUMINAMATH_GPT_evaluate_expression_m_4_evaluate_expression_m_negative_4_l162_16225


namespace NUMINAMATH_GPT_sum_of_a_and_b_l162_16211

theorem sum_of_a_and_b {a b : ℝ} (h : a^2 + b^2 + (a*b)^2 = 4*a*b - 1) : a + b = 2 ∨ a + b = -2 :=
sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l162_16211


namespace NUMINAMATH_GPT_smallest_tax_amount_is_professional_income_tax_l162_16274

def total_income : ℝ := 50000.00
def professional_deductions : ℝ := 35000.00

def tax_rate_ndfl : ℝ := 0.13
def tax_rate_simplified_income : ℝ := 0.06
def tax_rate_simplified_income_minus_expenditure : ℝ := 0.15
def tax_rate_professional_income : ℝ := 0.04

def ndfl_tax : ℝ := (total_income - professional_deductions) * tax_rate_ndfl
def simplified_tax_income : ℝ := total_income * tax_rate_simplified_income
def simplified_tax_income_minus_expenditure : ℝ := (total_income - professional_deductions) * tax_rate_simplified_income_minus_expenditure
def professional_income_tax : ℝ := total_income * tax_rate_professional_income

theorem smallest_tax_amount_is_professional_income_tax : 
  min (min ndfl_tax (min simplified_tax_income simplified_tax_income_minus_expenditure)) professional_income_tax = professional_income_tax := 
sorry

end NUMINAMATH_GPT_smallest_tax_amount_is_professional_income_tax_l162_16274


namespace NUMINAMATH_GPT_intersection_of_sets_l162_16213

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {0, 1, 2}) (hB : B = {1, 2, 3, 4}) :
  A ∩ B = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l162_16213


namespace NUMINAMATH_GPT_number_of_divisible_permutations_l162_16257

def digit_list := [1, 3, 1, 1, 5, 2, 1, 5, 2]
def count_permutations (d : List ℕ) (n : ℕ) : ℕ :=
  let fact := Nat.factorial
  let number := fact 8 / (fact 3 * fact 2 * fact 1)
  number

theorem number_of_divisible_permutations : count_permutations digit_list 2 = 3360 := by
  sorry

end NUMINAMATH_GPT_number_of_divisible_permutations_l162_16257


namespace NUMINAMATH_GPT_max_x5_l162_16201

theorem max_x5 (x1 x2 x3 x4 x5 : ℕ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h5 : 0 < x5) 
  (h : x1 + x2 + x3 + x4 + x5 ≤ x1 * x2 * x3 * x4 * x5) : x5 ≤ 5 :=
  sorry

end NUMINAMATH_GPT_max_x5_l162_16201


namespace NUMINAMATH_GPT_simplify_and_evaluate_l162_16279

theorem simplify_and_evaluate 
  (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l162_16279


namespace NUMINAMATH_GPT_find_a_solve_inequality_l162_16235

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem find_a (h : ∀ x : ℝ, f x a = -f (-x) a) : a = 1 := sorry

theorem solve_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : f x 1 > 3 := sorry

end NUMINAMATH_GPT_find_a_solve_inequality_l162_16235


namespace NUMINAMATH_GPT_larger_number_eq_1599_l162_16277

theorem larger_number_eq_1599 (L S : ℕ) (h1 : L - S = 1335) (h2 : L = 6 * S + 15) : L = 1599 :=
by 
  sorry

end NUMINAMATH_GPT_larger_number_eq_1599_l162_16277


namespace NUMINAMATH_GPT_jasmine_laps_per_afternoon_l162_16280

-- Defining the conditions
def swims_each_week (days_per_week : ℕ) := days_per_week = 5
def total_weeks := 5
def total_laps := 300

-- Main proof statement
theorem jasmine_laps_per_afternoon (d : ℕ) (l : ℕ) :
  swims_each_week d →
  total_weeks * d = 25 →
  total_laps = 300 →
  300 / 25 = l →
  l = 12 :=
by
  intros
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_jasmine_laps_per_afternoon_l162_16280


namespace NUMINAMATH_GPT_water_added_l162_16204

theorem water_added (initial_fullness : ℚ) (final_fullness : ℚ) (capacity : ℚ)
  (h1 : initial_fullness = 0.40) (h2 : final_fullness = 3 / 4) (h3 : capacity = 80) :
  (final_fullness * capacity - initial_fullness * capacity) = 28 := by
  sorry

end NUMINAMATH_GPT_water_added_l162_16204


namespace NUMINAMATH_GPT_six_digit_start_5_no_12_digit_perfect_square_l162_16276

theorem six_digit_start_5_no_12_digit_perfect_square :
  ∀ (n : ℕ), (500000 ≤ n ∧ n < 600000) → 
  (∀ (m : ℕ), n * 10^6 + m ≠ k^2) :=
by
  sorry

end NUMINAMATH_GPT_six_digit_start_5_no_12_digit_perfect_square_l162_16276


namespace NUMINAMATH_GPT_remainder_783245_div_7_l162_16200

theorem remainder_783245_div_7 :
  783245 % 7 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_783245_div_7_l162_16200


namespace NUMINAMATH_GPT_total_distance_hopped_l162_16222

def distance_hopped (rate: ℕ) (time: ℕ) : ℕ := rate * time

def spotted_rabbit_distance (time: ℕ) : ℕ :=
  let pattern := [8, 11, 16, 20, 9]
  let full_cycles := time / pattern.length
  let remaining_minutes := time % pattern.length
  let full_cycle_distance := full_cycles * pattern.sum
  let remaining_distance := (List.take remaining_minutes pattern).sum
  full_cycle_distance + remaining_distance

theorem total_distance_hopped :
  distance_hopped 15 12 + distance_hopped 12 12 + distance_hopped 18 12 + distance_hopped 10 12 + spotted_rabbit_distance 12 = 807 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_hopped_l162_16222


namespace NUMINAMATH_GPT_average_weight_of_a_and_b_l162_16266

-- Given conditions as Lean definitions
variable (A B C : ℝ)
variable (h1 : (A + B + C) / 3 = 45)
variable (h2 : (B + C) / 2 = 46)
variable (hB : B = 37)

-- The statement we want to prove
theorem average_weight_of_a_and_b : (A + B) / 2 = 40 := by
  sorry

end NUMINAMATH_GPT_average_weight_of_a_and_b_l162_16266


namespace NUMINAMATH_GPT_angle_ABC_is_83_degrees_l162_16270

theorem angle_ABC_is_83_degrees (A B C D K : Type)
  (angle_BAC : Real) (angle_CAD : Real) (angle_ACD : Real)
  (AB AC AD : Real) (angle_ABC : Real) :
  angle_BAC = 60 ∧ angle_CAD = 60 ∧ angle_ACD = 23 ∧ AB + AD = AC → 
  angle_ABC = 83 :=
by
  sorry

end NUMINAMATH_GPT_angle_ABC_is_83_degrees_l162_16270


namespace NUMINAMATH_GPT_hammerhead_teeth_fraction_l162_16282

theorem hammerhead_teeth_fraction (f : ℚ) : 
  let t := 180 
  let h := f * t
  let w := 2 * (t + h)
  w = 420 → f = (1 : ℚ) / 6 := by
  intros _ 
  sorry

end NUMINAMATH_GPT_hammerhead_teeth_fraction_l162_16282


namespace NUMINAMATH_GPT_unique_root_a_b_values_l162_16215

theorem unique_root_a_b_values {a b : ℝ} (h1 : ∀ x, x^2 + a * x + b = 0 ↔ x = 1) : a = -2 ∧ b = 1 := by
  sorry

end NUMINAMATH_GPT_unique_root_a_b_values_l162_16215


namespace NUMINAMATH_GPT_interest_rate_l162_16290

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t / 100

noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r/100)^t - P

theorem interest_rate (P t : ℝ) (diff : ℝ) (r : ℝ) (h : P = 1000) (t_eq : t = 4) 
  (diff_eq : diff = 64.10) : 
  compound_interest P r t - simple_interest P r t = diff → r = 10 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_l162_16290


namespace NUMINAMATH_GPT_moles_of_HCl_formed_l162_16252

-- Define the reaction
def balancedReaction (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

-- Number of moles given
def molesCH4 := 2
def molesCl2 := 4

-- Theorem statement
theorem moles_of_HCl_formed :
  ∀ CH4 Cl2 CH3Cl HCl : ℕ, balancedReaction CH4 Cl2 CH3Cl HCl →
  CH4 = molesCH4 →
  Cl2 = molesCl2 →
  HCl = 2 := sorry

end NUMINAMATH_GPT_moles_of_HCl_formed_l162_16252


namespace NUMINAMATH_GPT_addition_problem_l162_16214

theorem addition_problem (F I V N E : ℕ) (h1: F = 8) (h2: I % 2 = 0) 
  (h3: 1 ≤ F ∧ F ≤ 9) (h4: 1 ≤ I ∧ I ≤ 9) (h5: 1 ≤ V ∧ V ≤ 9) 
  (h6: 1 ≤ N ∧ N ≤ 9) (h7: 1 ≤ E ∧ E ≤ 9) 
  (h8: F ≠ I ∧ F ≠ V ∧ F ≠ N ∧ F ≠ E) 
  (h9: I ≠ V ∧ I ≠ N ∧ I ≠ E ∧ V ≠ N ∧ V ≠ E ∧ N ≠ E)
  (h10: 2 * F + 2 * I + 2 * V = 1000 * N + 100 * I + 10 * N + E):
  V = 5 :=
sorry

end NUMINAMATH_GPT_addition_problem_l162_16214


namespace NUMINAMATH_GPT_range_of_a_l162_16247

-- Define the function g(x) = x^3 - 3ax - a
def g (a x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x) which is g'(x) = 3x^2 - 3a
def g' (a x : ℝ) : ℝ := 3*x^2 - 3*a

theorem range_of_a (a : ℝ) : g a 0 * g a 1 < 0 → 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l162_16247


namespace NUMINAMATH_GPT_smallest_n_condition_l162_16206

theorem smallest_n_condition :
  ∃ n ≥ 2, ∃ (a : Fin n → ℤ), (Finset.sum Finset.univ a = 1990) ∧ (Finset.univ.prod a = 1990) ∧ (n = 5) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_condition_l162_16206


namespace NUMINAMATH_GPT_vector_magnitude_proof_l162_16217

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ := 
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude_proof
  (a b c : ℝ × ℝ)
  (h_a : a = (-2, 1))
  (h_b : b = (-2, 3))
  (h_c : ∃ m : ℝ, c = (m, -1) ∧ (m * b.1 + (-1) * b.2 = 0)) :
  vector_magnitude (a.1 - c.1, a.2 - c.2) = Real.sqrt 17 / 2 :=
by
  sorry

end NUMINAMATH_GPT_vector_magnitude_proof_l162_16217
