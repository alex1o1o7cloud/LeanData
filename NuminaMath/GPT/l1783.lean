import Mathlib

namespace NUMINAMATH_GPT_var_X_is_86_over_225_l1783_178360

/-- The probability of Person A hitting the target is 2/3. -/
def prob_A : ℚ := 2 / 3

/-- The probability of Person B hitting the target is 4/5. -/
def prob_B : ℚ := 4 / 5

/-- The events of A and B hitting or missing the target are independent. -/
def independent_events : Prop := true -- In Lean, independence would involve more complex definitions.

def prob_X (x : ℕ) : ℚ :=
  if x = 0 then (1 - prob_A) * (1 - prob_B)
  else if x = 1 then (1 - prob_A) * prob_B + (1 - prob_B) * prob_A
  else if x = 2 then prob_A * prob_B
  else 0

/-- Expected value of X -/
noncomputable def expect_X : ℚ :=
  0 * prob_X 0 + 1 * prob_X 1 + 2 * prob_X 2

/-- Variance of X -/
noncomputable def var_X : ℚ :=
  (0 - expect_X) ^ 2 * prob_X 0 +
  (1 - expect_X) ^ 2 * prob_X 1 +
  (2 - expect_X) ^ 2 * prob_X 2

theorem var_X_is_86_over_225 : var_X = 86 / 225 :=
by {
  sorry
}

end NUMINAMATH_GPT_var_X_is_86_over_225_l1783_178360


namespace NUMINAMATH_GPT_find_digits_l1783_178326

theorem find_digits :
  ∃ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1098) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_digits_l1783_178326


namespace NUMINAMATH_GPT_inequality_and_equality_l1783_178380

variables {x y z : ℝ}

theorem inequality_and_equality (x y z : ℝ) :
  (x^2 + y^4 + z^6 >= x * y^2 + y^2 * z^3 + x * z^3) ∧ (x^2 + y^4 + z^6 = x * y^2 + y^2 * z^3 + x * z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end NUMINAMATH_GPT_inequality_and_equality_l1783_178380


namespace NUMINAMATH_GPT_factor_expression_l1783_178387

theorem factor_expression (x : ℝ) : 84 * x^7 - 297 * x^13 = 3 * x^7 * (28 - 99 * x^6) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l1783_178387


namespace NUMINAMATH_GPT_stamps_total_l1783_178321

def Lizette_stamps : ℕ := 813
def Minerva_stamps : ℕ := Lizette_stamps - 125
def Jermaine_stamps : ℕ := Lizette_stamps + 217

def total_stamps : ℕ := Minerva_stamps + Lizette_stamps + Jermaine_stamps

theorem stamps_total :
  total_stamps = 2531 := by
  sorry

end NUMINAMATH_GPT_stamps_total_l1783_178321


namespace NUMINAMATH_GPT_avery_egg_cartons_l1783_178368

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end NUMINAMATH_GPT_avery_egg_cartons_l1783_178368


namespace NUMINAMATH_GPT_prob_allergic_prescribed_l1783_178372

def P (a : Prop) : ℝ := sorry

axiom P_conditional (A B : Prop) : P B > 0 → P (A ∧ B) = P A * P (B ∧ A) / P B

def A : Prop := sorry -- represent the event that a patient is prescribed Undetenin
def B : Prop := sorry -- represent the event that a patient is allergic to Undetenin

axiom P_A : P A = 0.10
axiom P_B_given_A : P (B ∧ A) / P A = 0.02
axiom P_B : P B = 0.04

theorem prob_allergic_prescribed : P (A ∧ B) / P B = 0.05 :=
by
  have h1 : P (A ∧ B) / P A = 0.10 * 0.02 := sorry -- using definition of P_A and P_B_given_A
  have h2 : P (A ∧ B) = 0.002 := sorry -- calculating the numerator P(B and A)
  exact sorry -- use the axiom P_B to complete the theorem

end NUMINAMATH_GPT_prob_allergic_prescribed_l1783_178372


namespace NUMINAMATH_GPT_num_signs_in_sign_language_l1783_178338

theorem num_signs_in_sign_language (n : ℕ) (h : n^2 - (n - 2)^2 = 888) : n = 223 := 
sorry

end NUMINAMATH_GPT_num_signs_in_sign_language_l1783_178338


namespace NUMINAMATH_GPT_problem_statement_l1783_178379

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem problem_statement : f (g (-3)) = 961 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1783_178379


namespace NUMINAMATH_GPT_part_one_solution_part_two_solution_l1783_178367

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part (1): "When a = 1, find the solution set of the inequality f(x) ≥ 3"
theorem part_one_solution (x : ℝ) : f x 1 ≥ 3 ↔ x ≤ 0 ∨ x ≥ 3 :=
by sorry

-- Part (2): "If f(x) ≥ 2a - 1, find the range of values for a"
theorem part_two_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ 2 * a - 1) ↔ a ≤ 1 :=
by sorry

end NUMINAMATH_GPT_part_one_solution_part_two_solution_l1783_178367


namespace NUMINAMATH_GPT_binomial_square_formula_l1783_178366

theorem binomial_square_formula (a b : ℝ) :
  let e1 := (4 * a + b) * (4 * a - 2 * b)
  let e2 := (a - 2 * b) * (2 * b - a)
  let e3 := (2 * a - b) * (-2 * a + b)
  let e4 := (a - b) * (a + b)
  (e4 = a^2 - b^2) :=
by
  sorry

end NUMINAMATH_GPT_binomial_square_formula_l1783_178366


namespace NUMINAMATH_GPT_candy_seller_initial_candies_l1783_178307

-- Given conditions
def num_clowns : ℕ := 4
def num_children : ℕ := 30
def candies_per_person : ℕ := 20
def candies_left : ℕ := 20

-- Question: What was the initial number of candies?
def total_people : ℕ := num_clowns + num_children
def total_candies_given_out : ℕ := total_people * candies_per_person
def initial_candies : ℕ := total_candies_given_out + candies_left

theorem candy_seller_initial_candies : initial_candies = 700 :=
by
  sorry

end NUMINAMATH_GPT_candy_seller_initial_candies_l1783_178307


namespace NUMINAMATH_GPT_third_stick_length_l1783_178332

theorem third_stick_length (x : ℝ) (h1 : 2 > 0) (h2 : 5 > 0) (h3 : 3 < x) (h4 : x < 7) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_third_stick_length_l1783_178332


namespace NUMINAMATH_GPT_function_value_proof_l1783_178342

theorem function_value_proof (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : ∀ x, f (x + 1) = -f (-x + 1))
    (h2 : ∀ x, f (x + 2) = f (-x + 2))
    (h3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b)
    (h4 : ∀ x y : ℝ, x - y - 3 = 0)
    : f (9/2) = 5/4 := by
  sorry

end NUMINAMATH_GPT_function_value_proof_l1783_178342


namespace NUMINAMATH_GPT_probability_intersection_of_diagonals_hendecagon_l1783_178399

-- Definition statements expressing the given conditions and required probability

def total_diagonals (n : ℕ) : ℕ := (Nat.choose n 2) - n

def ways_to_choose_2_diagonals (n : ℕ) : ℕ := Nat.choose (total_diagonals n) 2

def ways_sets_of_intersecting_diagonals (n : ℕ) : ℕ := Nat.choose n 4

def probability_intersection_lies_inside (n : ℕ) : ℚ :=
  ways_sets_of_intersecting_diagonals n / ways_to_choose_2_diagonals n

theorem probability_intersection_of_diagonals_hendecagon :
  probability_intersection_lies_inside 11 = 165 / 473 := 
by
  sorry

end NUMINAMATH_GPT_probability_intersection_of_diagonals_hendecagon_l1783_178399


namespace NUMINAMATH_GPT_volume_of_rectangular_solid_l1783_178369

variable {x y z : ℝ}
variable (hx : x * y = 3) (hy : x * z = 5) (hz : y * z = 15)

theorem volume_of_rectangular_solid : x * y * z = 15 :=
by sorry

end NUMINAMATH_GPT_volume_of_rectangular_solid_l1783_178369


namespace NUMINAMATH_GPT_cos_alpha_eq_l1783_178330

open Real

-- Define the angles and their conditions
variables (α β : ℝ)

-- Hypothesis and initial conditions
axiom ha1 : 0 < α ∧ α < π
axiom ha2 : 0 < β ∧ β < π
axiom h_cos_beta : cos β = -5 / 13
axiom h_sin_alpha_plus_beta : sin (α + β) = 3 / 5

-- The main theorem to prove
theorem cos_alpha_eq : cos α = 56 / 65 := sorry

end NUMINAMATH_GPT_cos_alpha_eq_l1783_178330


namespace NUMINAMATH_GPT_Jackie_apples_count_l1783_178378

variable (Adam_apples Jackie_apples : ℕ)

-- Conditions
axiom Adam_has_14_apples : Adam_apples = 14
axiom Adam_has_5_more_than_Jackie : Adam_apples = Jackie_apples + 5

-- Theorem to prove
theorem Jackie_apples_count : Jackie_apples = 9 := by
  -- Use the conditions to derive the answer
  sorry

end NUMINAMATH_GPT_Jackie_apples_count_l1783_178378


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1783_178373

theorem solution_set_of_inequality :
  {x : ℝ | 3 * x ^ 2 - 7 * x - 10 ≥ 0} = {x : ℝ | x ≥ (10 / 3) ∨ x ≤ -1} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1783_178373


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_div_four_l1783_178348

theorem tan_alpha_plus_pi_div_four (α : ℝ) (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3) : 
  Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_div_four_l1783_178348


namespace NUMINAMATH_GPT_total_weight_of_bars_l1783_178320

-- Definitions for weights of each gold bar
variables (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
variables (W1 W2 W3 W4 W5 W6 W7 W8 : ℝ)

-- Definitions for the weighings
axiom weight_C1_C2 : W1 = C1 + C2
axiom weight_C1_C3 : W2 = C1 + C3
axiom weight_C2_C3 : W3 = C2 + C3
axiom weight_C4_C5 : W4 = C4 + C5
axiom weight_C6_C7 : W5 = C6 + C7
axiom weight_C8_C9 : W6 = C8 + C9
axiom weight_C10_C11 : W7 = C10 + C11
axiom weight_C12_C13 : W8 = C12 + C13

-- Prove the total weight of all gold bars
theorem total_weight_of_bars :
  (C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13)
  = (W1 + W2 + W3) / 2 + W4 + W5 + W6 + W7 + W8 :=
by sorry

end NUMINAMATH_GPT_total_weight_of_bars_l1783_178320


namespace NUMINAMATH_GPT_simple_interest_sum_l1783_178390

theorem simple_interest_sum :
  let P := 1750
  let CI := 4000 * ((1 + (10 / 100))^2) - 4000
  let SI := (1 / 2) * CI
  SI = (P * 8 * 3) / 100 
  :=
by
  -- Definitions
  let P := 1750
  let CI := 4000 * ((1 + 10 / 100)^2) - 4000
  let SI := (1 / 2) * CI
  
  -- Claim
  have : SI = (P * 8 * 3) / 100 := sorry

  exact this

end NUMINAMATH_GPT_simple_interest_sum_l1783_178390


namespace NUMINAMATH_GPT_radius_correct_l1783_178397

open Real

noncomputable def radius_of_circle
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop) : ℝ := sorry

theorem radius_correct
  (tangent_length : ℝ) 
  (secant_internal_segment : ℝ) 
  (tangent_secant_perpendicular : Prop)
  (h1 : tangent_length = 12) 
  (h2 : secant_internal_segment = 10) 
  (h3 : tangent_secant_perpendicular) : radius_of_circle tangent_length secant_internal_segment tangent_secant_perpendicular = 13 := 
sorry

end NUMINAMATH_GPT_radius_correct_l1783_178397


namespace NUMINAMATH_GPT_system_of_linear_eq_l1783_178344

theorem system_of_linear_eq :
  ∃ (x y : ℝ), x + y = 5 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_system_of_linear_eq_l1783_178344


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1783_178365

-- Definitions of the vectors
def a (x : ℝ) : ℝ × ℝ := (1, 2 * x)
def b (x : ℝ) : ℝ × ℝ := (x, 3)
def c : ℝ × ℝ := (-2, 0)

-- Definitions for vector operations
def add_vec (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

noncomputable def part1 (x : ℝ) : Prop := parallel (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

noncomputable def part2 (x : ℝ) : Prop := perpendicular (add_vec (a x) (scalar_mul 2 (b x))) (add_vec (scalar_mul 2 (a x)) (scalar_mul (-1) c))

theorem problem_part1 : part1 2 ∧ part1 (-3 / 2) := sorry

theorem problem_part2 : part2 ((-4 + Real.sqrt 14) / 2) ∧ part2 ((-4 - Real.sqrt 14) / 2) := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1783_178365


namespace NUMINAMATH_GPT_girls_without_notebooks_l1783_178302

noncomputable def girls_in_class : Nat := 20
noncomputable def students_with_notebooks : Nat := 25
noncomputable def boys_with_notebooks : Nat := 16

theorem girls_without_notebooks : 
  (girls_in_class - (students_with_notebooks - boys_with_notebooks)) = 11 := by
  sorry

end NUMINAMATH_GPT_girls_without_notebooks_l1783_178302


namespace NUMINAMATH_GPT_roots_of_quadratic_l1783_178376

theorem roots_of_quadratic (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1783_178376


namespace NUMINAMATH_GPT_fractional_part_sum_leq_l1783_178391

noncomputable def fractional_part (z : ℝ) : ℝ :=
  z - (⌊z⌋ : ℝ)

theorem fractional_part_sum_leq (x y : ℝ) :
  fractional_part (x + y) ≤ fractional_part x + fractional_part y :=
by
  sorry

end NUMINAMATH_GPT_fractional_part_sum_leq_l1783_178391


namespace NUMINAMATH_GPT_find_divisor_l1783_178315

theorem find_divisor
  (n : ℕ) (h1 : n > 0)
  (h2 : (n + 1) % 6 = 4)
  (h3 : ∃ d : ℕ, n % d = 1) :
  ∃ d : ℕ, (n % d = 1) ∧ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1783_178315


namespace NUMINAMATH_GPT_problem_statement_l1783_178309

variable {a b c d : ℝ}

theorem problem_statement (h : a * d - b * c = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1783_178309


namespace NUMINAMATH_GPT_wristwatch_cost_proof_l1783_178329

-- Definition of the problem conditions
def allowance_per_week : ℕ := 5
def initial_weeks : ℕ := 10
def initial_savings : ℕ := 20
def additional_weeks : ℕ := 16

-- The total cost of the wristwatch
def wristwatch_cost : ℕ := 100

-- Let's state the proof problem
theorem wristwatch_cost_proof :
  (initial_savings + additional_weeks * allowance_per_week) = wristwatch_cost :=
by
  sorry

end NUMINAMATH_GPT_wristwatch_cost_proof_l1783_178329


namespace NUMINAMATH_GPT_ice_forms_inner_surface_in_winter_l1783_178312

-- Definitions based on conditions
variable (humid_air_inside : Prop) 
variable (heat_transfer_inner_surface : Prop) 
variable (heat_transfer_outer_surface : Prop) 
variable (temp_inner_surface_below_freezing : Prop) 
variable (condensation_inner_surface_below_freezing : Prop)
variable (ice_formation_inner_surface : Prop)
variable (cold_dry_air_outside : Prop)
variable (no_significant_condensation_outside : Prop)

-- Proof of the theorem
theorem ice_forms_inner_surface_in_winter :
  humid_air_inside ∧
  heat_transfer_inner_surface ∧
  heat_transfer_outer_surface ∧
  (¬sufficient_heating → temp_inner_surface_below_freezing) ∧
  (condensation_inner_surface_below_freezing ↔ (temp_inner_surface_below_freezing ∧ humid_air_inside)) ∧
  (ice_formation_inner_surface ↔ (condensation_inner_surface_below_freezing ∧ temp_inner_surface_below_freezing)) ∧
  (cold_dry_air_outside → ¬ice_formation_outer_surface)
  → ice_formation_inner_surface :=
sorry

end NUMINAMATH_GPT_ice_forms_inner_surface_in_winter_l1783_178312


namespace NUMINAMATH_GPT_maximum_omega_l1783_178381

noncomputable def f (omega varphi : ℝ) (x : ℝ) : ℝ :=
  Real.cos (omega * x + varphi)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a < x → x < y → y < b → f y ≤ f x

theorem maximum_omega (omega varphi : ℝ)
    (h0 : omega > 0)
    (h1 : 0 < varphi ∧ varphi < π)
    (h2 : is_odd_function (f omega varphi))
    (h3 : is_monotonically_decreasing (f omega varphi) (-π/3) (π/6)) :
  omega ≤ 3/2 :=
sorry

end NUMINAMATH_GPT_maximum_omega_l1783_178381


namespace NUMINAMATH_GPT_add_base7_l1783_178304

-- Define the two numbers in base 7 to be added.
def number1 : ℕ := 2 * 7 + 5
def number2 : ℕ := 5 * 7 + 4

-- Define the expected result in base 7.
def expected_sum : ℕ := 1 * 7^2 + 1 * 7 + 2

theorem add_base7 :
  let sum : ℕ := number1 + number2
  sum = expected_sum := sorry

end NUMINAMATH_GPT_add_base7_l1783_178304


namespace NUMINAMATH_GPT_cos_alpha_value_l1783_178350

theorem cos_alpha_value
  (a : ℝ) (h1 : π < a ∧ a < 3 * π / 2)
  (h2 : Real.tan a = 2) :
  Real.cos a = - (Real.sqrt 5) / 5 :=
sorry

end NUMINAMATH_GPT_cos_alpha_value_l1783_178350


namespace NUMINAMATH_GPT_distance_to_city_center_l1783_178301

theorem distance_to_city_center 
  (D : ℕ) 
  (H1 : D = 200 + 200 + D) 
  (H_total : 900 = 200 + 200 + D) : 
  D = 500 :=
by { sorry }

end NUMINAMATH_GPT_distance_to_city_center_l1783_178301


namespace NUMINAMATH_GPT_sum_of_numbers_in_ratio_l1783_178351

theorem sum_of_numbers_in_ratio (x : ℝ) (h1 : 8 * x - 3 * x = 20) : 3 * x + 8 * x = 44 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_in_ratio_l1783_178351


namespace NUMINAMATH_GPT_largest_y_l1783_178308

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

theorem largest_y (x y : ℕ) (hx : x ≥ y) (hy : y ≥ 3) 
  (h : (interior_angle x * 28) = (interior_angle y * 29)) :
  y = 57 :=
by
  sorry

end NUMINAMATH_GPT_largest_y_l1783_178308


namespace NUMINAMATH_GPT_cats_left_l1783_178385

theorem cats_left (siamese_cats : ℕ) (house_cats : ℕ) (cats_sold : ℕ) (total_initial_cats : ℕ) (remaining_cats : ℕ) :
  siamese_cats = 15 → house_cats = 49 → cats_sold = 19 → total_initial_cats = siamese_cats + house_cats → remaining_cats = total_initial_cats - cats_sold → remaining_cats = 45 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  rw [h4, h3] at h5
  exact h5

end NUMINAMATH_GPT_cats_left_l1783_178385


namespace NUMINAMATH_GPT_abc_sum_eq_sixteen_l1783_178361

theorem abc_sum_eq_sixteen (a b c : ℤ) (h1 : a ≠ b ∨ a ≠ c ∨ b ≠ c) (h2 : a ≥ 4 ∧ b ≥ 4 ∧ c ≥ 4) (h3 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by 
  sorry

end NUMINAMATH_GPT_abc_sum_eq_sixteen_l1783_178361


namespace NUMINAMATH_GPT_base7_sub_base5_to_base10_l1783_178300

def base7to10 (n : Nat) : Nat :=
  match n with
  | 52403 => 5 * 7^4 + 2 * 7^3 + 4 * 7^2 + 0 * 7^1 + 3 * 7^0
  | _ => 0

def base5to10 (n : Nat) : Nat :=
  match n with
  | 20345 => 2 * 5^4 + 0 * 5^3 + 3 * 5^2 + 4 * 5^1 + 5 * 5^0
  | _ => 0

theorem base7_sub_base5_to_base10 :
  base7to10 52403 - base5to10 20345 = 11540 :=
by
  sorry

end NUMINAMATH_GPT_base7_sub_base5_to_base10_l1783_178300


namespace NUMINAMATH_GPT_sector_perimeter_ratio_l1783_178362

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) 
  (h1 : α > 0) 
  (h2 : r > 0) 
  (h3 : R > 0) 
  (h4 : (1/2) * α * r^2 / ((1/2) * α * R^2) = 1/4) :
  (2 * r + α * r) / (2 * R + α * R) = 1 / 2 := 
sorry

end NUMINAMATH_GPT_sector_perimeter_ratio_l1783_178362


namespace NUMINAMATH_GPT_perpendicular_vectors_l1783_178340

noncomputable def a (k : ℝ) : ℝ × ℝ := (2 * k - 4, 3)
noncomputable def b (k : ℝ) : ℝ × ℝ := (-3, k)

theorem perpendicular_vectors (k : ℝ) (h : (2 * k - 4) * (-3) + 3 * k = 0) : k = 4 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_l1783_178340


namespace NUMINAMATH_GPT_Foster_Farms_donated_45_chickens_l1783_178358

def number_of_dressed_chickens_donated_by_Foster_Farms (C AS H BB D : ℕ) : Prop :=
  C + AS + H + BB + D = 375 ∧
  AS = 2 * C ∧
  H = 3 * C ∧
  BB = C ∧
  D = 2 * C - 30

theorem Foster_Farms_donated_45_chickens:
  ∃ C, number_of_dressed_chickens_donated_by_Foster_Farms C (2*C) (3*C) C (2*C - 30) ∧ C = 45 :=
by 
  sorry

end NUMINAMATH_GPT_Foster_Farms_donated_45_chickens_l1783_178358


namespace NUMINAMATH_GPT_total_interest_is_350_l1783_178317

-- Define the principal amounts, rates, and time
def principal1 : ℝ := 1000
def rate1 : ℝ := 0.03
def principal2 : ℝ := 1200
def rate2 : ℝ := 0.05
def time : ℝ := 3.888888888888889

-- Calculate the interest for one year for each loan
def interest_per_year1 : ℝ := principal1 * rate1
def interest_per_year2 : ℝ := principal2 * rate2

-- Calculate the total interest for the time period for each loan
def total_interest1 : ℝ := interest_per_year1 * time
def total_interest2 : ℝ := interest_per_year2 * time

-- Finally, calculate the total interest amount
def total_interest_amount : ℝ := total_interest1 + total_interest2

-- The proof problem: Prove that total_interest_amount == 350 Rs
theorem total_interest_is_350 : total_interest_amount = 350 := by
  sorry

end NUMINAMATH_GPT_total_interest_is_350_l1783_178317


namespace NUMINAMATH_GPT_find_the_number_l1783_178331

theorem find_the_number 
  (x y n : ℤ)
  (h : 19 * (x + y) + 17 = 19 * (-x + y) - n)
  (hx : x = 1) :
  n = -55 :=
by
  sorry

end NUMINAMATH_GPT_find_the_number_l1783_178331


namespace NUMINAMATH_GPT_gallons_added_in_fourth_hour_l1783_178347

-- Defining the conditions
def initial_volume : ℕ := 40
def loss_rate_per_hour : ℕ := 2
def add_in_third_hour : ℕ := 1
def remaining_after_fourth_hour : ℕ := 36

-- Prove the problem statement
theorem gallons_added_in_fourth_hour :
  ∃ (x : ℕ), initial_volume - 2 * 4 + 1 - loss_rate_per_hour + x = remaining_after_fourth_hour :=
sorry

end NUMINAMATH_GPT_gallons_added_in_fourth_hour_l1783_178347


namespace NUMINAMATH_GPT_ab_cd_zero_l1783_178346

theorem ab_cd_zero {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) (h3 : ac + bd = 0) : ab + cd = 0 :=
sorry

end NUMINAMATH_GPT_ab_cd_zero_l1783_178346


namespace NUMINAMATH_GPT_inches_of_rain_received_so_far_l1783_178336

def total_days_in_year : ℕ := 365
def days_left_in_year : ℕ := 100
def rain_per_day_initial_avg : ℝ := 2
def rain_per_day_required_avg : ℝ := 3

def total_annually_expected_rain : ℝ := rain_per_day_initial_avg * total_days_in_year
def days_passed_in_year : ℕ := total_days_in_year - days_left_in_year
def total_rain_needed_remaining : ℝ := rain_per_day_required_avg * days_left_in_year

variable (S : ℝ) -- inches of rain received so far

theorem inches_of_rain_received_so_far (S : ℝ) :
  S + total_rain_needed_remaining = total_annually_expected_rain → S = 430 :=
  by
  sorry

end NUMINAMATH_GPT_inches_of_rain_received_so_far_l1783_178336


namespace NUMINAMATH_GPT_triangle_is_right_triangle_l1783_178322

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : a ≠ b)
  (h₂ : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  (A_ne_B : A ≠ B)
  (hABC : A + B + C = Real.pi) :
  C = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_right_triangle_l1783_178322


namespace NUMINAMATH_GPT_side_c_possibilities_l1783_178343

theorem side_c_possibilities (A : ℝ) (a b c : ℝ) (hA : A = 30) (ha : a = 4) (hb : b = 4 * Real.sqrt 3) :
  c = 4 ∨ c = 8 :=
sorry

end NUMINAMATH_GPT_side_c_possibilities_l1783_178343


namespace NUMINAMATH_GPT_correct_assignment_statement_l1783_178356

noncomputable def is_assignment_statement (stmt : String) : Bool :=
  -- Assume a simplified function that interprets whether the statement is an assignment
  match stmt with
  | "6 = M" => false
  | "M = -M" => true
  | "B = A = 8" => false
  | "x - y = 0" => false
  | _ => false

theorem correct_assignment_statement :
  is_assignment_statement "M = -M" = true :=
by
  rw [is_assignment_statement]
  exact rfl

end NUMINAMATH_GPT_correct_assignment_statement_l1783_178356


namespace NUMINAMATH_GPT_polygon_sides_l1783_178354

theorem polygon_sides (s : ℕ) (h : 180 * (s - 2) = 720) : s = 6 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1783_178354


namespace NUMINAMATH_GPT_tangent_line_equation_parallel_to_given_line_l1783_178310

theorem tangent_line_equation_parallel_to_given_line :
  ∃ (x y : ℝ),  y = x^3 - 3 * x^2
    ∧ (3 * x^2 - 6 * x = -3)
    ∧ (y = -2)
    ∧ (3 * x + y - 1 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_equation_parallel_to_given_line_l1783_178310


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1783_178327

theorem solve_quadratic_eq (x : ℝ) :
  x^2 + 4 * x + 2 = 0 ↔ (x = -2 + Real.sqrt 2 ∨ x = -2 - Real.sqrt 2) :=
by
  -- This is a statement only. No proof is required.
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1783_178327


namespace NUMINAMATH_GPT_complement_union_A_B_l1783_178398

open Set

variable {U : Type*} [Preorder U] [BoundedOrder U]

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_union_A_B :
  compl (A ∪ B) = {x : ℝ | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_l1783_178398


namespace NUMINAMATH_GPT_number_of_valid_M_l1783_178303

def base_4_representation (M : ℕ) :=
  let c_3 := (M / 256) % 4
  let c_2 := (M / 64) % 4
  let c_1 := (M / 16) % 4
  let c_0 := M % 4
  (256 * c_3) + (64 * c_2) + (16 * c_1) + (4 * c_0)

def base_7_representation (M : ℕ) :=
  let d_3 := (M / 343) % 7
  let d_2 := (M / 49) % 7
  let d_1 := (M / 7) % 7
  let d_0 := M % 7
  (343 * d_3) + (49 * d_2) + (7 * d_1) + d_0

def valid_M (M T : ℕ) :=
  1000 ≤ M ∧ M < 10000 ∧ 
  T = base_4_representation M + base_7_representation M ∧ 
  (T % 100) = ((3 * M) % 100)

theorem number_of_valid_M : 
  ∃ n : ℕ, n = 81 ∧ ∀ M T, valid_M M T → n = (81 : ℕ) :=
sorry

end NUMINAMATH_GPT_number_of_valid_M_l1783_178303


namespace NUMINAMATH_GPT_find_quadruples_l1783_178339

theorem find_quadruples (x y z n : ℕ) : 
  x^2 + y^2 + z^2 + 1 = 2^n → 
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_quadruples_l1783_178339


namespace NUMINAMATH_GPT_chalk_breaking_probability_l1783_178371

/-- Given you start with a single piece of chalk of length 1,
    and every second you choose a piece of chalk uniformly at random and break it in half,
    until you have 8 pieces of chalk,
    prove that the probability of all pieces having length 1/8 is 1/63. -/
theorem chalk_breaking_probability :
  let initial_pieces := 1
  let final_pieces := 8
  let total_breaks := final_pieces - initial_pieces
  let favorable_sequences := 20 * 4
  let total_sequences := Nat.factorial total_breaks
  (initial_pieces = 1) →
  (final_pieces = 8) →
  (total_breaks = 7) →
  (favorable_sequences = 80) →
  (total_sequences = 5040) →
  (favorable_sequences / total_sequences = 1 / 63) :=
by
  intros
  sorry

end NUMINAMATH_GPT_chalk_breaking_probability_l1783_178371


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1783_178394

def A : Set ℕ := {0, 2}
def B : Set ℕ := {1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1783_178394


namespace NUMINAMATH_GPT_inequality_solution_l1783_178364

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (3 * x - 1) > 1 ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1783_178364


namespace NUMINAMATH_GPT_central_angle_radian_measure_l1783_178359

-- Definitions for the conditions
def circumference (r l : ℝ) : Prop := 2 * r + l = 8
def area (r l : ℝ) : Prop := (1/2) * l * r = 4
def radian_measure (l r θ : ℝ) : Prop := θ = l / r

-- Prove the radian measure of the central angle of the sector is 2
theorem central_angle_radian_measure (r l θ : ℝ) : 
  circumference r l → 
  area r l → 
  radian_measure l r θ → 
  θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_radian_measure_l1783_178359


namespace NUMINAMATH_GPT_proportion_correct_l1783_178388

theorem proportion_correct (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
sorry

end NUMINAMATH_GPT_proportion_correct_l1783_178388


namespace NUMINAMATH_GPT_xz_less_than_half_l1783_178316

theorem xz_less_than_half (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : xy + yz + zx = 1) : x * z < 1 / 2 :=
  sorry

end NUMINAMATH_GPT_xz_less_than_half_l1783_178316


namespace NUMINAMATH_GPT_arithmetic_seq_sixth_term_l1783_178395

theorem arithmetic_seq_sixth_term
  (a d : ℤ)
  (h1 : a + d = 14)
  (h2 : a + 3 * d = 32) : a + 5 * d = 50 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sixth_term_l1783_178395


namespace NUMINAMATH_GPT_cos_C_value_triangle_perimeter_l1783_178357

variables (A B C a b c : ℝ)
variables (cos_B : ℝ) (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3)
variables (dot_product_88 : a * b * (Real.cos C) = 88)

theorem cos_C_value (A B : ℝ) (a b : ℝ) (cos_B : ℝ) (cos_C : ℝ) (dot_product_88 : a * b * cos_C = 88) :
  A = 2 * B →
  cos_B = 2 / 3 →
  cos_C = 22 / 27 :=
sorry

theorem triangle_perimeter (A B C a b c : ℝ) (cos_B : ℝ)
  (A_eq_2B : A = 2 * B) (cos_B_val : cos_B = 2 / 3) (dot_product_88 : a * b * (Real.cos C) = 88)
  (a_val : a = 12) (b_val : b = 9) (c_val : c = 7) :
  a + b + c = 28 :=
sorry

end NUMINAMATH_GPT_cos_C_value_triangle_perimeter_l1783_178357


namespace NUMINAMATH_GPT_find_c_l1783_178328

-- Definitions for the conditions
def line1 (x y : ℝ) : Prop := 4 * y + 2 * x + 6 = 0
def line2 (x y : ℝ) (c : ℝ) : Prop := 5 * y + c * x + 4 = 0
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Main theorem
theorem find_c (c : ℝ) : 
  (∀ x y : ℝ, line1 x y → y = -1/2 * x - 3/2) ∧ 
  (∀ x y : ℝ, line2 x y c → y = -c/5 * x - 4/5) ∧ 
  perpendicular (-1/2) (-c/5) → 
  c = -10 := by
  sorry

end NUMINAMATH_GPT_find_c_l1783_178328


namespace NUMINAMATH_GPT_problem_statement_l1783_178334

noncomputable def given_expression (x y z : ℝ) : ℝ :=
  (45 + (23 / 89) * Real.sin x) * (4 * y^2 - 7 * z^3)

theorem problem_statement : given_expression (Real.pi / 6) 3 (-2) = 4186 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1783_178334


namespace NUMINAMATH_GPT_positive_integer_pairs_l1783_178370

theorem positive_integer_pairs (m n : ℕ) (p : ℕ) (hp_prime : Prime p) (h_diff : m - n = p) (h_square : ∃ k : ℕ, m * n = k^2) :
  ∃ p' : ℕ, (Prime p') ∧ m = (p' + 1) / 2 ^ 2 ∧ n = (p' - 1) / 2 ^ 2 :=
sorry

end NUMINAMATH_GPT_positive_integer_pairs_l1783_178370


namespace NUMINAMATH_GPT_n_minus_two_is_square_of_natural_number_l1783_178384

theorem n_minus_two_is_square_of_natural_number (n : ℕ) (h_n : n ≥ 3) (h_odd_m : Odd (1 / 2 * n * (n - 1))) :
  ∃ k : ℕ, n - 2 = k^2 := 
  by
  sorry

end NUMINAMATH_GPT_n_minus_two_is_square_of_natural_number_l1783_178384


namespace NUMINAMATH_GPT_sqrt_fraction_sum_l1783_178355

theorem sqrt_fraction_sum : 
    Real.sqrt ((1 / 25) + (1 / 36)) = (Real.sqrt 61) / 30 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_sum_l1783_178355


namespace NUMINAMATH_GPT_corina_problem_l1783_178393

variable (P Q : ℝ)

theorem corina_problem (h1 : P + Q = 16) (h2 : P - Q = 4) : P = 10 :=
sorry

end NUMINAMATH_GPT_corina_problem_l1783_178393


namespace NUMINAMATH_GPT_find_window_width_on_second_wall_l1783_178374

noncomputable def total_wall_area (width length height: ℝ) : ℝ :=
  4 * width * height

noncomputable def doorway_area (width height : ℝ) : ℝ :=
  width * height

noncomputable def window_area (width height : ℝ) : ℝ :=
  width * height

theorem find_window_width_on_second_wall :
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  total_area - first_doorway - second_doorway - window_area w window_height = area_to_paint
  → w = 6 :=
by
  let room_width := 20
  let room_length := 20
  let room_height := 8
  let first_doorway_width := 3
  let first_doorway_height := 7
  let second_doorway_width := 5
  let second_doorway_height := 7
  let window_height := 4
  let area_to_paint := 560
  let total_area := total_wall_area room_width room_length room_height
  let first_doorway := doorway_area first_doorway_width first_doorway_height
  let second_doorway := doorway_area second_doorway_width second_doorway_height
  sorry

end NUMINAMATH_GPT_find_window_width_on_second_wall_l1783_178374


namespace NUMINAMATH_GPT_sum_infinite_geometric_series_l1783_178377

theorem sum_infinite_geometric_series :
  ∑' (n : ℕ), (3 : ℝ) * ((1 / 3) ^ n) = (9 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_sum_infinite_geometric_series_l1783_178377


namespace NUMINAMATH_GPT_coordinates_of_B_l1783_178353

-- Definitions of the points and vectors are given as conditions.
def A : ℝ × ℝ := (-1, -1)
def a : ℝ × ℝ := (2, 3)

-- Statement of the problem translated to Lean
theorem coordinates_of_B (B : ℝ × ℝ) (h : B = (5, 8)) :
  (B.1 + 1, B.2 + 1) = (3 * a.1, 3 * a.2) :=
sorry

end NUMINAMATH_GPT_coordinates_of_B_l1783_178353


namespace NUMINAMATH_GPT_rectangle_ratio_l1783_178349

theorem rectangle_ratio (a b c d : ℝ) (h₀ : a = 4)
  (h₁ : b = (4 / 3)) (h₂ : c = (8 / 3)) (h₃ : d = 4) :
  (∃ XY YZ, XY * YZ = a * a ∧ XY / YZ = 0.9) :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1783_178349


namespace NUMINAMATH_GPT_number_of_people_in_room_l1783_178313

theorem number_of_people_in_room (P : ℕ) 
  (h1 : 1/4 * P = P / 4) 
  (h2 : 3/4 * P = 3 * P / 4) 
  (h3 : P / 4 = 20) : 
  P = 80 :=
sorry

end NUMINAMATH_GPT_number_of_people_in_room_l1783_178313


namespace NUMINAMATH_GPT_feed_days_l1783_178341

theorem feed_days (morning_food evening_food total_food : ℕ) (h1 : morning_food = 1) (h2 : evening_food = 1) (h3 : total_food = 32)
: (total_food / (morning_food + evening_food)) = 16 := by
  sorry

end NUMINAMATH_GPT_feed_days_l1783_178341


namespace NUMINAMATH_GPT_two_digit_solution_l1783_178314

def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

theorem two_digit_solution :
  ∃ (x y : ℕ), 
    two_digit_number x y = 24 ∧ 
    two_digit_number x y = x^3 + y^2 ∧ 
    0 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_solution_l1783_178314


namespace NUMINAMATH_GPT_yellow_area_is_1_5625_percent_l1783_178318

def square_flag_area (s : ℝ) : ℝ := s ^ 2

def cross_yellow_occupies_25_percent (s : ℝ) (w : ℝ) : Prop :=
  4 * w * s - 4 * w ^ 2 = 0.25 * s ^ 2

def yellow_area (s w : ℝ) : ℝ := 4 * w ^ 2

def percent_of_flag_area_is_yellow (s w : ℝ) : Prop :=
  yellow_area s w = 0.015625 * s ^ 2

theorem yellow_area_is_1_5625_percent (s w : ℝ) (h1: cross_yellow_occupies_25_percent s w) : 
  percent_of_flag_area_is_yellow s w :=
by sorry

end NUMINAMATH_GPT_yellow_area_is_1_5625_percent_l1783_178318


namespace NUMINAMATH_GPT_susan_initial_amount_l1783_178306

theorem susan_initial_amount :
  ∃ S: ℝ, (S - (1/5 * S + 1/4 * S + 120) = 1200) → S = 2400 :=
by
  sorry

end NUMINAMATH_GPT_susan_initial_amount_l1783_178306


namespace NUMINAMATH_GPT_agnes_weekly_hours_l1783_178325

-- Given conditions
def mila_hourly_rate : ℝ := 10
def agnes_hourly_rate : ℝ := 15
def mila_hours_per_month : ℝ := 48

-- Derived condition that Mila's earnings in a month equal Agnes's in a month
def mila_monthly_earnings : ℝ := mila_hourly_rate * mila_hours_per_month

-- Prove that Agnes must work 8 hours each week to match Mila's monthly earnings
theorem agnes_weekly_hours (A : ℝ) : 
  agnes_hourly_rate * 4 * A = mila_monthly_earnings → A = 8 := 
by
  intro h
  -- sorry here is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_agnes_weekly_hours_l1783_178325


namespace NUMINAMATH_GPT_incorrect_axis_symmetry_l1783_178345

noncomputable def quadratic_function (x : ℝ) : ℝ := - (x + 2)^2 - 3

theorem incorrect_axis_symmetry :
  (∀ x : ℝ, quadratic_function x < 0) ∧
  (∀ x : ℝ, x > -1 → (quadratic_function x < quadratic_function (-2))) ∧
  (¬∃ x : ℝ, quadratic_function x = 0) ∧
  (¬ ∀ x : ℝ, x = 2) →
  false :=
by
  sorry

end NUMINAMATH_GPT_incorrect_axis_symmetry_l1783_178345


namespace NUMINAMATH_GPT_pieces_per_package_l1783_178383

-- Definitions from conditions
def total_pieces_of_gum : ℕ := 486
def number_of_packages : ℕ := 27

-- Mathematical statement to prove
theorem pieces_per_package : total_pieces_of_gum / number_of_packages = 18 := sorry

end NUMINAMATH_GPT_pieces_per_package_l1783_178383


namespace NUMINAMATH_GPT_quadrilateral_perimeter_l1783_178305

noncomputable def EG (FH : ℝ) : ℝ := Real.sqrt ((FH + 5) ^ 2 + FH ^ 2)

theorem quadrilateral_perimeter 
  (EF FH GH : ℝ) 
  (h1 : EF = 12)
  (h2 : FH = 7)
  (h3 : GH = FH) :
  EF + FH + GH + EG FH = 26 + Real.sqrt 193 :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_quadrilateral_perimeter_l1783_178305


namespace NUMINAMATH_GPT_calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l1783_178323

theorem calculation_a_squared_plus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  a^2 + b^2 = 6 := by
  sorry

theorem calculation_a_minus_b_squared
  (a b : ℝ)
  (h1 : a + b = 2)
  (h2 : a * b = -1) :
  (a - b)^2 = 8 := by
  sorry

end NUMINAMATH_GPT_calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l1783_178323


namespace NUMINAMATH_GPT_min_geometric_ratio_l1783_178382

theorem min_geometric_ratio (q : ℝ) (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n * q)
(h2 : 1 < q) (h3 : q < 2) : q = 6 / 5 := by
  sorry

end NUMINAMATH_GPT_min_geometric_ratio_l1783_178382


namespace NUMINAMATH_GPT_sues_answer_l1783_178324

theorem sues_answer (x : ℕ) (hx : x = 6) : 
  let b := 2 * (x + 1)
  let s := 2 * (b - 1)
  s = 26 :=
by
  sorry

end NUMINAMATH_GPT_sues_answer_l1783_178324


namespace NUMINAMATH_GPT_detergent_for_9_pounds_l1783_178363

-- Define the given condition.
def detergent_per_pound : ℕ := 2

-- Define the total weight of clothes
def weight_of_clothes : ℕ := 9

-- Define the result of the detergent used.
def detergent_used (d : ℕ) (w : ℕ) : ℕ := d * w

-- Prove that the detergent used to wash 9 pounds of clothes is 18 ounces
theorem detergent_for_9_pounds :
  detergent_used detergent_per_pound weight_of_clothes = 18 := 
sorry

end NUMINAMATH_GPT_detergent_for_9_pounds_l1783_178363


namespace NUMINAMATH_GPT_valid_rearrangements_count_l1783_178335

noncomputable def count_valid_rearrangements : ℕ := sorry

theorem valid_rearrangements_count :
  count_valid_rearrangements = 7 :=
sorry

end NUMINAMATH_GPT_valid_rearrangements_count_l1783_178335


namespace NUMINAMATH_GPT_satisfying_lines_l1783_178337

theorem satisfying_lines (x y : ℝ) : (y^2 - 2*y = x^2 + 2*x) ↔ (y = x + 2 ∨ y = -x) :=
by
  sorry

end NUMINAMATH_GPT_satisfying_lines_l1783_178337


namespace NUMINAMATH_GPT_rides_on_roller_coaster_l1783_178375

-- Definitions based on the conditions given.
def roller_coaster_cost : ℕ := 17
def total_tickets : ℕ := 255
def tickets_spent_on_other_activities : ℕ := 78

-- The proof statement.
theorem rides_on_roller_coaster : (total_tickets - tickets_spent_on_other_activities) / roller_coaster_cost = 10 :=
by 
  sorry

end NUMINAMATH_GPT_rides_on_roller_coaster_l1783_178375


namespace NUMINAMATH_GPT_chairs_to_remove_l1783_178392

/-- Given conditions:
1. Each row holds 13 chairs.
2. There are 169 chairs initially.
3. There are 95 expected attendees.

Task: 
Prove that the number of chairs to be removed to ensure complete rows and minimize empty seats is 65. -/
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ)
  (h1 : chairs_per_row = 13)
  (h2 : total_chairs = 169)
  (h3 : expected_attendees = 95) :
  ∃ chairs_to_remove : ℕ, chairs_to_remove = 65 :=
by
  sorry -- proof omitted

end NUMINAMATH_GPT_chairs_to_remove_l1783_178392


namespace NUMINAMATH_GPT_value_of_frac_sum_l1783_178319

theorem value_of_frac_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : (x + y) / 3 = 11 / 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_frac_sum_l1783_178319


namespace NUMINAMATH_GPT_range_of_a_l1783_178311

noncomputable def has_root_in_R (f : ℝ → ℝ) : Prop :=
∃ x : ℝ, f x = 0

theorem range_of_a (a : ℝ) (h : has_root_in_R (λ x => 4 * x + a * 2^x + a + 1)) : a ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1783_178311


namespace NUMINAMATH_GPT_soccer_ball_seams_l1783_178352

theorem soccer_ball_seams 
  (num_pentagons : ℕ) 
  (num_hexagons : ℕ) 
  (sides_per_pentagon : ℕ) 
  (sides_per_hexagon : ℕ) 
  (total_pieces : ℕ) 
  (equal_sides : sides_per_pentagon = sides_per_hexagon)
  (total_pieces_eq : total_pieces = 32)
  (num_pentagons_eq : num_pentagons = 12)
  (num_hexagons_eq : num_hexagons = 20)
  (sides_per_pentagon_eq : sides_per_pentagon = 5)
  (sides_per_hexagon_eq : sides_per_hexagon = 6) :
  90 = (num_pentagons * sides_per_pentagon + num_hexagons * sides_per_hexagon) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_soccer_ball_seams_l1783_178352


namespace NUMINAMATH_GPT_cost_per_container_is_21_l1783_178389

-- Define the given problem conditions as Lean statements.

--  Let w be the number of weeks represented by 210 days.
def number_of_weeks (days: ℕ) : ℕ := days / 7
def weeks : ℕ := number_of_weeks 210

-- Let p be the total pounds of litter used over the number of weeks.
def pounds_per_week : ℕ := 15
def total_litter_pounds (weeks: ℕ) : ℕ := weeks * pounds_per_week
def total_pounds : ℕ := total_litter_pounds weeks

-- Let c be the number of 45-pound containers needed for the total pounds of litter.
def pounds_per_container : ℕ := 45
def number_of_containers (total_pounds pounds_per_container: ℕ) : ℕ := total_pounds / pounds_per_container
def containers : ℕ := number_of_containers total_pounds pounds_per_container

-- Given the total cost, find the cost per container.
def total_cost : ℕ := 210
def cost_per_container (total_cost containers: ℕ) : ℕ := total_cost / containers
def cost : ℕ := cost_per_container total_cost containers

-- Prove that the cost per container is 21.
theorem cost_per_container_is_21 : cost = 21 := by
  sorry

end NUMINAMATH_GPT_cost_per_container_is_21_l1783_178389


namespace NUMINAMATH_GPT_mini_toy_height_difference_l1783_178396

variables (H_standard H_toy H_mini_diff : ℝ)

def poodle_heights : Prop :=
  H_standard = 28 ∧ H_toy = 14 ∧ H_standard - 8 = H_mini_diff + H_toy

theorem mini_toy_height_difference (H_standard H_toy H_mini_diff: ℝ) (h: poodle_heights H_standard H_toy H_mini_diff) :
  H_mini_diff = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_mini_toy_height_difference_l1783_178396


namespace NUMINAMATH_GPT_find_f_neg3_l1783_178386

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom sum_equation : f 1 + f 2 + f 3 + f 4 + f 5 = 6

theorem find_f_neg3 : f (-3) = 6 := by
  sorry

end NUMINAMATH_GPT_find_f_neg3_l1783_178386


namespace NUMINAMATH_GPT_ralph_socks_problem_l1783_178333

theorem ralph_socks_problem :
  ∃ x y z : ℕ, x + y + z = 10 ∧ x + 2 * y + 4 * z = 30 ∧ 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_ralph_socks_problem_l1783_178333
