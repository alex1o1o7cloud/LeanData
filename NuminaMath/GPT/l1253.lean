import Mathlib

namespace greatest_three_digit_multiple_of_17_l1253_125376

open Nat

theorem greatest_three_digit_multiple_of_17 : ∃ n, n % 17 = 0 ∧ 100 ≤ n ∧ n ≤ 999 ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l1253_125376


namespace bathtub_problem_l1253_125347

theorem bathtub_problem (T : ℝ) (h1 : 1 / T - 1 / 12 = 1 / 60) : T = 10 := 
by {
  -- Sorry, skip the proof as requested
  sorry
}

end bathtub_problem_l1253_125347


namespace divide_plane_into_four_quadrants_l1253_125327

-- Definitions based on conditions
def perpendicular_axes (x y : ℝ → ℝ) : Prop :=
  (∀ t : ℝ, x t = t ∨ x t = 0) ∧ (∀ t : ℝ, y t = t ∨ y t = 0) ∧ ∀ t : ℝ, x t ≠ y t

-- The mathematical proof statement
theorem divide_plane_into_four_quadrants (x y : ℝ → ℝ) (hx : perpendicular_axes x y) :
  ∃ quadrants : ℕ, quadrants = 4 :=
by
  sorry

end divide_plane_into_four_quadrants_l1253_125327


namespace ratio_mom_pays_to_total_cost_l1253_125367

-- Definitions based on the conditions from the problem
def num_shirts := 4
def num_pants := 2
def num_jackets := 2
def cost_per_shirt := 8
def cost_per_pant := 18
def cost_per_jacket := 60
def amount_carrie_pays := 94

-- Calculate total costs based on given definitions
def cost_shirts := num_shirts * cost_per_shirt
def cost_pants := num_pants * cost_per_pant
def cost_jackets := num_jackets * cost_per_jacket
def total_cost := cost_shirts + cost_pants + cost_jackets

-- Amount Carrie's mom pays
def amount_mom_pays := total_cost - amount_carrie_pays

-- The proving statement
theorem ratio_mom_pays_to_total_cost : (amount_mom_pays : ℝ) / (total_cost : ℝ) = 1 / 2 :=
by
  sorry

end ratio_mom_pays_to_total_cost_l1253_125367


namespace polar_equation_l1253_125352

theorem polar_equation (y ρ θ : ℝ) (x : ℝ) 
  (h1 : y = ρ * Real.sin θ) 
  (h2 : x = ρ * Real.cos θ) 
  (h3 : y^2 = 12 * x) : 
  ρ * (Real.sin θ)^2 = 12 * Real.cos θ := 
by
  sorry

end polar_equation_l1253_125352


namespace max_area_of_inscribed_equilateral_triangle_l1253_125334

noncomputable def maxInscribedEquilateralTriangleArea : ℝ :=
  let length : ℝ := 12
  let width : ℝ := 15
  let max_area := 369 * Real.sqrt 3 - 540
  max_area

theorem max_area_of_inscribed_equilateral_triangle :
  maxInscribedEquilateralTriangleArea = 369 * Real.sqrt 3 - 540 := 
by
  sorry

end max_area_of_inscribed_equilateral_triangle_l1253_125334


namespace polar_curve_is_parabola_l1253_125384

theorem polar_curve_is_parabola (ρ θ : ℝ) (h : 3 * ρ * Real.sin θ ^ 2 + Real.cos θ = 0) : ∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 3 * y ^ 2 + x = 0 :=
by
  sorry

end polar_curve_is_parabola_l1253_125384


namespace find_m_value_l1253_125368

-- Definitions for the problem conditions are given below
variables (m : ℝ)

-- Conditions
def conditions := (6 < m) ∧ (m < 10) ∧ (4 = 2 * 2) ∧ (4 = (m - 2) - (10 - m))

-- Proof statement
theorem find_m_value : conditions m → m = 8 :=
sorry

end find_m_value_l1253_125368


namespace infinite_a_exists_l1253_125328

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ+), ∃ (m : ℕ+), n^6 + 3 * (a : ℕ) = m^3 :=
  sorry

end infinite_a_exists_l1253_125328


namespace no_solution_exists_l1253_125322

theorem no_solution_exists (p : ℝ) : (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) = (x - p) / (x - 8) → false) ↔ p = 7 :=
by sorry

end no_solution_exists_l1253_125322


namespace simplify_fraction_l1253_125310

theorem simplify_fraction (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b)) :=
by
  sorry

end simplify_fraction_l1253_125310


namespace remainder_when_divided_l1253_125380

theorem remainder_when_divided (L S R : ℕ) (h1: L - S = 1365) (h2: S = 270) (h3: L = 6 * S + R) : 
  R = 15 := 
by 
  sorry

end remainder_when_divided_l1253_125380


namespace min_value_expression_l1253_125303

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2*x + y = 1) : 
  ∃ (xy_min : ℝ), xy_min = 9 ∧ (∀ (x y : ℝ), 0 < x ∧ 0 < y ∧ 2*x + y = 1 → (x + 2*y)/(x*y) ≥ xy_min) :=
sorry

end min_value_expression_l1253_125303


namespace find_positive_x_l1253_125306

theorem find_positive_x :
  ∃ x : ℝ, x > 0 ∧ (1 / 2 * (4 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4))
  ∧ x = 21 + Real.sqrt 449 :=
by
  sorry

end find_positive_x_l1253_125306


namespace consecutive_even_legs_sum_l1253_125350

theorem consecutive_even_legs_sum (x : ℕ) (h : x % 2 = 0) (hx : x ^ 2 + (x + 2) ^ 2 = 34 ^ 2) : x + (x + 2) = 48 := by
  sorry

end consecutive_even_legs_sum_l1253_125350


namespace calculate_molar_mass_l1253_125398

-- Definitions from the conditions
def number_of_moles : ℝ := 8
def weight_in_grams : ℝ := 1600

-- Goal: Prove that the molar mass is 200 grams/mole
theorem calculate_molar_mass : (weight_in_grams / number_of_moles) = 200 :=
by
  sorry

end calculate_molar_mass_l1253_125398


namespace tangent_lines_through_point_l1253_125333

theorem tangent_lines_through_point {x y : ℝ} (h_circle : (x-1)^2 + (y-1)^2 = 1)
  (h_point : ∀ (x y: ℝ), (x, y) = (2, 4)) :
  (x = 2 ∨ 4 * x - 3 * y + 4 = 0) :=
sorry

end tangent_lines_through_point_l1253_125333


namespace points_lie_on_circle_l1253_125321

theorem points_lie_on_circle (t : ℝ) : 
  let x := Real.cos t
  let y := Real.sin t
  x^2 + y^2 = 1 :=
by
  sorry

end points_lie_on_circle_l1253_125321


namespace find_number_l1253_125386

def single_digit (n : ℕ) : Prop := n < 10
def greater_than_zero (n : ℕ) : Prop := n > 0
def less_than_two (n : ℕ) : Prop := n < 2

theorem find_number (n : ℕ) : 
  single_digit n ∧ greater_than_zero n ∧ less_than_two n → n = 1 :=
by
  sorry

end find_number_l1253_125386


namespace conditions_neither_necessary_nor_sufficient_l1253_125312

theorem conditions_neither_necessary_nor_sufficient :
  (¬(0 < x ∧ x < 2) ↔ (¬(-1 / 2 < x ∨ x < 1)) ∨ (¬(-1 / 2 < x ∧ x < 1))) :=
by sorry

end conditions_neither_necessary_nor_sufficient_l1253_125312


namespace second_polygon_sides_l1253_125378

/--
Given two regular polygons where:
- The first polygon has 42 sides.
- Each side of the first polygon is three times the length of each side of the second polygon.
- The perimeters of both polygons are equal.
Prove that the second polygon has 126 sides.
-/
theorem second_polygon_sides
  (s : ℝ) -- the side length of the second polygon
  (h1 : ∃ n : ℕ, n = 42) -- the first polygon has 42 sides
  (h2 : ∃ m : ℝ, m = 3 * s) -- the side length of the first polygon is three times the side length of the second polygon
  (h3 : ∃ k : ℕ, k * (3 * s) = n * s) -- the perimeters of both polygons are equal
  : ∃ n2 : ℕ, n2 = 126 := 
by
  sorry

end second_polygon_sides_l1253_125378


namespace maximum_s_squared_l1253_125390

-- Definitions based on our conditions
def semicircle_radius : ℝ := 5
def diameter_length : ℝ := 10

-- Statement of the problem (no proof, statement only)
theorem maximum_s_squared (A B C : ℝ×ℝ) (AC BC : ℝ) (h : AC + BC = s) :
    (A.2 = 0) ∧ (B.2 = 0) ∧ (dist A B = diameter_length) ∧
    (dist C (5,0) = semicircle_radius) ∧ (s = AC + BC) →
    s^2 ≤ 200 :=
sorry

end maximum_s_squared_l1253_125390


namespace largest_number_l1253_125346

-- Definitions based on the conditions
def numA := 0.893
def numB := 0.8929
def numC := 0.8931
def numD := 0.839
def numE := 0.8391

-- The statement to be proved 
theorem largest_number : numB = max numA (max numB (max numC (max numD numE))) := by
  sorry

end largest_number_l1253_125346


namespace inequality_solution_l1253_125388

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 > (3 * x - 2) / 2 - 1 → x < 2 :=
by
  intro h
  sorry

end inequality_solution_l1253_125388


namespace sum_f_x₁_f_x₂_lt_0_l1253_125325

variable (f : ℝ → ℝ)
variable (x₁ x₂ : ℝ)

-- Condition: y = f(x + 2) is an odd function
def odd_function_on_shifted_domain : Prop :=
  ∀ x, f (x + 2) = -f (-(x + 2))

-- Condition: f(x) is monotonically increasing for x > 2
def monotonically_increasing_for_x_gt_2 : Prop :=
  ∀ x₁ x₂, 2 < x₁ → x₁ < x₂ → f x₁ < f x₂

-- Condition: x₁ + x₂ < 4
def sum_lt_4 : Prop :=
  x₁ + x₂ < 4

-- Condition: (x₁-2)(x₂-2) < 0
def product_shift_lt_0 : Prop :=
  (x₁ - 2) * (x₂ - 2) < 0

-- Main theorem to prove f(x₁) + f(x₂) < 0
theorem sum_f_x₁_f_x₂_lt_0
  (h1 : odd_function_on_shifted_domain f)
  (h2 : monotonically_increasing_for_x_gt_2 f)
  (h3 : sum_lt_4 x₁ x₂)
  (h4 : product_shift_lt_0 x₁ x₂) :
  f x₁ + f x₂ < 0 := sorry

end sum_f_x₁_f_x₂_lt_0_l1253_125325


namespace value_of_a_l1253_125340

theorem value_of_a 
  (a b c d e : ℤ)
  (h1 : a + 4 = b + 2)
  (h2 : a + 2 = b)
  (h3 : a + c = 146)
  (he : e = 79)
  (h4 : e = d + 2)
  (h5 : d = c + 2)
  (h6 : c = b + 2) :
  a = 71 :=
by
  sorry

end value_of_a_l1253_125340


namespace family_e_initial_members_l1253_125307

theorem family_e_initial_members 
(a b c d f E : ℕ) 
(h_a : a = 7) 
(h_b : b = 8) 
(h_c : c = 10) 
(h_d : d = 13) 
(h_f : f = 10)
(h_avg : (a - 1 + b - 1 + c - 1 + d - 1 + E - 1 + f - 1) / 6 = 8) : 
E = 6 := 
by 
  sorry

end family_e_initial_members_l1253_125307


namespace machine_present_value_l1253_125379

/-- A machine depreciates at a certain rate annually.
    Given the future value after a certain number of years and the depreciation rate,
    prove the present value of the machine. -/
theorem machine_present_value
  (depreciation_rate : ℝ := 0.25)
  (future_value : ℝ := 54000)
  (years : ℕ := 3)
  (pv : ℝ := 128000) :
  (future_value = pv * (1 - depreciation_rate) ^ years) :=
sorry

end machine_present_value_l1253_125379


namespace simplify_expression_l1253_125338

theorem simplify_expression : 5 * (14 / 3) * (21 / -70) = - 35 / 2 := by
  sorry

end simplify_expression_l1253_125338


namespace quadrilateral_midpoints_area_l1253_125358

-- We set up the geometric context and define the problem in Lean 4.

noncomputable def area_of_midpoint_quadrilateral
  (AB CD : ℝ) (AD BC : ℝ)
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) : ℝ :=
  37.5

-- The theorem statement validating the area of the quadrilateral.
theorem quadrilateral_midpoints_area (AB CD AD BC : ℝ) 
  (h_AB_CD : AB = 15) (h_CD_AB : CD = 15)
  (h_AD_BC : AD = 10) (h_BC_AD : BC = 10)
  (mid_AB : Prop) (mid_BC : Prop) (mid_CD : Prop) (mid_DA : Prop) :
  area_of_midpoint_quadrilateral AB CD AD BC h_AB_CD h_CD_AB h_AD_BC h_BC_AD mid_AB mid_BC mid_CD mid_DA = 37.5 :=
by 
  sorry  -- Proof is omitted.

end quadrilateral_midpoints_area_l1253_125358


namespace inequality_proof_l1253_125357

variables {x y a b ε m : ℝ}

theorem inequality_proof (h1 : |x - a| < ε / (2 * m))
                        (h2 : |y - b| < ε / (2 * |a|))
                        (h3 : 0 < y ∧ y < m) :
                        |x * y - a * b| < ε :=
sorry

end inequality_proof_l1253_125357


namespace common_chord_is_linear_l1253_125309

-- Defining the equations of two intersecting circles
noncomputable def circle1 : ℝ → ℝ → ℝ := sorry
noncomputable def circle2 : ℝ → ℝ → ℝ := sorry

-- Defining a method to eliminate quadratic terms
noncomputable def eliminate_quadratic_terms (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Defining the linear equation representing the common chord
noncomputable def common_chord (eq1 eq2 : ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ := sorry

-- Statement of the problem
theorem common_chord_is_linear (circle1 circle2 : ℝ → ℝ → ℝ) :
  common_chord circle1 circle2 = eliminate_quadratic_terms circle1 circle2 := sorry

end common_chord_is_linear_l1253_125309


namespace ratio_of_arithmetic_sums_l1253_125304

theorem ratio_of_arithmetic_sums : 
  let a1 := 4
  let d1 := 4
  let l1 := 48
  let a2 := 2
  let d2 := 3
  let l2 := 35
  let n1 := (l1 - a1) / d1 + 1
  let n2 := (l2 - a2) / d2 + 1
  let S1 := n1 * (a1 + l1) / 2
  let S2 := n2 * (a2 + l2) / 2
  let ratio := S1 / S2
  ratio = 52 / 37 := by sorry

end ratio_of_arithmetic_sums_l1253_125304


namespace corn_syrup_amount_l1253_125323

-- Definitions based on given conditions
def flavoring_to_corn_syrup_standard := 1 / 12
def flavoring_to_water_standard := 1 / 30

def flavoring_to_corn_syrup_sport := (3 * flavoring_to_corn_syrup_standard)
def flavoring_to_water_sport := (1 / 2) * flavoring_to_water_standard

def common_factor := (30 : ℝ)

-- Amounts in sport formulation after adjustment
def flavoring_to_corn_syrup_ratio_sport := 1 / 4
def flavoring_to_water_ratio_sport := 1 / 60

def total_flavoring_corn_syrup := 15 -- Since ratio is 15:60:60 and given water is 15 ounces

theorem corn_syrup_amount (water_ounces : ℝ) :
  water_ounces = 15 → 
  (60 / 60) * water_ounces = 15 :=
by
  sorry

end corn_syrup_amount_l1253_125323


namespace quadratic_equation_C_has_real_solutions_l1253_125319

theorem quadratic_equation_C_has_real_solutions :
  ∀ (x : ℝ), ∃ (a b c : ℝ), a = 1 ∧ b = 3 ∧ c = -2 ∧ a*x^2 + b*x + c = 0 :=
by
  sorry

end quadratic_equation_C_has_real_solutions_l1253_125319


namespace f_positive_l1253_125317

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

axiom f_monotonically_decreasing : ∀ x y : ℝ, x < y → f x > f y
axiom inequality_condition : ∀ x : ℝ, (f x) / (f'' x) + x < 1

theorem f_positive : ∀ x : ℝ, f x > 0 :=
by sorry

end f_positive_l1253_125317


namespace ratio_of_cream_max_to_maxine_l1253_125343

def ounces_of_cream_in_max (coffee_sipped : ℕ) (cream_added: ℕ) : ℕ := cream_added

def ounces_of_remaining_cream_in_maxine (initial_coffee : ℚ) (cream_added: ℚ) (sipped : ℚ) : ℚ :=
  let total_mixture := initial_coffee + cream_added
  let remaining_mixture := total_mixture - sipped
  (initial_coffee / total_mixture) * cream_added

theorem ratio_of_cream_max_to_maxine :
  let max_cream := ounces_of_cream_in_max 4 3
  let maxine_cream := ounces_of_remaining_cream_in_maxine 16 3 5
  (max_cream : ℚ) / maxine_cream = 19 / 14 := by 
  sorry

end ratio_of_cream_max_to_maxine_l1253_125343


namespace complex_inequality_l1253_125351

theorem complex_inequality (m : ℝ) : 
  (m - 3 ≥ 0 ∧ m^2 - 9 = 0) → m = 3 := 
by
  sorry

end complex_inequality_l1253_125351


namespace point_in_second_quadrant_l1253_125372

theorem point_in_second_quadrant {x y : ℝ} (hx : x < 0) (hy : y > 0) : 
  ∃ q, q = 2 :=
by
  sorry

end point_in_second_quadrant_l1253_125372


namespace Q_polynomial_l1253_125391

def cos3x_using_cos2x (cos_α : ℝ) := (2 * cos_α^2 - 1) * cos_α - 2 * (1 - cos_α^2) * cos_α

def Q (x : ℝ) := 4 * x^3 - 3 * x

theorem Q_polynomial (α : ℝ) : Q (Real.cos α) = Real.cos (3 * α) := by
  rw [Real.cos_three_mul]
  sorry

end Q_polynomial_l1253_125391


namespace max_m_n_squared_l1253_125374

theorem max_m_n_squared (m n : ℤ) 
  (hmn : 1 ≤ m ∧ m ≤ 1981 ∧ 1 ≤ n ∧ n ≤ 1981)
  (h_eq : (n^2 - m*n - m^2)^2 = 1) : 
  m^2 + n^2 ≤ 3524578 :=
sorry

end max_m_n_squared_l1253_125374


namespace seq_le_n_squared_l1253_125397

theorem seq_le_n_squared (a : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (h_property : ∀ t, ∃ i j, t = a i ∨ t = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by {
  sorry
}

end seq_le_n_squared_l1253_125397


namespace perimeter_of_polygon_l1253_125330

-- Conditions
variables (a b : ℝ) (polygon_is_part_of_rectangle : 0 < a ∧ 0 < b)

-- Prove that if the polygon completes a rectangle with perimeter 28,
-- then the perimeter of the polygon is 28.
theorem perimeter_of_polygon (h : 2 * (a + b) = 28) : 2 * (a + b) = 28 :=
by
  exact h

end perimeter_of_polygon_l1253_125330


namespace range_of_a_l1253_125389

def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, ¬ (x^2 + (a-1)*x + 1 ≤ 0)
def proposition_q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a-1)^x < (a-1)^y

theorem range_of_a (a : ℝ) :
  ¬ (proposition_p a ∧ proposition_q a) ∧ (proposition_p a ∨ proposition_q a) →
  (-1 < a ∧ a ≤ 2) ∨ (3 ≤ a) :=
by sorry

end range_of_a_l1253_125389


namespace ajax_weight_after_exercise_l1253_125332

theorem ajax_weight_after_exercise :
  ∀ (initial_weight_kg : ℕ) (conversion_rate : ℝ) (daily_exercise_hours : ℕ) (exercise_loss_rate : ℝ) (days_in_week : ℕ) (weeks : ℕ),
    initial_weight_kg = 80 →
    conversion_rate = 2.2 →
    daily_exercise_hours = 2 →
    exercise_loss_rate = 1.5 →
    days_in_week = 7 →
    weeks = 2 →
    initial_weight_kg * conversion_rate - daily_exercise_hours * exercise_loss_rate * (days_in_week * weeks) = 134 :=
by
  intros
  sorry

end ajax_weight_after_exercise_l1253_125332


namespace tan_add_pi_over_3_l1253_125359

variable (y : ℝ)

theorem tan_add_pi_over_3 (h : Real.tan y = 3) : 
  Real.tan (y + π / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) := 
by
  sorry

end tan_add_pi_over_3_l1253_125359


namespace shortest_distance_from_origin_l1253_125361

noncomputable def shortest_distance_to_circle (x y : ℝ) : ℝ :=
  x^2 + 6 * x + y^2 - 8 * y + 18

theorem shortest_distance_from_origin :
  ∃ (d : ℝ), d = 5 - Real.sqrt 7 ∧ ∀ (x y : ℝ), shortest_distance_to_circle x y = 0 →
    (Real.sqrt ((x - 0)^2 + (y - 0)^2) - Real.sqrt ((x + 3)^2 + (y - 4)^2)) = d := sorry

end shortest_distance_from_origin_l1253_125361


namespace remainder_sum_mod_13_l1253_125356

theorem remainder_sum_mod_13 : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 :=
by
  sorry

end remainder_sum_mod_13_l1253_125356


namespace sufficient_but_not_necessary_condition_l1253_125313

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 → ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) ∧
  ¬ (a = 1 → (∀ x y : ℝ, ax + 2 * y - 1 = 0 ↔ ∃ x' y' : ℝ, x' + (a + 1) * y' + 4 = 0)) := 
sorry

end sufficient_but_not_necessary_condition_l1253_125313


namespace symmetric_circle_proof_l1253_125362

-- Define the original circle equation
def original_circle_eq (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def line_of_symmetry (x y : ℝ) : Prop :=
  y = x

-- Define the symmetric circle equation
def symmetric_circle_eq (x y : ℝ) : Prop :=
  x^2 + (y + 2)^2 = 5

-- The theorem to prove
theorem symmetric_circle_proof (x y : ℝ) :
  (original_circle_eq x y) ↔ (symmetric_circle_eq x y) :=
sorry

end symmetric_circle_proof_l1253_125362


namespace number_of_questions_in_test_l1253_125382

variable (n : ℕ) -- the total number of questions
variable (correct_answers : ℕ) -- the number of correct answers
variable (sections : ℕ) -- number of sections in the test
variable (questions_per_section : ℕ) -- number of questions per section
variable (percentage_correct : ℚ) -- percentage of correct answers

-- Given conditions
def conditions := 
  correct_answers = 32 ∧ 
  sections = 5 ∧ 
  questions_per_section * sections = n ∧ 
  (70 : ℚ) < percentage_correct ∧ 
  percentage_correct < 77 ∧ 
  percentage_correct * n = 3200

-- The main statement to prove
theorem number_of_questions_in_test : conditions n correct_answers sections questions_per_section percentage_correct → 
  n = 45 :=
by
  sorry

end number_of_questions_in_test_l1253_125382


namespace probability_of_A_losing_l1253_125331

variable (p_win p_draw p_lose : ℝ)

def probability_of_A_winning := p_win = (1/3)
def probability_of_draw := p_draw = (1/2)
def sum_of_probabilities := p_win + p_draw + p_lose = 1

theorem probability_of_A_losing
  (h1: probability_of_A_winning p_win)
  (h2: probability_of_draw p_draw)
  (h3: sum_of_probabilities p_win p_draw p_lose) :
  p_lose = (1/6) :=
sorry

end probability_of_A_losing_l1253_125331


namespace inequality_proof_l1253_125335

theorem inequality_proof (a b : ℝ) (h : (a = 0 ∨ b = 0 ∨ (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0))) :
  a^4 + 2*a^3*b + 2*a*b^3 + b^4 ≥ 6*a^2*b^2 :=
by
  sorry

end inequality_proof_l1253_125335


namespace common_root_unique_solution_l1253_125354

theorem common_root_unique_solution
  (p : ℝ) (h : ∃ x, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) :
  p = 3 :=
by sorry

end common_root_unique_solution_l1253_125354


namespace pairs_divisible_by_7_l1253_125341

theorem pairs_divisible_by_7 :
  (∃ (pairs : List (ℕ × ℕ)), 
    (∀ p ∈ pairs, (1 ≤ p.fst ∧ p.fst ≤ 1000) ∧ (1 ≤ p.snd ∧ p.snd ≤ 1000) ∧ (p.fst^2 + p.snd^2) % 7 = 0) ∧ 
    pairs.length = 20164) :=
sorry

end pairs_divisible_by_7_l1253_125341


namespace prove_value_of_question_l1253_125393

theorem prove_value_of_question :
  let a := 9548
  let b := 7314
  let c := 3362
  let value_of_question : ℕ := by 
    sorry -- Proof steps to show the computation.

  (a + b = value_of_question) ∧ (c + 13500 = value_of_question) :=
by {
  let a := 9548
  let b := 7314
  let c := 3362
  let sum_of_a_b := a + b
  let computed_question := sum_of_a_b - c
  sorry -- Proof steps to show sum_of_a_b and the final result.
}

end prove_value_of_question_l1253_125393


namespace simplify_expression_l1253_125336

variable (a b : ℤ)

theorem simplify_expression :
  (15 * a + 45 * b) + (20 * a + 35 * b) - (25 * a + 55 * b) + (30 * a - 5 * b) = 
  40 * a + 20 * b :=
by
  sorry

end simplify_expression_l1253_125336


namespace find_angle_A_find_bc_range_l1253_125324

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  (c * (a * Real.cos B - (1/2) * b) = a^2 - b^2) ∧ (A = Real.arccos (1/2))

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) :
  A = Real.pi / 3 := 
sorry

theorem find_bc_range (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) (ha : a = Real.sqrt 3) :
  b + c ∈ Set.Icc (Real.sqrt 3) (2 * Real.sqrt 3) := 
sorry

end find_angle_A_find_bc_range_l1253_125324


namespace difference_between_numbers_l1253_125387

-- Given definitions based on conditions
def sum_of_two_numbers (x y : ℝ) : Prop := x + y = 15
def difference_of_two_numbers (x y : ℝ) : Prop := x - y = 10
def difference_of_squares (x y : ℝ) : Prop := x^2 - y^2 = 150

theorem difference_between_numbers (x y : ℝ) 
  (h1 : sum_of_two_numbers x y) 
  (h2 : difference_of_two_numbers x y) 
  (h3 : difference_of_squares x y) :
  x - y = 10 :=
by
  sorry

end difference_between_numbers_l1253_125387


namespace oil_bill_for_January_l1253_125344

variables (J F : ℝ)

-- Conditions
def condition1 := F = (5 / 4) * J
def condition2 := (F + 45) / J = 3 / 2

theorem oil_bill_for_January (h1 : condition1 J F) (h2 : condition2 J F) : J = 180 :=
by sorry

end oil_bill_for_January_l1253_125344


namespace chess_tournament_total_games_l1253_125314

theorem chess_tournament_total_games (n : ℕ) (h : n = 10) : (n * (n - 1)) / 2 = 45 := by
  sorry

end chess_tournament_total_games_l1253_125314


namespace johannes_sells_48_kg_l1253_125366

-- Define Johannes' earnings
def earnings_wednesday : ℕ := 30
def earnings_friday : ℕ := 24
def earnings_today : ℕ := 42

-- Price per kilogram of cabbage
def price_per_kg : ℕ := 2

-- Prove that the total kilograms of cabbage sold is 48
theorem johannes_sells_48_kg :
  ((earnings_wednesday + earnings_friday + earnings_today) / price_per_kg) = 48 := by
  sorry

end johannes_sells_48_kg_l1253_125366


namespace nonzero_rational_pow_zero_l1253_125377

theorem nonzero_rational_pow_zero 
  (num : ℤ) (denom : ℤ) (hnum : num = -1241376497) (hdenom : denom = 294158749357) (h_nonzero: num ≠ 0 ∧ denom ≠ 0) :
  (num / denom : ℚ) ^ 0 = 1 := 
by 
  sorry

end nonzero_rational_pow_zero_l1253_125377


namespace jerry_trays_l1253_125349

theorem jerry_trays :
  ∀ (trays_from_table1 trays_from_table2 trips trays_per_trip : ℕ),
  trays_from_table1 = 9 →
  trays_from_table2 = 7 →
  trips = 2 →
  trays_from_table1 + trays_from_table2 = 16 →
  trays_per_trip = (trays_from_table1 + trays_from_table2) / trips →
  trays_per_trip = 8 :=
by
  intros
  sorry

end jerry_trays_l1253_125349


namespace minimum_f_zero_iff_t_is_2sqrt2_l1253_125300

noncomputable def f (x t : ℝ) : ℝ := 4 * x^4 - 6 * t * x^3 + (2 * t + 6) * x^2 - 3 * t * x + 1

theorem minimum_f_zero_iff_t_is_2sqrt2 :
  (∀ x > 0, f x t ≥ 0) ∧ (∃ x > 0, f x t = 0) ↔ t = 2 * Real.sqrt 2 := 
sorry

end minimum_f_zero_iff_t_is_2sqrt2_l1253_125300


namespace milo_dozen_eggs_l1253_125302

theorem milo_dozen_eggs (total_weight_pounds egg_weight_pounds dozen : ℕ) (h1 : total_weight_pounds = 6)
  (h2 : egg_weight_pounds = 1 / 16) (h3 : dozen = 12) :
  total_weight_pounds / egg_weight_pounds / dozen = 8 :=
by
  -- The proof would go here
  sorry

end milo_dozen_eggs_l1253_125302


namespace jimmy_bread_packs_needed_l1253_125301

theorem jimmy_bread_packs_needed 
  (sandwiches : ℕ)
  (slices_per_sandwich : ℕ)
  (initial_bread_slices : ℕ)
  (slices_per_pack : ℕ)
  (H1 : sandwiches = 8)
  (H2 : slices_per_sandwich = 2)
  (H3 : initial_bread_slices = 0)
  (H4 : slices_per_pack = 4) : 
  (8 * 2) / 4 = 4 := 
sorry

end jimmy_bread_packs_needed_l1253_125301


namespace distance_midpoint_parabola_y_axis_l1253_125385

theorem distance_midpoint_parabola_y_axis (x1 x2 : ℝ) (y1 y2 : ℝ) 
  (hA : y1 ^ 2 = x1) (hB : y2 ^ 2 = x2) 
  (h_focus : ∀ {p : ℝ × ℝ}, p = (x1, y1) ∨ p = (x2, y2) → |p.1 - 1/4| = |p.1 + 1/4|)
  (h_dist : |x1 - 1/4| + |x2 - 1/4| = 3) :
  abs ((x1 + x2) / 2) = 5 / 4 :=
by sorry

end distance_midpoint_parabola_y_axis_l1253_125385


namespace total_pencil_length_l1253_125395

-- Definitions from the conditions
def purple_length : ℕ := 3
def black_length : ℕ := 2
def blue_length : ℕ := 1

-- Proof statement
theorem total_pencil_length :
  purple_length + black_length + blue_length = 6 :=
by
  sorry

end total_pencil_length_l1253_125395


namespace least_possible_value_of_z_minus_x_l1253_125373

variables (x y z : ℤ)

-- Define the conditions
def even (n : ℤ) := ∃ k : ℤ, n = 2 * k
def odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

-- State the theorem
theorem least_possible_value_of_z_minus_x (h1 : x < y) (h2 : y < z) (h3 : y - x > 3) 
    (hx_even : even x) (hy_odd : odd y) (hz_odd : odd z) : z - x = 7 :=
sorry

end least_possible_value_of_z_minus_x_l1253_125373


namespace unit_vector_parallel_to_d_l1253_125339

theorem unit_vector_parallel_to_d (x y: ℝ): (4 * x - 3 * y = 0) ∧ (x^2 + y^2 = 1) → (x = 3/5 ∧ y = 4/5) ∨ (x = -3/5 ∧ y = -4/5) :=
by sorry

end unit_vector_parallel_to_d_l1253_125339


namespace reduced_price_l1253_125342

variable (P : ℝ)  -- the original price per kg
variable (reduction_factor : ℝ := 0.5)  -- 50% reduction
variable (extra_kgs : ℝ := 5)  -- 5 kgs more
variable (total_cost : ℝ := 800)  -- Rs. 800

theorem reduced_price :
  total_cost / (P * (1 - reduction_factor)) = total_cost / P + extra_kgs → 
  P / 2 = 80 :=
by
  sorry

end reduced_price_l1253_125342


namespace ways_to_go_from_first_to_fifth_l1253_125370

theorem ways_to_go_from_first_to_fifth (floors : ℕ) (staircases_per_floor : ℕ) (total_ways : ℕ) 
    (h1 : floors = 5) (h2 : staircases_per_floor = 2) (h3 : total_ways = 2^4) : total_ways = 16 :=
by
  sorry

end ways_to_go_from_first_to_fifth_l1253_125370


namespace neg_p_necessary_not_sufficient_neg_q_l1253_125329

def p (x : ℝ) : Prop := x^2 - 1 > 0
def q (x : ℝ) : Prop := (x + 1) * (x - 2) > 0
def not_p (x : ℝ) : Prop := ¬ (p x)
def not_q (x : ℝ) : Prop := ¬ (q x)

theorem neg_p_necessary_not_sufficient_neg_q : ∀ (x : ℝ), (not_q x → not_p x) ∧ ¬ (not_p x → not_q x) :=
by
  sorry

end neg_p_necessary_not_sufficient_neg_q_l1253_125329


namespace sector_area_l1253_125353

theorem sector_area (r θ : ℝ) (hr : r = 1) (hθ : θ = 2) : 
  (1 / 2) * r * r * θ = 1 := by
sorry

end sector_area_l1253_125353


namespace arithmetic_sequence_a2_value_l1253_125363

open Nat

theorem arithmetic_sequence_a2_value (a : ℕ → ℕ) (d : ℕ) 
  (h1 : a 1 = 3) 
  (h2 : a 2 + a 3 = 12) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n + d) : 
  a 2 = 5 :=
  sorry

end arithmetic_sequence_a2_value_l1253_125363


namespace total_number_of_fish_l1253_125337

theorem total_number_of_fish (fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1: fishbowls = 261) (h2: fish_per_bowl = 23) : 
  fishbowls * fish_per_bowl = 6003 := 
by
  sorry

end total_number_of_fish_l1253_125337


namespace tan_double_angle_l1253_125316

theorem tan_double_angle (α : ℝ) (h₁ : Real.sin α = 4/5) (h₂ : α ∈ Set.Ioc (π / 2) π) :
  Real.tan (2 * α) = 24 / 7 := 
  sorry

end tan_double_angle_l1253_125316


namespace percentage_error_x_percentage_error_y_l1253_125369

theorem percentage_error_x (x : ℝ) : 
  let correct_result := x * 10
  let erroneous_result := x / 10
  (correct_result - erroneous_result) / correct_result * 100 = 99 :=
by
  sorry

theorem percentage_error_y (y : ℝ) : 
  let correct_result := y + 15
  let erroneous_result := y - 15
  (correct_result - erroneous_result) / correct_result * 100 = (30 / (y + 15)) * 100 :=
by
  sorry

end percentage_error_x_percentage_error_y_l1253_125369


namespace train_length_l1253_125311

theorem train_length
  (speed_kmph : ℕ)
  (platform_length : ℕ)
  (time_seconds : ℕ)
  (speed_m_per_s : speed_kmph * 1000 / 3600 = 20)
  (distance_covered : 20 * time_seconds = 520)
  (platform_eq : platform_length = 280)
  (time_eq : time_seconds = 26) :
  ∃ L : ℕ, L = 240 := by
  sorry

end train_length_l1253_125311


namespace top_leftmost_rectangle_is_E_l1253_125308

def rectangle (w x y z : ℕ) : Prop := true

-- Define the rectangles according to the given conditions
def rectangle_A : Prop := rectangle 4 1 6 9
def rectangle_B : Prop := rectangle 1 0 3 6
def rectangle_C : Prop := rectangle 3 8 5 2
def rectangle_D : Prop := rectangle 7 5 4 8
def rectangle_E : Prop := rectangle 9 2 7 0

-- Prove that the top leftmost rectangle is E
theorem top_leftmost_rectangle_is_E : rectangle_E → True :=
by
  sorry

end top_leftmost_rectangle_is_E_l1253_125308


namespace min_value_of_frac_expr_l1253_125383

theorem min_value_of_frac_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (3 / a) + (2 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_of_frac_expr_l1253_125383


namespace maximum_value_of_n_with_positive_sequence_l1253_125318

theorem maximum_value_of_n_with_positive_sequence (a : ℕ → ℝ) (h_seq : ∀ n : ℕ, 0 < a n) 
    (h_arithmetic : ∀ n : ℕ, a (n + 1)^2 - a n^2 = 1) : ∃ n : ℕ, n = 24 ∧ a n < 5 :=
by
  sorry

end maximum_value_of_n_with_positive_sequence_l1253_125318


namespace sufficient_but_not_necessary_l1253_125392

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l1253_125392


namespace smallest_n_to_make_183_divisible_by_11_l1253_125381

theorem smallest_n_to_make_183_divisible_by_11 : ∃ n : ℕ, 183 + n % 11 = 0 ∧ n = 4 :=
by
  have h1 : 183 % 11 = 7 := 
    sorry
  let n := 11 - (183 % 11)
  have h2 : 183 + n % 11 = 0 :=
    sorry
  exact ⟨n, h2, sorry⟩

end smallest_n_to_make_183_divisible_by_11_l1253_125381


namespace wilson_theorem_non_prime_divisibility_l1253_125355

theorem wilson_theorem (p : ℕ) (h : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

theorem non_prime_divisibility (p : ℕ) (h : ¬ Nat.Prime p) : ¬ p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

end wilson_theorem_non_prime_divisibility_l1253_125355


namespace curve_C2_eq_l1253_125360

def curve_C (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def reflect_y_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := f (-x)
def reflect_x_axis (f : ℝ → ℝ) (x : ℝ) : ℝ := - (f x)

theorem curve_C2_eq (a b c : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, reflect_x_axis (reflect_y_axis (curve_C a b c)) x = -a * x^2 + b * x - c := by
  sorry

end curve_C2_eq_l1253_125360


namespace geom_series_first_term_l1253_125396

theorem geom_series_first_term (a r : ℝ) 
  (h1 : a / (1 - r) = 30)
  (h2 : a^2 / (1 - r^2) = 120) : 
  a = 120 / 17 :=
by
  sorry

end geom_series_first_term_l1253_125396


namespace charge_for_cat_l1253_125315

theorem charge_for_cat (D N_D N_C T C : ℝ) 
  (h1 : D = 60) (h2 : N_D = 20) (h3 : N_C = 60) (h4 : T = 3600)
  (h5 : 20 * D + 60 * C = T) :
  C = 40 := by
  sorry

end charge_for_cat_l1253_125315


namespace no_solution_for_x_l1253_125305

theorem no_solution_for_x (x : ℝ) :
  (1 / (x + 4) + 1 / (x - 4) = 1 / (x - 4)) → False :=
by
  sorry

end no_solution_for_x_l1253_125305


namespace percentage_square_area_in_rectangle_l1253_125371

variable (s : ℝ)
variable (W : ℝ) (L : ℝ)
variable (hW : W = 3 * s) -- Width is 3 times the side of the square
variable (hL : L = (3 / 2) * W) -- Length is 3/2 times the width

theorem percentage_square_area_in_rectangle :
  (s^2 / ((27 * s^2) / 2)) * 100 = 7.41 :=
by 
  sorry

end percentage_square_area_in_rectangle_l1253_125371


namespace min_omega_symmetry_l1253_125345

noncomputable def f (x : ℝ) : ℝ := Real.cos x

theorem min_omega_symmetry :
  ∃ (omega : ℝ), omega > 0 ∧ 
  (∀ x : ℝ, Real.cos (omega * (x - π / 12)) = Real.cos (omega * (2 * (π / 4) - x) - omega * π / 12) ) ∧ 
  (∀ ω_, ω_ > 0 → 
  (∀ x : ℝ, Real.cos (ω_ * (x - π / 12)) = Real.cos (ω_ * (2 * (π / 4) - x) - ω_ * π / 12) → 
  omega ≤ ω_)) ∧ omega = 6 :=
sorry

end min_omega_symmetry_l1253_125345


namespace value_of_x_l1253_125365

theorem value_of_x (x : ℝ) (h : x = 90 + (11 / 100) * 90) : x = 99.9 :=
by {
  sorry
}

end value_of_x_l1253_125365


namespace save_after_increase_l1253_125320

def monthly_saving_initial (salary : ℕ) (saving_percentage : ℕ) : ℕ :=
  salary * saving_percentage / 100

def monthly_expense_initial (salary : ℕ) (saving : ℕ) : ℕ :=
  salary - saving

def increase_by_percentage (amount : ℕ) (percentage : ℕ) : ℕ :=
  amount * percentage / 100

def new_expense (initial_expense : ℕ) (increase : ℕ) : ℕ :=
  initial_expense + increase

def new_saving (salary : ℕ) (expense : ℕ) : ℕ :=
  salary - expense

theorem save_after_increase (salary saving_percentage increase_percentage : ℕ) 
  (H_salary : salary = 5500) 
  (H_saving_percentage : saving_percentage = 20) 
  (H_increase_percentage : increase_percentage = 20) :
  new_saving salary (new_expense (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) (increase_by_percentage (monthly_expense_initial salary (monthly_saving_initial salary saving_percentage)) increase_percentage)) = 220 := 
by
  sorry

end save_after_increase_l1253_125320


namespace min_value_expr_l1253_125394

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 := 
sorry

end min_value_expr_l1253_125394


namespace cube_polygon_area_l1253_125399

theorem cube_polygon_area (cube_side : ℝ) 
  (A B C D : ℝ × ℝ × ℝ)
  (P Q R : ℝ × ℝ × ℝ)
  (hP : P = (10, 0, 0))
  (hQ : Q = (30, 0, 20))
  (hR : R = (30, 5, 30))
  (hA : A = (0, 0, 0))
  (hB : B = (30, 0, 0))
  (hC : C = (30, 0, 30))
  (hD : D = (30, 30, 30))
  (cube_length : cube_side = 30) :
  ∃ area, area = 450 := 
sorry

end cube_polygon_area_l1253_125399


namespace smallest_positive_perfect_square_div_by_2_3_5_l1253_125326

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l1253_125326


namespace problem_proof_l1253_125364

theorem problem_proof (x y z : ℝ) 
  (h1 : 1/x + 2/y + 3/z = 0) 
  (h2 : 1/x - 6/y - 5/z = 0) : 
  (x / y + y / z + z / x) = -1 := 
by
  sorry

end problem_proof_l1253_125364


namespace minimum_value_is_12_l1253_125348

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) : ℝ :=
(a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (a + d)) + (1 / (b + c)) + (1 / (b + d)) + (1 / (c + d)))

theorem minimum_value_is_12 (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) :
  smallest_possible_value a b c d h1 h2 h3 h4 h5 h6 h7 ≥ 12 :=
sorry

end minimum_value_is_12_l1253_125348


namespace candy_store_total_sales_l1253_125375

def price_per_pound_fudge : ℝ := 2.50
def pounds_fudge : ℕ := 20
def price_per_truffle : ℝ := 1.50
def dozens_truffles : ℕ := 5
def price_per_pretzel : ℝ := 2.00
def dozens_pretzels : ℕ := 3

theorem candy_store_total_sales :
  price_per_pound_fudge * pounds_fudge +
  price_per_truffle * (dozens_truffles * 12) +
  price_per_pretzel * (dozens_pretzels * 12) = 212.00 := by
  sorry

end candy_store_total_sales_l1253_125375
