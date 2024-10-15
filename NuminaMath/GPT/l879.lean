import Mathlib

namespace NUMINAMATH_GPT_area_of_region_l879_87929

theorem area_of_region : 
  (∃ (x y : ℝ), x^2 + y^2 + 6 * x - 4 * y - 11 = 0) -> 
  ∃ (A : ℝ), A = 24 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_area_of_region_l879_87929


namespace NUMINAMATH_GPT_almond_croissant_price_l879_87915

theorem almond_croissant_price (R : ℝ) (T : ℝ) (W : ℕ) (total_spent : ℝ) (regular_price : ℝ) (weeks_in_year : ℕ) :
  R = 3.50 →
  T = 468 →
  W = 52 →
  (total_spent = 468) →
  (weekly_regular : ℝ) = 52 * 3.50 →
  (almond_total_cost : ℝ) = (total_spent - weekly_regular) →
  (A : ℝ) = (almond_total_cost / 52) →
  A = 5.50 := by
  intros hR hT hW htotal_spent hweekly_regular halmond_total_cost hA
  sorry

end NUMINAMATH_GPT_almond_croissant_price_l879_87915


namespace NUMINAMATH_GPT_star_polygon_points_eq_24_l879_87916

theorem star_polygon_points_eq_24 (n : ℕ) 
  (A_i B_i : ℕ → ℝ) 
  (h_congruent_A : ∀ i j, A_i i = A_i j) 
  (h_congruent_B : ∀ i j, B_i i = B_i j) 
  (h_angle_difference : ∀ i, A_i i = B_i i - 15) : 
  n = 24 := 
sorry

end NUMINAMATH_GPT_star_polygon_points_eq_24_l879_87916


namespace NUMINAMATH_GPT_morales_sisters_revenue_l879_87966

variable (Gabriela Alba Maricela : Nat)
variable (trees_per_grove : Nat := 110)
variable (oranges_per_tree : (Nat × Nat × Nat) := (600, 400, 500))
variable (oranges_per_cup : Nat := 3)
variable (price_per_cup : Nat := 4)

theorem morales_sisters_revenue :
  let G := trees_per_grove * oranges_per_tree.fst
  let A := trees_per_grove * oranges_per_tree.snd
  let M := trees_per_grove * oranges_per_tree.snd.snd
  let total_oranges := G + A + M
  let total_cups := total_oranges / oranges_per_cup
  let total_revenue := total_cups * price_per_cup
  total_revenue = 220000 :=
by 
  sorry

end NUMINAMATH_GPT_morales_sisters_revenue_l879_87966


namespace NUMINAMATH_GPT_pencil_distribution_l879_87997

theorem pencil_distribution (C C' : ℕ) (pencils : ℕ) (remaining : ℕ) (less_per_class : ℕ) 
  (original_classes : C = 4) 
  (total_pencils : pencils = 172) 
  (remaining_pencils : remaining = 7) 
  (less_pencils : less_per_class = 28)
  (actual_classes : C' > C) 
  (distribution_mistake : (pencils - remaining) / C' + less_per_class = pencils / C) :
  C' = 11 := 
sorry

end NUMINAMATH_GPT_pencil_distribution_l879_87997


namespace NUMINAMATH_GPT_find_length_of_AB_l879_87923

-- Definitions of the conditions
def areas_ratio (A B C D : Point) (areaABC areaADC : ℝ) :=
  (areaABC / areaADC) = (7 / 3)

def total_length (A B C D : Point) (AB CD : ℝ) :=
  AB + CD = 280

-- Statement of the proof problem
theorem find_length_of_AB
  (A B C D : Point)
  (AB CD : ℝ)
  (areaABC areaADC : ℝ)
  (h_height_not_zero : h ≠ 0) -- Assumption to ensure height is non-zero
  (h_areas_ratio : areas_ratio A B C D areaABC areaADC)
  (h_total_length : total_length A B C D AB CD) :
  AB = 196 := sorry

end NUMINAMATH_GPT_find_length_of_AB_l879_87923


namespace NUMINAMATH_GPT_jumps_per_second_l879_87999

-- Define the conditions and known values
def record_jumps : ℕ := 54000
def hours : ℕ := 5
def seconds_per_hour : ℕ := 3600

-- Define the target question as a theorem to prove
theorem jumps_per_second :
  (record_jumps / (hours * seconds_per_hour)) = 3 := by
  sorry

end NUMINAMATH_GPT_jumps_per_second_l879_87999


namespace NUMINAMATH_GPT_x_intercept_of_quadratic_l879_87977

theorem x_intercept_of_quadratic (a b c : ℝ) (h_vertex : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 4 ∧ y = -2) 
(h_intercept : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 1 ∧ y = 0) : 
∃ x : ℝ, x = 7 ∧ ∃ y : ℝ, y = a * x^2 + b * x + c ∧ y = 0 :=
sorry

end NUMINAMATH_GPT_x_intercept_of_quadratic_l879_87977


namespace NUMINAMATH_GPT_integer_solutions_to_equation_l879_87990

-- Define the problem statement in Lean 4
theorem integer_solutions_to_equation :
  ∀ (x y : ℤ), (x ≠ 0) → (y ≠ 0) → (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 19) →
      (x, y) = (38, 38) ∨ (x, y) = (380, 20) ∨ (x, y) = (-342, 18) ∨ 
      (x, y) = (20, 380) ∨ (x, y) = (18, -342) :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_to_equation_l879_87990


namespace NUMINAMATH_GPT_problem_a9_b9_l879_87974

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Prove the goal
theorem problem_a9_b9 : a^9 + b^9 = 76 :=
by
  -- the proof will come here
  sorry

end NUMINAMATH_GPT_problem_a9_b9_l879_87974


namespace NUMINAMATH_GPT_sqrt_factorial_mul_factorial_l879_87985

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end NUMINAMATH_GPT_sqrt_factorial_mul_factorial_l879_87985


namespace NUMINAMATH_GPT_area_enclosed_by_cosine_l879_87961

theorem area_enclosed_by_cosine :
  ∫ x in -Real.pi..Real.pi, (1 + Real.cos x) = 2 * Real.pi := by
  sorry

end NUMINAMATH_GPT_area_enclosed_by_cosine_l879_87961


namespace NUMINAMATH_GPT_smallest_four_digit_number_l879_87905

theorem smallest_four_digit_number (N : ℕ) (a b : ℕ) (h1 : N = 100 * a + b) (h2 : N = (a + b)^2) (h3 : 1000 ≤ N) (h4 : N < 10000) : N = 2025 :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_number_l879_87905


namespace NUMINAMATH_GPT_squares_on_sides_of_triangle_l879_87913

theorem squares_on_sides_of_triangle (A B C : ℕ) (hA : A = 3^2) (hB : B = 4^2) (hC : C = 5^2) : 
  A + B = C :=
by 
  rw [hA, hB, hC] 
  exact Nat.add_comm 9 16 ▸ rfl

end NUMINAMATH_GPT_squares_on_sides_of_triangle_l879_87913


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l879_87919

-- Definitions and conditions
def hyperbola (x y a b : ℝ) : Prop :=
  (a > 0 ∧ b > 0 ∧ a > b) ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def regular_hexagon_side_length (a b c : ℝ) : Prop :=
  2 * a = (Real.sqrt 3 + 1) * c

-- Goal: Prove the eccentricity of the hyperbola
theorem eccentricity_of_hyperbola (a b c : ℝ) (x y : ℝ) :
  hyperbola x y a b →
  regular_hexagon_side_length a b c →
  2 * a = (Real.sqrt 3 + 1) * c →
  c ≠ 0 →
  a ≠ 0 →
  b ≠ 0 →
  (c / a = Real.sqrt 3 + 1) :=
by
  intros h_hyp h_hex h_eq h_c_ne_zero h_a_ne_zero h_b_ne_zero
  sorry -- Proof goes here

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l879_87919


namespace NUMINAMATH_GPT_minimize_total_cost_l879_87964

open Real

noncomputable def total_cost (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100) : ℝ :=
  (130 / x) * 2 * (2 + (x^2 / 360)) + (14 * 130 / x)

theorem minimize_total_cost :
  ∀ (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100),
  total_cost x h = (2340 / x) + (13 * x / 18)
  ∧ (x = 18 * sqrt 10 → total_cost x h = 26 * sqrt 10) :=
by
  sorry

end NUMINAMATH_GPT_minimize_total_cost_l879_87964


namespace NUMINAMATH_GPT_daves_earnings_l879_87903

theorem daves_earnings
  (hourly_wage : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (monday_earning : monday_hours * hourly_wage = 36)
  (tuesday_earning : tuesday_hours * hourly_wage = 12) :
  monday_hours * hourly_wage + tuesday_hours * hourly_wage = 48 :=
by
  sorry

end NUMINAMATH_GPT_daves_earnings_l879_87903


namespace NUMINAMATH_GPT_solve_problem_l879_87934

theorem solve_problem (Δ q : ℝ) (h1 : 2 * Δ + q = 134) (h2 : 2 * (Δ + q) + q = 230) : Δ = 43 := by
  sorry

end NUMINAMATH_GPT_solve_problem_l879_87934


namespace NUMINAMATH_GPT_add_55_result_l879_87991

theorem add_55_result (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 :=
sorry

end NUMINAMATH_GPT_add_55_result_l879_87991


namespace NUMINAMATH_GPT_tetrahedron_ratio_l879_87975

open Real

theorem tetrahedron_ratio (a b : ℝ) (h1 : a = PA ∧ PB = a) (h2 : PC = b ∧ AB = b ∧ BC = b ∧ CA = b) (h3 : a < b) :
  (sqrt 6 - sqrt 2) / 2 < a / b ∧ a / b < 1 :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_ratio_l879_87975


namespace NUMINAMATH_GPT_factorial_quotient_52_50_l879_87910

theorem factorial_quotient_52_50 : (Nat.factorial 52) / (Nat.factorial 50) = 2652 := 
by 
  sorry

end NUMINAMATH_GPT_factorial_quotient_52_50_l879_87910


namespace NUMINAMATH_GPT_sum_of_coefficients_l879_87967

-- Define a namespace to encapsulate the problem
namespace PolynomialCoefficients

-- Problem statement as a Lean theorem
theorem sum_of_coefficients (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  α^2005 + β^2005 = 1 :=
sorry -- Placeholder for the proof

end PolynomialCoefficients

end NUMINAMATH_GPT_sum_of_coefficients_l879_87967


namespace NUMINAMATH_GPT_triangle_rectangle_ratio_l879_87906

/--
An equilateral triangle and a rectangle both have perimeters of 60 inches.
The rectangle has a length to width ratio of 2:1.
We need to prove that the ratio of the length of the side of the triangle to
the length of the rectangle is 1.
-/
theorem triangle_rectangle_ratio
  (triangle_perimeter rectangle_perimeter : ℕ)
  (triangle_side rectangle_length rectangle_width : ℕ)
  (h1 : triangle_perimeter = 60)
  (h2 : rectangle_perimeter = 60)
  (h3 : rectangle_length = 2 * rectangle_width)
  (h4 : triangle_side = triangle_perimeter / 3)
  (h5 : rectangle_perimeter = 2 * rectangle_length + 2 * rectangle_width)
  (h6 : rectangle_width = 10)
  (h7 : rectangle_length = 20)
  : triangle_side / rectangle_length = 1 := 
sorry

end NUMINAMATH_GPT_triangle_rectangle_ratio_l879_87906


namespace NUMINAMATH_GPT_melanie_dimes_l879_87983

theorem melanie_dimes (original_dimes dad_dimes mom_dimes total_dimes : ℕ) :
  original_dimes = 7 →
  mom_dimes = 4 →
  total_dimes = 19 →
  (total_dimes = original_dimes + dad_dimes + mom_dimes) →
  dad_dimes = 8 :=
by
  intros h1 h2 h3 h4
  sorry -- The proof is omitted as instructed.

end NUMINAMATH_GPT_melanie_dimes_l879_87983


namespace NUMINAMATH_GPT_gcd_78_36_l879_87996

theorem gcd_78_36 : Nat.gcd 78 36 = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_78_36_l879_87996


namespace NUMINAMATH_GPT_min_value_of_exponential_l879_87955

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 3) : 2^x + 4^y = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_min_value_of_exponential_l879_87955


namespace NUMINAMATH_GPT_cos_sin_combination_l879_87914

theorem cos_sin_combination (x : ℝ) (h : 2 * Real.cos x + 3 * Real.sin x = 4) : 
  3 * Real.cos x - 2 * Real.sin x = 0 := 
by 
  sorry

end NUMINAMATH_GPT_cos_sin_combination_l879_87914


namespace NUMINAMATH_GPT_pyramid_values_l879_87937

theorem pyramid_values :
  ∃ (A B C D : ℕ),
    (A = 3000) ∧
    (D = 623) ∧
    (B = 700) ∧
    (C = 253) ∧
    (A = 1100 + 1800) ∧
    (D + 451 ≥ 1065) ∧ (D + 451 ≤ 1075) ∧ -- rounding to nearest ten
    (B + 440 ≥ 1050) ∧ (B + 440 ≤ 1150) ∧
    (B + 1070 ≥ 1700) ∧ (B + 1070 ≤ 1900) ∧
    (C + 188 ≥ 430) ∧ (C + 188 ≤ 450) ∧    -- rounding to nearest ten
    (C + 451 ≥ 695) ∧ (C + 451 ≤ 705) :=  -- using B = 700 for rounding range
sorry

end NUMINAMATH_GPT_pyramid_values_l879_87937


namespace NUMINAMATH_GPT_area_of_enclosed_figure_l879_87945

noncomputable def area_enclosed_by_curves : ℝ :=
  ∫ (x : ℝ) in (0 : ℝ)..(1 : ℝ), ((x)^(1/2) - x^2)

theorem area_of_enclosed_figure :
  area_enclosed_by_curves = (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_area_of_enclosed_figure_l879_87945


namespace NUMINAMATH_GPT_H_H_H_one_eq_three_l879_87968

noncomputable def H : ℝ → ℝ := sorry

theorem H_H_H_one_eq_three :
  H 1 = -3 ∧ H (-3) = 3 ∧ H 3 = 3 → H (H (H 1)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_H_H_H_one_eq_three_l879_87968


namespace NUMINAMATH_GPT_cubic_root_of_determinant_l879_87907

open Complex 
open Matrix

noncomputable def matrix_d (a b c n : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![
    ![b + n^3 * c, n * (c - b), n^2 * (b - c)],
    ![n^2 * (c - a), c + n^3 * a, n * (a - c)],
    ![n * (b - a), n^2 * (a - b), a + n^3 * b]
  ]

theorem cubic_root_of_determinant (a b c n : ℂ) (h : a * b * c = 1) :
  (det (matrix_d a b c n))^(1/3 : ℂ) = n^3 + 1 :=
  sorry

end NUMINAMATH_GPT_cubic_root_of_determinant_l879_87907


namespace NUMINAMATH_GPT_portions_of_milk_l879_87909

theorem portions_of_milk (liters_to_ml : ℕ) (total_liters : ℕ) (portion : ℕ) (total_volume_ml : ℕ) (num_portions : ℕ) :
  liters_to_ml = 1000 →
  total_liters = 2 →
  portion = 200 →
  total_volume_ml = total_liters * liters_to_ml →
  num_portions = total_volume_ml / portion →
  num_portions = 10 := by
  sorry

end NUMINAMATH_GPT_portions_of_milk_l879_87909


namespace NUMINAMATH_GPT_am_gm_inequality_example_am_gm_inequality_equality_condition_l879_87960

theorem am_gm_inequality_example (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 :=
sorry

theorem am_gm_inequality_equality_condition (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2) ↔ (x = 0 ∧ y = 0 ∨ x = 1 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_am_gm_inequality_example_am_gm_inequality_equality_condition_l879_87960


namespace NUMINAMATH_GPT_employee_n_weekly_wage_l879_87932

theorem employee_n_weekly_wage (Rm Rn : ℝ) (Hm Hn : ℝ) 
    (h1 : (Rm * Hm) + (Rn * Hn) = 770) 
    (h2 : (Rm * Hm) = 1.3 * (Rn * Hn)) :
    Rn * Hn = 335 :=
by
  sorry

end NUMINAMATH_GPT_employee_n_weekly_wage_l879_87932


namespace NUMINAMATH_GPT_linear_function_passing_quadrants_l879_87998

theorem linear_function_passing_quadrants (b : ℝ) :
  (∀ x : ℝ, (y = x + b) ∧ (y > 0 ↔ (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0))) →
  b > 0 :=
sorry

end NUMINAMATH_GPT_linear_function_passing_quadrants_l879_87998


namespace NUMINAMATH_GPT_cole_average_speed_l879_87979

noncomputable def cole_average_speed_to_work : ℝ :=
  let time_to_work := 1.2
  let return_trip_speed := 105
  let total_round_trip_time := 2
  let time_to_return := total_round_trip_time - time_to_work
  let distance_to_work := return_trip_speed * time_to_return
  distance_to_work / time_to_work

theorem cole_average_speed : cole_average_speed_to_work = 70 := by
  sorry

end NUMINAMATH_GPT_cole_average_speed_l879_87979


namespace NUMINAMATH_GPT_determine_C_l879_87901
noncomputable def A : ℕ := sorry
noncomputable def B : ℕ := sorry
noncomputable def C : ℕ := sorry

-- Conditions
axiom cond1 : A + B + 1 = C + 10
axiom cond2 : B = A + 2

-- Proof statement
theorem determine_C : C = 1 :=
by {
  -- using the given conditions, deduce that C must equal 1
  sorry
}

end NUMINAMATH_GPT_determine_C_l879_87901


namespace NUMINAMATH_GPT_mary_number_l879_87917

-- Definitions of the properties and conditions
def is_two_digit_number (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

def switch_digits (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def conditions_met (x : ℕ) : Prop :=
  is_two_digit_number x ∧ 91 ≤ switch_digits (4 * x - 7) ∧ switch_digits (4 * x - 7) ≤ 95

-- The statement to prove
theorem mary_number : ∃ x : ℕ, conditions_met x ∧ x = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_mary_number_l879_87917


namespace NUMINAMATH_GPT_general_term_formula_l879_87925

theorem general_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = 2 * a n - 1) : 
  ∀ n, a n = 2^(n-1) := 
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l879_87925


namespace NUMINAMATH_GPT_mushroom_picking_l879_87972

theorem mushroom_picking (n T : ℕ) (hn_min : n ≥ 5) (hn_max : n ≤ 7)
  (hmax : ∀ (M_max M_min : ℕ), M_max = T / 5 → M_min = T / 7 → 
    T ≠ 0 → M_max ≤ T / n ∧ M_min ≥ T / n) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_mushroom_picking_l879_87972


namespace NUMINAMATH_GPT_second_to_last_digit_of_special_number_l879_87976

theorem second_to_last_digit_of_special_number :
  ∀ (N : ℕ), (N % 10 = 0) ∧ (∃ k : ℕ, k > 0 ∧ N = 2 * 5^k) →
  (N / 10) % 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_second_to_last_digit_of_special_number_l879_87976


namespace NUMINAMATH_GPT_max_true_statements_l879_87936

theorem max_true_statements (y : ℝ) :
  (0 < y^3 ∧ y^3 < 2 → ∀ (y : ℝ),  y^3 > 2 → False) ∧
  ((-2 < y ∧ y < 0) → ∀ (y : ℝ), (0 < y ∧ y < 2) → False) →
  ∃ (s1 s2 : Prop), 
    ((0 < y^3 ∧ y^3 < 2) = s1 ∨ (y^3 > 2) = s1 ∨ (-2 < y ∧ y < 0) = s1 ∨ (0 < y ∧ y < 2) = s1 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s1) ∧
    ((0 < y^3 ∧ y^3 < 2) = s2 ∨ (y^3 > 2) = s2 ∨ (-2 < y ∧ y < 0) = s2 ∨ (0 < y ∧ y < 2) = s2 ∨ (0 < y - y^3 ∧ y - y^3 < 2) = s2) ∧ 
    (s1 ∧ s2) → 
    ∃ m : ℕ, m = 2 := 
sorry

end NUMINAMATH_GPT_max_true_statements_l879_87936


namespace NUMINAMATH_GPT_factorize_expression_l879_87954

variable (b : ℝ)

theorem factorize_expression : 2 * b^3 - 4 * b^2 + 2 * b = 2 * b * (b - 1)^2 := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l879_87954


namespace NUMINAMATH_GPT_hours_per_day_l879_87924

variable (M : ℕ)

noncomputable def H : ℕ := 9
noncomputable def D1 : ℕ := 24
noncomputable def Men2 : ℕ := 12
noncomputable def D2 : ℕ := 16

theorem hours_per_day (H_new : ℝ) : 
  (M * H * D1 : ℝ) = (Men2 * H_new * D2) → 
  H_new = (M * 9 : ℝ) / 8 := 
  sorry

end NUMINAMATH_GPT_hours_per_day_l879_87924


namespace NUMINAMATH_GPT_find_BC_length_l879_87942

noncomputable def area_triangle (A B C : ℝ) : ℝ :=
  1/2 * A * B * C

theorem find_BC_length (A B C : ℝ) (angleA : ℝ)
  (h1 : area_triangle 5 A (Real.sin (π / 6)) = 5 * Real.sqrt 3)
  (h2 : B = 5)
  (h3 : angleA = π / 6) :
  C = Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_find_BC_length_l879_87942


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l879_87944

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2023) :
    ∃ k : ℤ, (2023 - k * 360) = 223 ∧ 180 ≤ 223 ∧ 223 < 270 := by
sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l879_87944


namespace NUMINAMATH_GPT_divisor_of_100_by_quotient_9_and_remainder_1_l879_87984

theorem divisor_of_100_by_quotient_9_and_remainder_1 :
  ∃ d : ℕ, 100 = d * 9 + 1 ∧ d = 11 :=
by
  sorry

end NUMINAMATH_GPT_divisor_of_100_by_quotient_9_and_remainder_1_l879_87984


namespace NUMINAMATH_GPT_carla_gas_cost_l879_87959

theorem carla_gas_cost:
  let distance_grocery := 8
  let distance_school := 6
  let distance_bank := 12
  let distance_practice := 9
  let distance_dinner := 15
  let distance_home := 2 * distance_practice
  let total_distance := distance_grocery + distance_school + distance_bank + distance_practice + distance_dinner + distance_home
  let miles_per_gallon := 25
  let price_per_gallon_first := 2.35
  let price_per_gallon_second := 2.65
  let total_gallons := total_distance / miles_per_gallon
  let gallons_per_fill_up := total_gallons / 2
  let cost_first := gallons_per_fill_up * price_per_gallon_first
  let cost_second := gallons_per_fill_up * price_per_gallon_second
  let total_cost := cost_first + cost_second
  total_cost = 6.80 :=
by sorry

end NUMINAMATH_GPT_carla_gas_cost_l879_87959


namespace NUMINAMATH_GPT_certain_event_drawing_triangle_interior_angles_equal_180_deg_l879_87921

-- Define a triangle in the Euclidean space
structure Triangle (α : Type) [plane : TopologicalSpace α] :=
(a b c : α)

-- Define the sum of the interior angles of a triangle
noncomputable def sum_of_interior_angles {α : Type} [TopologicalSpace α] (T : Triangle α) : ℝ :=
180

-- The proof statement
theorem certain_event_drawing_triangle_interior_angles_equal_180_deg {α : Type} [TopologicalSpace α]
(T : Triangle α) : 
(sum_of_interior_angles T = 180) :=
sorry

end NUMINAMATH_GPT_certain_event_drawing_triangle_interior_angles_equal_180_deg_l879_87921


namespace NUMINAMATH_GPT_net_profit_100_patches_l879_87926

theorem net_profit_100_patches :
  let cost_per_patch := 1.25
  let num_patches_ordered := 100
  let selling_price_per_patch := 12.00
  let total_cost := cost_per_patch * num_patches_ordered
  let total_revenue := selling_price_per_patch * num_patches_ordered
  let net_profit := total_revenue - total_cost
  net_profit = 1075 :=
by
  sorry

end NUMINAMATH_GPT_net_profit_100_patches_l879_87926


namespace NUMINAMATH_GPT_tray_height_l879_87986

noncomputable def height_of_tray : ℝ :=
  let side_length := 120
  let cut_distance := 4 * Real.sqrt 2
  let angle := 45 * (Real.pi / 180)
  -- Define the function that calculates height based on given conditions
  
  sorry

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  side_length = 120 ∧ cut_distance = 4 * Real.sqrt 2 ∧ angle = 45 * (Real.pi / 180) →
  height_of_tray = 4 * Real.sqrt 2 :=
by
  intros
  unfold height_of_tray
  sorry

end NUMINAMATH_GPT_tray_height_l879_87986


namespace NUMINAMATH_GPT_right_angled_triangle_solution_l879_87939

-- Define the necessary constants
def t : ℝ := 504 -- area in cm^2
def c : ℝ := 65 -- hypotenuse in cm

-- The definitions of the right-angled triangle's properties
def is_right_angled_triangle (a b : ℝ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2 ∧ a * b = 2 * t

-- The proof problem statement
theorem right_angled_triangle_solution :
  ∃ (a b : ℝ), is_right_angled_triangle a b ∧ ((a = 63 ∧ b = 16) ∨ (a = 16 ∧ b = 63)) :=
sorry

end NUMINAMATH_GPT_right_angled_triangle_solution_l879_87939


namespace NUMINAMATH_GPT_donation_problem_l879_87930

theorem donation_problem
  (A B C D : Prop)
  (h1 : ¬A ↔ (B ∨ C ∨ D))
  (h2 : B ↔ D)
  (h3 : C ↔ ¬B) 
  (h4 : D ↔ ¬B): A := 
by
  sorry

end NUMINAMATH_GPT_donation_problem_l879_87930


namespace NUMINAMATH_GPT_times_reaching_35m_l879_87912

noncomputable def projectile_height (t : ℝ) : ℝ :=
  -4.9 * t^2 + 30 * t

theorem times_reaching_35m :
  ∃ t1 t2 : ℝ, (abs (t1 - 1.57) < 0.01 ∧ abs (t2 - 4.55) < 0.01) ∧
               projectile_height t1 = 35 ∧ projectile_height t2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_times_reaching_35m_l879_87912


namespace NUMINAMATH_GPT_jasmine_percentage_l879_87965

namespace ProofExample

variables (original_volume : ℝ) (initial_percent_jasmine : ℝ) (added_jasmine : ℝ) (added_water : ℝ)
variables (initial_jasmine : ℝ := initial_percent_jasmine * original_volume / 100)
variables (total_jasmine : ℝ := initial_jasmine + added_jasmine)
variables (total_volume : ℝ := original_volume + added_jasmine + added_water)
variables (final_percent_jasmine : ℝ := (total_jasmine / total_volume) * 100)

theorem jasmine_percentage 
  (h1 : original_volume = 80)
  (h2 : initial_percent_jasmine = 10)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 12)
  : final_percent_jasmine = 16 := 
sorry

end ProofExample

end NUMINAMATH_GPT_jasmine_percentage_l879_87965


namespace NUMINAMATH_GPT_find_m_l879_87908

noncomputable def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2*k + 1

theorem find_m (m : ℕ) (h₀ : 0 < m) (h₁ : (m ^ 2 - 2 * m - 3:ℤ) < 0) (h₂ : is_odd (m ^ 2 - 2 * m - 3)) : m = 2 := 
sorry

end NUMINAMATH_GPT_find_m_l879_87908


namespace NUMINAMATH_GPT_rectangle_other_side_length_l879_87987

/-- Theorem: Consider a rectangle with one side of length 10 cm. Another rectangle of dimensions 
10 cm x 1 cm fits diagonally inside this rectangle. We need to prove that the length 
of the other side of the larger rectangle is 2.96 cm. -/
theorem rectangle_other_side_length :
  ∃ (x : ℝ), (x ≠ 0) ∧ (0 < x) ∧ (10 * 10 - x * x = 1 * 1) ∧ x = 2.96 :=
sorry

end NUMINAMATH_GPT_rectangle_other_side_length_l879_87987


namespace NUMINAMATH_GPT_sum_difference_of_odd_and_even_integers_l879_87900

noncomputable def sum_of_first_n_odds (n : ℕ) : ℕ :=
  n * n

noncomputable def sum_of_first_n_evens (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_difference_of_odd_and_even_integers :
  sum_of_first_n_evens 50 - sum_of_first_n_odds 50 = 50 := 
by
  sorry

end NUMINAMATH_GPT_sum_difference_of_odd_and_even_integers_l879_87900


namespace NUMINAMATH_GPT_time_in_1876_minutes_from_6AM_is_116PM_l879_87993

def minutesToTime (startTime : Nat) (minutesToAdd : Nat) : Nat × Nat :=
  let totalMinutes := startTime + minutesToAdd
  let totalHours := totalMinutes / 60
  let remainderMinutes := totalMinutes % 60
  let resultHours := (totalHours % 24)
  (resultHours, remainderMinutes)

theorem time_in_1876_minutes_from_6AM_is_116PM :
  minutesToTime (6 * 60) 1876 = (13, 16) :=
  sorry

end NUMINAMATH_GPT_time_in_1876_minutes_from_6AM_is_116PM_l879_87993


namespace NUMINAMATH_GPT_nat_nums_division_by_7_l879_87938

theorem nat_nums_division_by_7 (n : ℕ) : 
  (∃ q r, n = 7 * q + r ∧ q = r ∧ 1 ≤ r ∧ r < 7) ↔ 
  n = 8 ∨ n = 16 ∨ n = 24 ∨ n = 32 ∨ n = 40 ∨ n = 48 := by
  sorry

end NUMINAMATH_GPT_nat_nums_division_by_7_l879_87938


namespace NUMINAMATH_GPT_yards_in_a_mile_l879_87941

def mile_eq_furlongs : Prop := 1 = 5 * 1
def furlong_eq_rods : Prop := 1 = 50 * 1
def rod_eq_yards : Prop := 1 = 5 * 1

theorem yards_in_a_mile (h1 : mile_eq_furlongs) (h2 : furlong_eq_rods) (h3 : rod_eq_yards) :
  1 * (5 * (50 * 5)) = 1250 :=
by
-- Given conditions, translate them:
-- h1 : 1 mile = 5 furlongs -> 1 * 1 = 5 * 1
-- h2 : 1 furlong = 50 rods -> 1 * 1 = 50 * 1
-- h3 : 1 rod = 5 yards -> 1 * 1 = 5 * 1
-- Prove that the number of yards in one mile is 1250
sorry

end NUMINAMATH_GPT_yards_in_a_mile_l879_87941


namespace NUMINAMATH_GPT_max_x_minus_y_l879_87957

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^3 + y^3) = x + y) : x - y ≤ (Real.sqrt 2 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_x_minus_y_l879_87957


namespace NUMINAMATH_GPT_area_of_inscribed_square_l879_87978

noncomputable def circle_eq (x y : ℝ) : Prop := 
  3*x^2 + 3*y^2 - 15*x + 9*y + 27 = 0

theorem area_of_inscribed_square :
  (∃ x y : ℝ, circle_eq x y) →
  ∃ s : ℝ, s^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_area_of_inscribed_square_l879_87978


namespace NUMINAMATH_GPT_no_three_parabolas_l879_87920

theorem no_three_parabolas (a b c : ℝ) : ¬ (b^2 > 4*a*c ∧ a^2 > 4*b*c ∧ c^2 > 4*a*b) := by
  sorry

end NUMINAMATH_GPT_no_three_parabolas_l879_87920


namespace NUMINAMATH_GPT_gcd_problem_l879_87956

open Int -- Open the integer namespace to use gcd.

theorem gcd_problem : Int.gcd (Int.gcd 188094 244122) 395646 = 6 :=
by
  -- provide the proof here
  sorry

end NUMINAMATH_GPT_gcd_problem_l879_87956


namespace NUMINAMATH_GPT_soccer_team_physics_players_l879_87949

-- Define the number of players on the soccer team
def total_players := 15

-- Define the number of players taking mathematics
def math_players := 10

-- Define the number of players taking both mathematics and physics
def both_subjects_players := 4

-- Define the number of players taking physics
def physics_players := total_players - math_players + both_subjects_players

-- The theorem to prove
theorem soccer_team_physics_players : physics_players = 9 :=
by
  -- using the conditions defined above
  sorry

end NUMINAMATH_GPT_soccer_team_physics_players_l879_87949


namespace NUMINAMATH_GPT_number_of_true_statements_is_two_l879_87940

def line_plane_geometry : Type :=
  -- Types representing lines and planes
  sorry

def l : line_plane_geometry := sorry
def alpha : line_plane_geometry := sorry
def m : line_plane_geometry := sorry
def beta : line_plane_geometry := sorry

def is_perpendicular (x y : line_plane_geometry) : Prop := sorry
def is_parallel (x y : line_plane_geometry) : Prop := sorry
def is_contained_in (x y : line_plane_geometry) : Prop := sorry

axiom l_perpendicular_alpha : is_perpendicular l alpha
axiom m_contained_in_beta : is_contained_in m beta

def statement_1 : Prop := is_parallel alpha beta → is_perpendicular l m
def statement_2 : Prop := is_perpendicular alpha beta → is_parallel l m
def statement_3 : Prop := is_parallel l m → is_perpendicular alpha beta

theorem number_of_true_statements_is_two : 
  (statement_1 ↔ true) ∧ (statement_2 ↔ false) ∧ (statement_3 ↔ true) := 
sorry

end NUMINAMATH_GPT_number_of_true_statements_is_two_l879_87940


namespace NUMINAMATH_GPT_mark_increase_reading_time_l879_87971

theorem mark_increase_reading_time : 
  (let hours_per_day := 2
   let days_per_week := 7
   let desired_weekly_hours := 18
   let current_weekly_hours := hours_per_day * days_per_week
   let increase_per_week := desired_weekly_hours - current_weekly_hours
   increase_per_week = 4) :=
by
  let hours_per_day := 2
  let days_per_week := 7
  let desired_weekly_hours := 18
  let current_weekly_hours := hours_per_day * days_per_week
  let increase_per_week := desired_weekly_hours - current_weekly_hours
  have h1 : current_weekly_hours = 14 := by norm_num
  have h2 : increase_per_week = desired_weekly_hours - current_weekly_hours := rfl
  have h3 : increase_per_week = 18 - 14 := by rw [h2, h1]
  have h4 : increase_per_week = 4 := by norm_num
  exact h4

end NUMINAMATH_GPT_mark_increase_reading_time_l879_87971


namespace NUMINAMATH_GPT_real_roots_system_l879_87933

theorem real_roots_system :
  ∃ (x y : ℝ), 
    (x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97) ↔ 
    (x, y) = (3, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (-3, -2) ∨ (x, y) = (-2, -3) := 
by 
  sorry

end NUMINAMATH_GPT_real_roots_system_l879_87933


namespace NUMINAMATH_GPT_sum_unchanged_difference_changes_l879_87918

-- Definitions from conditions
def original_sum (a b c : ℤ) := a + b + c
def new_first (a : ℤ) := a - 329
def new_second (b : ℤ) := b + 401

-- Problem statement for sum unchanged
theorem sum_unchanged (a b c : ℤ) (h : original_sum a b c = 1281) :
  original_sum (new_first a) (new_second b) (c - 72) = 1281 := by
  sorry

-- Definitions for difference condition
def abs_diff (x y : ℤ) := abs (x - y)
def alter_difference (a b c : ℤ) :=
  abs_diff (new_first a) (new_second b) + abs_diff (new_first a) c + abs_diff b c

-- Problem statement addressing the difference
theorem difference_changes (a b c : ℤ) (h : original_sum a b c = 1281) :
  alter_difference a b c = abs_diff (new_first a) (new_second b) + abs_diff (c - 730) (new_first a) + abs_diff (c - 730) (new_first a) := by
  sorry

end NUMINAMATH_GPT_sum_unchanged_difference_changes_l879_87918


namespace NUMINAMATH_GPT_number_of_aquariums_l879_87988

theorem number_of_aquariums (total_animals animals_per_aquarium : ℕ) (h1 : total_animals = 40) (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_aquariums_l879_87988


namespace NUMINAMATH_GPT_fraction_solved_l879_87963

theorem fraction_solved (N f : ℝ) (h1 : N * f^2 = 6^3) (h2 : N * f^2 = 7776) : f = 1 / 6 :=
by sorry

end NUMINAMATH_GPT_fraction_solved_l879_87963


namespace NUMINAMATH_GPT_arithmetic_sequence_identification_l879_87992

variable (a : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_identification (h : is_arithmetic a d) :
  (is_arithmetic (fun n => a n + 3) d) ∧
  ¬ (is_arithmetic (fun n => a n ^ 2) d) ∧
  (is_arithmetic (fun n => a (n + 1) - a n) d) ∧
  (is_arithmetic (fun n => 2 * a n) (2 * d)) ∧
  (is_arithmetic (fun n => 2 * a n + n) (2 * d + 1)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_identification_l879_87992


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l879_87904

structure Point := (x y : ℝ)

def A := Point.mk 2 3
def B := Point.mk 9 3
def C := Point.mk 4 12

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * ((B.x - A.x) * (C.y - A.y))

theorem area_of_triangle_ABC :
  area_of_triangle A B C = 31.5 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l879_87904


namespace NUMINAMATH_GPT_inequality_reciprocal_l879_87935

theorem inequality_reciprocal (a b : ℝ)
  (h : a * b > 0) : a > b ↔ 1 / a < 1 / b := 
sorry

end NUMINAMATH_GPT_inequality_reciprocal_l879_87935


namespace NUMINAMATH_GPT_weight_of_one_bowling_ball_l879_87973

def weight_of_one_canoe : ℕ := 35

def ten_bowling_balls_equal_four_canoes (W: ℕ) : Prop :=
  ∀ w, (10 * w = 4 * W)

theorem weight_of_one_bowling_ball (W: ℕ) (h : W = weight_of_one_canoe) : 
  (10 * 14 = 4 * W) → 14 = 140 / 10 :=
by
  intros H
  sorry

end NUMINAMATH_GPT_weight_of_one_bowling_ball_l879_87973


namespace NUMINAMATH_GPT_green_more_than_red_l879_87980

def red_peaches : ℕ := 7
def green_peaches : ℕ := 8

theorem green_more_than_red : green_peaches - red_peaches = 1 := by
  sorry

end NUMINAMATH_GPT_green_more_than_red_l879_87980


namespace NUMINAMATH_GPT_trajectory_equation_l879_87970

theorem trajectory_equation (x y : ℝ) (M O A : ℝ × ℝ)
    (hO : O = (0, 0)) (hA : A = (3, 0))
    (h_ratio : dist M O / dist M A = 1 / 2) : 
    x^2 + y^2 + 2 * x - 3 = 0 :=
by
  -- Definition of points
  let M := (x, y)
  exact sorry

end NUMINAMATH_GPT_trajectory_equation_l879_87970


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l879_87995

variable (a b c e : ℝ)

-- The hyperbola definition and conditions.
def hyperbola (a b : ℝ) := (a > 0) ∧ (b > 0) ∧ (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)

-- Eccentricity is greater than 1 and less than the specified upper bound
def eccentricity_range (e : ℝ) := 1 < e ∧ e < (2 * Real.sqrt 3) / 3

-- Main theorem statement: Given the hyperbola with conditions, prove eccentricity lies in the specified range.
theorem eccentricity_of_hyperbola (h : hyperbola a b) (h_line : ∀ (x y : ℝ), y = x * (Real.sqrt 3) / 3 - 0 -> y^2 ≤ (c^2 - x^2 * a^2)) :
  eccentricity_range e :=
sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l879_87995


namespace NUMINAMATH_GPT_solutions_to_deqs_l879_87958

noncomputable def x1 (t : ℝ) : ℝ := -1 / t^2
noncomputable def x2 (t : ℝ) : ℝ := -t * Real.log t

theorem solutions_to_deqs (t : ℝ) (ht : 0 < t) :
  (deriv x1 t = 2 * t * (x1 t)^2) ∧ (deriv x2 t = x2 t / t - 1) :=
by
  sorry

end NUMINAMATH_GPT_solutions_to_deqs_l879_87958


namespace NUMINAMATH_GPT_extreme_values_sin_2x0_l879_87962

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos (Real.pi / 2 + x)^2 - 
  2 * Real.sin (Real.pi + x) * Real.cos x - Real.sqrt 3

-- Part (1)
theorem extreme_values : 
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), 1 ≤ f x ∧ f x ≤ 2) :=
sorry

-- Part (2)
theorem sin_2x0 (x0 : ℝ) (h : x0 ∈ Set.Icc (3 * Real.pi / 4) Real.pi) (hx : f (x0 - Real.pi / 6) = 10 / 13) : 
  Real.sin (2 * x0) = - (5 + 12 * Real.sqrt 3) / 26 :=
sorry

end NUMINAMATH_GPT_extreme_values_sin_2x0_l879_87962


namespace NUMINAMATH_GPT_coolant_left_l879_87994

theorem coolant_left (initial_volume : ℝ) (initial_concentration : ℝ) (x : ℝ) (replacement_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 19 ∧ 
  initial_concentration = 0.30 ∧ 
  replacement_concentration = 0.80 ∧ 
  final_concentration = 0.50 ∧ 
  (0.30 * initial_volume - 0.30 * x + 0.80 * x = 0.50 * initial_volume) →
  initial_volume - x = 11.4 :=
by sorry

end NUMINAMATH_GPT_coolant_left_l879_87994


namespace NUMINAMATH_GPT_probability_point_in_cube_l879_87950

noncomputable def volume_cube (s : ℝ) : ℝ := s ^ 3

noncomputable def radius_sphere (d : ℝ) : ℝ := d / 2

noncomputable def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

theorem probability_point_in_cube :
  let s := 1 -- side length of the cube
  let v_cube := volume_cube s
  let d := Real.sqrt 3 -- diagonal of the cube
  let r := radius_sphere d
  let v_sphere := volume_sphere r
  v_cube / v_sphere = (2 * Real.sqrt 3) / (3 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_probability_point_in_cube_l879_87950


namespace NUMINAMATH_GPT_percentage_of_sikh_boys_l879_87948

-- Define the conditions
def total_boys : ℕ := 650
def muslim_boys : ℕ := (44 * total_boys) / 100
def hindu_boys : ℕ := (28 * total_boys) / 100
def other_boys : ℕ := 117
def sikh_boys : ℕ := total_boys - (muslim_boys + hindu_boys + other_boys)

-- Define and prove the theorem
theorem percentage_of_sikh_boys : (sikh_boys * 100) / total_boys = 10 :=
by
  have h_muslims: muslim_boys = 286 := by sorry
  have h_hindus: hindu_boys = 182 := by sorry
  have h_total: muslim_boys + hindu_boys + other_boys = 585 := by sorry
  have h_sikhs: sikh_boys = 65 := by sorry
  have h_percentage: (65 * 100) / 650 = 10 := by sorry
  exact h_percentage

end NUMINAMATH_GPT_percentage_of_sikh_boys_l879_87948


namespace NUMINAMATH_GPT_polyhedron_with_n_edges_l879_87911

noncomputable def construct_polyhedron_with_n_edges (n : ℤ) : Prop :=
  ∃ (k : ℤ) (m : ℤ), (k = 8 ∨ k = 9 ∨ k = 10) ∧ (n = k + 3 * m)

theorem polyhedron_with_n_edges (n : ℤ) (h : n ≥ 8) : 
  construct_polyhedron_with_n_edges n :=
sorry

end NUMINAMATH_GPT_polyhedron_with_n_edges_l879_87911


namespace NUMINAMATH_GPT_distance_focus_directrix_l879_87981

theorem distance_focus_directrix (θ : ℝ) : 
  (∃ d : ℝ, (∀ (ρ : ℝ), ρ = 5 / (3 - 2 * Real.cos θ)) ∧ d = 5 / 2) :=
sorry

end NUMINAMATH_GPT_distance_focus_directrix_l879_87981


namespace NUMINAMATH_GPT_magic_square_base_l879_87989

theorem magic_square_base :
  ∃ b : ℕ, (b + 1 + (b + 5) + 2 = 9 + (b + 3)) ∧ b = 3 :=
by
  use 3
  -- Proof in Lean goes here
  sorry

end NUMINAMATH_GPT_magic_square_base_l879_87989


namespace NUMINAMATH_GPT_smallest_n_for_4n_square_and_5n_cube_l879_87928

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_4n_square_and_5n_cube_l879_87928


namespace NUMINAMATH_GPT_check_correct_digit_increase_l879_87969

-- Definition of the numbers involved
def number1 : ℕ := 732
def number2 : ℕ := 648
def number3 : ℕ := 985
def given_sum : ℕ := 2455
def calc_sum : ℕ := number1 + number2 + number3
def difference : ℕ := given_sum - calc_sum

-- Specify the smallest digit that needs to be increased by 1
def smallest_digit_to_increase : ℕ := 8

-- Theorem to check the validity of the problem's claim
theorem check_correct_digit_increase :
  (smallest_digit_to_increase = 8) →
  (calc_sum + 10 = given_sum - 80) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_check_correct_digit_increase_l879_87969


namespace NUMINAMATH_GPT_number_of_integer_solutions_l879_87951

theorem number_of_integer_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x + 3)^2 ≤ 4) ∧ S.card = 5 := by
  sorry

end NUMINAMATH_GPT_number_of_integer_solutions_l879_87951


namespace NUMINAMATH_GPT_daily_reading_goal_l879_87922

-- Define the problem conditions
def total_days : ℕ := 30
def goal_pages : ℕ := 600
def busy_days_13_16 : ℕ := 4
def busy_days_20_25 : ℕ := 6
def flight_day : ℕ := 1
def flight_pages : ℕ := 100

-- Define the mathematical equivalent proof problem in Lean 4
theorem daily_reading_goal :
  (total_days - busy_days_13_16 - busy_days_20_25 - flight_day) * 27 + flight_pages >= goal_pages :=
by
  sorry

end NUMINAMATH_GPT_daily_reading_goal_l879_87922


namespace NUMINAMATH_GPT_dorothy_score_l879_87902

theorem dorothy_score (T I D : ℝ) 
  (hT : T = 2 * I)
  (hI : I = (3 / 5) * D)
  (hSum : T + I + D = 252) : 
  D = 90 := 
by {
  sorry
}

end NUMINAMATH_GPT_dorothy_score_l879_87902


namespace NUMINAMATH_GPT_fifth_number_in_eighth_row_l879_87927

theorem fifth_number_in_eighth_row : 
  (∀ n : ℕ, ∃ k : ℕ, k = n * n ∧ 
    ∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
      k - (n - m) = 54 → m = 5 ∧ n = 8) := by sorry

end NUMINAMATH_GPT_fifth_number_in_eighth_row_l879_87927


namespace NUMINAMATH_GPT_money_distribution_l879_87952

theorem money_distribution (a b : ℝ) 
  (h1 : 4 * a - b = 40)
  (h2 : 6 * a + b = 110) :
  a = 15 ∧ b = 20 :=
by
  sorry

end NUMINAMATH_GPT_money_distribution_l879_87952


namespace NUMINAMATH_GPT_num_points_within_and_on_boundary_is_six_l879_87953

noncomputable def num_points_within_boundary : ℕ :=
  let points := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]
  points.length

theorem num_points_within_and_on_boundary_is_six :
  num_points_within_boundary = 6 :=
  by
    -- proof steps would go here
    sorry

end NUMINAMATH_GPT_num_points_within_and_on_boundary_is_six_l879_87953


namespace NUMINAMATH_GPT_no_largest_integer_exists_l879_87931

/--
  Define a predicate to check whether an integer is a non-square.
-/
def is_non_square (n : ℕ) : Prop :=
  ¬ ∃ m : ℕ, m * m = n

/--
  Define the main theorem which states that there is no largest positive integer
  that cannot be expressed as the sum of a positive integral multiple of 36
  and a positive non-square integer less than 36.
-/
theorem no_largest_integer_exists : ¬ ∃ (n : ℕ), 
  ∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b < 36 ∧ is_non_square b →
  n ≠ 36 * a + b :=
sorry

end NUMINAMATH_GPT_no_largest_integer_exists_l879_87931


namespace NUMINAMATH_GPT_sequence_values_l879_87982

variable {a1 a2 b2 : ℝ}

theorem sequence_values
  (arithmetic : 2 * a1 = 1 + a2 ∧ 2 * a2 = a1 + 4)
  (geometric : b2 ^ 2 = 1 * 4) :
  (a1 + a2) / b2 = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sequence_values_l879_87982


namespace NUMINAMATH_GPT_cube_largest_ne_sum_others_l879_87947

theorem cube_largest_ne_sum_others (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 :=
by
  sorry

end NUMINAMATH_GPT_cube_largest_ne_sum_others_l879_87947


namespace NUMINAMATH_GPT_candies_of_different_flavors_l879_87946

theorem candies_of_different_flavors (total_treats chewing_gums chocolate_bars : ℕ) (h1 : total_treats = 155) (h2 : chewing_gums = 60) (h3 : chocolate_bars = 55) :
  total_treats - (chewing_gums + chocolate_bars) = 40 := 
by 
  sorry

end NUMINAMATH_GPT_candies_of_different_flavors_l879_87946


namespace NUMINAMATH_GPT_problem_statement_l879_87943

variable {x y : ℝ}

def star (a b : ℝ) : ℝ := (a + b)^2

theorem problem_statement (x y : ℝ) : star ((x + y)^2) ((y + x)^2) = 4 * (x + y)^4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l879_87943
