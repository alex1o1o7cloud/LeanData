import Mathlib

namespace NUMINAMATH_GPT_solution_to_fractional_equation_l182_18285

theorem solution_to_fractional_equation (x : ℝ) (h : x ≠ 3 ∧ x ≠ 1) :
  (x / (x - 3) = (x + 1) / (x - 1)) ↔ (x = -3) :=
by
  sorry

end NUMINAMATH_GPT_solution_to_fractional_equation_l182_18285


namespace NUMINAMATH_GPT_a4_equals_8_l182_18213

variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {n : ℕ}

-- Defining the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) := ∀ n, a (n + 1) = a n * r

-- Given conditions as hypotheses
variable (h_geometric : geometric_sequence a r)
variable (h_root_2 : a 2 * a 6 = 64)
variable (h_roots_eq : ∀ x, x^2 - 34 * x + 64 = 0 → (x = a 2 ∨ x = a 6))

-- The statement to prove
theorem a4_equals_8 : a 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_a4_equals_8_l182_18213


namespace NUMINAMATH_GPT_sum_max_min_value_f_l182_18287

noncomputable def f (x : ℝ) : ℝ := ((x + 1) ^ 2 + x) / (x ^ 2 + 1)

theorem sum_max_min_value_f : 
  let M := (⨆ x : ℝ, f x)
  let m := (⨅ x : ℝ, f x)
  M + m = 2 :=
by
-- Proof to be filled in
  sorry

end NUMINAMATH_GPT_sum_max_min_value_f_l182_18287


namespace NUMINAMATH_GPT_total_revenue_full_price_l182_18255

theorem total_revenue_full_price (f d p : ℕ) (h1 : f + d = 200) (h2 : f * p + d * (3 * p) / 4 = 2800) : 
  f * p = 680 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_total_revenue_full_price_l182_18255


namespace NUMINAMATH_GPT_polynomial_evaluation_l182_18297

-- Define operations using Lean syntax
def star (a b : ℚ) := a + b
def otimes (a b : ℚ) := a - b

-- Define a function to represent the polynomial expression
def expression (a b : ℚ) := star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b)

theorem polynomial_evaluation (a b : ℚ) (ha : a = 5) (hb : b = 3) : expression a b = 435 := by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l182_18297


namespace NUMINAMATH_GPT_solution_set_l182_18263

variable {f : ℝ → ℝ}
variable (h1 : ∀ x, x < 0 → x * deriv f x - 2 * f x > 0)
variable (h2 : ∀ x, x < 0 → f x ≠ 0)

theorem solution_set (h3 : ∀ x, -2024 < x ∧ x < -2023 → f (x + 2023) - (x + 2023)^2 * f (-1) < 0) :
    {x : ℝ | f (x + 2023) - (x + 2023)^2 * f (-1) < 0} = {x : ℝ | -2024 < x ∧ x < -2023} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l182_18263


namespace NUMINAMATH_GPT_half_way_fraction_l182_18234

def half_way_between (a b : ℚ) : ℚ := (a + b) / 2

theorem half_way_fraction : 
  half_way_between (1/3) (3/4) = 13/24 :=
by 
  -- Proof follows from the calculation steps, but we leave it unproved.
  sorry

end NUMINAMATH_GPT_half_way_fraction_l182_18234


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l182_18207

theorem point_in_fourth_quadrant (θ : ℝ) (h : -1 < Real.cos θ ∧ Real.cos θ < 0) :
    ∃ (x y : ℝ), x = Real.sin (Real.cos θ) ∧ y = Real.cos (Real.cos θ) ∧ x < 0 ∧ y > 0 :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l182_18207


namespace NUMINAMATH_GPT_second_supplier_more_cars_l182_18265

-- Define the constants and conditions given in the problem
def total_production := 5650000
def first_supplier := 1000000
def fourth_fifth_supplier := 325000

-- Define the unknown variable for the second supplier
noncomputable def second_supplier : ℕ := sorry

-- Define the equation based on the conditions
def equation := first_supplier + second_supplier + (first_supplier + second_supplier) + (4 * fourth_fifth_supplier / 2) = total_production

-- Prove that the second supplier receives 500,000 more cars than the first supplier
theorem second_supplier_more_cars : 
  ∃ X : ℕ, equation → (X = first_supplier + 500000) :=
sorry

end NUMINAMATH_GPT_second_supplier_more_cars_l182_18265


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l182_18277

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x

noncomputable def g (x : ℝ) : ℝ := Real.log ((x + 2) / (x - 2))

theorem problem_part_1 :
  ∀ (x₁ x₂ : ℝ), 0 < x₂ ∧ x₂ < x₁ → Real.log x₁ + 2 * x₁ > Real.log x₂ + 2 * x₂ :=
sorry

theorem problem_part_2 :
  ∃ k : ℕ, ∀ (x₁ : ℝ), 0 < x₁ ∧ x₁ < 1 → (∃ (x₂ : ℝ), x₂ ∈ Set.Ioo (k : ℝ) (k + 1) ∧ Real.log x₁ + 2 * x₁ < Real.log ((x₂ + 2) / (x₂ - 2))) → k = 2 :=
sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l182_18277


namespace NUMINAMATH_GPT_cost_of_pen_l182_18247

theorem cost_of_pen 
  (total_amount_spent : ℕ)
  (total_items : ℕ)
  (number_of_pencils : ℕ)
  (cost_of_pencil : ℕ)
  (cost_of_pen : ℕ)
  (h1 : total_amount_spent = 2000)
  (h2 : total_items = 36)
  (h3 : number_of_pencils = 16)
  (h4 : cost_of_pencil = 25)
  (remaining_amount_spent : ℕ)
  (number_of_pens : ℕ)
  (h5 : remaining_amount_spent = total_amount_spent - (number_of_pencils * cost_of_pencil))
  (h6 : number_of_pens = total_items - number_of_pencils)
  (total_cost_of_pens : ℕ)
  (h7 : total_cost_of_pens = remaining_amount_spent)
  (h8 : total_cost_of_pens = number_of_pens * cost_of_pen)
  : cost_of_pen = 80 := by
  sorry

end NUMINAMATH_GPT_cost_of_pen_l182_18247


namespace NUMINAMATH_GPT_point_P_trajectory_circle_l182_18294

noncomputable def trajectory_of_point_P (d h1 h2 : ℝ) (x y : ℝ) : Prop :=
  (x - d/2)^2 + y^2 = (h1^2 + h2^2) / (2 * (h2/h1)^(2/3))

theorem point_P_trajectory_circle :
  ∀ (d h1 h2 x y : ℝ),
  d = 20 →
  h1 = 15 →
  h2 = 10 →
  (∃ x y, trajectory_of_point_P d h1 h2 x y) →
  (∃ x y, (x - 16)^2 + y^2 = 24^2) :=
by
  intros d h1 h2 x y hd hh1 hh2 hxy
  sorry

end NUMINAMATH_GPT_point_P_trajectory_circle_l182_18294


namespace NUMINAMATH_GPT_is_composite_1010_pattern_l182_18256

theorem is_composite_1010_pattern (k : ℕ) (h : k ≥ 2) : (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ (1010^k + 101 = a * b)) :=
  sorry

end NUMINAMATH_GPT_is_composite_1010_pattern_l182_18256


namespace NUMINAMATH_GPT_smallest_number_of_people_l182_18278

theorem smallest_number_of_people (N : ℕ) :
  (∃ (N : ℕ), ∀ seats : ℕ, seats = 80 → N ≤ 80 → ∀ n : ℕ, n > N → (∃ m : ℕ, (m < N) ∧ ((seats + m) % 80 < seats))) → N = 20 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_people_l182_18278


namespace NUMINAMATH_GPT_probability_Z_l182_18296

theorem probability_Z (p_X p_Y p_Z : ℚ)
  (hX : p_X = 2 / 5)
  (hY : p_Y = 1 / 4)
  (hTotal : p_X + p_Y + p_Z = 1) :
  p_Z = 7 / 20 := by sorry

end NUMINAMATH_GPT_probability_Z_l182_18296


namespace NUMINAMATH_GPT_log_problem_l182_18235

open Real

theorem log_problem : 2 * log 5 + log 4 = 2 := by
  sorry

end NUMINAMATH_GPT_log_problem_l182_18235


namespace NUMINAMATH_GPT_find_number_l182_18270

-- Define the condition given in the problem
def condition (x : ℤ) := 13 * x - 272 = 105

-- Prove that given the condition, x equals 29
theorem find_number : ∃ x : ℤ, condition x ∧ x = 29 :=
by
  use 29
  unfold condition
  sorry

end NUMINAMATH_GPT_find_number_l182_18270


namespace NUMINAMATH_GPT_average_salary_increase_l182_18257

theorem average_salary_increase :
  let avg_salary := 1200
  let num_employees := 20
  let manager_salary := 3300
  let new_num_people := num_employees + 1
  let total_salary := num_employees * avg_salary
  let new_total_salary := total_salary + manager_salary
  let new_avg_salary := new_total_salary / new_num_people
  let increase := new_avg_salary - avg_salary
  increase = 100 :=
by
  sorry

end NUMINAMATH_GPT_average_salary_increase_l182_18257


namespace NUMINAMATH_GPT_class_A_students_l182_18224

variable (A B : ℕ)

theorem class_A_students 
    (h1 : A = (5 * B) / 7)
    (h2 : A + 3 = (4 * (B - 3)) / 5) :
    A = 45 :=
sorry

end NUMINAMATH_GPT_class_A_students_l182_18224


namespace NUMINAMATH_GPT_lorie_total_bills_l182_18281

-- Definitions for the conditions
def initial_hundred_bills := 2
def hundred_to_fifty (bills : Nat) : Nat := bills * 2 / 100
def hundred_to_ten (bills : Nat) : Nat := (bills / 2) / 10
def hundred_to_five (bills : Nat) : Nat := (bills / 2) / 5

-- Statement of the problem
theorem lorie_total_bills : 
  let fifty_bills := hundred_to_fifty 100
  let ten_bills := hundred_to_ten 100
  let five_bills := hundred_to_five 100
  fifty_bills + ten_bills + five_bills = 2 + 5 + 10 :=
sorry

end NUMINAMATH_GPT_lorie_total_bills_l182_18281


namespace NUMINAMATH_GPT_cost_large_bulb_l182_18216

def small_bulbs : Nat := 3
def cost_small_bulb : Nat := 8
def total_budget : Nat := 60
def amount_left : Nat := 24

theorem cost_large_bulb (cost_large_bulb : Nat) :
  total_budget - amount_left - small_bulbs * cost_small_bulb = cost_large_bulb →
  cost_large_bulb = 12 := by
  sorry

end NUMINAMATH_GPT_cost_large_bulb_l182_18216


namespace NUMINAMATH_GPT_finite_solutions_to_equation_l182_18282

theorem finite_solutions_to_equation :
  ∃ (n : ℕ), ∀ (a b c : ℕ), (a > 0 ∧ b > 0 ∧ c > 0) ∧ (1 / (a:ℚ) + 1 / (b:ℚ) + 1 / (c:ℚ) = 1 / 1983) → 
  (a ≤ n ∧ b ≤ n ∧ c ≤ n) :=
sorry

end NUMINAMATH_GPT_finite_solutions_to_equation_l182_18282


namespace NUMINAMATH_GPT_percentage_of_difference_is_50_l182_18238

noncomputable def percentage_of_difference (x y : ℝ) (p : ℝ) :=
  (p / 100) * (x - y) = 0.20 * (x + y)

noncomputable def y_is_percentage_of_x (x y : ℝ) :=
  y = 0.42857142857142854 * x

theorem percentage_of_difference_is_50 (x y : ℝ) (p : ℝ)
  (h1 : percentage_of_difference x y p)
  (h2 : y_is_percentage_of_x x y) :
  p = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_difference_is_50_l182_18238


namespace NUMINAMATH_GPT_trigonometric_fraction_value_l182_18226

theorem trigonometric_fraction_value (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end NUMINAMATH_GPT_trigonometric_fraction_value_l182_18226


namespace NUMINAMATH_GPT_mrs_hilt_money_l182_18217

-- Definitions and given conditions
def cost_of_pencil := 5  -- in cents
def number_of_pencils := 10

-- The theorem we need to prove
theorem mrs_hilt_money : cost_of_pencil * number_of_pencils = 50 := by
  sorry

end NUMINAMATH_GPT_mrs_hilt_money_l182_18217


namespace NUMINAMATH_GPT_triangles_with_vertex_A_l182_18272

theorem triangles_with_vertex_A : 
  ∃ (A : Point) (remaining_points : Finset Point), 
    (remaining_points.card = 8) → 
    (∃ (n : ℕ), n = (Nat.choose 8 2) ∧ n = 28) :=
by
  sorry

end NUMINAMATH_GPT_triangles_with_vertex_A_l182_18272


namespace NUMINAMATH_GPT_quadratic_vertex_coordinates_l182_18204

theorem quadratic_vertex_coordinates (x y : ℝ) (h : y = 2 * x^2 - 4 * x + 5) : (x, y) = (1, 3) :=
sorry

end NUMINAMATH_GPT_quadratic_vertex_coordinates_l182_18204


namespace NUMINAMATH_GPT_triangle_area_less_than_sqrt3_div_3_l182_18280

-- Definitions for a triangle and its properties
structure Triangle :=
  (a b c : ℝ)
  (ha hb hc : ℝ)
  (area : ℝ)

def valid_triangle (Δ : Triangle) : Prop :=
  0 < Δ.a ∧ 0 < Δ.b ∧ 0 < Δ.c ∧ Δ.ha < 1 ∧ Δ.hb < 1 ∧ Δ.hc < 1

theorem triangle_area_less_than_sqrt3_div_3 (Δ : Triangle) (h : valid_triangle Δ) : Δ.area < (Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_triangle_area_less_than_sqrt3_div_3_l182_18280


namespace NUMINAMATH_GPT_simplify_expression_l182_18231

theorem simplify_expression (a b : ℕ) (h₁ : a = 2999) (h₂ : b = 3000) :
  b^3 - a * b^2 - a^2 * b + a^3 = 5999 :=
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l182_18231


namespace NUMINAMATH_GPT_least_positive_integer_reducible_fraction_l182_18260

theorem least_positive_integer_reducible_fraction :
  ∃ n : ℕ, n > 0 ∧ gcd (n - 17) (7 * n + 4) > 1 ∧ (∀ m : ℕ, m > 0 ∧ gcd (m - 17) (7 * m + 4) > 1 → n ≤ m) :=
by sorry

end NUMINAMATH_GPT_least_positive_integer_reducible_fraction_l182_18260


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l182_18203

theorem isosceles_triangle_base_length (P B : ℕ) (hP : P = 13) (hB : B = 3) :
    ∃ S : ℕ, S ≠ 3 ∧ S = 3 :=
by
    sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l182_18203


namespace NUMINAMATH_GPT_find_x_value_l182_18206

theorem find_x_value :
  ∃ x : ℝ, (75 * x + (18 + 12) * 6 / 4 - 11 * 8 = 2734) ∧ (x = 37.03) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_value_l182_18206


namespace NUMINAMATH_GPT_determine_g_x2_l182_18276

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem determine_g_x2 (x : ℝ) (h : x^2 ≠ 4) : g (x^2) = (2 * x^2 + 3) / (x^2 - 2) :=
by sorry

end NUMINAMATH_GPT_determine_g_x2_l182_18276


namespace NUMINAMATH_GPT_unique_rectangle_exists_l182_18228

theorem unique_rectangle_exists (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_rectangle_exists_l182_18228


namespace NUMINAMATH_GPT_number_of_three_digit_multiples_of_9_with_odd_digits_l182_18221

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

def consists_only_of_odd_digits (n : ℕ) : Prop :=
  (∀ d ∈ (n.digits 10), d % 2 = 1)

theorem number_of_three_digit_multiples_of_9_with_odd_digits :
  ∃ t, t = 11 ∧
  (∀ n, is_three_digit_number n ∧ is_multiple_of_9 n ∧ consists_only_of_odd_digits n) → 1 ≤ t ∧ t ≤ 11 :=
sorry

end NUMINAMATH_GPT_number_of_three_digit_multiples_of_9_with_odd_digits_l182_18221


namespace NUMINAMATH_GPT_determine_set_A_l182_18214

-- Define the function f as described
def f (n : ℕ) (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else (x - 1) / 2 + 2^(n - 1)

-- Define the set A
def A (n : ℕ) : Set ℕ :=
  { x | (Nat.iterate (f n) n x) = x }

-- State the theorem
theorem determine_set_A (n : ℕ) (hn : n > 0) :
    A n = { x | 1 ≤ x ∧ x ≤ 2^n } :=
sorry

end NUMINAMATH_GPT_determine_set_A_l182_18214


namespace NUMINAMATH_GPT_range_of_7a_minus_5b_l182_18225

theorem range_of_7a_minus_5b (a b : ℝ) (h1 : 5 ≤ a - b ∧ a - b ≤ 27) (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  36 ≤ 7 * a - 5 * b ∧ 7 * a - 5 * b ≤ 192 :=
sorry

end NUMINAMATH_GPT_range_of_7a_minus_5b_l182_18225


namespace NUMINAMATH_GPT_sum_of_coeffs_binomial_eq_32_l182_18246

noncomputable def sum_of_coeffs_binomial (x : ℝ) : ℝ :=
  (3 * x - 1 / Real.sqrt x)^5

theorem sum_of_coeffs_binomial_eq_32 :
  sum_of_coeffs_binomial 1 = 32 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_binomial_eq_32_l182_18246


namespace NUMINAMATH_GPT_solution_volume_l182_18261

theorem solution_volume (concentration volume_acid volume_solution : ℝ) 
  (h_concentration : concentration = 0.25) 
  (h_acid : volume_acid = 2.5) 
  (h_formula : concentration = volume_acid / volume_solution) : 
  volume_solution = 10 := 
by
  sorry

end NUMINAMATH_GPT_solution_volume_l182_18261


namespace NUMINAMATH_GPT_grocer_second_month_sale_l182_18222

theorem grocer_second_month_sale (sale_1 sale_3 sale_4 sale_5 sale_6 avg_sale n : ℕ) 
(h1 : sale_1 = 6435) 
(h3 : sale_3 = 6855) 
(h4 : sale_4 = 7230) 
(h5 : sale_5 = 6562) 
(h6 : sale_6 = 7391) 
(havg : avg_sale = 6900) 
(hn : n = 6) : 
  sale_2 = 6927 :=
by
  sorry

end NUMINAMATH_GPT_grocer_second_month_sale_l182_18222


namespace NUMINAMATH_GPT_pages_to_read_l182_18215

variable (E P_Science P_Civics P_Chinese Total : ℕ)
variable (h_Science : P_Science = 16)
variable (h_Civics : P_Civics = 8)
variable (h_Chinese : P_Chinese = 12)
variable (h_Total : Total = 14)

theorem pages_to_read :
  (E / 4) + (P_Science / 4) + (P_Civics / 4) + (P_Chinese / 4) = Total → 
  E = 20 := by
  sorry

end NUMINAMATH_GPT_pages_to_read_l182_18215


namespace NUMINAMATH_GPT_last_three_digits_of_2_pow_10000_l182_18252

theorem last_three_digits_of_2_pow_10000 (h : 2^500 ≡ 1 [MOD 1250]) : (2^10000) % 1000 = 1 :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_2_pow_10000_l182_18252


namespace NUMINAMATH_GPT_walt_age_l182_18232

-- Conditions
variables (T W : ℕ)
axiom h1 : T = 3 * W
axiom h2 : T + 12 = 2 * (W + 12)

-- Goal: Prove W = 12
theorem walt_age : W = 12 :=
sorry

end NUMINAMATH_GPT_walt_age_l182_18232


namespace NUMINAMATH_GPT_smallest_possible_gcd_l182_18248

theorem smallest_possible_gcd (m n p : ℕ) (h1 : Nat.gcd m n = 180) (h2 : Nat.gcd m p = 240) :
  ∃ k, k = Nat.gcd n p ∧ k = 60 := by
  sorry

end NUMINAMATH_GPT_smallest_possible_gcd_l182_18248


namespace NUMINAMATH_GPT_functional_equation_solution_l182_18298

theorem functional_equation_solution (f : ℚ → ℚ)
  (H : ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y) :
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l182_18298


namespace NUMINAMATH_GPT_total_students_l182_18284

-- Define the condition that the sum of boys (75) and girls (G) is the total number of students (T)
def sum_boys_girls (G T : ℕ) := 75 + G = T

-- Define the condition that the number of girls (G) equals 75% of the total number of students (T)
def girls_percentage (G T : ℕ) := G = Nat.div (3 * T) 4

-- State the theorem that given the above conditions, the total number of students (T) is 300
theorem total_students (G T : ℕ) (h1 : sum_boys_girls G T) (h2 : girls_percentage G T) : T = 300 := 
sorry

end NUMINAMATH_GPT_total_students_l182_18284


namespace NUMINAMATH_GPT_alice_daily_savings_l182_18275

theorem alice_daily_savings :
  ∀ (d total_days : ℕ) (dime_value : ℝ),
  d = 4 → total_days = 40 → dime_value = 0.10 →
  (d * dime_value) / total_days = 0.01 :=
by
  intros d total_days dime_value h_d h_total_days h_dime_value
  sorry

end NUMINAMATH_GPT_alice_daily_savings_l182_18275


namespace NUMINAMATH_GPT_abs_of_neg_square_add_l182_18223

theorem abs_of_neg_square_add (a b : ℤ) : |-a^2 + b| = 10 :=
by
  sorry

end NUMINAMATH_GPT_abs_of_neg_square_add_l182_18223


namespace NUMINAMATH_GPT_right_triangle_area_hypotenuse_30_deg_l182_18227

theorem right_triangle_area_hypotenuse_30_deg
  (h : Real)
  (θ : Real)
  (A : Real)
  (H1 : θ = 30)
  (H2 : h = 12)
  : A = 18 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_right_triangle_area_hypotenuse_30_deg_l182_18227


namespace NUMINAMATH_GPT_least_possible_value_l182_18243

theorem least_possible_value (y q p : ℝ) (h1: 5 < y) (h2: y < 7)
  (hq: q = 7) (hp: p = 5) : q - p = 2 :=
by
  sorry

end NUMINAMATH_GPT_least_possible_value_l182_18243


namespace NUMINAMATH_GPT_halfway_between_one_third_and_one_fifth_l182_18266

theorem halfway_between_one_third_and_one_fifth : (1/3 + 1/5) / 2 = 4/15 := 
by 
  sorry

end NUMINAMATH_GPT_halfway_between_one_third_and_one_fifth_l182_18266


namespace NUMINAMATH_GPT_combined_resistance_parallel_l182_18233

theorem combined_resistance_parallel (R1 R2 R3 R : ℝ)
  (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6)
  (h4 : 1/R = 1/R1 + 1/R2 + 1/R3) :
  R = 15/13 := 
by
  sorry

end NUMINAMATH_GPT_combined_resistance_parallel_l182_18233


namespace NUMINAMATH_GPT_find_xyz_l182_18268

theorem find_xyz (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11)
  (h₃ : x + y + z = 3) :
  x * y * z = 16 / 3 := 
  sorry

end NUMINAMATH_GPT_find_xyz_l182_18268


namespace NUMINAMATH_GPT_cos_2alpha_value_l182_18267

noncomputable def cos_double_angle (α : ℝ) : ℝ := Real.cos (2 * α)

theorem cos_2alpha_value (α : ℝ): 
  (∃ a : ℝ, α = Real.arctan (-3) + 2 * a * Real.pi) → cos_double_angle α = -4 / 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cos_2alpha_value_l182_18267


namespace NUMINAMATH_GPT_car_gas_consumption_l182_18249

theorem car_gas_consumption
  (miles_today : ℕ)
  (miles_tomorrow : ℕ)
  (total_gallons : ℕ)
  (h1 : miles_today = 400)
  (h2 : miles_tomorrow = miles_today + 200)
  (h3 : total_gallons = 4000)
  : (∃ g : ℕ, 400 * g + (400 + 200) * g = total_gallons ∧ g = 4) :=
by
  sorry

end NUMINAMATH_GPT_car_gas_consumption_l182_18249


namespace NUMINAMATH_GPT_thirteenth_result_is_128_l182_18209

theorem thirteenth_result_is_128 
  (avg_all : ℕ → ℕ → ℕ) (avg_first : ℕ → ℕ → ℕ) (avg_last : ℕ → ℕ → ℕ) :
  avg_all 25 20 = (avg_first 12 14) + (avg_last 12 17) + 128 :=
by
  sorry

end NUMINAMATH_GPT_thirteenth_result_is_128_l182_18209


namespace NUMINAMATH_GPT_find_x_l182_18264

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l182_18264


namespace NUMINAMATH_GPT_contest_paths_correct_l182_18251

noncomputable def count_contest_paths : Nat := sorry

theorem contest_paths_correct : count_contest_paths = 127 := sorry

end NUMINAMATH_GPT_contest_paths_correct_l182_18251


namespace NUMINAMATH_GPT_quadrilateral_area_l182_18283

-- Define the dimensions of the rectangles
variables (AB BC EF FG : ℝ)
variables (AFCH_area : ℝ)

-- State the conditions explicitly
def conditions : Prop :=
  (AB = 9) ∧ 
  (BC = 5) ∧ 
  (EF = 3) ∧ 
  (FG = 10)

-- State the theorem to prove
theorem quadrilateral_area (h: conditions AB BC EF FG) : 
  AFCH_area = 52.5 := 
sorry

end NUMINAMATH_GPT_quadrilateral_area_l182_18283


namespace NUMINAMATH_GPT_obtain_x_squared_obtain_xy_l182_18293

theorem obtain_x_squared (x y : ℝ) (hx : x ≠ 1) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x^2 :=
by
  sorry

theorem obtain_xy (x y : ℝ) (hx0 : 0 < x) (hy0 : 0 < y) :
  ∃ (k : ℝ), k = x * y :=
by
  sorry

end NUMINAMATH_GPT_obtain_x_squared_obtain_xy_l182_18293


namespace NUMINAMATH_GPT_boxes_calculation_proof_l182_18208

variable (baskets : ℕ) (eggs_per_basket : ℕ) (eggs_per_box : ℕ)
variable (total_eggs : ℕ := baskets * eggs_per_basket)
variable (boxes_needed : ℕ := total_eggs / eggs_per_box)

theorem boxes_calculation_proof :
  baskets = 21 →
  eggs_per_basket = 48 →
  eggs_per_box = 28 →
  boxes_needed = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boxes_calculation_proof_l182_18208


namespace NUMINAMATH_GPT_deer_meat_distribution_l182_18202

theorem deer_meat_distribution (a d : ℕ) (H1 : a = 100) :
  ∀ (Dafu Bugeng Zanbao Shangzao Gongshe : ℕ),
    Dafu = a - 2 * d →
    Bugeng = a - d →
    Zanbao = a →
    Shangzao = a + d →
    Gongshe = a + 2 * d →
    Dafu + Bugeng + Zanbao + Shangzao + Gongshe = 500 →
    Bugeng + Zanbao + Shangzao = 300 :=
by
  intros Dafu Bugeng Zanbao Shangzao Gongshe hDafu hBugeng hZanbao hShangzao hGongshe hSum
  sorry

end NUMINAMATH_GPT_deer_meat_distribution_l182_18202


namespace NUMINAMATH_GPT_complex_fraction_simplification_l182_18218

theorem complex_fraction_simplification : 
  ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h_imag_unit
  sorry

end NUMINAMATH_GPT_complex_fraction_simplification_l182_18218


namespace NUMINAMATH_GPT_three_pow_1234_mod_5_l182_18242

theorem three_pow_1234_mod_5 : (3^1234) % 5 = 4 := 
by 
  have h1 : 3^4 % 5 = 1 := by norm_num
  sorry

end NUMINAMATH_GPT_three_pow_1234_mod_5_l182_18242


namespace NUMINAMATH_GPT_number_of_pairs_exterior_angles_l182_18220

theorem number_of_pairs_exterior_angles (m n : ℕ) :
  (3 ≤ m ∧ 3 ≤ n ∧ 360 = m * n) ↔ 20 = 20 := 
by sorry

end NUMINAMATH_GPT_number_of_pairs_exterior_angles_l182_18220


namespace NUMINAMATH_GPT_quadratic_completion_l182_18237

theorem quadratic_completion 
    (x : ℝ) 
    (h : 16*x^2 - 32*x - 512 = 0) : 
    ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by sorry

end NUMINAMATH_GPT_quadratic_completion_l182_18237


namespace NUMINAMATH_GPT_expand_expression_l182_18258

theorem expand_expression (x y : ℝ) : (3 * x + 15) * (4 * y + 12) = 12 * x * y + 36 * x + 60 * y + 180 := 
  sorry

end NUMINAMATH_GPT_expand_expression_l182_18258


namespace NUMINAMATH_GPT_part1_part2_l182_18250

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a + 1 + Real.log x) / x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.exp (1 - x) + 1

theorem part1 (a : ℝ) (x : ℝ) (h : 0 < x) : (∀ x > 0, f a x ≤ Real.exp 1) → a ≤ 1 := 
sorry

theorem part2 (a : ℝ) : (∃! x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 = g x1 ∧ f a x2 = g x2 ∧ f a x3 = g x3) → a = 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l182_18250


namespace NUMINAMATH_GPT_cans_per_person_on_second_day_l182_18269

theorem cans_per_person_on_second_day :
  ∀ (initial_stock : ℕ) (people_first_day : ℕ) (cans_taken_first_day : ℕ)
    (restock_first_day : ℕ) (people_second_day : ℕ)
    (restock_second_day : ℕ) (total_cans_given : ℕ) (cans_per_person_second_day : ℚ),
    cans_taken_first_day = 1 →
    initial_stock = 2000 →
    people_first_day = 500 →
    restock_first_day = 1500 →
    people_second_day = 1000 →
    restock_second_day = 3000 →
    total_cans_given = 2500 →
    cans_per_person_second_day = total_cans_given / people_second_day →
    cans_per_person_second_day = 2.5 := by
  sorry

end NUMINAMATH_GPT_cans_per_person_on_second_day_l182_18269


namespace NUMINAMATH_GPT_evaluate_expression_simplified_l182_18299

theorem evaluate_expression_simplified (x : ℝ) (h : x = Real.sqrt 2) : 
  (x + 3) ^ 2 + (x + 2) * (x - 2) - x * (x + 6) = 7 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_simplified_l182_18299


namespace NUMINAMATH_GPT_hyperbola_focus_l182_18244

-- Definition of the hyperbola equation and foci
def is_hyperbola (x y : ℝ) (k : ℝ) : Prop :=
  x^2 - k * y^2 = 1

-- Definition of the hyperbola having a focus at (3, 0) and the value of k
def has_focus_at (k : ℝ) : Prop :=
  ∃ x y : ℝ, is_hyperbola x y k ∧ (x, y) = (3, 0)

theorem hyperbola_focus (k : ℝ) (h : has_focus_at k) : k = 1 / 8 :=
  sorry

end NUMINAMATH_GPT_hyperbola_focus_l182_18244


namespace NUMINAMATH_GPT_set_C_is_pythagorean_triple_l182_18295

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem set_C_is_pythagorean_triple : is_pythagorean_triple 5 12 13 :=
sorry

end NUMINAMATH_GPT_set_C_is_pythagorean_triple_l182_18295


namespace NUMINAMATH_GPT_no_common_root_l182_18241

theorem no_common_root (a b c d : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) :
  ¬∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) := 
sorry

end NUMINAMATH_GPT_no_common_root_l182_18241


namespace NUMINAMATH_GPT_sufficiency_condition_a_gt_b_sq_gt_sq_l182_18236

theorem sufficiency_condition_a_gt_b_sq_gt_sq (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a > b → a^2 > b^2) ∧ (∀ (h : a^2 > b^2), ∃ c > 0, ∃ d > 0, c^2 > d^2 ∧ ¬(c > d)) :=
by
  sorry

end NUMINAMATH_GPT_sufficiency_condition_a_gt_b_sq_gt_sq_l182_18236


namespace NUMINAMATH_GPT_compare_x_y_l182_18211

variable (a b : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (a_ne_b : a ≠ b)

noncomputable def x : ℝ := (Real.sqrt a + Real.sqrt b) / Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt (a + b)

theorem compare_x_y : y a b > x a b := sorry

end NUMINAMATH_GPT_compare_x_y_l182_18211


namespace NUMINAMATH_GPT_mistaken_divisor_is_12_l182_18291

-- Definitions based on conditions
def correct_divisor : ℕ := 21
def correct_quotient : ℕ := 36
def mistaken_quotient : ℕ := 63

-- The mistaken divisor  is computed as:
def mistaken_divisor : ℕ := correct_quotient * correct_divisor / mistaken_quotient

-- The theorem to prove the mistaken divisor is 12
theorem mistaken_divisor_is_12 : mistaken_divisor = 12 := by
  sorry

end NUMINAMATH_GPT_mistaken_divisor_is_12_l182_18291


namespace NUMINAMATH_GPT_total_bags_sold_l182_18229

theorem total_bags_sold (first_week second_week third_week fourth_week total : ℕ) 
  (h1 : first_week = 15) 
  (h2 : second_week = 3 * first_week) 
  (h3 : third_week = 20) 
  (h4 : fourth_week = 20) 
  (h5 : total = first_week + second_week + third_week + fourth_week) : 
  total = 100 := 
sorry

end NUMINAMATH_GPT_total_bags_sold_l182_18229


namespace NUMINAMATH_GPT_initial_bananas_per_child_l182_18274

theorem initial_bananas_per_child : 
  ∀ (B n m x : ℕ), 
  n = 740 → 
  m = 370 → 
  (B = n * x) → 
  (B = (n - m) * (x + 2)) → 
  x = 2 := 
by
  intros B n m x h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_initial_bananas_per_child_l182_18274


namespace NUMINAMATH_GPT_polynomial_determination_l182_18273

theorem polynomial_determination (P : Polynomial ℝ) :
  (∀ X : ℝ, P.eval (X^2) = (X^2 + 1) * P.eval X) →
  (∃ a : ℝ, ∀ X : ℝ, P.eval X = a * (X^2 - 1)) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_determination_l182_18273


namespace NUMINAMATH_GPT_all_acute_angles_in_first_quadrant_l182_18210

def terminal_side_same (θ₁ θ₂ : ℝ) : Prop := 
  ∃ (k : ℤ), θ₁ = θ₂ + 360 * k

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

def first_quadrant_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem all_acute_angles_in_first_quadrant :
  ∀ θ : ℝ, acute_angle θ → first_quadrant_angle θ :=
by
  intros θ h
  exact h

end NUMINAMATH_GPT_all_acute_angles_in_first_quadrant_l182_18210


namespace NUMINAMATH_GPT_twelve_pow_six_mod_nine_eq_zero_l182_18292

theorem twelve_pow_six_mod_nine_eq_zero : (∃ n : ℕ, 0 ≤ n ∧ n < 9 ∧ 12^6 ≡ n [MOD 9]) → 12^6 ≡ 0 [MOD 9] :=
by
  sorry

end NUMINAMATH_GPT_twelve_pow_six_mod_nine_eq_zero_l182_18292


namespace NUMINAMATH_GPT_trivia_team_points_l182_18271

theorem trivia_team_points (total_members: ℕ) (total_points: ℕ) (points_per_member: ℕ) (members_showed_up: ℕ) (members_did_not_show_up: ℕ):
  total_members = 7 → 
  total_points = 20 → 
  points_per_member = 4 → 
  members_showed_up = total_points / points_per_member → 
  members_did_not_show_up = total_members - members_showed_up → 
  members_did_not_show_up = 2 := 
by 
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_trivia_team_points_l182_18271


namespace NUMINAMATH_GPT_range_of_g_l182_18212

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 3 → 29 ≤ g x ∧ g x ≤ 93 :=
by
  sorry

end NUMINAMATH_GPT_range_of_g_l182_18212


namespace NUMINAMATH_GPT_simplify_trig_identity_l182_18201

open Real

theorem simplify_trig_identity (x y : ℝ) :
  sin x ^ 2 + sin (x + y) ^ 2 - 2 * sin x * sin y * sin (x + y) = sin y ^ 2 := 
sorry

end NUMINAMATH_GPT_simplify_trig_identity_l182_18201


namespace NUMINAMATH_GPT_find_pairs_l182_18262

theorem find_pairs (p a : ℕ) (hp_prime : Nat.Prime p) (hp_ge_2 : p ≥ 2) (ha_ge_1 : a ≥ 1) (h_p_ne_a : p ≠ a) :
  (a + p) ∣ (a^2 + p^2) → (a = p ∧ p = p) ∨ (a = p^2 - p ∧ p = p) ∨ (a = 2 * p^2 - p ∧ p = p) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l182_18262


namespace NUMINAMATH_GPT_incorrect_calculation_l182_18230

theorem incorrect_calculation
  (ξ η : ℝ)
  (Eξ : ℝ)
  (Eη : ℝ)
  (E_min : ℝ)
  (hEξ : Eξ = 3)
  (hEη : Eη = 5)
  (hE_min : E_min = 3.67) :
  E_min > Eξ :=
by
  sorry

end NUMINAMATH_GPT_incorrect_calculation_l182_18230


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_value_l182_18289

theorem arithmetic_sequence_a2_value 
  (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) = a n + 3)
  (h2 : S n = n * (a 1 + a n) / 2)
  (hS13 : S 13 = 156) :
  a 2 = -3 := 
    sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_value_l182_18289


namespace NUMINAMATH_GPT_sticks_difference_l182_18288

def sticks_picked_up : ℕ := 14
def sticks_left : ℕ := 4

theorem sticks_difference : (sticks_picked_up - sticks_left) = 10 := by
  sorry

end NUMINAMATH_GPT_sticks_difference_l182_18288


namespace NUMINAMATH_GPT_flowers_per_pot_l182_18200

def total_gardens : ℕ := 10
def pots_per_garden : ℕ := 544
def total_flowers : ℕ := 174080

theorem flowers_per_pot  :
  (total_flowers / (total_gardens * pots_per_garden)) = 32 :=
by
  -- Here would be the place to provide the proof, but we use sorry for now
  sorry

end NUMINAMATH_GPT_flowers_per_pot_l182_18200


namespace NUMINAMATH_GPT_unique_solution_xy_l182_18205

theorem unique_solution_xy
  (x y : ℕ)
  (h1 : (x^3 + y) % (x^2 + y^2) = 0)
  (h2 : (y^3 + x) % (x^2 + y^2) = 0) :
  x = 1 ∧ y = 1 := sorry

end NUMINAMATH_GPT_unique_solution_xy_l182_18205


namespace NUMINAMATH_GPT_sculpture_cost_in_chinese_yuan_l182_18219

theorem sculpture_cost_in_chinese_yuan
  (usd_to_nad : ℝ)
  (usd_to_cny : ℝ)
  (cost_nad : ℝ)
  (h1 : usd_to_nad = 8)
  (h2 : usd_to_cny = 5)
  (h3 : cost_nad = 160) :
  (cost_nad / usd_to_nad) * usd_to_cny = 100 :=
by
  sorry

end NUMINAMATH_GPT_sculpture_cost_in_chinese_yuan_l182_18219


namespace NUMINAMATH_GPT_quadruples_positive_integers_l182_18259

theorem quadruples_positive_integers (x y z n : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ n > 0 ∧ (x^2 + y^2 + z^2 + 1 = 2^n) →
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ n = 0) ∨ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 1 ∧ z = 0 ∧ n = 1) ∨ 
  (x = 0 ∧ y = 0 ∧ z = 1 ∧ n = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 2) :=
sorry

end NUMINAMATH_GPT_quadruples_positive_integers_l182_18259


namespace NUMINAMATH_GPT_xiao_hua_spent_7_yuan_l182_18240

theorem xiao_hua_spent_7_yuan :
  ∃ (a b c d: ℕ), a + b + c + d = 30 ∧
                   ((a = 5 ∧ b = 5 ∧ c = 10 ∧ d = 10) ∨
                    (a = 5 ∧ b = 10 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 5 ∧ c = 5 ∧ d = 10) ∨
                    (a = 10 ∧ b = 10 ∧ c = 5 ∧ d = 5) ∨
                    (a = 5 ∧ b = 10 ∧ c = 10 ∧ d = 5) ∨
                    (a = 10 ∧ b = 5 ∧ c = 10 ∧ d = 5)) ∧
                   10 * c + 15 * a + 25 * b + 40 * d = 700 :=
by {
  sorry
}

end NUMINAMATH_GPT_xiao_hua_spent_7_yuan_l182_18240


namespace NUMINAMATH_GPT_max_smoothie_servings_l182_18239

-- Define the constants based on the problem conditions
def servings_per_recipe := 4
def bananas_per_recipe := 3
def yogurt_per_recipe := 1 -- cup
def honey_per_recipe := 2 -- tablespoons
def strawberries_per_recipe := 2 -- cups

-- Define the total amount of ingredients Lynn has
def total_bananas := 12
def total_yogurt := 6 -- cups
def total_honey := 16 -- tablespoons (since 1 cup = 16 tablespoons)
def total_strawberries := 8 -- cups

-- Define the calculation for the number of servings each ingredient can produce
def servings_from_bananas := (total_bananas / bananas_per_recipe) * servings_per_recipe
def servings_from_yogurt := (total_yogurt / yogurt_per_recipe) * servings_per_recipe
def servings_from_honey := (total_honey / honey_per_recipe) * servings_per_recipe
def servings_from_strawberries := (total_strawberries / strawberries_per_recipe) * servings_per_recipe

-- Define the minimum number of servings that can be made based on all ingredients
def max_servings := min servings_from_bananas (min servings_from_yogurt (min servings_from_honey servings_from_strawberries))

theorem max_smoothie_servings : max_servings = 16 :=
by
  sorry

end NUMINAMATH_GPT_max_smoothie_servings_l182_18239


namespace NUMINAMATH_GPT_total_weight_full_bucket_l182_18286

theorem total_weight_full_bucket (x y p q : ℝ)
  (h1 : x + (3 / 4) * y = p)
  (h2 : x + (1 / 3) * y = q) :
  x + y = (8 * p - 11 * q) / 5 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_full_bucket_l182_18286


namespace NUMINAMATH_GPT_set_subset_of_inter_union_l182_18245

variable {α : Type} [Nonempty α]
variables {A B C : Set α}

-- The main theorem based on the problem statement
theorem set_subset_of_inter_union (h : A ∩ B = B ∪ C) : C ⊆ B :=
by
  sorry

end NUMINAMATH_GPT_set_subset_of_inter_union_l182_18245


namespace NUMINAMATH_GPT_div_pow_eq_l182_18254

theorem div_pow_eq (n : ℕ) (h : n = 16 ^ 2023) : n / 4 = 4 ^ 4045 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_div_pow_eq_l182_18254


namespace NUMINAMATH_GPT_Ganesh_avg_speed_l182_18279

theorem Ganesh_avg_speed (D : ℝ) : 
  (∃ (V : ℝ), (39.6 = (2 * D) / ((D / 44) + (D / V))) ∧ V = 36) :=
by
  sorry

end NUMINAMATH_GPT_Ganesh_avg_speed_l182_18279


namespace NUMINAMATH_GPT_range_of_b_l182_18253

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3^x + b

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, f x b ≥ 0) ↔ b ≤ -1 :=
by sorry

end NUMINAMATH_GPT_range_of_b_l182_18253


namespace NUMINAMATH_GPT_smallest_num_is_1113805958_l182_18290

def smallest_num (n : ℕ) : Prop :=
  (n + 5) % 19 = 0 ∧ (n + 5) % 73 = 0 ∧ (n + 5) % 101 = 0 ∧ (n + 5) % 89 = 0

theorem smallest_num_is_1113805958 : ∃ n, smallest_num n ∧ n = 1113805958 :=
by
  use 1113805958
  unfold smallest_num
  simp
  sorry

end NUMINAMATH_GPT_smallest_num_is_1113805958_l182_18290
