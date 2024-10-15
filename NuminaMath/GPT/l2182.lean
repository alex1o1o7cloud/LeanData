import Mathlib

namespace NUMINAMATH_GPT_faye_age_l2182_218206

theorem faye_age (D E C F : ℤ)
  (h1 : D = E - 4)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (hD : D = 18) :
  F = 21 :=
by
  sorry

end NUMINAMATH_GPT_faye_age_l2182_218206


namespace NUMINAMATH_GPT_marthas_bedroom_size_l2182_218299

theorem marthas_bedroom_size (M J : ℕ) 
  (h1 : M + J = 300)
  (h2 : J = M + 60) :
  M = 120 := 
sorry

end NUMINAMATH_GPT_marthas_bedroom_size_l2182_218299


namespace NUMINAMATH_GPT_segment_length_294_l2182_218240

theorem segment_length_294
  (A B P Q : ℝ)   -- Define points A, B, P, Q on the real line
  (h1 : P = A + (3 / 8) * (B - A))   -- P divides AB in the ratio 3:5
  (h2 : Q = A + (4 / 11) * (B - A))  -- Q divides AB in the ratio 4:7
  (h3 : Q - P = 3)                   -- The length of PQ is 3
  : B - A = 294 := 
sorry

end NUMINAMATH_GPT_segment_length_294_l2182_218240


namespace NUMINAMATH_GPT_train_speed_conversion_l2182_218244

-- Define the speed of the train in meters per second.
def speed_mps : ℝ := 37.503

-- Definition of the conversion factor between m/s and km/h.
def conversion_factor : ℝ := 3.6

-- Define the expected speed of the train in kilometers per hour.
def expected_speed_kmph : ℝ := 135.0108

-- Prove that the speed in km/h is the expected value.
theorem train_speed_conversion :
  (speed_mps * conversion_factor = expected_speed_kmph) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_conversion_l2182_218244


namespace NUMINAMATH_GPT_range_of_expression_l2182_218274

noncomputable def f (x : ℝ) := |Real.log x / Real.log 2|

theorem range_of_expression (a b : ℝ) (h_f_eq : f a = f b) (h_a_lt_b : a < b) :
  f a = f b → a < b → (∃ c > 3, c = (2 / a) + (1 / b)) := by
  sorry

end NUMINAMATH_GPT_range_of_expression_l2182_218274


namespace NUMINAMATH_GPT_value_subtracted_l2182_218221

theorem value_subtracted (n v : ℝ) (h1 : 2 * n - v = -12) (h2 : n = -10.0) : v = -8 :=
by
  sorry

end NUMINAMATH_GPT_value_subtracted_l2182_218221


namespace NUMINAMATH_GPT_factorize_1_factorize_2_l2182_218246

theorem factorize_1 {x : ℝ} : 2*x^2 - 4*x = 2*x*(x - 2) := 
by sorry

theorem factorize_2 {a b x y : ℝ} : a^2*(x - y) + b^2*(y - x) = (x - y) * (a + b) * (a - b) := 
by sorry

end NUMINAMATH_GPT_factorize_1_factorize_2_l2182_218246


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l2182_218275

open Set

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 ^ 2}

noncomputable def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.1 ^ 2 + (p.2 - a) ^ 2 ≤ 1}

theorem necessary_and_sufficient_condition (a : ℝ) :
  N a ⊆ M ↔ a ≥ 5 / 4 := sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l2182_218275


namespace NUMINAMATH_GPT_correct_product_l2182_218202

-- We define the conditions
def number1 : ℝ := 0.85
def number2 : ℝ := 3.25
def without_decimal_points_prod : ℕ := 27625

-- We state the problem
theorem correct_product (h1 : (85 : ℕ) * (325 : ℕ) = without_decimal_points_prod)
                        (h2 : number1 * number2 * 10000 = (without_decimal_points_prod : ℝ)) :
  number1 * number2 = 2.7625 :=
by sorry

end NUMINAMATH_GPT_correct_product_l2182_218202


namespace NUMINAMATH_GPT_fraction_to_decimal_l2182_218293

theorem fraction_to_decimal : (22 / 8 : ℝ) = 2.75 := 
sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2182_218293


namespace NUMINAMATH_GPT_quadratic_root_value_l2182_218223
-- Import the entirety of the necessary library

-- Define the quadratic equation with one root being -1
theorem quadratic_root_value 
    (m : ℝ)
    (h1 : ∀ x : ℝ, x^2 + m * x + 3 = 0)
    (root1 : -1 ∈ {x : ℝ | x^2 + m * x + 3 = 0}) :
    m = 4 ∧ ∃ root2 : ℝ, root2 = -3 ∧ root2 ∈ {x : ℝ | x^2 + m * x + 3 = 0} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_value_l2182_218223


namespace NUMINAMATH_GPT_possible_values_of_a_plus_b_l2182_218235

variable (a b : ℤ)

theorem possible_values_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = a) :
  (a + b = 0 ∨ a + b = 4 ∨ a + b = -4) :=
sorry

end NUMINAMATH_GPT_possible_values_of_a_plus_b_l2182_218235


namespace NUMINAMATH_GPT_fraction_irreducible_l2182_218264

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_irreducible_l2182_218264


namespace NUMINAMATH_GPT_max_inscribed_triangle_area_l2182_218273

theorem max_inscribed_triangle_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ A, A = (3 * Real.sqrt 3 / 4) * a * b := 
sorry

end NUMINAMATH_GPT_max_inscribed_triangle_area_l2182_218273


namespace NUMINAMATH_GPT_oblique_line_plane_angle_range_l2182_218209

/-- 
An oblique line intersects the plane at an angle other than a right angle. 
The angle cannot be $0$ radians or $\frac{\pi}{2}$ radians.
-/
theorem oblique_line_plane_angle_range (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) : 
  0 < θ ∧ θ < π / 2 :=
by {
  exact ⟨h₀, h₁⟩
}

end NUMINAMATH_GPT_oblique_line_plane_angle_range_l2182_218209


namespace NUMINAMATH_GPT_no_integer_solution_l2182_218283

theorem no_integer_solution (x y z : ℤ) (n : ℕ) (h1 : Prime (x + y)) (h2 : Odd n) : ¬ (x^n + y^n = z^n) :=
sorry

end NUMINAMATH_GPT_no_integer_solution_l2182_218283


namespace NUMINAMATH_GPT_first_shaded_square_in_each_column_l2182_218257

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem first_shaded_square_in_each_column : 
  ∃ n, triangular_number n = 120 ∧ ∀ m < n, ¬ ∀ k < 8, ∃ j ≤ m, ((triangular_number j) % 8) = k := 
by
  sorry

end NUMINAMATH_GPT_first_shaded_square_in_each_column_l2182_218257


namespace NUMINAMATH_GPT_solution_for_equation_l2182_218228

theorem solution_for_equation (m n : ℕ) (h : 0 < m ∧ 0 < n ∧ 2 * m^2 = 3 * n^3) :
  ∃ k : ℕ, 0 < k ∧ m = 18 * k^3 ∧ n = 6 * k^2 :=
by sorry

end NUMINAMATH_GPT_solution_for_equation_l2182_218228


namespace NUMINAMATH_GPT_table_to_chair_ratio_l2182_218227

noncomputable def price_chair : ℤ := 20
noncomputable def price_table : ℤ := 60
noncomputable def price_couch : ℤ := 300

theorem table_to_chair_ratio 
  (h1 : price_couch = 300)
  (h2 : price_couch = 5 * price_table)
  (h3 : price_chair + price_table + price_couch = 380)
  : price_table / price_chair = 3 := 
by 
  sorry

end NUMINAMATH_GPT_table_to_chair_ratio_l2182_218227


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l2182_218231

noncomputable def a_b_conditions (a b: ℝ) : Prop :=
∃ (a b: ℝ), ¬((a - b > 0) → (a^2 - b^2 > 0)) ∧ ¬((a^2 - b^2 > 0) → (a - b > 0))

theorem neither_sufficient_nor_necessary (a b: ℝ) : a_b_conditions a b :=
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l2182_218231


namespace NUMINAMATH_GPT_price_second_day_is_81_percent_l2182_218249

-- Define the original price P (for the sake of clarity in the proof statement)
variable (P : ℝ)

-- Define the reductions
def first_reduction (P : ℝ) : ℝ := P - 0.1 * P
def second_reduction (P : ℝ) : ℝ := first_reduction P - 0.1 * first_reduction P

-- Question translated to Lean statement
theorem price_second_day_is_81_percent (P : ℝ) : 
  (second_reduction P / P) * 100 = 81 := by
  sorry

end NUMINAMATH_GPT_price_second_day_is_81_percent_l2182_218249


namespace NUMINAMATH_GPT_probability_winning_probability_not_winning_l2182_218226

section Lottery

variable (p1 p2 p3 : ℝ)
variable (h1 : p1 = 0.1)
variable (h2 : p2 = 0.2)
variable (h3 : p3 = 0.4)

theorem probability_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  p1 + p2 + p3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

theorem probability_not_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  1 - (p1 + p2 + p3) = 0.3 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end Lottery

end NUMINAMATH_GPT_probability_winning_probability_not_winning_l2182_218226


namespace NUMINAMATH_GPT_true_discount_correct_l2182_218203

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  BD / (1 + (BD / FV))

theorem true_discount_correct
  (FV BD : ℝ)
  (hFV : FV = 2260)
  (hBD : BD = 428.21) :
  true_discount FV BD = 360.00 :=
by
  sorry

end NUMINAMATH_GPT_true_discount_correct_l2182_218203


namespace NUMINAMATH_GPT_recorded_expenditure_l2182_218217

-- Define what it means to record an income and an expenditure
def record_income (y : ℝ) : ℝ := y
def record_expenditure (y : ℝ) : ℝ := -y

-- Define specific instances for the problem
def income_recorded_as : ℝ := 20
def expenditure_value : ℝ := 75

-- Given condition
axiom income_condition : record_income income_recorded_as = 20

-- Theorem to prove the recorded expenditure
theorem recorded_expenditure : record_expenditure expenditure_value = -75 := by
  sorry

end NUMINAMATH_GPT_recorded_expenditure_l2182_218217


namespace NUMINAMATH_GPT_laptop_cost_l2182_218263

theorem laptop_cost (L : ℝ) (smartphone_cost : ℝ) (total_cost : ℝ) (change : ℝ) (n_laptops n_smartphones : ℕ) 
  (hl_smartphone : smartphone_cost = 400) 
  (hl_laptops : n_laptops = 2) 
  (hl_smartphones : n_smartphones = 4) 
  (hl_total : total_cost = 3000)
  (hl_change : change = 200) 
  (hl_total_spent : total_cost - change = 2 * L + 4 * smartphone_cost) : 
  L = 600 :=
by 
  sorry

end NUMINAMATH_GPT_laptop_cost_l2182_218263


namespace NUMINAMATH_GPT_calculate_salary_l2182_218272

-- Define the constants and variables
def food_percentage : ℝ := 0.35
def rent_percentage : ℝ := 0.25
def clothes_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def recreational_percentage : ℝ := 0.15
def emergency_fund : ℝ := 3000
def total_percentage : ℝ := food_percentage + rent_percentage + clothes_percentage + transportation_percentage + recreational_percentage

-- Define the salary
def salary (S : ℝ) : Prop :=
  (total_percentage - 1) * S = emergency_fund

-- The theorem stating the salary is 60000
theorem calculate_salary : ∃ S : ℝ, salary S ∧ S = 60000 :=
by
  use 60000
  unfold salary total_percentage
  sorry

end NUMINAMATH_GPT_calculate_salary_l2182_218272


namespace NUMINAMATH_GPT_total_games_l2182_218260

variable (L : ℕ) -- Number of games the team lost

-- Define the number of wins
def Wins := 3 * L + 14

theorem total_games (h_wins : Wins = 101) : (Wins + L = 130) :=
by
  sorry

end NUMINAMATH_GPT_total_games_l2182_218260


namespace NUMINAMATH_GPT_farmer_goats_l2182_218295

theorem farmer_goats (G C P : ℕ) (h1 : P = 2 * C) (h2 : C = G + 4) (h3 : G + C + P = 56) : G = 11 :=
by
  sorry

end NUMINAMATH_GPT_farmer_goats_l2182_218295


namespace NUMINAMATH_GPT_no_such_function_exists_l2182_218253

open Classical

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (f 0 > 0) ∧ (∀ (x y : ℝ), f (x + y) ≥ f x + y * f (f x)) :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l2182_218253


namespace NUMINAMATH_GPT_polygon_sides_l2182_218252

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) : n = 8 :=
by sorry

end NUMINAMATH_GPT_polygon_sides_l2182_218252


namespace NUMINAMATH_GPT_equal_frac_implies_x_zero_l2182_218232

theorem equal_frac_implies_x_zero (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
sorry

end NUMINAMATH_GPT_equal_frac_implies_x_zero_l2182_218232


namespace NUMINAMATH_GPT_cos_at_min_distance_l2182_218258

noncomputable def cosAtMinimumDistance (t : ℝ) (ht : t < 0) : ℝ :=
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  if distance = Real.sqrt 5 then
    x / distance
  else
    0 -- some default value given the condition distance is not sqrt(5), which is impossible in this context

theorem cos_at_min_distance (t : ℝ) (ht : t < 0) :
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  distance = Real.sqrt 5 → cosAtMinimumDistance t ht = - 2 * Real.sqrt 5 / 5 :=
by
  let x := t / 2 + 2 / t
  let y := 1
  let distance := Real.sqrt (x ^ 2 + y ^ 2)
  sorry

end NUMINAMATH_GPT_cos_at_min_distance_l2182_218258


namespace NUMINAMATH_GPT_reflection_slope_intercept_l2182_218251

noncomputable def reflect_line_slope_intercept (k : ℝ) (hk1 : k ≠ 0) (hk2 : k ≠ -1) : ℝ × ℝ :=
  let slope := (1 : ℝ) / k
  let intercept := (k - 1) / k
  (slope, intercept)

theorem reflection_slope_intercept {k : ℝ} (hk1 : k ≠ 0) (hk2 : k ≠ -1) :
  reflect_line_slope_intercept k hk1 hk2 = (1/k, (k-1)/k) := by
  sorry

end NUMINAMATH_GPT_reflection_slope_intercept_l2182_218251


namespace NUMINAMATH_GPT_equation1_equation2_equation3_equation4_l2182_218262

-- 1. Solve: 2(2x-1)^2 = 8
theorem equation1 (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ (x = 3/2) ∨ (x = -1/2) :=
sorry

-- 2. Solve: 2x^2 + 3x - 2 = 0
theorem equation2 (x : ℝ) : 2 * x^2 + 3 * x - 2 = 0 ↔ (x = 1/2) ∨ (x = -2) :=
sorry

-- 3. Solve: x(2x-7) = 3(2x-7)
theorem equation3 (x : ℝ) : x * (2 * x - 7) = 3 * (2 * x - 7) ↔ (x = 7/2) ∨ (x = 3) :=
sorry

-- 4. Solve: 2y^2 + 8y - 1 = 0
theorem equation4 (y : ℝ) : 2 * y^2 + 8 * y - 1 = 0 ↔ (y = (-4 + 3 * Real.sqrt 2) / 2) ∨ (y = (-4 - 3 * Real.sqrt 2) / 2) :=
sorry

end NUMINAMATH_GPT_equation1_equation2_equation3_equation4_l2182_218262


namespace NUMINAMATH_GPT_problem_I_problem_II_l2182_218269

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4 * a * x + 1
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6 * a^2 * Real.log x + 2 * b + 1
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_I (a : ℝ) (ha : a > 0) :
  ∃ b, b = 5 / 2 * a^2 - 3 * a^2 * Real.log a ∧ ∀ b', b' ≤ 3 / 2 * Real.exp (2 / 3) :=
sorry

theorem problem_II (a x₁ x₂ : ℝ) (ha : a ≥ Real.sqrt 3 - 1) (hx : 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂) :
  (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8 :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2182_218269


namespace NUMINAMATH_GPT_range_of_m_for_ellipse_l2182_218229

-- Define the equation of the ellipse
def ellipse_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- The theorem to prove
theorem range_of_m_for_ellipse (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  5 < m :=
sorry

end NUMINAMATH_GPT_range_of_m_for_ellipse_l2182_218229


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l2182_218265

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hh : h = 4) : 15 * Real.pi = Real.pi * r * (Real.sqrt (r^2 + h^2)) :=
by
  -- Prove that 15π = π * r * sqrt(r^2 + h^2) for r = 3 and h = 4
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l2182_218265


namespace NUMINAMATH_GPT_negation_equivalence_l2182_218266

theorem negation_equivalence (x : ℝ) :
  (¬ (x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (x < 1 → x^2 - 4*x + 2 < -1) :=
by
  sorry

end NUMINAMATH_GPT_negation_equivalence_l2182_218266


namespace NUMINAMATH_GPT_base_conversion_l2182_218284

theorem base_conversion (x : ℕ) (h : 4 * x + 7 = 71) : x = 16 := 
by {
  sorry
}

end NUMINAMATH_GPT_base_conversion_l2182_218284


namespace NUMINAMATH_GPT_min_value_two_x_plus_y_l2182_218259

theorem min_value_two_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y + 2 * x * y = 5 / 4) : 2 * x + y ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_two_x_plus_y_l2182_218259


namespace NUMINAMATH_GPT_mart_income_percentage_l2182_218250

variables (T J M : ℝ)

theorem mart_income_percentage (h1 : M = 1.60 * T) (h2 : T = 0.50 * J) :
  M = 0.80 * J :=
by
  sorry

end NUMINAMATH_GPT_mart_income_percentage_l2182_218250


namespace NUMINAMATH_GPT_radius_of_circle_zero_l2182_218294

theorem radius_of_circle_zero (x y : ℝ) :
    (x^2 + 4*x + y^2 - 2*y + 5 = 0) → 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_circle_zero_l2182_218294


namespace NUMINAMATH_GPT_tg_ctg_sum_l2182_218255

theorem tg_ctg_sum (x : Real) 
  (h : Real.cos x ≠ 0 ∧ Real.sin x ≠ 0 ∧ 1 / Real.cos x - 1 / Real.sin x = 4 * Real.sqrt 3) :
  (Real.sin x / Real.cos x + Real.cos x / Real.sin x = 8 ∨ Real.sin x / Real.cos x + Real.cos x / Real.sin x = -6) :=
sorry

end NUMINAMATH_GPT_tg_ctg_sum_l2182_218255


namespace NUMINAMATH_GPT_zero_of_function_l2182_218205

theorem zero_of_function : ∃ x : Real, 4 * x - 2 = 0 ∧ x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_zero_of_function_l2182_218205


namespace NUMINAMATH_GPT_time_spent_on_type_a_l2182_218219

theorem time_spent_on_type_a (num_questions : ℕ) 
                             (exam_duration : ℕ)
                             (type_a_count : ℕ)
                             (time_ratio : ℕ)
                             (type_b_count : ℕ)
                             (x : ℕ)
                             (total_time : ℕ) :
  num_questions = 200 ∧
  exam_duration = 180 ∧
  type_a_count = 20 ∧
  time_ratio = 2 ∧
  type_b_count = 180 ∧
  total_time = 36 →
  time_ratio * x * type_a_count + x * type_b_count = exam_duration →
  total_time = 36 :=
by
  sorry

end NUMINAMATH_GPT_time_spent_on_type_a_l2182_218219


namespace NUMINAMATH_GPT_fraction_red_knights_magical_l2182_218212

theorem fraction_red_knights_magical (total_knights : ℕ) (fraction_red fraction_magical : ℚ)
  (fraction_red_twice_fraction_blue : ℚ) 
  (h_total_knights : total_knights > 0)
  (h_fraction_red : fraction_red = 2 / 7)
  (h_fraction_magical : fraction_magical = 1 / 6)
  (h_relation : fraction_red_twice_fraction_blue = 2)
  (h_magic_eq : (total_knights : ℚ) * fraction_magical = 
    total_knights * fraction_red * fraction_red_twice_fraction_blue * fraction_magical / 2 + 
    total_knights * (1 - fraction_red) * fraction_magical / 2) :
  total_knights * (fraction_red * fraction_red_twice_fraction_blue / (fraction_red * fraction_red_twice_fraction_blue + (1 - fraction_red) / 2)) = 
  total_knights * 7 / 27 := 
sorry

end NUMINAMATH_GPT_fraction_red_knights_magical_l2182_218212


namespace NUMINAMATH_GPT_inequality_always_true_l2182_218216

theorem inequality_always_true (a : ℝ) (x : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  x^2 + (a - 4) * x + 4 - 2 * a > 0 → (x < 1 ∨ x > 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_always_true_l2182_218216


namespace NUMINAMATH_GPT_cone_volume_and_surface_area_l2182_218289

noncomputable def cone_volume (slant_height height : ℝ) : ℝ := 
  1 / 3 * Real.pi * (Real.sqrt (slant_height^2 - height^2))^2 * height

noncomputable def cone_surface_area (slant_height height : ℝ) : ℝ :=
  Real.pi * (Real.sqrt (slant_height^2 - height^2)) * (Real.sqrt (slant_height^2 - height^2) + slant_height)

theorem cone_volume_and_surface_area :
  (cone_volume 15 9 = 432 * Real.pi) ∧ (cone_surface_area 15 9 = 324 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_and_surface_area_l2182_218289


namespace NUMINAMATH_GPT_sqrt_product_eq_l2182_218225

theorem sqrt_product_eq :
  (16 ^ (1 / 4) : ℝ) * (64 ^ (1 / 2)) = 16 := by
  sorry

end NUMINAMATH_GPT_sqrt_product_eq_l2182_218225


namespace NUMINAMATH_GPT_length_of_chord_l2182_218279

theorem length_of_chord (r AB : ℝ) (h1 : r = 6) (h2 : 0 < AB) (h3 : AB <= 2 * r) : AB ≠ 14 :=
by
  sorry

end NUMINAMATH_GPT_length_of_chord_l2182_218279


namespace NUMINAMATH_GPT_barbara_candies_l2182_218281

theorem barbara_candies : (9 + 18) = 27 :=
by
  sorry

end NUMINAMATH_GPT_barbara_candies_l2182_218281


namespace NUMINAMATH_GPT_hexagon_shaded_area_l2182_218247

-- Given conditions
variable (A B C D T : ℝ)
variable (h₁ : A = 2)
variable (h₂ : B = 3)
variable (h₃ : C = 4)
variable (h₄ : T = 20)
variable (h₅ : A + B + C + D = T)

-- The goal is to prove that the area of the shaded region (D) is 11 cm².
theorem hexagon_shaded_area : D = 11 := by
  sorry

end NUMINAMATH_GPT_hexagon_shaded_area_l2182_218247


namespace NUMINAMATH_GPT_find_number_l2182_218211

-- Let's define the condition
def condition (x : ℝ) : Prop := x * 99999 = 58293485180

-- Statement to be proved
theorem find_number : ∃ x : ℝ, condition x ∧ x = 582.935 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l2182_218211


namespace NUMINAMATH_GPT_angle_measure_l2182_218261

variable (x : ℝ)

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem angle_measure (h : supplement x = 8 * complement x) : x = 540 / 7 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l2182_218261


namespace NUMINAMATH_GPT_chloe_boxes_l2182_218288

/-- Chloe was unboxing some of her old winter clothes. She found some boxes of clothing and
inside each box, there were 2 scarves and 6 mittens. Chloe had a total of 32 pieces of
winter clothing. How many boxes of clothing did Chloe find? -/
theorem chloe_boxes (boxes : ℕ) (total_clothing : ℕ) (pieces_per_box : ℕ) :
  pieces_per_box = 8 -> total_clothing = 32 -> total_clothing / pieces_per_box = boxes -> boxes = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_chloe_boxes_l2182_218288


namespace NUMINAMATH_GPT_find_13th_result_l2182_218224

theorem find_13th_result 
  (average_25 : ℕ → ℝ) (h1 : average_25 25 = 19)
  (average_first_12 : ℕ → ℝ) (h2 : average_first_12 12 = 14)
  (average_last_12 : ℕ → ℝ) (h3 : average_last_12 12 = 17) :
    let totalSum_25 := 25 * average_25 25
    let totalSum_first_12 := 12 * average_first_12 12
    let totalSum_last_12 := 12 * average_last_12 12
    let result_13 := totalSum_25 - totalSum_first_12 - totalSum_last_12
    result_13 = 103 :=
  by sorry

end NUMINAMATH_GPT_find_13th_result_l2182_218224


namespace NUMINAMATH_GPT_quadratic_solution_l2182_218287

theorem quadratic_solution (a : ℝ) (h : 2^2 - 3 * 2 + a = 0) : 2 * a - 1 = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_solution_l2182_218287


namespace NUMINAMATH_GPT_correct_system_of_equations_l2182_218214

theorem correct_system_of_equations : 
  ∃ (x y : ℕ), x + y = 12 ∧ 4 * x + 3 * y = 40 := by
  -- we are stating the existence of x and y that satisfy both equations given as conditions.
  sorry

end NUMINAMATH_GPT_correct_system_of_equations_l2182_218214


namespace NUMINAMATH_GPT_part1_part2_l2182_218215

open Real

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - 1

-- Define the function g for the second part
def g (x : ℝ) : ℝ := |x - 2| + |x + 3|

theorem part1 (m : ℝ) : (∀ x, f x m ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 :=
  by sorry

theorem part2 (t x: ℝ) (h: ∀ x: ℝ, f x 2 + f (x + 5) 2 ≥ t - 2) : t ≤ 5 :=
  by sorry

end NUMINAMATH_GPT_part1_part2_l2182_218215


namespace NUMINAMATH_GPT_prime_check_for_d1_prime_check_for_d2_l2182_218292

-- Define d1 and d2
def d1 : ℕ := 9^4 - 9^3 + 9^2 - 9 + 1
def d2 : ℕ := 9^4 - 9^2 + 1

-- Prime checking function
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Stating the conditions and proofs
theorem prime_check_for_d1 : ¬ is_prime d1 :=
by {
  -- condition: ten 8's in base nine is divisible by d1 (5905) is not used here directly
  sorry
}

theorem prime_check_for_d2 : is_prime d2 :=
by {
  -- condition: twelve 8's in base nine is divisible by d2 (6481) is not used here directly
  sorry
}

end NUMINAMATH_GPT_prime_check_for_d1_prime_check_for_d2_l2182_218292


namespace NUMINAMATH_GPT_shaded_region_area_l2182_218296

-- Definitions of known conditions
def grid_section_1_area : ℕ := 3 * 3
def grid_section_2_area : ℕ := 4 * 5
def grid_section_3_area : ℕ := 5 * 6

def total_grid_area : ℕ := grid_section_1_area + grid_section_2_area + grid_section_3_area

def base_of_unshaded_triangle : ℕ := 15
def height_of_unshaded_triangle : ℕ := 6

def unshaded_triangle_area : ℕ := (base_of_unshaded_triangle * height_of_unshaded_triangle) / 2

-- Statement of the problem
theorem shaded_region_area : (total_grid_area - unshaded_triangle_area) = 14 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2182_218296


namespace NUMINAMATH_GPT_cost_of_agricultural_equipment_max_units_of_type_A_l2182_218241

-- Define cost equations
variables (x y : ℝ)

-- Define conditions as hypotheses
def condition1 : Prop := 2 * x + y = 4.2
def condition2 : Prop := x + 3 * y = 5.1

-- Prove the costs are respectively 1.5 and 1.2
theorem cost_of_agricultural_equipment (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 1.5 ∧ y = 1.2 := sorry

-- Define the maximum units constraint
def total_cost (m : ℕ) : ℝ := 1.5 * m + 1.2 * (2 * m - 3)

-- Prove the maximum units of type A is 3
theorem max_units_of_type_A (m : ℕ) (h : total_cost m ≤ 10) : m ≤ 3 := sorry

end NUMINAMATH_GPT_cost_of_agricultural_equipment_max_units_of_type_A_l2182_218241


namespace NUMINAMATH_GPT_sum_of_squares_l2182_218238

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2182_218238


namespace NUMINAMATH_GPT_g_is_correct_l2182_218290

noncomputable def g : ℝ → ℝ := sorry

axiom g_0 : g 0 = 2

axiom g_functional_eq : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2

theorem g_is_correct : ∀ x : ℝ, g x = 2 - 2 * x := 
by 
  sorry

end NUMINAMATH_GPT_g_is_correct_l2182_218290


namespace NUMINAMATH_GPT_no_arith_geo_progression_S1_S2_S3_l2182_218207

noncomputable def S_1 (A B C : Point) : ℝ := sorry -- area of triangle ABC
noncomputable def S_2 (A B E : Point) : ℝ := sorry -- area of triangle ABE
noncomputable def S_3 (A B D : Point) : ℝ := sorry -- area of triangle ABD

def bisecting_plane (A B D C E : Point) : Prop := sorry -- plane bisects dihedral angle at AB

theorem no_arith_geo_progression_S1_S2_S3 (A B C D E : Point) 
(h_bisect : bisecting_plane A B D C E) :
¬ (∃ (S1 S2 S3 : ℝ), S1 = S_1 A B C ∧ S2 = S_2 A B E ∧ S3 = S_3 A B D ∧ 
  (S2 = (S1 + S3) / 2 ∨ S2^2 = S1 * S3 )) :=
sorry

end NUMINAMATH_GPT_no_arith_geo_progression_S1_S2_S3_l2182_218207


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2182_218270

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 + x - 2 = 0) : x^2 + x + 2023 = 2025 := 
  sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h : a + b = 5) : 2 * (a + b) - 4 * a - 4 * b + 21 = 11 := 
  sorry

-- Problem 3
theorem problem3 (a b : ℝ) (h1 : a^2 + 3 * a * b = 20) (h2 : b^2 + 5 * a * b = 8) : 2 * a^2 - b^2 + a * b = 32 := 
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2182_218270


namespace NUMINAMATH_GPT_smallest_lucky_number_exists_l2182_218234

theorem smallest_lucky_number_exists :
  ∃ (a b c d N: ℕ), 
  N = a^2 + b^2 ∧ 
  N = c^2 + d^2 ∧ 
  a - c = 7 ∧ 
  d - b = 13 ∧ 
  N = 545 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_lucky_number_exists_l2182_218234


namespace NUMINAMATH_GPT_percentage_increase_l2182_218298

variables {a b : ℝ} -- Assuming a and b are real numbers

-- Define the conditions explicitly
def initial_workers := a
def workers_left := b
def remaining_workers := a - b

-- Define the theorem for percentage increase in daily performance
theorem percentage_increase (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
  (100 * b) / (a - b) = (100 * a * b) / (a * (a - b)) :=
by
  sorry -- Proof will be filled in as needed

end NUMINAMATH_GPT_percentage_increase_l2182_218298


namespace NUMINAMATH_GPT_marks_lost_per_wrong_answer_l2182_218208

theorem marks_lost_per_wrong_answer (score_per_correct : ℕ) (total_questions : ℕ) 
(total_score : ℕ) (correct_attempts : ℕ) (wrong_attempts : ℕ) (marks_lost_total : ℕ)
(H1 : score_per_correct = 4)
(H2 : total_questions = 75)
(H3 : total_score = 125)
(H4 : correct_attempts = 40)
(H5 : wrong_attempts = total_questions - correct_attempts)
(H6 : marks_lost_total = (correct_attempts * score_per_correct) - total_score)
: (marks_lost_total / wrong_attempts) = 1 := by
  sorry

end NUMINAMATH_GPT_marks_lost_per_wrong_answer_l2182_218208


namespace NUMINAMATH_GPT_part1_part2_l2182_218285

open Set Real

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | 2 ≤ x ∧ x < 5 }
def setB : Set ℝ := { x | 1 < x ∧ x < 8 }
def setC (a : ℝ) : Set ℝ := { x | x < a - 1 ∨ x > a }

-- Conditions:
-- - Complement of A
def complementA : Set ℝ := { x | x < 2 ∨ x ≥ 5 }

-- Question parts:
-- (1) Finding intersection of complementA and B
theorem part1 : (complementA ∩ setB) = { x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 8) } := sorry

-- (2) Finding range of a for specific condition on C
theorem part2 (a : ℝ) : (setA ∪ setC a = univ) → (a ≤ 2 ∨ a > 6) := sorry

end NUMINAMATH_GPT_part1_part2_l2182_218285


namespace NUMINAMATH_GPT_coordinates_of_N_l2182_218239

theorem coordinates_of_N
  (M : ℝ × ℝ)
  (a : ℝ × ℝ)
  (x y : ℝ)
  (hM : M = (5, -6))
  (ha : a = (1, -2))
  (hMN : (x - M.1, y - M.2) = (-3 * a.1, -3 * a.2)) :
  (x, y) = (2, 0) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_N_l2182_218239


namespace NUMINAMATH_GPT_no_real_roots_range_l2182_218248

theorem no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_range_l2182_218248


namespace NUMINAMATH_GPT_clothing_weight_removed_l2182_218276

/-- 
In a suitcase, the initial ratio of books to clothes to electronics, by weight measured in pounds, 
is 7:4:3. The electronics weight 9 pounds. Someone removes some pounds of clothing, doubling the ratio of books to clothes. 
This theorem verifies the weight of clothing removed is 1.5 pounds.
-/
theorem clothing_weight_removed 
  (B C E : ℕ) 
  (initial_ratio : B / 7 = C / 4 ∧ C / 4 = E / 3)
  (E_val : E = 9)
  (new_ratio : ∃ x : ℝ, B / (C - x) = 2) : 
  ∃ x : ℝ, x = 1.5 := 
sorry

end NUMINAMATH_GPT_clothing_weight_removed_l2182_218276


namespace NUMINAMATH_GPT_wrongly_recorded_height_l2182_218282

theorem wrongly_recorded_height 
  (avg_incorrect : ℕ → ℕ → ℕ)
  (avg_correct : ℕ → ℕ → ℕ)
  (boy_count : ℕ)
  (incorrect_avg_height : ℕ) 
  (correct_avg_height : ℕ) 
  (actual_height : ℕ) 
  (correct_total_height : ℕ) 
  (incorrect_total_height: ℕ)
  (x : ℕ) :
  avg_incorrect boy_count incorrect_avg_height = incorrect_total_height →
  avg_correct boy_count correct_avg_height = correct_total_height →
  incorrect_total_height - x + actual_height = correct_total_height →
  x = 176 := 
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_wrongly_recorded_height_l2182_218282


namespace NUMINAMATH_GPT_x1_sufficient_not_necessary_l2182_218254

theorem x1_sufficient_not_necessary : (x : ℝ) → (x = 1 ↔ (x - 1) * (x + 2) = 0) ∧ ∀ x, (x = 1 ∨ x = -2) → (x - 1) * (x + 2) = 0 ∧ (∀ y, (y - 1) * (y + 2) = 0 → (y = 1 ∨ y = -2)) :=
by
  sorry

end NUMINAMATH_GPT_x1_sufficient_not_necessary_l2182_218254


namespace NUMINAMATH_GPT_smallest_possible_floor_sum_l2182_218200

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end NUMINAMATH_GPT_smallest_possible_floor_sum_l2182_218200


namespace NUMINAMATH_GPT_part1_l2182_218237

noncomputable def f (a x : ℝ) : ℝ := a * x - 2 * Real.log x + 2 * (1 + a) + (a - 2) / x

theorem part1 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 1 ≤ x → f a x ≥ 0) ↔ 1 ≤ a :=
sorry

end NUMINAMATH_GPT_part1_l2182_218237


namespace NUMINAMATH_GPT_range_of_m_l2182_218213

theorem range_of_m 
    (m : ℝ) (x : ℝ)
    (p : x^2 - 8 * x - 20 > 0)
    (q : (x - (1 - m)) * (x - (1 + m)) > 0)
    (h : ∀ x, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
    0 < m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l2182_218213


namespace NUMINAMATH_GPT_exponential_function_fixed_point_l2182_218204

theorem exponential_function_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end NUMINAMATH_GPT_exponential_function_fixed_point_l2182_218204


namespace NUMINAMATH_GPT_find_unknown_number_l2182_218267

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l2182_218267


namespace NUMINAMATH_GPT_binom_15_3_eq_455_l2182_218220

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement: Prove that binom 15 3 = 455
theorem binom_15_3_eq_455 : binom 15 3 = 455 := sorry

end NUMINAMATH_GPT_binom_15_3_eq_455_l2182_218220


namespace NUMINAMATH_GPT_bill_and_harry_nuts_l2182_218230

theorem bill_and_harry_nuts {Bill Harry Sue : ℕ} 
    (h1 : Bill = 6 * Harry) 
    (h2 : Harry = 2 * Sue) 
    (h3 : Sue = 48) : 
    Bill + Harry = 672 := 
by
  sorry

end NUMINAMATH_GPT_bill_and_harry_nuts_l2182_218230


namespace NUMINAMATH_GPT_max_area_of_triangle_l2182_218243

theorem max_area_of_triangle (AB AC BC : ℝ) : 
  AB = 4 → AC = 2 * BC → 
  ∃ (S : ℝ), (∀ (S' : ℝ), S' ≤ S) ∧ S = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_l2182_218243


namespace NUMINAMATH_GPT_multiplication_subtraction_difference_l2182_218218

theorem multiplication_subtraction_difference (x n : ℕ) (h₁ : x = 5) (h₂ : 3 * x = (16 - x) + n) : n = 4 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_multiplication_subtraction_difference_l2182_218218


namespace NUMINAMATH_GPT_milk_production_per_cow_l2182_218278

theorem milk_production_per_cow :
  ∀ (total_cows : ℕ) (milk_price_per_gallon butter_price_per_stick total_earnings : ℝ)
    (customers customer_milk_demand gallons_per_butter : ℕ),
  total_cows = 12 →
  milk_price_per_gallon = 3 →
  butter_price_per_stick = 1.5 →
  total_earnings = 144 →
  customers = 6 →
  customer_milk_demand = 6 →
  gallons_per_butter = 2 →
  (∀ (total_milk_sold_to_customers produced_milk used_for_butter : ℕ),
    total_milk_sold_to_customers = customers * customer_milk_demand →
    produced_milk = total_milk_sold_to_customers + used_for_butter →
    used_for_butter = (total_earnings - (total_milk_sold_to_customers * milk_price_per_gallon)) / butter_price_per_stick / gallons_per_butter →
    produced_milk / total_cows = 4)
:= by sorry

end NUMINAMATH_GPT_milk_production_per_cow_l2182_218278


namespace NUMINAMATH_GPT_james_total_cost_is_100_l2182_218286

def cost_of_shirts (number_of_shirts : Nat) (cost_per_shirt : Nat) : Nat :=
  number_of_shirts * cost_per_shirt

def cost_of_pants (number_of_pants : Nat) (cost_per_pants : Nat) : Nat :=
  number_of_pants * cost_per_pants

def total_cost (number_of_shirts : Nat) (number_of_pants : Nat) (cost_per_shirt : Nat) (cost_per_pants : Nat) : Nat :=
  cost_of_shirts number_of_shirts cost_per_shirt + cost_of_pants number_of_pants cost_per_pants

theorem james_total_cost_is_100 : 
  total_cost 10 (10 / 2) 6 8 = 100 :=
by
  sorry

end NUMINAMATH_GPT_james_total_cost_is_100_l2182_218286


namespace NUMINAMATH_GPT_square_perimeter_equals_66_88_l2182_218236

noncomputable def circle_perimeter : ℝ := 52.5

noncomputable def circle_radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def circle_diameter (r : ℝ) : ℝ := 2 * r

noncomputable def square_side_length (d : ℝ) : ℝ := d

noncomputable def square_perimeter (s : ℝ) : ℝ := 4 * s

theorem square_perimeter_equals_66_88 :
  square_perimeter (square_side_length (circle_diameter (circle_radius circle_perimeter))) = 66.88 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_square_perimeter_equals_66_88_l2182_218236


namespace NUMINAMATH_GPT_remainder_of_large_number_l2182_218291

noncomputable def X (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 4 => 16
  | 5 => 32
  | 6 => 64
  | 7 => 128
  | 8 => 256
  | 9 => 512
  | 10 => 1024
  | 11 => 2048
  | 12 => 4096
  | 13 => 8192
  | _ => 0

noncomputable def concatenate_X (k : ℕ) : ℕ :=
  if k = 5 then 
    100020004000800160032
  else if k = 11 then 
    100020004000800160032006401280256051210242048
  else if k = 13 then 
    10002000400080016003200640128025605121024204840968192
  else 
    0

theorem remainder_of_large_number :
  (concatenate_X 13) % (concatenate_X 5) = 40968192 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_large_number_l2182_218291


namespace NUMINAMATH_GPT_rows_seat_7_students_are_5_l2182_218271

-- Definitions based on provided conditions
def total_students : Nat := 53
def total_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_students = 6 * six_seat_rows + 7 * seven_seat_rows

-- To prove the number of rows seating exactly 7 students is 5
def number_of_7_seat_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_rows six_seat_rows seven_seat_rows ∧ seven_seat_rows = 5

-- Statement to be proved
theorem rows_seat_7_students_are_5 : ∃ (six_seat_rows seven_seat_rows : Nat), number_of_7_seat_rows six_seat_rows seven_seat_rows := 
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_rows_seat_7_students_are_5_l2182_218271


namespace NUMINAMATH_GPT_solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l2182_218277

variable (a : ℝ) (x : ℝ)

def inequality (a x : ℝ) : Prop :=
  a * x^2 - (a + 2) * x + 2 < 0

theorem solve_inequality_when_a_lt_2 (h : a < 2) :
  (a = 0 → ∀ x, x > 1 → inequality a x) ∧
  (a < 0 → ∀ x, x < 2 / a ∨ x > 1 → inequality a x) ∧
  (0 < a ∧ a < 2 → ∀ x, 1 < x ∧ x < 2 / a → inequality a x) := 
sorry

theorem find_a_range_when_x_in_2_3 :
  (∀ x, 2 ≤ x ∧ x ≤ 3 → inequality a x) → a < 2 / 3 :=
sorry

end NUMINAMATH_GPT_solve_inequality_when_a_lt_2_find_a_range_when_x_in_2_3_l2182_218277


namespace NUMINAMATH_GPT_Grandfather_age_correct_l2182_218280

-- Definitions based on the conditions
def Yuna_age : Nat := 9
def Father_age (Yuna_age : Nat) : Nat := Yuna_age + 27
def Grandfather_age (Father_age : Nat) : Nat := Father_age + 23

-- The theorem stating the problem to prove
theorem Grandfather_age_correct : Grandfather_age (Father_age Yuna_age) = 59 := by
  sorry

end NUMINAMATH_GPT_Grandfather_age_correct_l2182_218280


namespace NUMINAMATH_GPT_part1_part2_l2182_218222

variable (a b : ℝ)

-- Part (1)
theorem part1 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h : a ≠ b) :
  A + B > 0 := sorry

-- Part (2)
theorem part2 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h: a * b = 1) : 
  A - B = -4 := sorry

end NUMINAMATH_GPT_part1_part2_l2182_218222


namespace NUMINAMATH_GPT_distinct_values_least_count_l2182_218233

theorem distinct_values_least_count (total_integers : ℕ) (mode_count : ℕ) (unique_mode : Prop) 
  (h1 : total_integers = 3200)
  (h2 : mode_count = 17)
  (h3 : unique_mode):
  ∃ (least_count : ℕ), least_count = 200 := by
  sorry

end NUMINAMATH_GPT_distinct_values_least_count_l2182_218233


namespace NUMINAMATH_GPT_barbara_total_cost_l2182_218242

-- Define conditions
def steak_weight : ℝ := 4.5
def steak_price_per_pound : ℝ := 15.0
def chicken_weight : ℝ := 1.5
def chicken_price_per_pound : ℝ := 8.0

-- Define total cost formula
def total_cost := (steak_weight * steak_price_per_pound) + (chicken_weight * chicken_price_per_pound)

-- Prove that the total cost equals $79.50
theorem barbara_total_cost : total_cost = 79.50 := by
  sorry

end NUMINAMATH_GPT_barbara_total_cost_l2182_218242


namespace NUMINAMATH_GPT_star_point_angle_l2182_218256

theorem star_point_angle (n : ℕ) (h : n > 4) (h₁ : n ≥ 3) :
  ∃ θ : ℝ, θ = (n-2) * 180 / n :=
by
  sorry

end NUMINAMATH_GPT_star_point_angle_l2182_218256


namespace NUMINAMATH_GPT_common_chord_properties_l2182_218297

noncomputable def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 1 = 0

noncomputable def length_common_chord : ℝ := 2 * Real.sqrt 5

theorem common_chord_properties :
  (∀ x y : ℝ, 
    x^2 + y^2 + 2 * x + 8 * y - 8 = 0 ∧
    x^2 + y^2 - 4 * x - 4 * y - 2 = 0 →
    line_equation x y) ∧ 
  length_common_chord = 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_common_chord_properties_l2182_218297


namespace NUMINAMATH_GPT_odd_function_f_2_eq_2_l2182_218210

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^2 + 3 * x else -(if -x < 0 then (-x)^2 + 3 * (-x) else x^2 + 3 * x)

theorem odd_function_f_2_eq_2 : f 2 = 2 :=
by
  -- sorry will be used to skip the actual proof
  sorry

end NUMINAMATH_GPT_odd_function_f_2_eq_2_l2182_218210


namespace NUMINAMATH_GPT_num_articles_cost_price_l2182_218245

theorem num_articles_cost_price (N C S : ℝ) (h1 : N * C = 50 * S) (h2 : (S - C) / C * 100 = 10) : N = 55 := 
sorry

end NUMINAMATH_GPT_num_articles_cost_price_l2182_218245


namespace NUMINAMATH_GPT_polynomial_evaluation_l2182_218268

theorem polynomial_evaluation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) : 6 * x^2 + 9 * y + 8 = 23 :=
sorry

end NUMINAMATH_GPT_polynomial_evaluation_l2182_218268


namespace NUMINAMATH_GPT_mr_bird_on_time_58_mph_l2182_218201

def mr_bird_travel_speed_exactly_on_time (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) : ℝ :=
  58

theorem mr_bird_on_time_58_mph (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) :
  mr_bird_travel_speed_exactly_on_time d t h₁ h₂ = 58 := 
  by
  sorry

end NUMINAMATH_GPT_mr_bird_on_time_58_mph_l2182_218201
