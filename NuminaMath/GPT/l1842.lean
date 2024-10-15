import Mathlib

namespace NUMINAMATH_GPT_canteen_distances_l1842_184203

theorem canteen_distances 
  (B G C : ℝ)
  (hB : B = 600)
  (hBG : G = 800)
  (hBC_eq_2x : ∃ x, C = 2 * x ∧ B = G + x + x) :
  G = 800 / 3 :=
by
  sorry

end NUMINAMATH_GPT_canteen_distances_l1842_184203


namespace NUMINAMATH_GPT_value_is_correct_l1842_184298

-- Define the mean and standard deviation
def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

-- Define the value that is 2 standard deviations less than the mean
def value : ℝ := mean - 2 * std_dev

-- Theorem stating that value = 11.0
theorem value_is_correct : value = 11.0 := by
  sorry

end NUMINAMATH_GPT_value_is_correct_l1842_184298


namespace NUMINAMATH_GPT_multiple_of_second_number_l1842_184246

def main : IO Unit := do
  IO.println s!"Proof problem statement in Lean 4."

theorem multiple_of_second_number (x m : ℕ) 
  (h1 : 19 = m * x + 3) 
  (h2 : 19 + x = 27) : 
  m = 2 := 
sorry

end NUMINAMATH_GPT_multiple_of_second_number_l1842_184246


namespace NUMINAMATH_GPT_tree_boy_growth_ratio_l1842_184296

theorem tree_boy_growth_ratio 
    (initial_tree_height final_tree_height initial_boy_height final_boy_height : ℕ) 
    (h₀ : initial_tree_height = 16) 
    (h₁ : final_tree_height = 40) 
    (h₂ : initial_boy_height = 24) 
    (h₃ : final_boy_height = 36) 
:
  (final_tree_height - initial_tree_height) / (final_boy_height - initial_boy_height) = 2 := 
by {
    -- Definitions and given conditions used in the statement part of the proof
    sorry
}

end NUMINAMATH_GPT_tree_boy_growth_ratio_l1842_184296


namespace NUMINAMATH_GPT_find_X_l1842_184257

def r (X Y : ℕ) : ℕ := X^2 + Y^2

theorem find_X (X : ℕ) (h : r X 7 = 338) : X = 17 := by
  sorry

end NUMINAMATH_GPT_find_X_l1842_184257


namespace NUMINAMATH_GPT_derive_units_equivalent_to_velocity_l1842_184290

-- Define the unit simplifications
def watt := 1 * (1 * (1 * (1 / 1)))
def newton := 1 * (1 * (1 / (1 * 1)))

-- Define the options
def option_A := watt / newton
def option_B := newton / watt
def option_C := watt / (newton * newton)
def option_D := (watt * watt) / newton
def option_E := (newton * newton) / (watt * watt)

-- Define what it means for a unit to be equivalent to velocity
def is_velocity (unit : ℚ) : Prop := unit = (1 * (1 / 1))

theorem derive_units_equivalent_to_velocity :
  is_velocity option_A ∧ 
  ¬ is_velocity option_B ∧ 
  ¬ is_velocity option_C ∧ 
  ¬ is_velocity option_D ∧ 
  ¬ is_velocity option_E := 
by sorry

end NUMINAMATH_GPT_derive_units_equivalent_to_velocity_l1842_184290


namespace NUMINAMATH_GPT_acorns_given_is_correct_l1842_184210

-- Define initial conditions
def initial_acorns : ℕ := 16
def remaining_acorns : ℕ := 9

-- Define the number of acorns given to her sister
def acorns_given : ℕ := initial_acorns - remaining_acorns

-- Theorem statement
theorem acorns_given_is_correct : acorns_given = 7 := by
  sorry

end NUMINAMATH_GPT_acorns_given_is_correct_l1842_184210


namespace NUMINAMATH_GPT_petya_wins_max_margin_l1842_184201

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end NUMINAMATH_GPT_petya_wins_max_margin_l1842_184201


namespace NUMINAMATH_GPT_find_a_l1842_184204

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 3^x + a / (3^x + 1)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, 3^x + a / (3^x + 1) ≥ 5) ∧ (∃ x : ℝ, 3^x + a / (3^x + 1) = 5) 
  → a = 9 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1842_184204


namespace NUMINAMATH_GPT_range_of_a_l1842_184295

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 * Real.exp (-x1) = a) 
    ∧ (x2^2 * Real.exp (-x2) = a) ∧ (x3^2 * Real.exp (-x3) = a)) ↔ (0 < a ∧ a < 4 * Real.exp (-2)) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1842_184295


namespace NUMINAMATH_GPT_nancy_crystal_beads_l1842_184260

-- Definitions of given conditions
def price_crystal : ℕ := 9
def price_metal : ℕ := 10
def sets_metal : ℕ := 2
def total_spent : ℕ := 29

-- Statement of the proof problem
theorem nancy_crystal_beads : ∃ x : ℕ, price_crystal * x + price_metal * sets_metal = total_spent ∧ x = 1 := by
  sorry

end NUMINAMATH_GPT_nancy_crystal_beads_l1842_184260


namespace NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1842_184231

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), (n > 0) ∧ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (n * α^2 + a * α + b = 0) ∧ (n * β^2 + a * β + b = 0)
 ) ∧ (∀ (m : ℕ), m > 0 ∧ m < n → ¬ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (m * α^2 + a * α + b = 0) ∧ (m * β^2 + a * β + b = 0))) := 
sorry

end NUMINAMATH_GPT_smallest_n_satisfying_conditions_l1842_184231


namespace NUMINAMATH_GPT_c_range_l1842_184200

open Real

theorem c_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1)
  (h2 : 1 / (a + b) + 1 / c = 1) : 1 < c ∧ c ≤ 4 / 3 := 
sorry

end NUMINAMATH_GPT_c_range_l1842_184200


namespace NUMINAMATH_GPT_solution_interval_l1842_184224

theorem solution_interval (x : ℝ) : (x^2 / (x - 5)^2 > 0) ↔ (x ∈ Set.Iio 0 ∪ Set.Ioi 0 ∩ Set.Iio 5 ∪ Set.Ioi 5) :=
by
  sorry

end NUMINAMATH_GPT_solution_interval_l1842_184224


namespace NUMINAMATH_GPT_value_of_coefficients_l1842_184211

theorem value_of_coefficients (a₀ a₁ a₂ a₃ : ℤ) (x : ℤ) :
  (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 →
  x = -1 →
  (a₀ + a₂) - (a₁ + a₃) = -1 :=
by
  sorry

end NUMINAMATH_GPT_value_of_coefficients_l1842_184211


namespace NUMINAMATH_GPT_percent_correct_l1842_184251

theorem percent_correct (x : ℕ) : 
  (5 * 100.0 / 7) = 71.43 :=
by
  sorry

end NUMINAMATH_GPT_percent_correct_l1842_184251


namespace NUMINAMATH_GPT_triangle_inequality_l1842_184259

theorem triangle_inequality (a b c S : ℝ)
  (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)   -- a, b, c are sides of a non-isosceles triangle
  (S_def : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  (a^3) / ((a - b) * (a - c)) + (b^3) / ((b - c) * (b - a)) + (c^3) / ((c - a) * (c - b)) > 2 * 3^(3/4) * S :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1842_184259


namespace NUMINAMATH_GPT_group_made_l1842_184213

-- Definitions based on the problem's conditions
def teachers_made : Nat := 28
def total_products : Nat := 93

-- Theorem to prove that the group made 65 recycled materials
theorem group_made : total_products - teachers_made = 65 := by
  sorry

end NUMINAMATH_GPT_group_made_l1842_184213


namespace NUMINAMATH_GPT_xy_diff_square_l1842_184294

theorem xy_diff_square (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_xy_diff_square_l1842_184294


namespace NUMINAMATH_GPT_number_of_teams_l1842_184229

theorem number_of_teams (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end NUMINAMATH_GPT_number_of_teams_l1842_184229


namespace NUMINAMATH_GPT_complement_set_P_l1842_184223

open Set

theorem complement_set_P (P : Set ℝ) (hP : P = {x : ℝ | x ≥ 1}) : Pᶜ = {x : ℝ | x < 1} :=
sorry

end NUMINAMATH_GPT_complement_set_P_l1842_184223


namespace NUMINAMATH_GPT_truck_distance_on_7_gallons_l1842_184261

theorem truck_distance_on_7_gallons :
  ∀ (d : ℝ) (g₁ g₂ : ℝ), d = 240 → g₁ = 5 → g₂ = 7 → (d / g₁) * g₂ = 336 :=
by
  intros d g₁ g₂ h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end NUMINAMATH_GPT_truck_distance_on_7_gallons_l1842_184261


namespace NUMINAMATH_GPT_hyperbola_real_axis_length_l1842_184258

theorem hyperbola_real_axis_length :
  (∃ a : ℝ, (∀ x y : ℝ, (x^2 / 9 - y^2 = 1) → (2 * a = 6))) :=
sorry

end NUMINAMATH_GPT_hyperbola_real_axis_length_l1842_184258


namespace NUMINAMATH_GPT_no_solutions_to_equation_l1842_184244

theorem no_solutions_to_equation : ¬∃ x : ℝ, (x ≠ 0) ∧ (x ≠ 5) ∧ ((2 * x ^ 2 - 10 * x) / (x ^ 2 - 5 * x) = x - 3) :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_to_equation_l1842_184244


namespace NUMINAMATH_GPT_slope_of_intersection_points_l1842_184266

theorem slope_of_intersection_points : 
  (∀ t : ℝ, ∃ x y : ℝ, (2 * x + 3 * y = 10 * t + 4) ∧ (x + 4 * y = 3 * t + 3)) → 
  (∀ t1 t2 : ℝ, t1 ≠ t2 → ((2 * ((10 * t1 + 4)  / 2) + 3 * ((-5/3 * t1 - 2/3)) = (10 * t1 + 4)) ∧ (2 * ((10 * t2 + 4) / 2) + 3 * ((-5/3 * t2 - 2/3)) = (10 * t2 + 4))) → 
  (31 * (((-5/3 * t1 - 2/3) - (-5/3 * t2 - 2/3)) / ((10 * t1 + 4) / 2 - (10 * t2 + 4) / 2)) = -4)) :=
sorry

end NUMINAMATH_GPT_slope_of_intersection_points_l1842_184266


namespace NUMINAMATH_GPT_man_age_difference_l1842_184263

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) : M - S = 24 :=
by
  sorry

end NUMINAMATH_GPT_man_age_difference_l1842_184263


namespace NUMINAMATH_GPT_expression_equality_l1842_184272

theorem expression_equality :
  (2^1001 + 5^1002)^2 - (2^1001 - 5^1002)^2 = 40 * 10^1001 := 
by
  sorry

end NUMINAMATH_GPT_expression_equality_l1842_184272


namespace NUMINAMATH_GPT_max_ratio_a_c_over_b_d_l1842_184286

-- Given conditions as Lean definitions
variables {a b c d : ℝ}
variable (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
variable (h2 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3 / 8)

-- The statement to prove the maximum value of the given expression
theorem max_ratio_a_c_over_b_d : ∃ t : ℝ, t = (a + c) / (b + d) ∧ t ≤ 3 :=
by {
  -- The proof of this theorem is omitted.
  sorry
}

end NUMINAMATH_GPT_max_ratio_a_c_over_b_d_l1842_184286


namespace NUMINAMATH_GPT_distinct_real_roots_form_geometric_progression_eq_170_l1842_184275

theorem distinct_real_roots_form_geometric_progression_eq_170 
  (a : ℝ) :
  (∃ (u : ℝ) (v : ℝ) (hu : u ≠ 0) (hv : v ≠ 0) (hv1 : |v| ≠ 1), 
  (16 * u^12 + (2 * a + 17) * u^6 * v^3 - a * u^9 * v - a * u^3 * v^9 + 16 = 0)) 
  → a = 170 :=
by sorry

end NUMINAMATH_GPT_distinct_real_roots_form_geometric_progression_eq_170_l1842_184275


namespace NUMINAMATH_GPT_count_yellow_balls_l1842_184230

theorem count_yellow_balls (total white green yellow red purple : ℕ) (prob : ℚ)
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_red : red = 9)
  (h_purple : purple = 3)
  (h_prob : prob = 0.88) :
  yellow = 8 :=
by
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_count_yellow_balls_l1842_184230


namespace NUMINAMATH_GPT_transform_equation_l1842_184238

theorem transform_equation (x : ℝ) (h₁ : x ≠ 3 / 2) (h₂ : 5 - 3 * x = 1) :
  x = 4 / 3 :=
sorry

end NUMINAMATH_GPT_transform_equation_l1842_184238


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_is_32_l1842_184262

-- Defining the geometric sequence conditions
variables (a r : ℝ)

def third_term := a * r^2 = 18
def fourth_term := a * r^3 = 24
def fifth_term := a * r^4

theorem geometric_sequence_fifth_term_is_32 (h1 : third_term a r) (h2 : fourth_term a r) : 
  fifth_term a r = 32 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_is_32_l1842_184262


namespace NUMINAMATH_GPT_book_cost_l1842_184252

theorem book_cost (x : ℝ) 
  (h1 : Vasya_has = x - 150)
  (h2 : Tolya_has = x - 200)
  (h3 : (x - 150) + (x - 200) / 2 = x + 100) : x = 700 :=
sorry

end NUMINAMATH_GPT_book_cost_l1842_184252


namespace NUMINAMATH_GPT_min_x_minus_y_l1842_184293

theorem min_x_minus_y {x y : ℝ} (hx : 0 ≤ x) (hx2 : x ≤ 2 * Real.pi) (hy : 0 ≤ y) (hy2 : y ≤ 2 * Real.pi)
    (h : 2 * Real.sin x * Real.cos y - Real.sin x + Real.cos y = 1 / 2) : 
    x - y = -Real.pi / 2 := 
sorry

end NUMINAMATH_GPT_min_x_minus_y_l1842_184293


namespace NUMINAMATH_GPT_annual_interest_rate_continuous_compounding_l1842_184299

noncomputable def continuous_compounding_rate (A P : ℝ) (t : ℝ) : ℝ :=
  (Real.log (A / P)) / t

theorem annual_interest_rate_continuous_compounding :
  continuous_compounding_rate 8500 5000 10 = (Real.log (1.7)) / 10 :=
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_continuous_compounding_l1842_184299


namespace NUMINAMATH_GPT_triangle_inequality_sum_l1842_184250

theorem triangle_inequality_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (c / (a + b)) + (a / (b + c)) + (b / (c + a)) > 1 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_sum_l1842_184250


namespace NUMINAMATH_GPT_range_of_a_l1842_184245

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x > x ^ 2 + a) → a < -8 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1842_184245


namespace NUMINAMATH_GPT_smallest_positive_difference_l1842_184253

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) : (∃ n : ℤ, n > 0 ∧ n = a - b) → n = 17 :=
by sorry

end NUMINAMATH_GPT_smallest_positive_difference_l1842_184253


namespace NUMINAMATH_GPT_courses_students_problem_l1842_184217

theorem courses_students_problem :
  let courses := Fin 6 -- represent 6 courses
  let students := Fin 20 -- represent 20 students
  (∀ (C C' : courses), ∀ (S : Finset students), S.card = 5 → 
    ¬ ((∀ s ∈ S, ∃ s_courses : Finset courses, C ∈ s_courses ∧ C' ∈ s_courses) ∨ 
       (∀ s ∈ S, ∃ s_courses : Finset courses, C ∉ s_courses ∧ C' ∉ s_courses))) :=
by sorry

end NUMINAMATH_GPT_courses_students_problem_l1842_184217


namespace NUMINAMATH_GPT_domain_of_f_l1842_184239

theorem domain_of_f (x : ℝ) : (1 - x > 0) ∧ (2 * x + 1 > 0) ↔ - (1 / 2 : ℝ) < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1842_184239


namespace NUMINAMATH_GPT_proof_part1_proof_part2_l1842_184256

-- Proof problem for the first part (1)
theorem proof_part1 (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := 
by
  sorry

-- Proof problem for the second part (2)
theorem proof_part2 (a : ℝ) : a * (a - 2) - 2 * a * (1 - 3 * a) = 7 * a^2 - 4 * a := 
by
  sorry

end NUMINAMATH_GPT_proof_part1_proof_part2_l1842_184256


namespace NUMINAMATH_GPT_range_of_a_l1842_184280

noncomputable def f (x a : ℝ) : ℝ := 
  (1 / 2) * (Real.cos x + Real.sin x) * (Real.cos x - Real.sin x - 4 * a) + (4 * a - 3) * x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
  0 ≤ (Real.cos (2 * x) - 2 * a * (Real.sin x - Real.cos x) + 4 * a - 3)) ↔ (a ≥ 1.5) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1842_184280


namespace NUMINAMATH_GPT_kaeli_problems_per_day_l1842_184254

-- Definitions based on conditions
def problems_solved_per_day_marie_pascale : ℕ := 4
def total_problems_marie_pascale : ℕ := 72
def total_problems_kaeli : ℕ := 126

-- Number of days both took should be the same
def number_of_days : ℕ := total_problems_marie_pascale / problems_solved_per_day_marie_pascale

-- Kaeli solves 54 more problems than Marie-Pascale
def extra_problems_kaeli : ℕ := 54

-- Definition that Kaeli's total problems solved is that of Marie-Pascale plus 54
axiom kaeli_total_problems (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : True

-- Now to find x, the problems solved per day by Kaeli
def x : ℕ := total_problems_kaeli / number_of_days

-- Prove that x = 7
theorem kaeli_problems_per_day (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : x = 7 := by
  sorry

end NUMINAMATH_GPT_kaeli_problems_per_day_l1842_184254


namespace NUMINAMATH_GPT_find_speed_range_l1842_184208

noncomputable def runningErrorB (v : ℝ) : ℝ := abs ((300 / v) - 7)
noncomputable def runningErrorC (v : ℝ) : ℝ := abs ((480 / v) - 11)

theorem find_speed_range (v : ℝ) :
  (runningErrorB v + runningErrorC v ≤ 2) →
  33.33 ≤ v ∧ v ≤ 48.75 := sorry

end NUMINAMATH_GPT_find_speed_range_l1842_184208


namespace NUMINAMATH_GPT_weight_of_a_l1842_184269

theorem weight_of_a (a b c d e : ℝ)
  (h1 : (a + b + c) / 3 = 84)
  (h2 : (a + b + c + d) / 4 = 80)
  (h3 : e = d + 8)
  (h4 : (b + c + d + e) / 4 = 79) :
  a = 80 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_a_l1842_184269


namespace NUMINAMATH_GPT_perfect_square_k_l1842_184234

theorem perfect_square_k (a b k : ℝ) (h : ∃ c : ℝ, a^2 + 2*(k-3)*a*b + 9*b^2 = (a + c*b)^2) : 
  k = 6 ∨ k = 0 := 
sorry

end NUMINAMATH_GPT_perfect_square_k_l1842_184234


namespace NUMINAMATH_GPT_binomial_expansion_terms_l1842_184264

theorem binomial_expansion_terms (x n : ℝ) (hn : n = 8) : 
  ∃ t, t = 3 :=
  sorry

end NUMINAMATH_GPT_binomial_expansion_terms_l1842_184264


namespace NUMINAMATH_GPT_total_daily_salary_l1842_184225

def manager_salary : ℕ := 5
def clerk_salary : ℕ := 2
def num_managers : ℕ := 2
def num_clerks : ℕ := 3

theorem total_daily_salary : num_managers * manager_salary + num_clerks * clerk_salary = 16 := by
    sorry

end NUMINAMATH_GPT_total_daily_salary_l1842_184225


namespace NUMINAMATH_GPT_find_a_l1842_184270

-- Define what it means for P(X = k) to be given by a particular function
def P (X : ℕ) (a : ℕ) := X / (2 * a)

-- Define the condition on the probabilities
def sum_of_probabilities_is_one (a : ℕ) :=
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) + 4 / (2 * a)) = 1

-- The theorem to prove
theorem find_a (a : ℕ) (h : sum_of_probabilities_is_one a) : a = 5 :=
by sorry

end NUMINAMATH_GPT_find_a_l1842_184270


namespace NUMINAMATH_GPT_mean_goals_correct_l1842_184242

-- Definitions based on problem conditions
def players_with_3_goals := 4
def players_with_4_goals := 3
def players_with_5_goals := 1
def players_with_6_goals := 2

-- The total number of goals scored
def total_goals := (3 * players_with_3_goals) + (4 * players_with_4_goals) + (5 * players_with_5_goals) + (6 * players_with_6_goals)

-- The total number of players
def total_players := players_with_3_goals + players_with_4_goals + players_with_5_goals + players_with_6_goals

-- The mean number of goals
def mean_goals := total_goals.toFloat / total_players.toFloat

theorem mean_goals_correct : mean_goals = 4.1 := by
  sorry

end NUMINAMATH_GPT_mean_goals_correct_l1842_184242


namespace NUMINAMATH_GPT_slope_of_tangent_at_point_l1842_184267

theorem slope_of_tangent_at_point (x : ℝ) (y : ℝ) (h_curve : y = x^3)
    (h_slope : 3*x^2 = 3) : (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
sorry

end NUMINAMATH_GPT_slope_of_tangent_at_point_l1842_184267


namespace NUMINAMATH_GPT_solve_for_a_l1842_184285

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = -2 → x^2 - a * x + 7 = 0) → a = -11 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_a_l1842_184285


namespace NUMINAMATH_GPT_tan_square_proof_l1842_184206

theorem tan_square_proof (θ : ℝ) (h : Real.tan θ = 2) : 
  1 / (Real.sin θ ^ 2 - Real.cos θ ^ 2) = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_square_proof_l1842_184206


namespace NUMINAMATH_GPT_minimum_distance_proof_l1842_184273

noncomputable def minimum_distance_AB : ℝ :=
  let f (x : ℝ) := x^2 - Real.log x
  let x_min := Real.sqrt 2 / 2
  let min_dist := (5 + Real.log 2) / 4
  min_dist

theorem minimum_distance_proof :
  ∃ a : ℝ, a = minimum_distance_AB :=
by
  use (5 + Real.log 2) / 4
  sorry

end NUMINAMATH_GPT_minimum_distance_proof_l1842_184273


namespace NUMINAMATH_GPT_red_pencils_count_l1842_184287

theorem red_pencils_count 
  (packs : ℕ) 
  (pencils_per_pack : ℕ) 
  (extra_packs : ℕ) 
  (extra_pencils_per_pack : ℕ)
  (total_red_pencils : ℕ) 
  (h1 : packs = 15)
  (h2 : pencils_per_pack = 1)
  (h3 : extra_packs = 3)
  (h4 : extra_pencils_per_pack = 2)
  (h5 : total_red_pencils = packs * pencils_per_pack + extra_packs * extra_pencils_per_pack) : 
  total_red_pencils = 21 := 
  by sorry

end NUMINAMATH_GPT_red_pencils_count_l1842_184287


namespace NUMINAMATH_GPT_part1_part2_l1842_184219

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- Problem (1)
theorem part1 (a : ℝ) (h : a = 1) : 
  ∀ x : ℝ, f x a ≥ 2 ↔ x ≤ 0 ∨ x ≥ 2 := 
  sorry

-- Problem (2)
theorem part2 (a : ℝ) (h : a > 1) : 
  (∀ x : ℝ, f x a + abs (x - 1) ≥ 2) ↔ a ≥ 3 := 
  sorry

end NUMINAMATH_GPT_part1_part2_l1842_184219


namespace NUMINAMATH_GPT_range_of_x_l1842_184248

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x :
  ∀ x : ℝ, (f x > f (2*x - 1)) ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1842_184248


namespace NUMINAMATH_GPT_machine_A_sprockets_per_hour_l1842_184240

theorem machine_A_sprockets_per_hour :
  ∀ (A T : ℝ),
    (T > 0 ∧
    (∀ P Q, P = 1.1 * A ∧ Q = 330 / P ∧ Q = 330 / A + 10) →
      A = 3) := 
by
  intro A T
  intro h
  sorry

end NUMINAMATH_GPT_machine_A_sprockets_per_hour_l1842_184240


namespace NUMINAMATH_GPT_line_through_points_l1842_184212

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 8)) (h2 : (x2, y2) = (5, 2)) :
  ∃ m b : ℝ, (∀ x, y = m * x + b → (x, y) = (2,8) ∨ (x, y) = (5, 2)) ∧ (m + b = 10) :=
by
  sorry

end NUMINAMATH_GPT_line_through_points_l1842_184212


namespace NUMINAMATH_GPT_number_of_students_more_than_pets_l1842_184218

theorem number_of_students_more_than_pets 
  (students_per_classroom pets_per_classroom num_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : pets_per_classroom = 3)
  (h3 : num_classrooms = 5) :
  (students_per_classroom * num_classrooms) - (pets_per_classroom * num_classrooms) = 85 := 
by
  sorry

end NUMINAMATH_GPT_number_of_students_more_than_pets_l1842_184218


namespace NUMINAMATH_GPT_probability_eq_l1842_184283

noncomputable def probability_exactly_two_one_digit_and_three_two_digit : ℚ := 
  let n := 5
  let p_one_digit := 9 / 20
  let p_two_digit := 11 / 20
  let binomial_coeff := Nat.choose 5 2
  (binomial_coeff * p_one_digit^2 * p_two_digit^3)

theorem probability_eq : probability_exactly_two_one_digit_and_three_two_digit = 539055 / 1600000 := 
  sorry

end NUMINAMATH_GPT_probability_eq_l1842_184283


namespace NUMINAMATH_GPT_solve_equations_l1842_184277

theorem solve_equations (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 4 * a + 3 * b = 39) : a + b = 82 / 7 := by
  sorry

end NUMINAMATH_GPT_solve_equations_l1842_184277


namespace NUMINAMATH_GPT_vanaspati_percentage_l1842_184274

theorem vanaspati_percentage (Q : ℝ) (h1 : 0.60 * Q > 0) (h2 : Q + 10 > 0) (h3 : Q = 10) :
    let total_ghee := Q + 10
    let pure_ghee := 0.60 * Q + 10
    let pure_ghee_fraction := pure_ghee / total_ghee
    pure_ghee_fraction = 0.80 → 
    let vanaspati_fraction := 1 - pure_ghee_fraction
    vanaspati_fraction * 100 = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_vanaspati_percentage_l1842_184274


namespace NUMINAMATH_GPT_find_circle_center_l1842_184265

-- Define the conditions as hypotheses
def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 40
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line_center_constraint (x y : ℝ) : Prop := 3 * x - 4 * y = 0

-- Define the function for the equidistant line
def line_eq (x y : ℝ) : Prop := 5 * x - 2 * y = 25

-- Prove that the center of the circle satisfying the given conditions is (50/7, 75/14)
theorem find_circle_center (x y : ℝ) 
(h1 : line_eq x y)
(h2 : line_center_constraint x y) : 
(x = 50 / 7 ∧ y = 75 / 14) :=
sorry

end NUMINAMATH_GPT_find_circle_center_l1842_184265


namespace NUMINAMATH_GPT_lcm_18_24_30_l1842_184243

theorem lcm_18_24_30 :
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  (∀ x > 0, x ∣ a ∧ x ∣ b ∧ x ∣ c → x ∣ lcm) ∧ (∀ y > 0, y ∣ lcm → y ∣ a ∧ y ∣ b ∧ y ∣ c) :=
by {
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  sorry
}

end NUMINAMATH_GPT_lcm_18_24_30_l1842_184243


namespace NUMINAMATH_GPT_correct_option_for_ruler_length_l1842_184228

theorem correct_option_for_ruler_length (A B C D : String) (correct_answer : String) : 
  A = "two times as longer as" ∧ 
  B = "twice the length of" ∧ 
  C = "three times longer of" ∧ 
  D = "twice long than" ∧ 
  correct_answer = B := 
by
  sorry

end NUMINAMATH_GPT_correct_option_for_ruler_length_l1842_184228


namespace NUMINAMATH_GPT_methane_required_l1842_184249

def mole_of_methane (moles_of_oxygen : ℕ) : ℕ := 
  if moles_of_oxygen = 2 then 1 else 0

theorem methane_required (moles_of_oxygen : ℕ) : 
  moles_of_oxygen = 2 → mole_of_methane moles_of_oxygen = 1 := 
by 
  intros h
  simp [mole_of_methane, h]

end NUMINAMATH_GPT_methane_required_l1842_184249


namespace NUMINAMATH_GPT_number_of_silverware_per_setting_l1842_184268

-- Conditions
def silverware_weight_per_piece := 4   -- in ounces
def plates_per_setting := 2
def plate_weight := 12  -- in ounces
def tables := 15
def settings_per_table := 8
def backup_settings := 20
def total_weight := 5040  -- in ounces

-- Let's define variables in our conditions
def settings := tables * settings_per_table + backup_settings
def plates_weight_per_setting := plates_per_setting * plate_weight
def total_silverware_weight (S : Nat) := S * silverware_weight_per_piece * settings
def total_plate_weight := plates_weight_per_setting * settings

-- Define the required proof statement
theorem number_of_silverware_per_setting : 
  ∃ S : Nat, (total_silverware_weight S + total_plate_weight = total_weight) ∧ S = 3 :=
by {
  sorry -- proof will be provided here
}

end NUMINAMATH_GPT_number_of_silverware_per_setting_l1842_184268


namespace NUMINAMATH_GPT_cars_no_air_conditioning_l1842_184276

variables {A R AR : Nat}

/-- Given a total of 100 cars, of which at least 51 have racing stripes,
and the greatest number of cars that could have air conditioning but not racing stripes is 49,
prove that the number of cars that do not have air conditioning is 49. -/
theorem cars_no_air_conditioning :
  ∀ (A R AR : ℕ), 
  (A = AR + 49) → 
  (R ≥ 51) → 
  (AR ≤ R) → 
  (AR ≤ 51) → 
  (100 - A = 49) :=
by
  intros A R AR h1 h2 h3 h4
  exact sorry

end NUMINAMATH_GPT_cars_no_air_conditioning_l1842_184276


namespace NUMINAMATH_GPT_johnson_vincent_work_together_l1842_184284

theorem johnson_vincent_work_together (work : Type) (time_johnson : ℕ) (time_vincent : ℕ) (combined_time : ℕ) :
  time_johnson = 10 → time_vincent = 40 → combined_time = 8 → 
  (1 / time_johnson + 1 / time_vincent) = 1 / combined_time :=
by
  intros h_johnson h_vincent h_combined
  sorry

end NUMINAMATH_GPT_johnson_vincent_work_together_l1842_184284


namespace NUMINAMATH_GPT_inequality_ab2_bc2_ca2_leq_27_div_8_l1842_184215

theorem inequality_ab2_bc2_ca2_leq_27_div_8 (a b c : ℝ) (h : a ≥ b) (h1 : b ≥ c) (h2 : c ≥ 0) (h3 : a + b + c = 3) :
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end NUMINAMATH_GPT_inequality_ab2_bc2_ca2_leq_27_div_8_l1842_184215


namespace NUMINAMATH_GPT_no_solution_eqn_l1842_184297

theorem no_solution_eqn (m : ℝ) : (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) ↔ m = 6 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_eqn_l1842_184297


namespace NUMINAMATH_GPT_sin_2x_eq_7_div_25_l1842_184288

theorem sin_2x_eq_7_div_25 (x : ℝ) (h : Real.sin (Real.pi / 4 - x) = 3 / 5) :
    Real.sin (2 * x) = 7 / 25 := by
  sorry

end NUMINAMATH_GPT_sin_2x_eq_7_div_25_l1842_184288


namespace NUMINAMATH_GPT_arithmetic_sequence_an_12_l1842_184220

theorem arithmetic_sequence_an_12 {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 9)
  (h_a6 : a 6 = 15) :
  a 12 = 27 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_an_12_l1842_184220


namespace NUMINAMATH_GPT_proof_of_a_b_and_T_l1842_184233

-- Define sequences and the given conditions

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 2 * n

def S (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := 1 / ((b n)^2 - 1)

def T (n : ℕ) : ℚ := (n : ℚ) / (2 * n + 1)

axiom b_condition : ∀ n : ℕ, n > 0 → (b n + 2 * n = 2 * (b (n-1)) + 4)

axiom S_condition : ∀ n : ℕ, S n = 2^n - 1

theorem proof_of_a_b_and_T (n : ℕ) (h : n > 0) : 
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = 2 * k) ∧ 
  (∀ k, T k = (k : ℚ) / (2 * k + 1)) := by
  sorry

end NUMINAMATH_GPT_proof_of_a_b_and_T_l1842_184233


namespace NUMINAMATH_GPT_expression_simplified_l1842_184247

theorem expression_simplified (d : ℤ) (h : d ≠ 0) :
  let a := 24
  let b := 61
  let c := 96
  a + b + c = 181 ∧ 
  (15 * d ^ 2 + 7 * d + 15 + (3 * d + 9) ^ 2 = a * d ^ 2 + b * d + c) := by
{
  sorry
}

end NUMINAMATH_GPT_expression_simplified_l1842_184247


namespace NUMINAMATH_GPT_angle_variance_less_than_bound_l1842_184281

noncomputable def angle_variance (α β γ : ℝ) : ℝ :=
  (1/3) * ((α - (2 * Real.pi / 3))^2 + (β - (2 * Real.pi / 3))^2 + (γ - (2 * Real.pi / 3))^2)

theorem angle_variance_less_than_bound (O A B C : ℝ → ℝ) :
  ∀ α β γ : ℝ, α + β + γ = 2 * Real.pi ∧ α ≥ β ∧ β ≥ γ → angle_variance α β γ < 2 * Real.pi^2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_angle_variance_less_than_bound_l1842_184281


namespace NUMINAMATH_GPT_johns_overall_loss_l1842_184232

noncomputable def johns_loss_percentage : ℝ :=
  let cost_A := 1000 * 2
  let cost_B := 1500 * 3
  let cost_C := 2000 * 4
  let discount_A := 0.1
  let discount_B := 0.15
  let discount_C := 0.2
  let cost_A_after_discount := cost_A * (1 - discount_A)
  let cost_B_after_discount := cost_B * (1 - discount_B)
  let cost_C_after_discount := cost_C * (1 - discount_C)
  let total_cost_after_discount := cost_A_after_discount + cost_B_after_discount + cost_C_after_discount
  let import_tax_rate := 0.08
  let import_tax := total_cost_after_discount * import_tax_rate
  let total_cost_incl_tax := total_cost_after_discount + import_tax
  let cost_increase_rate_C := 0.04
  let new_cost_C := 2000 * (4 + 4 * cost_increase_rate_C)
  let adjusted_total_cost := cost_A_after_discount + cost_B_after_discount + new_cost_C
  let total_selling_price := (800 * 3) + (70 * 3 + 1400 * 3.5 + 900 * 5) + (130 * 2.5 + 130 * 3 + 130 * 5)
  let gain_or_loss := total_selling_price - adjusted_total_cost
  let loss_percentage := (gain_or_loss / adjusted_total_cost) * 100
  loss_percentage

theorem johns_overall_loss : abs (johns_loss_percentage + 4.09) < 0.01 := sorry

end NUMINAMATH_GPT_johns_overall_loss_l1842_184232


namespace NUMINAMATH_GPT_days_B_to_complete_remaining_work_l1842_184271

/-- 
  Given that:
  - A can complete a work in 20 days.
  - B can complete the same work in 12 days.
  - A and B worked together for 3 days before A left.
  
  We need to prove that B will require 7.2 days to complete the remaining work alone. 
--/
theorem days_B_to_complete_remaining_work : 
  (∃ (A_rate B_rate combined_rate work_done_in_3_days remaining_work d_B : ℚ), 
   A_rate = (1 / 20) ∧
   B_rate = (1 / 12) ∧
   combined_rate = A_rate + B_rate ∧
   work_done_in_3_days = 3 * combined_rate ∧
   remaining_work = 1 - work_done_in_3_days ∧
   d_B = remaining_work / B_rate ∧
   d_B = 7.2) := 
by 
  sorry

end NUMINAMATH_GPT_days_B_to_complete_remaining_work_l1842_184271


namespace NUMINAMATH_GPT_wrongly_read_number_l1842_184291

theorem wrongly_read_number 
  (S_initial : ℕ) (S_correct : ℕ) (correct_num : ℕ) (num_count : ℕ) 
  (h_initial : S_initial = num_count * 18) 
  (h_correct : S_correct = num_count * 19) 
  (h_correct_num : correct_num = 36) 
  (h_diff : S_correct - S_initial = correct_num - wrong_num) 
  (h_num_count : num_count = 10) 
  : wrong_num = 26 :=
sorry

end NUMINAMATH_GPT_wrongly_read_number_l1842_184291


namespace NUMINAMATH_GPT_min_value_of_x2_plus_y2_l1842_184289

-- Define the problem statement
theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : 3 * x + y = 10) : x^2 + y^2 ≥ 10 :=
sorry

end NUMINAMATH_GPT_min_value_of_x2_plus_y2_l1842_184289


namespace NUMINAMATH_GPT_a_n_formula_l1842_184255

variable {a : ℕ+ → ℝ}  -- Defining a_n as a sequence from positive natural numbers to real numbers
variable {S : ℕ+ → ℝ}  -- Defining S_n as a sequence from positive natural numbers to real numbers

-- Given conditions
axiom S_def (n : ℕ+) : S n = a n / 2 + 1 / a n - 1
axiom a_pos (n : ℕ+) : a n > 0

-- Conjecture to be proved
theorem a_n_formula (n : ℕ+) : a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) := 
sorry -- proof to be done

end NUMINAMATH_GPT_a_n_formula_l1842_184255


namespace NUMINAMATH_GPT_option_c_correct_l1842_184226

theorem option_c_correct (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end NUMINAMATH_GPT_option_c_correct_l1842_184226


namespace NUMINAMATH_GPT_area_proportions_and_point_on_line_l1842_184236

theorem area_proportions_and_point_on_line (T : ℝ × ℝ) :
  (∃ r s : ℝ, T = (r, s) ∧ s = -(5 / 3) * r + 10 ∧ 1 / 2 * 6 * s = 7.5) 
  ↔ T.1 + T.2 = 7 :=
by { sorry }

end NUMINAMATH_GPT_area_proportions_and_point_on_line_l1842_184236


namespace NUMINAMATH_GPT_find_parameters_l1842_184292

noncomputable def cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + 27

def deriv_cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + b

theorem find_parameters
  (a b : ℝ)
  (h1 : deriv_cubic_function a b (-1) = 0)
  (h2 : deriv_cubic_function a b 3 = 0) :
  a = -3 ∧ b = -9 :=
by
  -- leaving proof as sorry since the task doesn't require proving
  sorry

end NUMINAMATH_GPT_find_parameters_l1842_184292


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1842_184278

theorem problem1 (x : ℤ) (h : 263 - x = 108) : x = 155 :=
by sorry

theorem problem2 (x : ℤ) (h : 25 * x = 1950) : x = 78 :=
by sorry

theorem problem3 (x : ℤ) (h : x / 15 = 64) : x = 960 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1842_184278


namespace NUMINAMATH_GPT_exists_three_numbers_sum_to_zero_l1842_184235

theorem exists_three_numbers_sum_to_zero (s : Finset ℤ) (h_card : s.card = 101) (h_abs : ∀ x ∈ s, |x| ≤ 99) :
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_three_numbers_sum_to_zero_l1842_184235


namespace NUMINAMATH_GPT_graham_crackers_leftover_l1842_184279

-- Definitions for the problem conditions
def initial_boxes_graham := 14
def initial_packets_oreos := 15
def initial_ounces_cream_cheese := 36

def boxes_per_cheesecake := 2
def packets_per_cheesecake := 3
def ounces_per_cheesecake := 4

-- Define the statement that needs to be proved
theorem graham_crackers_leftover :
  initial_boxes_graham - (min (initial_boxes_graham / boxes_per_cheesecake) (min (initial_packets_oreos / packets_per_cheesecake) (initial_ounces_cream_cheese / ounces_per_cheesecake)) * boxes_per_cheesecake) = 4 :=
by sorry

end NUMINAMATH_GPT_graham_crackers_leftover_l1842_184279


namespace NUMINAMATH_GPT_climbing_stairs_l1842_184282

noncomputable def total_methods_to_climb_stairs : ℕ :=
  (Nat.choose 8 5) + (Nat.choose 8 6) + (Nat.choose 8 7) + 1

theorem climbing_stairs (n : ℕ := 9) (min_steps : ℕ := 6) (max_steps : ℕ := 9)
  (H1 : min_steps ≤ n)
  (H2 : n ≤ max_steps)
  : total_methods_to_climb_stairs = 93 := by
  sorry

end NUMINAMATH_GPT_climbing_stairs_l1842_184282


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l1842_184227

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  ∑' n : ℕ, a * r ^ n = 20 / 21 := by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l1842_184227


namespace NUMINAMATH_GPT_leap_day_2040_is_tuesday_l1842_184207

-- Define the given condition that 29th February 2012 is Wednesday
def feb_29_2012_is_wednesday : Prop := sorry

-- Define the calculation of the day of the week for February 29, 2040
def day_of_feb_29_2040 (initial_day : Nat) : Nat := (10228 % 7 + initial_day) % 7

-- Define the proof statement
theorem leap_day_2040_is_tuesday : feb_29_2012_is_wednesday →
  (day_of_feb_29_2040 3 = 2) := -- Here, 3 represents Wednesday and 2 represents Tuesday
sorry

end NUMINAMATH_GPT_leap_day_2040_is_tuesday_l1842_184207


namespace NUMINAMATH_GPT_domain_of_g_l1842_184222

def f : ℝ → ℝ := sorry  -- Placeholder for the function f

noncomputable def g (x : ℝ) : ℝ := f (x - 1) / Real.sqrt (2 * x + 1)

theorem domain_of_g :
  ∀ x : ℝ, g x ≠ 0 → (-1/2 < x ∧ x ≤ 3) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_domain_of_g_l1842_184222


namespace NUMINAMATH_GPT_registration_methods_l1842_184221

-- Define the number of students and groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem stating the total number of different registration methods
theorem registration_methods : (num_groups ^ num_students) = 81 := 
by sorry

end NUMINAMATH_GPT_registration_methods_l1842_184221


namespace NUMINAMATH_GPT_number_of_social_science_papers_selected_is_18_l1842_184205

def total_social_science_papers : ℕ := 54
def total_humanities_papers : ℕ := 60
def total_other_papers : ℕ := 39
def total_selected_papers : ℕ := 51

def number_of_social_science_papers_selected : ℕ :=
  (total_social_science_papers * total_selected_papers) / (total_social_science_papers + total_humanities_papers + total_other_papers)

theorem number_of_social_science_papers_selected_is_18 :
  number_of_social_science_papers_selected = 18 :=
by 
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_number_of_social_science_papers_selected_is_18_l1842_184205


namespace NUMINAMATH_GPT_parallel_lines_regular_ngon_l1842_184216

def closed_n_hop_path (n : ℕ) (a : Fin (n + 1) → Fin n) : Prop :=
∀ i j : Fin n, a (i + 1) + a i = a (j + 1) + a j → i = j

theorem parallel_lines_regular_ngon (n : ℕ) (a : Fin (n + 1) → Fin n):
  (Even n → ∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j) ∧
  (Odd n → ¬(∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j ∧ ∀ k l : Fin n, k ≠ l → a (k + 1) + k ≠ a (l + 1) + l)) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_regular_ngon_l1842_184216


namespace NUMINAMATH_GPT_initial_dragon_fruits_remaining_kiwis_l1842_184237

variable (h d k : ℕ)    -- h: initial number of cantaloupes, d: initial number of dragon fruits, k: initial number of kiwis
variable (d_rem : ℕ)    -- d_rem: remaining number of dragon fruits after all cantaloupes are used up
variable (k_rem : ℕ)    -- k_rem: remaining number of kiwis after all cantaloupes are used up

axiom condition1 : d = 3 * h + 10
axiom condition2 : k = 2 * d
axiom condition3 : d_rem = 130
axiom condition4 : (d - d_rem) = 2 * h
axiom condition5 : k_rem = k - 10 * h

theorem initial_dragon_fruits (h : ℕ) (d : ℕ) (k : ℕ) (d_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  d_rem = 130 →
  2 * h + d_rem = d → 
  h = 120 → 
  d = 370 :=
by 
  intros
  sorry

theorem remaining_kiwis (h : ℕ) (d : ℕ) (k : ℕ) (k_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  h = 120 →
  k_rem = k - 10 * h → 
  k_rem = 140 :=
by 
  intros
  sorry

end NUMINAMATH_GPT_initial_dragon_fruits_remaining_kiwis_l1842_184237


namespace NUMINAMATH_GPT_smallest_part_when_divided_l1842_184241

theorem smallest_part_when_divided (total : ℝ) (a b c : ℝ) (h_total : total = 150)
                                   (h_a : a = 3) (h_b : b = 5) (h_c : c = 7/2) :
                                   min (min (3 * (total / (a + b + c))) (5 * (total / (a + b + c)))) ((7/2) * (total / (a + b + c))) = 3 * (total / (a + b + c)) :=
by
  -- Mathematical steps have been omitted
  sorry

end NUMINAMATH_GPT_smallest_part_when_divided_l1842_184241


namespace NUMINAMATH_GPT_sum_even_integers_12_to_46_l1842_184209

theorem sum_even_integers_12_to_46 : 
  let a1 := 12
  let d := 2
  let an := 46
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 522 := 
by
  let a1 := 12 
  let d := 2 
  let an := 46
  let n := (an - a1) / d + 1 
  let Sn := n * (a1 + an) / 2
  sorry

end NUMINAMATH_GPT_sum_even_integers_12_to_46_l1842_184209


namespace NUMINAMATH_GPT_value_of_mathematics_l1842_184214

def letter_value (n : ℕ) : ℤ :=
  -- The function to assign values based on position modulo 8
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 0 => 0
  | _ => 0 -- This case is practically unreachable

def letter_position (c : Char) : ℕ :=
  -- The function to find the position of a character in the alphabet
  c.toNat - 'a'.toNat + 1

def value_of_word (word : String) : ℤ :=
  -- The function to calculate the sum of values of letters in the word
  word.foldr (fun c acc => acc + letter_value (letter_position c)) 0

theorem value_of_mathematics : value_of_word "mathematics" = 6 := 
  by
    sorry -- Proof to be completed

end NUMINAMATH_GPT_value_of_mathematics_l1842_184214


namespace NUMINAMATH_GPT_sheepdog_speed_l1842_184202

theorem sheepdog_speed 
  (T : ℝ) (t : ℝ) (sheep_speed : ℝ) (initial_distance : ℝ)
  (total_distance_speed : ℝ) :
  T = 20  →
  t = 20 →
  sheep_speed = 12 →
  initial_distance = 160 →
  total_distance_speed = 20 →
  total_distance_speed * T = initial_distance + sheep_speed * t := 
by sorry

end NUMINAMATH_GPT_sheepdog_speed_l1842_184202
