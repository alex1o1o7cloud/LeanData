import Mathlib

namespace NUMINAMATH_GPT_maura_seashells_l842_84236

theorem maura_seashells (original_seashells given_seashells remaining_seashells : ℕ)
  (h1 : original_seashells = 75) 
  (h2 : remaining_seashells = 57) 
  (h3 : given_seashells = original_seashells - remaining_seashells) :
  given_seashells = 18 := by
  -- Lean will use 'sorry' as a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_maura_seashells_l842_84236


namespace NUMINAMATH_GPT_three_equal_of_four_l842_84254

theorem three_equal_of_four (a b c d : ℕ) 
  (h1 : (a + b)^2 ∣ c * d) 
  (h2 : (a + c)^2 ∣ b * d) 
  (h3 : (a + d)^2 ∣ b * c) 
  (h4 : (b + c)^2 ∣ a * d) 
  (h5 : (b + d)^2 ∣ a * c) 
  (h6 : (c + d)^2 ∣ a * b) : 
  (a = b ∧ b = c) ∨ (a = b ∧ b = d) ∨ (a = c ∧ c = d) ∨ (b = c ∧ c = d) := 
sorry

end NUMINAMATH_GPT_three_equal_of_four_l842_84254


namespace NUMINAMATH_GPT_athlete_more_stable_l842_84263

theorem athlete_more_stable (var_A var_B : ℝ) 
                                (h1 : var_A = 0.024) 
                                (h2 : var_B = 0.008) 
                                (h3 : var_A > var_B) : 
  var_B < var_A :=
by
  exact h3

end NUMINAMATH_GPT_athlete_more_stable_l842_84263


namespace NUMINAMATH_GPT_students_apply_colleges_l842_84298

    -- Define that there are 5 students
    def students : Nat := 5

    -- Each student has 3 choices of colleges
    def choices_per_student : Nat := 3

    -- The number of different ways the students can apply
    def number_of_ways : Nat := choices_per_student ^ students

    theorem students_apply_colleges : number_of_ways = 3 ^ 5 :=
    by
        -- Proof will be done here
        sorry
    
end NUMINAMATH_GPT_students_apply_colleges_l842_84298


namespace NUMINAMATH_GPT_suggestions_difference_l842_84212

def mashed_potatoes_suggestions : ℕ := 408
def pasta_suggestions : ℕ := 305
def bacon_suggestions : ℕ := 137
def grilled_vegetables_suggestions : ℕ := 213
def sushi_suggestions : ℕ := 137

theorem suggestions_difference :
  let highest := mashed_potatoes_suggestions
  let lowest := bacon_suggestions
  highest - lowest = 271 :=
by
  sorry

end NUMINAMATH_GPT_suggestions_difference_l842_84212


namespace NUMINAMATH_GPT_find_inverse_modulo_l842_84250

theorem find_inverse_modulo :
  113 * 113 ≡ 1 [MOD 114] :=
by
  sorry

end NUMINAMATH_GPT_find_inverse_modulo_l842_84250


namespace NUMINAMATH_GPT_frac_difference_l842_84255

theorem frac_difference (m n : ℝ) (h : m^2 - n^2 = m * n) : (n / m) - (m / n) = -1 :=
sorry

end NUMINAMATH_GPT_frac_difference_l842_84255


namespace NUMINAMATH_GPT_parabola_intersects_x_axis_l842_84289

theorem parabola_intersects_x_axis :
  ∀ m : ℝ, (m^2 - m - 1 = 0) → (-2 * m^2 + 2 * m + 2023 = 2021) :=
by 
intros m hm
/-
  Given condition: m^2 - m - 1 = 0
  We need to show: -2 * m^2 + 2 * m + 2023 = 2021
-/
sorry

end NUMINAMATH_GPT_parabola_intersects_x_axis_l842_84289


namespace NUMINAMATH_GPT_commutative_l842_84271

variable (R : Type) [NonAssocRing R]
variable (star : R → R → R)

axiom assoc : ∀ x y z : R, star (star x y) z = star x (star y z)
axiom comm_left : ∀ x y z : R, star (star x y) z = star (star y z) x
axiom distinct : ∀ {x y : R}, x ≠ y → ∃ z : R, star z x ≠ star z y

theorem commutative (x y : R) : star x y = star y x := sorry

end NUMINAMATH_GPT_commutative_l842_84271


namespace NUMINAMATH_GPT_line_circle_intersection_l842_84209

theorem line_circle_intersection (x y : ℝ) (h1 : 7 * x + 5 * y = 14) (h2 : x^2 + y^2 = 4) :
  ∃ (p q : ℝ), (7 * p + 5 * q = 14) ∧ (p^2 + q^2 = 4) ∧ (7 * p + 5 * q = 14) ∧ (p ≠ q) :=
sorry

end NUMINAMATH_GPT_line_circle_intersection_l842_84209


namespace NUMINAMATH_GPT_a7_plus_a11_l842_84294

variable {a : ℕ → ℤ} (d : ℤ) (a₁ : ℤ)

-- Definitions based on given conditions
def S_n (n : ℕ) := (n * (2 * a₁ + (n - 1) * d)) / 2
def a_n (n : ℕ) := a₁ + (n - 1) * d

-- Condition: S_17 = 51
axiom h : S_n 17 = 51

-- Theorem to prove the question is equivalent to the answer
theorem a7_plus_a11 (h : S_n 17 = 51) : a_n 7 + a_n 11 = 6 :=
by
  -- This is where you'd fill in the actual proof, but we'll use sorry for now
  sorry

end NUMINAMATH_GPT_a7_plus_a11_l842_84294


namespace NUMINAMATH_GPT_correct_calculation_l842_84257

theorem correct_calculation (x : ℕ) (h : 954 - x = 468) : 954 + x = 1440 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l842_84257


namespace NUMINAMATH_GPT_least_area_of_triangles_l842_84269

-- Define the points A, B, C, D of the unit square
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (0, 1)

-- Define the function s(M, N) as the least area of the triangles having their vertices in the set {A, B, C, D, M, N}
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

noncomputable def s (M N : ℝ × ℝ) : ℝ :=
  min (min (min (min (min (triangle_area A B M) (triangle_area A B N)) (triangle_area A C M)) (triangle_area A C N)) (min (triangle_area A D M) (triangle_area A D N)))
    (min (min (min (triangle_area B C M) (triangle_area B C N)) (triangle_area B D M)) (min (triangle_area B D N) (min (triangle_area C D M) (triangle_area C D N))))

-- Define the statement to prove
theorem least_area_of_triangles (M N : ℝ × ℝ)
  (hM : M.1 > 0 ∧ M.1 < 1 ∧ M.2 > 0 ∧ M.2 < 1)
  (hN : N.1 > 0 ∧ N.1 < 1 ∧ N.2 > 0 ∧ N.2 < 1)
  (hMN : (M ≠ A ∨ N ≠ A) ∧ (M ≠ B ∨ N ≠ B) ∧ (M ≠ C ∨ N ≠ C) ∧ (M ≠ D ∨ N ≠ D))
  : s M N ≤ 1 / 8 := 
sorry

end NUMINAMATH_GPT_least_area_of_triangles_l842_84269


namespace NUMINAMATH_GPT_hyperbola_equation_l842_84260

-- Fixed points F_1 and F_2
def F1 : ℝ × ℝ := (5, 0)
def F2 : ℝ × ℝ := (-5, 0)

-- Condition: The absolute value of the difference in distances from P to F1 and F2 is 6
def distance_condition (P : ℝ × ℝ) : Prop :=
  abs ((dist P F1) - (dist P F2)) = 6

theorem hyperbola_equation : 
  ∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ ∀ (x y : ℝ), distance_condition (x, y) → 
  (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1 :=
by
  -- We state the conditions and result derived from them
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l842_84260


namespace NUMINAMATH_GPT_solve_for_x_l842_84299

theorem solve_for_x : ∀ (x : ℤ), (5 * x - 2) * 4 = (3 * (6 * x - 6)) → x = -5 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l842_84299


namespace NUMINAMATH_GPT_lines_intersect_at_l842_84267

noncomputable def line1 (x : ℚ) : ℚ := (-2 / 3) * x + 2
noncomputable def line2 (x : ℚ) : ℚ := -2 * x + (3 / 2)

theorem lines_intersect_at :
  ∃ (x y : ℚ), line1 x = y ∧ line2 x = y ∧ x = (3 / 8) ∧ y = (7 / 4) :=
sorry

end NUMINAMATH_GPT_lines_intersect_at_l842_84267


namespace NUMINAMATH_GPT_simplify_expression_l842_84220

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l842_84220


namespace NUMINAMATH_GPT_max_rectangle_area_l842_84238

theorem max_rectangle_area (l w : ℕ) (h : 2 * (l + w) = 120) : l * w ≤ 900 :=
by 
  sorry

end NUMINAMATH_GPT_max_rectangle_area_l842_84238


namespace NUMINAMATH_GPT_binomial_expansion_coefficient_x_l842_84259

theorem binomial_expansion_coefficient_x :
  (∃ (c : ℕ), (x : ℝ) → (x + 1/x^(1/2))^7 = c * x + (rest)) ∧ c = 35 := by
  sorry

end NUMINAMATH_GPT_binomial_expansion_coefficient_x_l842_84259


namespace NUMINAMATH_GPT_B_completes_remaining_work_in_12_days_l842_84278

-- Definitions for conditions.
def work_rate_a := 1/15
def work_rate_b := 1/18
def days_worked_by_a := 5

-- Calculation of work done by A and the remaining work for B
def work_done_by_a := days_worked_by_a * work_rate_a
def remaining_work := 1 - work_done_by_a

-- Proof statement
theorem B_completes_remaining_work_in_12_days : 
  ∀ (work_rate_a work_rate_b : ℚ), 
    work_rate_a = 1/15 → 
    work_rate_b = 1/18 → 
    days_worked_by_a = 5 → 
    work_done_by_a = days_worked_by_a * work_rate_a → 
    remaining_work = 1 - work_done_by_a → 
    (remaining_work / work_rate_b) = 12 :=
by 
  intros 
  sorry

end NUMINAMATH_GPT_B_completes_remaining_work_in_12_days_l842_84278


namespace NUMINAMATH_GPT_slope_of_asymptotes_l842_84245

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (x - 2)^2 / 144 - (y + 3)^2 / 81 = 1

-- The theorem stating the slope of the asymptotes
theorem slope_of_asymptotes : ∀ x y : ℝ, hyperbola x y → (∃ m : ℝ, m = 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_slope_of_asymptotes_l842_84245


namespace NUMINAMATH_GPT_minimum_red_chips_l842_84239

variable (w b r : ℕ)

-- Define the conditions
def condition1 : Prop := b ≥ 3 * w / 4
def condition2 : Prop := b ≤ r / 4
def condition3 : Prop := 60 ≤ w + b ∧ w + b ≤ 80

-- Prove the minimum number of red chips r is 108
theorem minimum_red_chips (H1 : condition1 w b) (H2 : condition2 b r) (H3 : condition3 w b) : r ≥ 108 := 
sorry

end NUMINAMATH_GPT_minimum_red_chips_l842_84239


namespace NUMINAMATH_GPT_cone_volume_l842_84272

theorem cone_volume (S : ℝ) (h_S : S = 12 * Real.pi) (h_lateral : ∃ r : ℝ, S = 3 * Real.pi * r^2) :
    ∃ V : ℝ, V = (8 * Real.sqrt 3 * Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_l842_84272


namespace NUMINAMATH_GPT_combined_tax_rate_l842_84202

theorem combined_tax_rate
  (Mork_income : ℝ)
  (Mindy_income : ℝ)
  (h1 : Mindy_income = 3 * Mork_income)
  (Mork_tax_rate : ℝ := 0.30)
  (Mindy_tax_rate : ℝ := 0.20) :
  (Mork_tax_rate * Mork_income + Mindy_tax_rate * Mindy_income) / (Mork_income + Mindy_income) * 100 = 22.5 :=
by
  sorry

end NUMINAMATH_GPT_combined_tax_rate_l842_84202


namespace NUMINAMATH_GPT_division_value_l842_84261

theorem division_value (a b c : ℝ) 
  (h1 : a / b = 5 / 3) 
  (h2 : b / c = 7 / 2) : 
  c / a = 6 / 35 := 
by
  sorry

end NUMINAMATH_GPT_division_value_l842_84261


namespace NUMINAMATH_GPT_find_first_number_l842_84279

-- Definitions from conditions
variable (x : ℕ) -- Let the first number be x
variable (y : ℕ) -- Let the second number be y

-- Given conditions in the problem
def condition1 : Prop := y = 43
def condition2 : Prop := x + 2 * y = 124

-- The proof target
theorem find_first_number (h1 : condition1 y) (h2 : condition2 x y) : x = 38 := by
  sorry

end NUMINAMATH_GPT_find_first_number_l842_84279


namespace NUMINAMATH_GPT_simplest_radical_l842_84234

theorem simplest_radical (r1 r2 r3 r4 : ℝ) 
  (h1 : r1 = Real.sqrt 3) 
  (h2 : r2 = Real.sqrt 4)
  (h3 : r3 = Real.sqrt 8)
  (h4 : r4 = Real.sqrt (1 / 2)) : r1 = Real.sqrt 3 :=
  by sorry

end NUMINAMATH_GPT_simplest_radical_l842_84234


namespace NUMINAMATH_GPT_no_factors_of_p_l842_84219

open Polynomial

noncomputable def p : Polynomial ℝ := X^4 - 4 * X^2 + 16
noncomputable def optionA : Polynomial ℝ := X^2 + 4
noncomputable def optionB : Polynomial ℝ := X + 2
noncomputable def optionC : Polynomial ℝ := X^2 - 4*X + 4
noncomputable def optionD : Polynomial ℝ := X^2 - 4

theorem no_factors_of_p (h : Polynomial ℝ) : h ≠ p / optionA ∧ h ≠ p / optionB ∧ h ≠ p / optionC ∧ h ≠ p / optionD := by
  sorry

end NUMINAMATH_GPT_no_factors_of_p_l842_84219


namespace NUMINAMATH_GPT_digit_divisible_by_3_l842_84225

theorem digit_divisible_by_3 (d : ℕ) (h : d < 10) : (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end NUMINAMATH_GPT_digit_divisible_by_3_l842_84225


namespace NUMINAMATH_GPT_regular_pay_per_hour_l842_84210

theorem regular_pay_per_hour (R : ℝ) (h : 40 * R + 11 * (2 * R) = 186) : R = 3 :=
by
  sorry

end NUMINAMATH_GPT_regular_pay_per_hour_l842_84210


namespace NUMINAMATH_GPT_negation_of_proposition_l842_84231

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l842_84231


namespace NUMINAMATH_GPT_depth_B_is_correct_l842_84247

-- Given: Diver A is at a depth of -55 meters.
def depth_A : ℤ := -55

-- Given: Diver B is 5 meters above diver A.
def offset : ℤ := 5

-- Prove: The depth of diver B
theorem depth_B_is_correct : (depth_A + offset) = -50 :=
by
  sorry

end NUMINAMATH_GPT_depth_B_is_correct_l842_84247


namespace NUMINAMATH_GPT_central_angle_of_sector_l842_84297

theorem central_angle_of_sector (r S : ℝ) (h_r : r = 2) (h_S : S = 4) : 
  ∃ α : ℝ, α = 2 ∧ S = (1/2) * α * r^2 := 
by 
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l842_84297


namespace NUMINAMATH_GPT_contrapositive_example_l842_84221

theorem contrapositive_example (x : ℝ) : (x = 1 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_example_l842_84221


namespace NUMINAMATH_GPT_quadratic_union_nonempty_l842_84288

theorem quadratic_union_nonempty (a : ℝ) :
  (∃ x : ℝ, x^2 - (a-2)*x - 2*a + 4 = 0) ∨ (∃ y : ℝ, y^2 + (2*a-3)*y + 2*a^2 - a - 3 = 0) ↔
    a ≤ -6 ∨ (-7/2) ≤ a ∧ a ≤ (3/2) ∨ a ≥ 2 :=
sorry

end NUMINAMATH_GPT_quadratic_union_nonempty_l842_84288


namespace NUMINAMATH_GPT_no_natural_m_n_exists_l842_84296

theorem no_natural_m_n_exists (m n : ℕ) : 
  (0.07 = (1 : ℝ) / m + (1 : ℝ) / n) → False :=
by
  -- Normally, the proof would go here, but it's not required by the prompt
  sorry

end NUMINAMATH_GPT_no_natural_m_n_exists_l842_84296


namespace NUMINAMATH_GPT_contractor_fine_amount_l842_84244

def total_days := 30
def daily_earning := 25
def total_earnings := 360
def days_absent := 12
def days_worked := total_days - days_absent
def fine_per_absent_day (x : ℝ) : Prop :=
  (daily_earning * days_worked) - (x * days_absent) = total_earnings

theorem contractor_fine_amount : ∃ x : ℝ, fine_per_absent_day x := by
  use 7.5
  sorry

end NUMINAMATH_GPT_contractor_fine_amount_l842_84244


namespace NUMINAMATH_GPT_diameter_of_lake_l842_84237

-- Given conditions: the radius of the circular lake
def radius : ℝ := 7

-- The proof problem: proving the diameter of the lake is 14 meters
theorem diameter_of_lake : 2 * radius = 14 :=
by
  sorry

end NUMINAMATH_GPT_diameter_of_lake_l842_84237


namespace NUMINAMATH_GPT_employee_salary_l842_84290

theorem employee_salary (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 528) : Y = 240 :=
by
  sorry

end NUMINAMATH_GPT_employee_salary_l842_84290


namespace NUMINAMATH_GPT_crows_eat_worms_l842_84223

theorem crows_eat_worms (worms_eaten_by_3_crows_in_1_hour : ℕ) 
                        (crows_eating_worms_constant : worms_eaten_by_3_crows_in_1_hour = 30)
                        (number_of_crows : ℕ) 
                        (observation_time_hours : ℕ) :
                        number_of_crows = 5 ∧ observation_time_hours = 2 →
                        (number_of_crows * worms_eaten_by_3_crows_in_1_hour / 3) * observation_time_hours = 100 :=
by
  sorry

end NUMINAMATH_GPT_crows_eat_worms_l842_84223


namespace NUMINAMATH_GPT_Iris_shorts_l842_84262

theorem Iris_shorts :
  ∃ s, (3 * 10) + s * 6 + (4 * 12) = 90 ∧ s = 2 := 
by
  existsi 2
  sorry

end NUMINAMATH_GPT_Iris_shorts_l842_84262


namespace NUMINAMATH_GPT_system_of_inequalities_l842_84292

theorem system_of_inequalities (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : (0.5 < p ∧ p < 5 / 9) :=
by sorry

end NUMINAMATH_GPT_system_of_inequalities_l842_84292


namespace NUMINAMATH_GPT_four_digit_div_by_14_l842_84229

theorem four_digit_div_by_14 (n : ℕ) (h₁ : 9450 + n < 10000) :
  (∃ k : ℕ, 9450 + n = 14 * k) ↔ (n = 8) := by
  sorry

end NUMINAMATH_GPT_four_digit_div_by_14_l842_84229


namespace NUMINAMATH_GPT_interest_rate_A_l842_84246

-- Definitions for the conditions
def principal : ℝ := 1000
def rate_C : ℝ := 0.115
def time_period : ℝ := 3
def gain_B : ℝ := 45

-- Main theorem to prove
theorem interest_rate_A {R : ℝ} (h1 : gain_B = (principal * rate_C * time_period - principal * (R / 100) * time_period)) : R = 10 := 
by
  sorry

end NUMINAMATH_GPT_interest_rate_A_l842_84246


namespace NUMINAMATH_GPT_calculate_visits_to_water_fountain_l842_84285

-- Define the distance from the desk to the fountain
def distance_desk_to_fountain : ℕ := 30

-- Define the total distance Mrs. Hilt walked
def total_distance_walked : ℕ := 120

-- Define the distance of a round trip (desk to fountain and back)
def round_trip_distance : ℕ := 2 * distance_desk_to_fountain

-- Define the number of round trips and hence the number of times to water fountain
def number_of_visits : ℕ := total_distance_walked / round_trip_distance

theorem calculate_visits_to_water_fountain:
    number_of_visits = 2 := 
by
    sorry

end NUMINAMATH_GPT_calculate_visits_to_water_fountain_l842_84285


namespace NUMINAMATH_GPT_part1_part2_l842_84286

theorem part1 (p : ℝ) (h : p = 2 / 5) : 
  (p^2 + 2 * (3 / 5) * p^2) = 0.352 :=
by 
  rw [h]
  sorry

theorem part2 (p : ℝ) (h : p = 2 / 5) : 
  (4 * (1 / (11.32 * p^4)) + 5 * (2.4 / (11.32 * p^4)) + 6 * (3.6 / (11.32 * p^4)) + 7 * (2.16 / (11.32 * p^4))) = 4.834 :=
by 
  rw [h]
  sorry

end NUMINAMATH_GPT_part1_part2_l842_84286


namespace NUMINAMATH_GPT_num_candidates_appeared_each_state_l842_84216

-- Definitions
def candidates_appear : ℕ := 8000
def sel_pct_A : ℚ := 0.06
def sel_pct_B : ℚ := 0.07
def additional_selections_B : ℕ := 80

-- Proof Problem Statement
theorem num_candidates_appeared_each_state (x : ℕ) 
  (h1 : x = candidates_appear) 
  (h2 : sel_pct_A * ↑x = 0.06 * ↑x) 
  (h3 : sel_pct_B * ↑x = 0.07 * ↑x) 
  (h4 : sel_pct_B * ↑x = sel_pct_A * ↑x + additional_selections_B) : 
  x = candidates_appear := sorry

end NUMINAMATH_GPT_num_candidates_appeared_each_state_l842_84216


namespace NUMINAMATH_GPT_Ricciana_run_distance_l842_84206

def Ricciana_jump : ℕ := 4

def Margarita_run : ℕ := 18

def Margarita_jump (Ricciana_jump : ℕ) : ℕ := 2 * Ricciana_jump - 1

def Margarita_total_distance (Margarita_run Margarita_jump : ℕ) : ℕ := Margarita_run + Margarita_jump

def Ricciana_total_distance (Ricciana_run Ricciana_jump : ℕ) : ℕ := Ricciana_run + Ricciana_jump

theorem Ricciana_run_distance (R : ℕ) 
  (Ricciana_total : ℕ := R + Ricciana_jump) 
  (Margarita_total : ℕ := Margarita_run + Margarita_jump Ricciana_jump) 
  (h : Margarita_total = Ricciana_total + 1) : 
  R = 20 :=
by
  sorry

end NUMINAMATH_GPT_Ricciana_run_distance_l842_84206


namespace NUMINAMATH_GPT_reflection_ray_equation_l842_84211

theorem reflection_ray_equation (x y : ℝ) : (y = 2 * x + 1) → (∃ (x' y' : ℝ), y' = x ∧ y = 2 * x' + 1 ∧ x - 2 * y - 1 = 0) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_reflection_ray_equation_l842_84211


namespace NUMINAMATH_GPT_donations_received_l842_84218

def profit : Nat := 960
def half_profit: Nat := profit / 2
def goal: Nat := 610
def extra: Nat := 180
def total_needed: Nat := goal + extra
def donations: Nat := total_needed - half_profit

theorem donations_received :
  donations = 310 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_donations_received_l842_84218


namespace NUMINAMATH_GPT_n_five_minus_n_divisible_by_30_l842_84249

theorem n_five_minus_n_divisible_by_30 (n : ℤ) : 30 ∣ (n^5 - n) :=
sorry

end NUMINAMATH_GPT_n_five_minus_n_divisible_by_30_l842_84249


namespace NUMINAMATH_GPT_min_value_expression_l842_84264

noncomputable def f (x y : ℝ) : ℝ := 
  (x + 1 / y) * (x + 1 / y - 2023) + (y + 1 / x) * (y + 1 / x - 2023)

theorem min_value_expression : ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ f x y = -2048113 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l842_84264


namespace NUMINAMATH_GPT_scientific_notation_correct_l842_84228

theorem scientific_notation_correct : 1630000 = 1.63 * 10^6 :=
by sorry

end NUMINAMATH_GPT_scientific_notation_correct_l842_84228


namespace NUMINAMATH_GPT_vasya_number_l842_84235

theorem vasya_number (a b c d : ℕ) (h1 : a * b = 21) (h2 : b * c = 20) (h3 : ∃ x, x ∈ [4, 7] ∧ a ≠ c ∧ b = 7 ∧ c = 4 ∧ d = 5) : (1000 * a + 100 * b + 10 * c + d) = 3745 :=
sorry

end NUMINAMATH_GPT_vasya_number_l842_84235


namespace NUMINAMATH_GPT_molecular_weight_X_l842_84226

theorem molecular_weight_X (Ba_weight : ℝ) (total_molecular_weight : ℝ) (X_weight : ℝ) 
  (h1 : Ba_weight = 137) 
  (h2 : total_molecular_weight = 171) 
  (h3 : total_molecular_weight - Ba_weight * 1 = 2 * X_weight) : 
  X_weight = 17 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_X_l842_84226


namespace NUMINAMATH_GPT_range_f_subset_interval_l842_84242

-- Define the function f on real numbers
def f : ℝ → ℝ := sorry

-- The given condition for all real numbers x and y such that x > y
axiom condition (x y : ℝ) (h : x > y) : (f x)^2 ≤ f y

-- The main theorem that needs to be proven
theorem range_f_subset_interval : ∀ x, 0 ≤ f x ∧ f x ≤ 1 := 
by
  intro x
  apply And.intro
  -- Proof for 0 ≤ f x
  sorry
  -- Proof for f x ≤ 1
  sorry

end NUMINAMATH_GPT_range_f_subset_interval_l842_84242


namespace NUMINAMATH_GPT_solution_set_of_inverse_inequality_l842_84205

open Function

variable {f : ℝ → ℝ}

theorem solution_set_of_inverse_inequality 
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_A : f (-2) = 2)
  (h_B : f 2 = -2)
  : { x : ℝ | |(invFun f (x + 1))| ≤ 2 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inverse_inequality_l842_84205


namespace NUMINAMATH_GPT_variance_male_greater_than_female_l842_84252

noncomputable def male_scores : List ℝ := [87, 95, 89, 93, 91]
noncomputable def female_scores : List ℝ := [89, 94, 94, 89, 94]

-- Function to calculate the variance of scores
noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := scores.sum / n
  (scores.map (λ x => (x - mean) ^ 2)).sum / n

-- We assert the problem statement
theorem variance_male_greater_than_female :
  variance male_scores > variance female_scores :=
by
  sorry

end NUMINAMATH_GPT_variance_male_greater_than_female_l842_84252


namespace NUMINAMATH_GPT_convert_scientific_notation_l842_84273

theorem convert_scientific_notation (a : ℝ) (b : ℤ) (h : a = 6.03 ∧ b = 5) : a * 10^b = 603000 := by
  cases h with
  | intro ha hb =>
    rw [ha, hb]
    sorry

end NUMINAMATH_GPT_convert_scientific_notation_l842_84273


namespace NUMINAMATH_GPT_abs_ineq_solution_range_l842_84293

theorem abs_ineq_solution_range (a : ℝ) :
  (∃ x : ℝ, |x + 1| - |x - 2| > a) → a < 3 :=
by
  sorry

end NUMINAMATH_GPT_abs_ineq_solution_range_l842_84293


namespace NUMINAMATH_GPT_not_perfect_square_l842_84251

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, (3^n + 2 * 17^n) = k^2 :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_l842_84251


namespace NUMINAMATH_GPT_floor_ceil_inequality_l842_84201

theorem floor_ceil_inequality 
  (a b c : ℝ)
  (h : ⌈a⌉ + ⌈b⌉ + ⌈c⌉ + ⌊a + b⌋ + ⌊b + c⌋ + ⌊c + a⌋ = 2020) :
  ⌊a⌋ + ⌊b⌋ + ⌊c⌋ + ⌈a + b + c⌉ ≥ 1346 := 
by
  sorry 

end NUMINAMATH_GPT_floor_ceil_inequality_l842_84201


namespace NUMINAMATH_GPT_division_proof_l842_84282

-- Defining the given conditions
def total_books := 1200
def first_div := 3
def second_div := 4
def final_books_per_category := 15

-- Calculating the number of books per each category after each division
def books_per_first_category := total_books / first_div
def books_per_second_group := books_per_first_category / second_div

-- Correcting the third division to ensure each part has 15 books
def third_div := books_per_second_group / final_books_per_category
def rounded_parts := (books_per_second_group : ℕ) / final_books_per_category -- Rounded to the nearest integer

-- The number of final parts must be correct to ensure the total final categories
def final_division := first_div * second_div * rounded_parts

-- Required proof statement
theorem division_proof : final_division = 84 ∧ books_per_second_group = final_books_per_category :=
by 
  sorry

end NUMINAMATH_GPT_division_proof_l842_84282


namespace NUMINAMATH_GPT_multiply_expression_l842_84227

-- Definitions of variables
def a (x y : ℝ) := 3 * x^2
def b (x y : ℝ) := 4 * y^3

-- Theorem statement
theorem multiply_expression (x y : ℝ) :
  ((a x y) - (b x y)) * ((a x y)^2 + (a x y) * (b x y) + (b x y)^2) = 27 * x^6 - 64 * y^9 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_multiply_expression_l842_84227


namespace NUMINAMATH_GPT_value_of_b_pos_sum_for_all_x_l842_84224

noncomputable def f (b : ℝ) (x : ℝ) := 3 * x^2 - 2 * x + b
noncomputable def g (b : ℝ) (x : ℝ) := x^2 + b * x - 1
noncomputable def sum_f_g (b : ℝ) (x : ℝ) := f b x + g b x

theorem value_of_b (b : ℝ) (h : ∀ x : ℝ, (sum_f_g b x = 4 * x^2 + (b - 2) * x + (b - 1))) :
  b = 2 := 
sorry

theorem pos_sum_for_all_x :
  ∀ x : ℝ, 4 * x^2 + 1 > 0 := 
sorry

end NUMINAMATH_GPT_value_of_b_pos_sum_for_all_x_l842_84224


namespace NUMINAMATH_GPT_find_equation_of_line_l842_84287

theorem find_equation_of_line 
  (l : ℝ → ℝ → Prop)
  (h_intersect : ∃ x y : ℝ, 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ l x y)
  (h_parallel : ∀ x y : ℝ, l x y → 4 * x - 3 * y - 6 = 0) :
  ∀ x y : ℝ, l x y ↔ 4 * x - 3 * y - 6 = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_equation_of_line_l842_84287


namespace NUMINAMATH_GPT_complement_of_A_l842_84283

-- Definition of the universal set U and the set A
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | (x - 1) * (x + 2) > 0}

-- Theorem statement for the complement of A in U
theorem complement_of_A:
  (U \ A) = {x | -2 ≤ x ∧ x ≤ 1} :=
  sorry

end NUMINAMATH_GPT_complement_of_A_l842_84283


namespace NUMINAMATH_GPT_cylinder_volume_transformation_l842_84230

-- Define the original volume of the cylinder
def original_volume (V: ℝ) := V = 5

-- Define the transformation of quadrupling the dimensions of the cylinder
def new_volume (V V': ℝ) := V' = 64 * V

-- The goal is to show that under these conditions, the new volume is 320 gallons
theorem cylinder_volume_transformation (V V': ℝ) (h: original_volume V) (h': new_volume V V'):
  V' = 320 :=
by
  -- Proof is left as an exercise
  sorry

end NUMINAMATH_GPT_cylinder_volume_transformation_l842_84230


namespace NUMINAMATH_GPT_sequence_formula_and_sum_l842_84215

def arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) :=
  ∀ m n k, m < n → n < k → a n^2 = a m * a k

def Sn (S : ℕ → ℝ) (a : ℕ → ℝ) :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem sequence_formula_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ geometric_sequence a → ∀ n, a n = 4 * n - 2) ∧
  (arithmetic_geometric_sequence a 4 ∧ a 1 = 2 ∧ (∀ n, S n = (n * (4 * n)) / 2) → ∃ n > 0, S n > 60 * n + 800 ∧ n = 41) ∧
  (arithmetic_geometric_sequence a 0 ∧ a 1 = 2 ∧ (∀ n, S n = 2 * n) → ∀ n > 0, ¬ (S n > 60 * n + 800)) :=
by sorry

end NUMINAMATH_GPT_sequence_formula_and_sum_l842_84215


namespace NUMINAMATH_GPT_locus_of_P_l842_84295

variables {x y : ℝ}
variables {x0 y0 : ℝ}

-- The initial ellipse equation
def ellipse (x y : ℝ) : Prop :=
  x^2 / 20 + y^2 / 16 = 1

-- Point M is on the ellipse
def point_M (x0 y0 : ℝ) : Prop :=
  ellipse x0 y0

-- The equation of P, symmetric to transformations applied to point Q derived from M
theorem locus_of_P 
  (hx0 : x0^2 / 20 + y0^2 / 16 = 1) :
  ∃ x y, (x^2 / 20 + y^2 / 36 = 1) ∧ y ≠ 0 :=
sorry

end NUMINAMATH_GPT_locus_of_P_l842_84295


namespace NUMINAMATH_GPT_lambda_value_l842_84233

-- Definitions provided in the conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (e1 e2 : V) (A B C D : V)
-- Non-collinear vectors e1 and e2
variables (h_non_collinear : ∃ a b : ℝ, a ≠ b ∧ a • e1 + b • e2 ≠ 0)
-- Given vectors AB, BC, CD
variables (AB BC CD : V)
variables (lambda : ℝ)
-- Vector definitions based on given conditions
variables (h1 : AB = 2 • e1 + e2)
variables (h2 : BC = -e1 + 3 • e2)
variables (h3 : CD = lambda • e1 - e2)
-- Collinearity condition of points A, B, D
variables (collinear : ∃ β : ℝ, AB = β • (BC + CD))

-- The proof goal
theorem lambda_value (h1 : AB = 2 • e1 + e2) (h2 : BC = -e1 + 3 • e2) (h3 : CD = lambda • e1 - e2) (collinear : ∃ β : ℝ, AB = β • (BC + CD)) : lambda = 5 := 
sorry

end NUMINAMATH_GPT_lambda_value_l842_84233


namespace NUMINAMATH_GPT_find_complex_z_modulus_of_z_l842_84256

open Complex

theorem find_complex_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    z = -1 + 3 * I := by 
  sorry

theorem modulus_of_z (z : ℂ) (h1 : (z - (0 + 3 * I)).im = 0) (h2 : ((z - (0 + 5 * I)) / (2 - I)).re = 0) : 
    Complex.abs (z / (1 - I)) = Real.sqrt 5 := by 
  sorry

end NUMINAMATH_GPT_find_complex_z_modulus_of_z_l842_84256


namespace NUMINAMATH_GPT_column_1000_is_B_l842_84291

-- Definition of the column pattern
def columnPattern : List String := ["B", "C", "D", "E", "F", "E", "D", "C", "B", "A"]

-- Function to determine the column for a given integer
def columnOf (n : Nat) : String :=
  columnPattern.get! ((n - 2) % 10)

-- The theorem we want to prove
theorem column_1000_is_B : columnOf 1000 = "B" :=
by
  sorry

end NUMINAMATH_GPT_column_1000_is_B_l842_84291


namespace NUMINAMATH_GPT_product_ab_l842_84270

noncomputable def median_of_four_numbers (a b : ℕ) := 3
noncomputable def mean_of_four_numbers (a b : ℕ) := 4

theorem product_ab (a b : ℕ)
  (h1 : 1 + 2 + a + b = 4 * 4)
  (h2 : median_of_four_numbers a b = 3)
  (h3 : mean_of_four_numbers a b = 4) : (a * b = 36) :=
by sorry

end NUMINAMATH_GPT_product_ab_l842_84270


namespace NUMINAMATH_GPT_min_m_plus_inv_m_min_frac_expr_l842_84277

-- Sub-problem (1): Minimum value of m + 1/m for m > 0.
theorem min_m_plus_inv_m (m : ℝ) (h : m > 0) : m + 1/m = 2 :=
sorry

-- Sub-problem (2): Minimum value of (x^2 + x - 5)/(x - 2) for x > 2.
theorem min_frac_expr (x : ℝ) (h : x > 2) : (x^2 + x - 5)/(x - 2) = 7 :=
sorry

end NUMINAMATH_GPT_min_m_plus_inv_m_min_frac_expr_l842_84277


namespace NUMINAMATH_GPT_john_buys_packs_l842_84276

theorem john_buys_packs :
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  total_packs = 360 :=
by
  let classes := 6
  let students_per_class := 30
  let packs_per_student := 2
  let total_students := classes * students_per_class
  let total_packs := total_students * packs_per_student
  show total_packs = 360
  sorry

end NUMINAMATH_GPT_john_buys_packs_l842_84276


namespace NUMINAMATH_GPT_susan_age_in_5_years_l842_84207

variable (J N S X : ℕ)

-- Conditions
axiom h1 : J - 8 = 2 * (N - 8)
axiom h2 : J + X = 37
axiom h3 : S = N - 3

-- Theorem statement
theorem susan_age_in_5_years : S + 5 = N + 2 :=
by sorry

end NUMINAMATH_GPT_susan_age_in_5_years_l842_84207


namespace NUMINAMATH_GPT_sequence_property_l842_84213

theorem sequence_property (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 9) : 7 * n * 15873 = n * 111111 :=
by sorry

end NUMINAMATH_GPT_sequence_property_l842_84213


namespace NUMINAMATH_GPT_chandu_work_days_l842_84284

theorem chandu_work_days (W : ℝ) (c : ℝ) 
  (anand_rate : ℝ := W / 7) 
  (bittu_rate : ℝ := W / 8) 
  (chandu_rate : ℝ := W / c) 
  (completed_in_7_days : 3 * anand_rate + 2 * bittu_rate + 2 * chandu_rate = W) : 
  c = 7 :=
by
  sorry

end NUMINAMATH_GPT_chandu_work_days_l842_84284


namespace NUMINAMATH_GPT_find_principal_l842_84217

theorem find_principal (R P : ℝ) (h₁ : (P * R * 10) / 100 = P * R * 0.1)
  (h₂ : (P * (R + 3) * 10) / 100 = P * (R + 3) * 0.1)
  (h₃ : P * 0.1 * (R + 3) - P * 0.1 * R = 300) : 
  P = 1000 := 
sorry

end NUMINAMATH_GPT_find_principal_l842_84217


namespace NUMINAMATH_GPT_log_equality_l842_84265

theorem log_equality (x : ℝ) : (8 : ℝ)^x = 16 ↔ x = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_log_equality_l842_84265


namespace NUMINAMATH_GPT_new_three_digit_number_l842_84240

theorem new_three_digit_number (t u : ℕ) (h1 : t < 10) (h2 : u < 10) :
  let original := 10 * t + u
  let new_number := (original * 10) + 2
  new_number = 100 * t + 10 * u + 2 :=
by
  sorry

end NUMINAMATH_GPT_new_three_digit_number_l842_84240


namespace NUMINAMATH_GPT_board_rook_placement_l842_84243

-- Define the color function for the board
def color (n i j : ℕ) : ℕ :=
  min (i + j - 1) (2 * n - i - j + 1)

-- Conditions: It is possible to place n rooks such that no two attack each other and 
-- no two rooks stand on cells of the same color
def non_attacking_rooks (n : ℕ) (rooks : Fin n → Fin n) : Prop :=
  ∀ i j : Fin n, i ≠ j → rooks i ≠ rooks j ∧ color n i.val (rooks i).val ≠ color n j.val (rooks j).val

-- Main theorem to be proven
theorem board_rook_placement (n : ℕ) :
  (∃ rooks : Fin n → Fin n, non_attacking_rooks n rooks) →
  n % 4 = 0 ∨ n % 4 = 1 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_board_rook_placement_l842_84243


namespace NUMINAMATH_GPT_measure_of_angle_B_l842_84200

-- Define the conditions and the goal as a theorem
theorem measure_of_angle_B (A B C : ℝ)
  (h1 : C = 2 * B)
  (h2 : A = 3 * B)
  (triangle_angle_sum : A + B + C = 180) : B = 30 :=
by
  -- Substitute the conditions into Lean to express and prove the statement
  sorry

end NUMINAMATH_GPT_measure_of_angle_B_l842_84200


namespace NUMINAMATH_GPT_shuttle_speed_l842_84266

theorem shuttle_speed (speed_kps : ℕ) (conversion_factor : ℕ) (speed_kph : ℕ) :
  speed_kps = 2 → conversion_factor = 3600 → speed_kph = speed_kps * conversion_factor → speed_kph = 7200 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_shuttle_speed_l842_84266


namespace NUMINAMATH_GPT_solution_set_inequality_l842_84280

-- Statement of the problem
theorem solution_set_inequality :
  {x : ℝ | 1 / x < 1 / 2} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l842_84280


namespace NUMINAMATH_GPT_julie_hours_per_week_school_year_l842_84253

-- Defining the assumptions
variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_earnings : ℝ)
variable (school_year_weeks : ℕ) (school_year_earnings : ℝ)

-- Assuming the given values
def assumptions : Prop :=
  summer_hours_per_week = 36 ∧ 
  summer_weeks = 10 ∧ 
  summer_earnings = 4500 ∧ 
  school_year_weeks = 45 ∧ 
  school_year_earnings = 4500

-- Proving that Julie must work 8 hours per week during the school year to make another $4500
theorem julie_hours_per_week_school_year : 
  assumptions summer_hours_per_week summer_weeks summer_earnings school_year_weeks school_year_earnings →
  (school_year_earnings / (summer_earnings / (summer_hours_per_week * summer_weeks)) / school_year_weeks = 8) :=
by
  sorry

end NUMINAMATH_GPT_julie_hours_per_week_school_year_l842_84253


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l842_84208
open Locale

variables {l m : Line} {α β : Plane}

def perp (l : Line) (p : Plane) : Prop := sorry
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

theorem necessary_but_not_sufficient_condition (h1 : perp l α) (h2 : subset m β) (h3 : perp l m) :
  ∃ (α : Plane) (β : Plane), parallel α β ∧ (perp l α → perp l β) ∧ (parallel α β → perp l β)  :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l842_84208


namespace NUMINAMATH_GPT_planes_contain_at_least_three_midpoints_l842_84281

-- Define the cube structure and edge midpoints
structure Cube where
  edges : Fin 12

def midpoints (c : Cube) : Set (Fin 12) := { e | true }

-- Define the total planes considering the constraints
noncomputable def planes : ℕ := 4 + 18 + 56

-- The proof goal
theorem planes_contain_at_least_three_midpoints :
  planes = 81 := by
  sorry

end NUMINAMATH_GPT_planes_contain_at_least_three_midpoints_l842_84281


namespace NUMINAMATH_GPT_unique_perpendicular_line_through_point_l842_84275

-- Definitions of the geometric entities and their relationships
structure Point := (x : ℝ) (y : ℝ)

structure Line := (m : ℝ) (b : ℝ)

-- A function to check if a point lies on a given line
def point_on_line (P : Point) (l : Line) : Prop := P.y = l.m * P.x + l.b

-- A function to represent that a line is perpendicular to another line at a given point
def perpendicular_lines_at_point (P : Point) (l1 l2 : Line) : Prop :=
  l1.m = -(1 / l2.m) ∧ point_on_line P l1 ∧ point_on_line P l2

-- The statement to be proved
theorem unique_perpendicular_line_through_point (P : Point) (l : Line) (h : point_on_line P l) :
  ∃! l' : Line, perpendicular_lines_at_point P l' l :=
by
  sorry

end NUMINAMATH_GPT_unique_perpendicular_line_through_point_l842_84275


namespace NUMINAMATH_GPT_hyperbola_condition_l842_84241

theorem hyperbola_condition (k : ℝ) : (k > 1) -> ( ∀ x y : ℝ, (k - 1) * (k + 1) > 0 ↔ ( ∃ x y : ℝ, (k > 1) ∧ ((x * x) / (k - 1) - (y * y) / (k + 1)) = 1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_condition_l842_84241


namespace NUMINAMATH_GPT_train_speed_l842_84248

theorem train_speed (length : ℕ) (time : ℕ) (h1 : length = 1600) (h2 : time = 40) : length / time = 40 := 
by
  -- use the given conditions here
  sorry

end NUMINAMATH_GPT_train_speed_l842_84248


namespace NUMINAMATH_GPT_quadratic_complete_square_l842_84274

theorem quadratic_complete_square (x d e: ℝ) (h : x^2 - 26 * x + 129 = (x + d)^2 + e) : 
d + e = -53 := sorry

end NUMINAMATH_GPT_quadratic_complete_square_l842_84274


namespace NUMINAMATH_GPT_multiple_of_sum_squares_l842_84258

theorem multiple_of_sum_squares (a b c : ℕ) (h1 : a < 2017) (h2 : b < 2017) (h3 : c < 2017) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a)
    (h7 : ∃ k1, a^3 - b^3 = k1 * 2017) (h8 : ∃ k2, b^3 - c^3 = k2 * 2017) (h9 : ∃ k3, c^3 - a^3 = k3 * 2017) :
    ∃ k, a^2 + b^2 + c^2 = k * (a + b + c) :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_sum_squares_l842_84258


namespace NUMINAMATH_GPT_max_remaining_area_l842_84203

theorem max_remaining_area (original_area : ℕ) (rec1 : ℕ × ℕ) (rec2 : ℕ × ℕ) (rec3 : ℕ × ℕ)
  (rec4 : ℕ × ℕ) (total_area_cutout : ℕ):
  original_area = 132 →
  rec1 = (1, 4) →
  rec2 = (2, 2) →
  rec3 = (2, 3) →
  rec4 = (2, 3) →
  total_area_cutout = 20 →
  original_area - total_area_cutout = 112 :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_remaining_area_l842_84203


namespace NUMINAMATH_GPT_tangent_lines_to_curve_l842_84204

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the general form of a tangent line
def tangent_line (x : ℝ) (y : ℝ) (m : ℝ) (x0 : ℝ) (y0 : ℝ) : Prop :=
  y - y0 = m * (x - x0)

-- Define the conditions
def condition1 : Prop :=
  tangent_line 1 1 3 1 1

def condition2 : Prop :=
  tangent_line 1 1 (3/4) (-1/2) ((-1/2)^3)

-- Define the equations of the tangent lines
def line1 : Prop :=
  ∀ x y : ℝ, 3 * x - y - 2 = 0

def line2 : Prop :=
  ∀ x y : ℝ, 3 * x - 4 * y + 1 = 0

-- The final theorem statement
theorem tangent_lines_to_curve :
  (condition1 → line1) ∧ (condition2 → line2) :=
  by
    sorry -- Placeholder for proof

end NUMINAMATH_GPT_tangent_lines_to_curve_l842_84204


namespace NUMINAMATH_GPT_arithmetic_sequence_max_sum_l842_84268

theorem arithmetic_sequence_max_sum (a : ℕ → ℝ) (d : ℝ) (m : ℕ) (S : ℕ → ℝ):
  (∀ n, a n = a 1 + (n - 1) * d) → 
  3 * a 8 = 5 * a m → 
  a 1 > 0 →
  (∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) →
  (∀ n, S n ≤ S 20) →
  m = 13 := 
by {
  -- State the corresponding solution steps leading to the proof.
  sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_max_sum_l842_84268


namespace NUMINAMATH_GPT_sum_mod_7_remainder_l842_84214

def sum_to (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem sum_mod_7_remainder : (sum_to 140) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_7_remainder_l842_84214


namespace NUMINAMATH_GPT_part_a_l842_84232

theorem part_a (x : ℝ) (hx : x > 0) :
  ∃ color : ℕ, ∃ p1 p2 : ℝ × ℝ, (p1 = p2 ∨ x = dist p1 p2) :=
sorry

end NUMINAMATH_GPT_part_a_l842_84232


namespace NUMINAMATH_GPT_increasing_function_range_of_a_l842_84222

variable {f : ℝ → ℝ}

theorem increasing_function_range_of_a (a : ℝ) (h : ∀ x : ℝ, 3 * a * x^2 ≥ 0) : a > 0 :=
sorry

end NUMINAMATH_GPT_increasing_function_range_of_a_l842_84222
