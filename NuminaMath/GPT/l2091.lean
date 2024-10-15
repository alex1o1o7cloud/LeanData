import Mathlib

namespace NUMINAMATH_GPT_jars_water_fraction_l2091_209129

theorem jars_water_fraction (S L W : ℝ) (h1 : W = 1/6 * S) (h2 : W = 1/5 * L) : 
  (2 * W / L) = 2 / 5 :=
by
  -- We are only stating the theorem here, not proving it.
  sorry

end NUMINAMATH_GPT_jars_water_fraction_l2091_209129


namespace NUMINAMATH_GPT_min_value_of_derivative_l2091_209189

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * a * x^2 + (1 / a) * x

noncomputable def f' (a : ℝ) : ℝ := 3 * 2^2 + 4 * a * 2 + (1 / a)

theorem min_value_of_derivative (a : ℝ) (h : a > 0) : 
  f' a ≥ 12 + 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_derivative_l2091_209189


namespace NUMINAMATH_GPT_scale_length_l2091_209198

theorem scale_length (length_of_part : ℕ) (number_of_parts : ℕ) (h1 : number_of_parts = 2) (h2 : length_of_part = 40) :
  number_of_parts * length_of_part = 80 := 
by
  sorry

end NUMINAMATH_GPT_scale_length_l2091_209198


namespace NUMINAMATH_GPT_evaluate_expression_right_to_left_l2091_209136

variable (a b c d : ℝ)

theorem evaluate_expression_right_to_left:
  (a * b + c - d) = (a * (b + c - d)) :=
by {
  -- Group operations from right to left according to the given condition
  sorry
}

end NUMINAMATH_GPT_evaluate_expression_right_to_left_l2091_209136


namespace NUMINAMATH_GPT_percentage_less_than_l2091_209152

namespace PercentProblem

noncomputable def A (C : ℝ) : ℝ := 0.65 * C
noncomputable def B (C : ℝ) : ℝ := 0.8923076923076923 * A C

theorem percentage_less_than (C : ℝ) (hC : C ≠ 0) : (C - B C) / C = 0.42 :=
by
  sorry

end PercentProblem

end NUMINAMATH_GPT_percentage_less_than_l2091_209152


namespace NUMINAMATH_GPT_domain_of_h_l2091_209190

open Real

theorem domain_of_h : ∀ x : ℝ, |x - 5| + |x + 2| ≠ 0 := by
  intro x
  sorry

end NUMINAMATH_GPT_domain_of_h_l2091_209190


namespace NUMINAMATH_GPT_probability_blue_ball_l2091_209181

-- Define the probabilities of drawing a red and yellow ball
def P_red : ℝ := 0.48
def P_yellow : ℝ := 0.35

-- Define the total probability formula in this sample space
def total_probability (P_red P_yellow P_blue : ℝ) : Prop :=
  P_red + P_yellow + P_blue = 1

-- The theorem we need to prove
theorem probability_blue_ball :
  ∃ P_blue : ℝ, total_probability P_red P_yellow P_blue ∧ P_blue = 0.17 :=
sorry

end NUMINAMATH_GPT_probability_blue_ball_l2091_209181


namespace NUMINAMATH_GPT_carol_sold_cupcakes_l2091_209188

variable (initial_cupcakes := 30) (additional_cupcakes := 28) (final_cupcakes := 49)

theorem carol_sold_cupcakes : (initial_cupcakes + additional_cupcakes - final_cupcakes = 9) :=
by sorry

end NUMINAMATH_GPT_carol_sold_cupcakes_l2091_209188


namespace NUMINAMATH_GPT_calculation_correct_l2091_209113

theorem calculation_correct (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b :=
by sorry

end NUMINAMATH_GPT_calculation_correct_l2091_209113


namespace NUMINAMATH_GPT_rock_paper_scissors_score_divisible_by_3_l2091_209162

theorem rock_paper_scissors_score_divisible_by_3 
  (R : ℕ) 
  (rock_shown : ℕ) 
  (scissors_shown : ℕ) 
  (paper_shown : ℕ)
  (points : ℕ)
  (h_equal_shows : 3 * ((rock_shown + scissors_shown + paper_shown) / 3) = rock_shown + scissors_shown + paper_shown)
  (h_points_awarded : ∀ (r s p : ℕ), r + s + p = 3 → (r = 2 ∧ s = 1 ∧ p = 0) ∨ (r = 0 ∧ s = 2 ∧ p = 1) ∨ (r = 1 ∧ s = 0 ∧ p = 2) → points % 3 = 0) :
  points % 3 = 0 := 
sorry

end NUMINAMATH_GPT_rock_paper_scissors_score_divisible_by_3_l2091_209162


namespace NUMINAMATH_GPT_smallest_nat_satisfying_conditions_l2091_209163

theorem smallest_nat_satisfying_conditions : 
  ∃ x : ℕ, 
  (x % 4 = 2) ∧ 
  (x % 5 = 2) ∧ 
  (x % 6 = 2) ∧ 
  (x % 12 = 2) ∧ 
  (∀ y : ℕ, (y % 4 = 2) ∧ (y % 5 = 2) ∧ (y % 6 = 2) ∧ (y % 12 = 2) → x ≤ y) :=
  sorry

end NUMINAMATH_GPT_smallest_nat_satisfying_conditions_l2091_209163


namespace NUMINAMATH_GPT_birth_date_16_Jan_1993_l2091_209180

noncomputable def year_of_birth (current_date : Nat) (age_years : Nat) :=
  current_date - age_years * 365

noncomputable def month_of_birth (current_date : Nat) (age_years : Nat) (age_months : Nat) :=
  current_date - (age_years * 12 + age_months) * 30

theorem birth_date_16_Jan_1993 :
  let boy_age_years := 10
  let boy_age_months := 1
  let current_date := 16 + 31 * 12 * 2003 -- 16th February 2003 represented in days
  let full_months_lived := boy_age_years * 12 + boy_age_months
  full_months_lived - boy_age_years = 111 → 
  year_of_birth current_date boy_age_years = 1993 ∧ month_of_birth current_date boy_age_years boy_age_months = 31 * 1 * 1993 := 
sorry

end NUMINAMATH_GPT_birth_date_16_Jan_1993_l2091_209180


namespace NUMINAMATH_GPT_cyclic_points_exist_l2091_209185

noncomputable def f (x : ℝ) : ℝ := 
if x < (1 / 3) then 
  2 * x + (1 / 3) 
else 
  (3 / 2) * (1 - x)

theorem cyclic_points_exist :
  ∃ (x0 x1 x2 x3 x4 : ℝ), 
  0 ≤ x0 ∧ x0 ≤ 1 ∧
  0 ≤ x1 ∧ x1 ≤ 1 ∧
  0 ≤ x2 ∧ x2 ≤ 1 ∧
  0 ≤ x3 ∧ x3 ≤ 1 ∧
  0 ≤ x4 ∧ x4 ≤ 1 ∧
  x0 ≠ x1 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ x4 ≠ x0 ∧
  f x0 = x1 ∧ f x1 = x2 ∧ f x2 = x3 ∧ f x3 = x4 ∧ f x4 = x0 :=
sorry

end NUMINAMATH_GPT_cyclic_points_exist_l2091_209185


namespace NUMINAMATH_GPT_CodgerNeedsTenPairs_l2091_209125

def CodgerHasThreeFeet : Prop := true

def ShoesSoldInPairs : Prop := true

def ShoesSoldInEvenNumberedPairs : Prop := true

def CodgerOwnsOneThreePieceSet : Prop := true

-- Main theorem stating Codger needs 10 pairs of shoes to have 7 complete 3-piece sets
theorem CodgerNeedsTenPairs (h1 : CodgerHasThreeFeet) (h2 : ShoesSoldInPairs)
  (h3 : ShoesSoldInEvenNumberedPairs) (h4 : CodgerOwnsOneThreePieceSet) : 
  ∃ pairsToBuy : ℕ, pairsToBuy = 10 := 
by {
  -- We have to prove codger needs 10 pairs of shoes to have 7 complete 3-piece sets
  sorry
}

end NUMINAMATH_GPT_CodgerNeedsTenPairs_l2091_209125


namespace NUMINAMATH_GPT_infinitely_many_sum_form_l2091_209169

theorem infinitely_many_sum_form {a : ℕ → ℕ} (h : ∀ n, a n < a (n + 1)) :
  ∀ i, ∃ᶠ n in at_top, ∃ r s j, r > 0 ∧ s > 0 ∧ i < j ∧ a n = r * a i + s * a j := 
by
  sorry

end NUMINAMATH_GPT_infinitely_many_sum_form_l2091_209169


namespace NUMINAMATH_GPT_pages_difference_l2091_209112

def second_chapter_pages : ℕ := 18
def third_chapter_pages : ℕ := 3

theorem pages_difference : second_chapter_pages - third_chapter_pages = 15 := by 
  sorry

end NUMINAMATH_GPT_pages_difference_l2091_209112


namespace NUMINAMATH_GPT_license_plate_possibilities_count_l2091_209107

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

def digits : Finset Char := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

theorem license_plate_possibilities_count : 
  (vowels.card * digits.card * 2 = 100) := 
by {
  -- vowels.card = 5 because there are 5 vowels.
  -- digits.card = 10 because there are 10 digits.
  -- 2 because the middle character must match either the first vowel or the last digit.
  sorry
}

end NUMINAMATH_GPT_license_plate_possibilities_count_l2091_209107


namespace NUMINAMATH_GPT_pencils_ratio_l2091_209192

theorem pencils_ratio (C J : ℕ) (hJ : J = 18) 
    (hJ_to_A : J_to_A = J / 3) (hJ_left : J_left = J - J_to_A)
    (hJ_left_eq : J_left = C + 3) :
    (C : ℚ) / (J : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_pencils_ratio_l2091_209192


namespace NUMINAMATH_GPT_triangle_inequality_l2091_209140

theorem triangle_inequality
  (α β γ a b c : ℝ)
  (h_angles_sum : α + β + γ = Real.pi)
  (h_pos_angles : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_pos_sides : a > 0 ∧ b > 0 ∧ c > 0) :
  a * (1 / β + 1 / γ) + b * (1 / γ + 1 / α) + c * (1 / α + 1 / β) ≥ 2 * (a / α + b / β + c / γ) := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l2091_209140


namespace NUMINAMATH_GPT_perfect_square_quotient_l2091_209194

theorem perfect_square_quotient {a b : ℕ} (hpos: 0 < a ∧ 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end NUMINAMATH_GPT_perfect_square_quotient_l2091_209194


namespace NUMINAMATH_GPT_sum_a1_to_a5_l2091_209148

-- Define the conditions
def equation_holds (x a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  x^5 + 2 = a0 + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 + a4 * (x - 1)^4 + a5 * (x - 1)^5

-- State the theorem
theorem sum_a1_to_a5 (a0 a1 a2 a3 a4 a5 : ℝ) (h : ∀ x : ℝ, equation_holds x a0 a1 a2 a3 a4 a5) :
  a1 + a2 + a3 + a4 + a5 = 31 :=
by
  sorry

end NUMINAMATH_GPT_sum_a1_to_a5_l2091_209148


namespace NUMINAMATH_GPT_set_inter_complement_l2091_209133

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}

theorem set_inter_complement :
  A ∩ (U \ B) = {1, 3} :=
by
  sorry

end NUMINAMATH_GPT_set_inter_complement_l2091_209133


namespace NUMINAMATH_GPT_area_difference_zero_l2091_209118

theorem area_difference_zero
  (AG CE : ℝ)
  (s : ℝ)
  (area_square area_rectangle : ℝ)
  (h1 : AG = 2)
  (h2 : CE = 2)
  (h3 : s = 2)
  (h4 : area_square = s^2)
  (h5 : area_rectangle = 2 * 2) :
  (area_square - area_rectangle = 0) :=
by sorry

end NUMINAMATH_GPT_area_difference_zero_l2091_209118


namespace NUMINAMATH_GPT_range_of_x_plus_y_l2091_209174

theorem range_of_x_plus_y (x y : ℝ) (hx1 : y = 3 * ⌊x⌋ + 4) (hx2 : y = 4 * ⌊x - 3⌋ + 7) (hxnint : ¬ ∃ z : ℤ, x = z): 
  40 < x + y ∧ x + y < 41 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_plus_y_l2091_209174


namespace NUMINAMATH_GPT_days_to_complete_work_l2091_209131

variable {P W D : ℕ}

axiom condition_1 : 2 * P * 3 = W / 2
axiom condition_2 : P * D = W

theorem days_to_complete_work : D = 12 :=
by
  -- As an axiom or sorry is used, the proof is omitted.
  sorry

end NUMINAMATH_GPT_days_to_complete_work_l2091_209131


namespace NUMINAMATH_GPT_brick_wall_completion_time_l2091_209176

def rate (hours : ℚ) : ℚ := 1 / hours

/-- Avery can build a brick wall in 3 hours. -/
def avery_rate : ℚ := rate 3
/-- Tom can build a brick wall in 2.5 hours. -/
def tom_rate : ℚ := rate 2.5
/-- Catherine can build a brick wall in 4 hours. -/
def catherine_rate : ℚ := rate 4
/-- Derek can build a brick wall in 5 hours. -/
def derek_rate : ℚ := rate 5

/-- Combined rate for Avery, Tom, and Catherine working together. -/
def combined_rate_1 : ℚ := avery_rate + tom_rate + catherine_rate
/-- Combined rate for Tom and Catherine working together. -/
def combined_rate_2 : ℚ := tom_rate + catherine_rate
/-- Combined rate for Tom, Catherine, and Derek working together. -/
def combined_rate_3 : ℚ := tom_rate + catherine_rate + derek_rate

/-- Total time taken to complete the wall. -/
def total_time (t : ℚ) : Prop :=
  t = 2

theorem brick_wall_completion_time (t : ℚ) : total_time t :=
by
  sorry

end NUMINAMATH_GPT_brick_wall_completion_time_l2091_209176


namespace NUMINAMATH_GPT_rice_in_each_container_l2091_209104

variable (weight_in_pounds : ℚ := 35 / 2)
variable (num_containers : ℕ := 4)
variable (pound_to_oz : ℕ := 16)

theorem rice_in_each_container :
  (weight_in_pounds * pound_to_oz) / num_containers = 70 :=
by
  sorry

end NUMINAMATH_GPT_rice_in_each_container_l2091_209104


namespace NUMINAMATH_GPT_all_have_perp_property_l2091_209142

def M₁ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, x^3 - 2 * x^2 + 3)}
def M₂ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.log (2 - x) / Real.log 2)}
def M₃ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 2 - 2^x)}
def M₄ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 1 - Real.sin x)}

def perp_property (M : Set (ℝ × ℝ)) : Prop :=
∀ p ∈ M, ∃ q ∈ M, p.1 * q.1 + p.2 * q.2 = 0

-- Theorem statement
theorem all_have_perp_property :
  perp_property M₁ ∧ perp_property M₂ ∧ perp_property M₃ ∧ perp_property M₄ :=
sorry

end NUMINAMATH_GPT_all_have_perp_property_l2091_209142


namespace NUMINAMATH_GPT_given_problem_l2091_209183

noncomputable def improper_fraction_5_2_7 : ℚ := 37 / 7
noncomputable def improper_fraction_6_1_3 : ℚ := 19 / 3
noncomputable def improper_fraction_3_1_2 : ℚ := 7 / 2
noncomputable def improper_fraction_2_1_5 : ℚ := 11 / 5

theorem given_problem :
  71 * (improper_fraction_5_2_7 - improper_fraction_6_1_3) / (improper_fraction_3_1_2 + improper_fraction_2_1_5) = -13 - 37 / 1197 := 
  sorry

end NUMINAMATH_GPT_given_problem_l2091_209183


namespace NUMINAMATH_GPT_contest_score_difference_l2091_209120

theorem contest_score_difference :
  let percent_50 := 0.05
  let percent_60 := 0.20
  let percent_70 := 0.25
  let percent_80 := 0.30
  let percent_90 := 1 - (percent_50 + percent_60 + percent_70 + percent_80)
  let mean := (percent_50 * 50) + (percent_60 * 60) + (percent_70 * 70) + (percent_80 * 80) + (percent_90 * 90)
  let median := 70
  median - mean = -4 :=
by
  sorry

end NUMINAMATH_GPT_contest_score_difference_l2091_209120


namespace NUMINAMATH_GPT_work_completion_l2091_209122

theorem work_completion (A B C : ℝ) (h₁ : A + B = 1 / 18) (h₂ : B + C = 1 / 24) (h₃ : A + C = 1 / 36) : 
  1 / (A + B + C) = 16 := 
by
  sorry

end NUMINAMATH_GPT_work_completion_l2091_209122


namespace NUMINAMATH_GPT_quadratic_root_property_l2091_209126

theorem quadratic_root_property (m n : ℝ)
  (hmn : m^2 + m - 2021 = 0)
  (hn : n^2 + n - 2021 = 0) :
  m^2 + 2 * m + n = 2020 :=
by sorry

end NUMINAMATH_GPT_quadratic_root_property_l2091_209126


namespace NUMINAMATH_GPT_negation_of_square_positivity_l2091_209165

theorem negation_of_square_positivity :
  (¬ ∀ n : ℕ, n * n > 0) ↔ (∃ n : ℕ, n * n ≤ 0) :=
  sorry

end NUMINAMATH_GPT_negation_of_square_positivity_l2091_209165


namespace NUMINAMATH_GPT_range_of_a_l2091_209121

theorem range_of_a (x a : ℝ) (h₁ : x > 1) (h₂ : a ≤ x + 1 / (x - 1)) : 
  a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l2091_209121


namespace NUMINAMATH_GPT_find_k_l2091_209110

open Real

def vector := ℝ × ℝ

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k
  (a b : vector)
  (h_a : a = (2, -1))
  (h_b : b = (-1, 4))
  (h_perpendicular : dot_product (a.1 - k * b.1, a.2 + 4 * k) (3, -5) = 0) :
  k = -11/17 := sorry

end NUMINAMATH_GPT_find_k_l2091_209110


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_polynomial_l2091_209164

-- Define the quadratic polynomial
def quadratic_polynomial (x : ℝ) : ℝ := x^2 + 14 * x + 3

-- Statement to prove
theorem minimum_value_of_quadratic_polynomial : ∃ x : ℝ, quadratic_polynomial x = quadratic_polynomial (-7) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_polynomial_l2091_209164


namespace NUMINAMATH_GPT_valid_numbers_eq_l2091_209155

-- Definition of the number representation
def is_valid_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999 ∧
  ∃ (a b c : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    x = 100 * a + 10 * b + c ∧
    x = a^3 + b^3 + c^3

-- The theorem to prove
theorem valid_numbers_eq : 
  {x : ℕ | is_valid_number x} = {153, 407} :=
by
  sorry

end NUMINAMATH_GPT_valid_numbers_eq_l2091_209155


namespace NUMINAMATH_GPT_blown_out_sand_dunes_l2091_209199

theorem blown_out_sand_dunes (p_remain p_lucky p_both : ℝ) (h_rem: p_remain = 1 / 3) (h_luck: p_lucky = 2 / 3)
(h_both: p_both = 0.08888888888888889) : 
  ∃ N : ℕ, N = 8 :=
by
  sorry

end NUMINAMATH_GPT_blown_out_sand_dunes_l2091_209199


namespace NUMINAMATH_GPT_complex_equality_l2091_209130

theorem complex_equality (a b : ℝ) (i : ℂ) (h : i^2 = -1) (h_eq : a - b * i = (1 + i) * i^3) : a = 1 ∧ b = -1 :=
by sorry

end NUMINAMATH_GPT_complex_equality_l2091_209130


namespace NUMINAMATH_GPT_boxes_of_bolts_purchased_l2091_209134

theorem boxes_of_bolts_purchased 
  (bolts_per_box : ℕ) 
  (nuts_per_box : ℕ) 
  (num_nut_boxes : ℕ) 
  (leftover_bolts : ℕ) 
  (leftover_nuts : ℕ) 
  (total_bolts_nuts_used : ℕ)
  (B : ℕ) :
  bolts_per_box = 11 →
  nuts_per_box = 15 →
  num_nut_boxes = 3 →
  leftover_bolts = 3 →
  leftover_nuts = 6 →
  total_bolts_nuts_used = 113 →
  B = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boxes_of_bolts_purchased_l2091_209134


namespace NUMINAMATH_GPT_quadratic_zeros_l2091_209138

theorem quadratic_zeros (a b : ℝ) (h1 : (4 - 2 * a + b = 0)) (h2 : (9 + 3 * a + b = 0)) : a + b = -7 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_zeros_l2091_209138


namespace NUMINAMATH_GPT_intersection_point_exists_l2091_209158

theorem intersection_point_exists
  (m n a b : ℝ)
  (h1 : m * a + 2 * m * b = 5)
  (h2 : n * a - 2 * n * b = 7)
  : (∃ x y : ℝ, 
    (y = (5 / (2 * m)) - (1 / 2) * x) ∧ 
    (y = (1 / 2) * x - (7 / (2 * n))) ∧
    (x = a) ∧ (y = b)) :=
sorry

end NUMINAMATH_GPT_intersection_point_exists_l2091_209158


namespace NUMINAMATH_GPT_number_of_boys_l2091_209115

theorem number_of_boys
  (x y : ℕ) 
  (h1 : x + y = 43)
  (h2 : 24 * x + 27 * y = 1101) : 
  x = 20 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l2091_209115


namespace NUMINAMATH_GPT_charlies_age_22_l2091_209102

variable (A : ℕ) (C : ℕ)

theorem charlies_age_22 (h1 : C = 2 * A + 8) (h2 : C = 22) : A = 7 := by
  sorry

end NUMINAMATH_GPT_charlies_age_22_l2091_209102


namespace NUMINAMATH_GPT_skipping_rates_l2091_209106

theorem skipping_rates (x y : ℕ) (h₀ : 300 / (x + 19) = 270 / x) (h₁ : y = x + 19) :
  x = 171 ∧ y = 190 := by
  sorry

end NUMINAMATH_GPT_skipping_rates_l2091_209106


namespace NUMINAMATH_GPT_proof_f_values_l2091_209109

def f (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 7
  else
    x^2 - 2

theorem proof_f_values :
  f (-2) = 3 ∧ f (3) = 7 :=
by
  sorry

end NUMINAMATH_GPT_proof_f_values_l2091_209109


namespace NUMINAMATH_GPT_quadruples_characterization_l2091_209172

/-- Proving the characterization of quadruples (a, b, c, d) of non-negative integers 
such that ab = 2(1 + cd) and there exists a non-degenerate triangle with sides (a - c), 
(b - d), and (c + d). -/
theorem quadruples_characterization :
  ∀ (a b c d : ℕ), 
    ab = 2 * (1 + cd) ∧ 
    (a - c) + (b - d) > c + d ∧ 
    (a - c) + (c + d) > b - d ∧ 
    (b - d) + (c + d) > a - c ∧
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 → 
    (a = 1 ∧ b = 2 ∧ c = 0 ∧ d = 1) ∨ 
    (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 0) :=
by sorry

end NUMINAMATH_GPT_quadruples_characterization_l2091_209172


namespace NUMINAMATH_GPT_solve_equation_l2091_209147

theorem solve_equation (x : ℝ) (h : (x - 1) / 2 = 1 - (x + 2) / 3) : x = 1 :=
sorry

end NUMINAMATH_GPT_solve_equation_l2091_209147


namespace NUMINAMATH_GPT_fraction_to_decimal_l2091_209177

theorem fraction_to_decimal (n d : ℕ) (hn : n = 53) (hd : d = 160) (gcd_nd : Nat.gcd n d = 1)
  (prime_factorization_d : ∃ k l : ℕ, d = 2^k * 5^l) : ∃ dec : ℚ, (n:ℚ) / (d:ℚ) = dec ∧ dec = 0.33125 :=
by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l2091_209177


namespace NUMINAMATH_GPT_leak_drain_time_l2091_209167

theorem leak_drain_time :
  ∀ (P L : ℝ),
  P = 1/6 →
  P - L = 1/12 →
  (1/L) = 12 :=
by
  intros P L hP hPL
  sorry

end NUMINAMATH_GPT_leak_drain_time_l2091_209167


namespace NUMINAMATH_GPT_exists_n_good_but_not_succ_good_l2091_209117

def S (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

def n_good (n : ℕ) (a : ℕ) : Prop :=
  ∃ (a_seq : Fin (n + 1) → ℕ), 
    a_seq n = a ∧ (∀ i : Fin n, a_seq (Fin.succ i) = a_seq i - S (a_seq i))

theorem exists_n_good_but_not_succ_good (n : ℕ) : 
  ∃ a, n_good n a ∧ ¬ n_good (n + 1) a := 
sorry

end NUMINAMATH_GPT_exists_n_good_but_not_succ_good_l2091_209117


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2091_209124

def point : ℝ × ℝ := (3, -4)

def isFirstQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0
def isSecondQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0
def isThirdQuadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0
def isFourthQuadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : isFourthQuadrant point :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2091_209124


namespace NUMINAMATH_GPT_solve_puzzle_l2091_209191

theorem solve_puzzle (x1 x2 x3 x4 x5 x6 x7 x8 : ℕ) : 
  (8 + x1 + x2 = 20) →
  (x1 + x2 + x3 = 20) →
  (x2 + x3 + x4 = 20) →
  (x3 + x4 + x5 = 20) →
  (x4 + x5 + 5 = 20) →
  (x5 + 5 + x6 = 20) →
  (5 + x6 + x7 = 20) →
  (x6 + x7 + x8 = 20) →
  (x1 = 7 ∧ x2 = 5 ∧ x3 = 8 ∧ x4 = 7 ∧ x5 = 5 ∧ x6 = 8 ∧ x7 = 7 ∧ x8 = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_puzzle_l2091_209191


namespace NUMINAMATH_GPT_fraction_of_yard_occupied_l2091_209166

noncomputable def area_triangle_flower_bed : ℝ := 
  2 * (0.5 * (10:ℝ) * (10:ℝ))

noncomputable def area_circular_flower_bed : ℝ := 
  Real.pi * (2:ℝ)^2

noncomputable def total_area_flower_beds : ℝ := 
  area_triangle_flower_bed + area_circular_flower_bed

noncomputable def area_yard : ℝ := 
  (40:ℝ) * (10:ℝ)

noncomputable def fraction_occupied := 
  total_area_flower_beds / area_yard

theorem fraction_of_yard_occupied : 
  fraction_occupied = 0.2814 := 
sorry

end NUMINAMATH_GPT_fraction_of_yard_occupied_l2091_209166


namespace NUMINAMATH_GPT_card_draw_count_l2091_209150

theorem card_draw_count : 
  let total_cards := 12
  let red_cards := 3
  let yellow_cards := 3
  let blue_cards := 3
  let green_cards := 3
  let total_ways := Nat.choose total_cards 3
  let invalid_same_color := 4 * Nat.choose 3 3
  let invalid_two_red := Nat.choose red_cards 2 * Nat.choose (total_cards - red_cards) 1
  total_ways - invalid_same_color - invalid_two_red = 189 :=
by
  sorry

end NUMINAMATH_GPT_card_draw_count_l2091_209150


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2091_209196

theorem quadratic_inequality_solution (x : ℝ) : 2 * x^2 - 5 * x - 3 ≥ 0 ↔ x ≤ -1/2 ∨ x ≥ 3 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2091_209196


namespace NUMINAMATH_GPT_fourth_boy_payment_l2091_209119

theorem fourth_boy_payment (a b c d : ℝ) 
  (h₁ : a = (1 / 2) * (b + c + d)) 
  (h₂ : b = (1 / 3) * (a + c + d)) 
  (h₃ : c = (1 / 4) * (a + b + d)) 
  (h₄ : a + b + c + d = 60) : 
  d = 13 := 
sorry

end NUMINAMATH_GPT_fourth_boy_payment_l2091_209119


namespace NUMINAMATH_GPT_Marcus_walking_speed_l2091_209195

def bath_time : ℕ := 20  -- in minutes
def blow_dry_time : ℕ := bath_time / 2  -- in minutes
def trail_distance : ℝ := 3  -- in miles
def total_dog_time : ℕ := 60  -- in minutes

theorem Marcus_walking_speed :
  let walking_time := total_dog_time - (bath_time + blow_dry_time)
  let walking_time_hours := (walking_time:ℝ) / 60
  (trail_distance / walking_time_hours) = 6 := by
  sorry

end NUMINAMATH_GPT_Marcus_walking_speed_l2091_209195


namespace NUMINAMATH_GPT_calculate_expression_l2091_209153

theorem calculate_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) * (3 + 2) = X :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2091_209153


namespace NUMINAMATH_GPT_final_amount_correct_l2091_209171

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3
def shoes_cost : ℝ := wallet_cost + purse_cost + 7
def total_cost_before_discount : ℝ := wallet_cost + purse_cost + shoes_cost
def discount_rate : ℝ := 0.10
def discounted_amount : ℝ := total_cost_before_discount * discount_rate
def final_amount : ℝ := total_cost_before_discount - discounted_amount

theorem final_amount_correct :
  final_amount = 198.90 := by
  -- Here we would provide the proof of the theorem
  sorry

end NUMINAMATH_GPT_final_amount_correct_l2091_209171


namespace NUMINAMATH_GPT_tutors_next_together_l2091_209156

-- Define the conditions given in the problem
def Elisa_work_days := 5
def Frank_work_days := 6
def Giselle_work_days := 8
def Hector_work_days := 9

-- Theorem statement to prove the number of days until they all work together again
theorem tutors_next_together (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = Elisa_work_days) 
  (h2 : d2 = Frank_work_days) 
  (h3 : d3 = Giselle_work_days) 
  (h4 : d4 = Hector_work_days) : 
  Nat.lcm (Nat.lcm (Nat.lcm d1 d2) d3) d4 = 360 := 
by
  -- Translate the problem statement into Lean terms and structure
  sorry

end NUMINAMATH_GPT_tutors_next_together_l2091_209156


namespace NUMINAMATH_GPT_interval_between_births_l2091_209160

variables {A1 A2 A3 A4 A5 : ℝ}
variable {x : ℝ}

def ages (A1 A2 A3 A4 A5 : ℝ) := A1 + A2 + A3 + A4 + A5 = 50
def youngest (A1 : ℝ) := A1 = 4
def interval (x : ℝ) := x = 3.4

theorem interval_between_births
  (h_age_sum: ages A1 A2 A3 A4 A5)
  (h_youngest: youngest A1)
  (h_ages: A2 = A1 + x ∧ A3 = A1 + 2 * x ∧ A4 = A1 + 3 * x ∧ A5 = A1 + 4 * x) :
  interval x :=
by {
  sorry
}

end NUMINAMATH_GPT_interval_between_births_l2091_209160


namespace NUMINAMATH_GPT_find_constant_t_l2091_209111

theorem find_constant_t (a : ℕ → ℝ) (S : ℕ → ℝ) (t : ℝ) :
  (∀ n, S n = t + 5^n) ∧ (∀ n ≥ 2, a n = S n - S (n - 1)) ∧ (a 1 = S 1) ∧ 
  (∃ q, ∀ n ≥ 1, a (n + 1) = q * a n) → 
  t = -1 := by
  sorry

end NUMINAMATH_GPT_find_constant_t_l2091_209111


namespace NUMINAMATH_GPT_find_k_l2091_209145

noncomputable def curve_C (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1

noncomputable def line_eq (k x y : ℝ) : Prop :=
  y = k * x + 1

theorem find_k (k : ℝ) :
  (∃ A B : ℝ × ℝ, (curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧ line_eq k A.1 A.2 ∧ line_eq k B.1 B.2 ∧ 
   (A.1 * B.1 + A.2 * B.2 = 0))) ↔ (k = 1/2 ∨ k = -1/2) :=
sorry

end NUMINAMATH_GPT_find_k_l2091_209145


namespace NUMINAMATH_GPT_find_B_l2091_209146

theorem find_B (A B : Nat) (hA : A ≤ 9) (hB : B ≤ 9) (h_eq : 6 * A + 10 * B + 2 = 77) : B = 1 :=
by
-- proof steps would go here
sorry

end NUMINAMATH_GPT_find_B_l2091_209146


namespace NUMINAMATH_GPT_Jane_age_l2091_209175

theorem Jane_age (x : ℕ) 
  (h1 : ∃ n1 : ℕ, x - 1 = n1 ^ 2) 
  (h2 : ∃ n2 : ℕ, x + 1 = n2 ^ 3) : 
  x = 26 :=
sorry

end NUMINAMATH_GPT_Jane_age_l2091_209175


namespace NUMINAMATH_GPT_students_behind_yoongi_l2091_209116

theorem students_behind_yoongi (total_students jungkoo_position students_between_jungkook_yoongi : ℕ) 
    (h1 : total_students = 20)
    (h2 : jungkoo_position = 3)
    (h3 : students_between_jungkook_yoongi = 5) : 
    (total_students - (jungkoo_position + students_between_jungkook_yoongi + 1)) = 11 :=
by
  sorry

end NUMINAMATH_GPT_students_behind_yoongi_l2091_209116


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_eq_l2091_209128

theorem axis_of_symmetry_parabola_eq : ∀ (x y p : ℝ), 
  y = -2 * x^2 → 
  (x^2 = -2 * p * y) → 
  (p = 1/4) →  
  (y = p / 2) → 
  y = 1 / 8 := by 
  intros x y p h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_eq_l2091_209128


namespace NUMINAMATH_GPT_license_plates_count_l2091_209149

def number_of_license_plates : ℕ :=
  let digit_choices := 10^5
  let letter_block_choices := 3 * 26^2
  let block_positions := 6
  digit_choices * letter_block_choices * block_positions

theorem license_plates_count : number_of_license_plates = 1216800000 := by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_license_plates_count_l2091_209149


namespace NUMINAMATH_GPT_angle_sum_triangle_l2091_209103

theorem angle_sum_triangle (x : ℝ) 
  (h1 : 70 + 70 + x = 180) : 
  x = 40 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_triangle_l2091_209103


namespace NUMINAMATH_GPT_gifts_needed_l2091_209173

def num_teams : ℕ := 7
def num_gifts_per_team : ℕ := 2

theorem gifts_needed (h1 : num_teams = 7) (h2 : num_gifts_per_team = 2) : num_teams * num_gifts_per_team = 14 := 
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_gifts_needed_l2091_209173


namespace NUMINAMATH_GPT_hexagon_internal_angle_A_l2091_209108

theorem hexagon_internal_angle_A
  (B C D E F : ℝ) 
  (hB : B = 134) 
  (hC : C = 98) 
  (hD : D = 120) 
  (hE : E = 139) 
  (hF : F = 109) 
  (H : B + C + D + E + F + A = 720) : A = 120 := 
sorry

end NUMINAMATH_GPT_hexagon_internal_angle_A_l2091_209108


namespace NUMINAMATH_GPT_maximum_candies_after_20_hours_l2091_209168

-- Define a function to compute the sum of the digits
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

-- Define the recursive function to model the candy process
def candies_after_hours (n : ℕ) (hours : ℕ) : ℕ :=
  if hours = 0 then n 
  else candies_after_hours (n + sum_of_digits n) (hours - 1)

theorem maximum_candies_after_20_hours :
  candies_after_hours 1 20 = 148 :=
sorry

end NUMINAMATH_GPT_maximum_candies_after_20_hours_l2091_209168


namespace NUMINAMATH_GPT_determine_c_l2091_209182

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem determine_c (a b : ℝ) (m c : ℝ) 
  (h1 : ∀ x, 0 ≤ x → f x a b = x^2 + a * x + b)
  (h2 : ∃ m : ℝ, ∀ x : ℝ, f x a b < c ↔ m < x ∧ x < m + 6) :
  c = 9 :=
sorry

end NUMINAMATH_GPT_determine_c_l2091_209182


namespace NUMINAMATH_GPT_solution_set_of_quadratic_l2091_209184

theorem solution_set_of_quadratic (a b x : ℝ) (h1 : a = 5) (h2 : b = -6) :
  (2 ≤ x ∧ x ≤ 3) → (bx^2 - ax - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_l2091_209184


namespace NUMINAMATH_GPT_ellipse_slope_condition_l2091_209179

theorem ellipse_slope_condition (a b x y x₀ y₀ : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h_ellipse1 : x^2 / a^2 + y^2 / b^2 = 1) 
  (h_ellipse2 : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (hA : x ≠ x₀ ∨ y ≠ y₀) 
  (hB : x ≠ -x₀ ∨ y ≠ -y₀) :
  ((y - y₀) / (x - x₀)) * ((y + y₀) / (x + x₀)) = -b^2 / a^2 := 
sorry

end NUMINAMATH_GPT_ellipse_slope_condition_l2091_209179


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2091_209159

theorem sufficient_not_necessary (b c: ℝ) : (c < 0) → ∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2091_209159


namespace NUMINAMATH_GPT_k_not_possible_l2091_209170

theorem k_not_possible (S : ℕ → ℚ) (a b : ℕ → ℚ) (n k : ℕ) (k_gt_2 : k > 2) :
  (S n = (n^2 + n) / 2) →
  (a n = S n - S (n - 1)) →
  (b n = 1 / a n) →
  (2 * b (n + 2) = b n + b (n + k)) →
  k ≠ 4 ∧ k ≠ 10 :=
by
  -- Proof goes here (skipped)
  sorry

end NUMINAMATH_GPT_k_not_possible_l2091_209170


namespace NUMINAMATH_GPT_probability_three_heads_l2091_209135

noncomputable def fair_coin_flip: ℝ := 1 / 2

theorem probability_three_heads :
  (fair_coin_flip * fair_coin_flip * fair_coin_flip) = 1 / 8 :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_probability_three_heads_l2091_209135


namespace NUMINAMATH_GPT_original_volume_l2091_209100

variable {π : Real} (r h : Real)

theorem original_volume (hπ : π ≠ 0) (hr : r ≠ 0) (hh : h ≠ 0) (condition : 3 * π * r^2 * h = 180) : π * r^2 * h = 60 := by
  sorry

end NUMINAMATH_GPT_original_volume_l2091_209100


namespace NUMINAMATH_GPT_tower_count_l2091_209137

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binom (n k : Nat) : Nat :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def multinomialCoeff (n : Nat) (ks : List Nat) : Nat :=
  factorial n / List.foldr (fun k acc => acc * factorial k) 1 ks

theorem tower_count :
  let totalCubes := 9
  let usedCubes := 8
  let redCubes := 2
  let blueCubes := 3
  let greenCubes := 4
  multinomialCoeff totalCubes [redCubes, blueCubes, greenCubes] = 1260 :=
by
  sorry

end NUMINAMATH_GPT_tower_count_l2091_209137


namespace NUMINAMATH_GPT_compare_pow_value_l2091_209197

theorem compare_pow_value : 
  ∀ (x : ℝ) (n : ℕ), x = 0.01 → n = 1000 → (1 + x)^n > 1000 := 
by 
  intros x n hx hn
  rw [hx, hn]
  sorry

end NUMINAMATH_GPT_compare_pow_value_l2091_209197


namespace NUMINAMATH_GPT_corner_cells_different_colors_l2091_209105

theorem corner_cells_different_colors 
  (colors : Fin 4 → Prop)
  (painted : (Fin 100 × Fin 100) → Fin 4)
  (h : ∀ (i j : Fin 99), 
    ∃ f g h k, 
      f ≠ g ∧ f ≠ h ∧ f ≠ k ∧
      g ≠ h ∧ g ≠ k ∧ 
      h ≠ k ∧ 
      painted (i, j) = f ∧ 
      painted (i.succ, j) = g ∧ 
      painted (i, j.succ) = h ∧ 
      painted (i.succ, j.succ) = k) :
  painted (0, 0) ≠ painted (99, 0) ∧
  painted (0, 0) ≠ painted (0, 99) ∧
  painted (0, 0) ≠ painted (99, 99) ∧
  painted (99, 0) ≠ painted (0, 99) ∧
  painted (99, 0) ≠ painted (99, 99) ∧
  painted (0, 99) ≠ painted (99, 99) :=
  sorry

end NUMINAMATH_GPT_corner_cells_different_colors_l2091_209105


namespace NUMINAMATH_GPT_inequality_solution_range_of_a_l2091_209101

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

def range_y := Set.Icc (-2 : ℝ) 2

def subset_property (a : ℝ) : Prop := 
  Set.Icc a (2 * a - 1) ⊆ range_y

theorem inequality_solution (x : ℝ) :
  f x ≤ x^2 - 3 * x + 1 ↔ x ≤ 1 ∨ x ≥ 3 := sorry

theorem range_of_a (a : ℝ) :
  subset_property a ↔ 1 ≤ a ∧ a ≤ 3 / 2 := sorry

end NUMINAMATH_GPT_inequality_solution_range_of_a_l2091_209101


namespace NUMINAMATH_GPT_sandy_jacket_price_l2091_209139

noncomputable def discounted_shirt_price (initial_shirt_price discount_percentage : ℝ) : ℝ :=
  initial_shirt_price - (initial_shirt_price * discount_percentage / 100)

noncomputable def money_left (initial_money additional_money discounted_price : ℝ) : ℝ :=
  initial_money + additional_money - discounted_price

noncomputable def jacket_price_before_tax (remaining_money tax_percentage : ℝ) : ℝ :=
  remaining_money / (1 + tax_percentage / 100)

theorem sandy_jacket_price :
  let initial_money := 13.99
  let initial_shirt_price := 12.14
  let discount_percentage := 5.0
  let additional_money := 7.43
  let tax_percentage := 10.0
  
  let discounted_price := discounted_shirt_price initial_shirt_price discount_percentage
  let remaining_money := money_left initial_money additional_money discounted_price
  
  jacket_price_before_tax remaining_money tax_percentage = 8.99 := sorry

end NUMINAMATH_GPT_sandy_jacket_price_l2091_209139


namespace NUMINAMATH_GPT_find_three_fifths_of_neg_twelve_sevenths_l2091_209187

def a : ℚ := -12 / 7
def b : ℚ := 3 / 5
def c : ℚ := -36 / 35

theorem find_three_fifths_of_neg_twelve_sevenths : b * a = c := by 
  -- sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_find_three_fifths_of_neg_twelve_sevenths_l2091_209187


namespace NUMINAMATH_GPT_Margie_distance_on_25_dollars_l2091_209123

theorem Margie_distance_on_25_dollars
  (miles_per_gallon : ℝ)
  (cost_per_gallon : ℝ)
  (amount_spent : ℝ) :
  miles_per_gallon = 40 →
  cost_per_gallon = 5 →
  amount_spent = 25 →
  (amount_spent / cost_per_gallon) * miles_per_gallon = 200 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_Margie_distance_on_25_dollars_l2091_209123


namespace NUMINAMATH_GPT_divisible_by_two_of_square_l2091_209144

theorem divisible_by_two_of_square {a : ℤ} (h : 2 ∣ a^2) : 2 ∣ a :=
sorry

end NUMINAMATH_GPT_divisible_by_two_of_square_l2091_209144


namespace NUMINAMATH_GPT_remaining_pencils_total_l2091_209151

-- Definitions corresponding to the conditions:
def J : ℝ := 300
def J_d : ℝ := 0.30 * J
def J_r : ℝ := J - J_d

def V : ℝ := 2 * J
def V_d : ℝ := 125
def V_r : ℝ := V - V_d

def S : ℝ := 450
def S_d : ℝ := 0.60 * S
def S_r : ℝ := S - S_d

-- Proving the remaining pencils add up to the required amount:
theorem remaining_pencils_total : J_r + V_r + S_r = 865 := by
  sorry

end NUMINAMATH_GPT_remaining_pencils_total_l2091_209151


namespace NUMINAMATH_GPT_optionD_is_quadratic_l2091_209114

variable (x : ℝ)

-- Original equation in Option D
def optionDOriginal := (x^2 + 2 * x = 2 * x^2 - 1)

-- Rearranged form of Option D's equation
def optionDRearranged := (-x^2 + 2 * x + 1 = 0)

theorem optionD_is_quadratic : optionDOriginal x → optionDRearranged x :=
by
  intro h
  -- The proof steps would go here, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_optionD_is_quadratic_l2091_209114


namespace NUMINAMATH_GPT_integer_satisfies_mod_and_range_l2091_209141

theorem integer_satisfies_mod_and_range :
  ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-150 ≡ n [ZMOD 25]) → n = 0 :=
by
  sorry

end NUMINAMATH_GPT_integer_satisfies_mod_and_range_l2091_209141


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2091_209154

theorem solution_set_of_inequality : {x : ℝ | x^2 + x - 2 < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2091_209154


namespace NUMINAMATH_GPT_smallest_x_abs_eq_18_l2091_209161

theorem smallest_x_abs_eq_18 : 
  ∃ x : ℝ, (|2 * x + 5| = 18) ∧ (∀ y : ℝ, (|2 * y + 5| = 18) → x ≤ y) :=
sorry

end NUMINAMATH_GPT_smallest_x_abs_eq_18_l2091_209161


namespace NUMINAMATH_GPT_no_convex_27gon_with_distinct_integer_angles_l2091_209143

noncomputable def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

def is_convex (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i, angles i < 180

def all_distinct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i j, i ≠ j → angles i ≠ angles j

def sum_is_correct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  Finset.sum (Finset.univ : Finset (Fin n)) angles = sum_of_interior_angles n

theorem no_convex_27gon_with_distinct_integer_angles :
  ¬ ∃ (angles : Fin 27 → ℕ), is_convex 27 angles ∧ all_distinct 27 angles ∧ sum_is_correct 27 angles :=
by
  sorry

end NUMINAMATH_GPT_no_convex_27gon_with_distinct_integer_angles_l2091_209143


namespace NUMINAMATH_GPT_expand_polynomial_l2091_209127

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l2091_209127


namespace NUMINAMATH_GPT_rectangle_area_error_percentage_l2091_209186

theorem rectangle_area_error_percentage (L W : ℝ) :
  let L' := 1.10 * L
  let W' := 0.95 * W
  let A := L * W 
  let A' := L' * W'
  let error := A' - A
  let error_percentage := (error / A) * 100
  error_percentage = 4.5 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_error_percentage_l2091_209186


namespace NUMINAMATH_GPT_seventh_triangular_number_is_28_l2091_209157

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem seventh_triangular_number_is_28 : triangular_number 7 = 28 :=
by
  /- proof goes here -/
  sorry

end NUMINAMATH_GPT_seventh_triangular_number_is_28_l2091_209157


namespace NUMINAMATH_GPT_factorization_of_x10_minus_1024_l2091_209178

theorem factorization_of_x10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x - 2) * (x^4 + 2 * x^3 + 4 * x^2 + 8 * x + 16) :=
by sorry

end NUMINAMATH_GPT_factorization_of_x10_minus_1024_l2091_209178


namespace NUMINAMATH_GPT_pizza_area_percentage_increase_l2091_209132

theorem pizza_area_percentage_increase :
  let r1 := 6
  let r2 := 4
  let A1 := Real.pi * r1^2
  let A2 := Real.pi * r2^2
  let deltaA := A1 - A2
  let N := (deltaA / A2) * 100
  N = 125 := by
  sorry

end NUMINAMATH_GPT_pizza_area_percentage_increase_l2091_209132


namespace NUMINAMATH_GPT_honda_day_shift_production_l2091_209193

theorem honda_day_shift_production (S : ℕ) (day_shift_production : ℕ)
  (h1 : day_shift_production = 4 * S)
  (h2 : day_shift_production + S = 5500) :
  day_shift_production = 4400 :=
sorry

end NUMINAMATH_GPT_honda_day_shift_production_l2091_209193
