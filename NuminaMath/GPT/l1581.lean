import Mathlib

namespace NUMINAMATH_GPT_team_A_minimum_workers_l1581_158187

-- Define the variables and conditions for the problem.
variables (A B c : ℕ)

-- Condition 1: If team A lends 90 workers to team B, Team B will have twice as many workers as Team A.
def condition1 : Prop :=
  2 * (A - 90) = B + 90

-- Condition 2: If team B lends c workers to team A, Team A will have six times as many workers as Team B.
def condition2 : Prop :=
  A + c = 6 * (B - c)

-- Define the proof goal.
theorem team_A_minimum_workers (h1 : condition1 A B) (h2 : condition2 A B c) : 
  153 ≤ A :=
sorry

end NUMINAMATH_GPT_team_A_minimum_workers_l1581_158187


namespace NUMINAMATH_GPT_LeonaEarnsGivenHourlyRate_l1581_158133

theorem LeonaEarnsGivenHourlyRate :
  (∀ (c: ℝ) (t h e: ℝ), 
    (c = 24.75) → 
    (t = 3) → 
    (h = c / t) → 
    (e = h * 5) →
    e = 41.25) :=
by
  intros c t h e h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_LeonaEarnsGivenHourlyRate_l1581_158133


namespace NUMINAMATH_GPT_utility_bills_total_correct_l1581_158189

-- Define the number and values of the bills
def fifty_dollar_bills : Nat := 3
def ten_dollar_bills : Nat := 2
def value_fifty_dollar_bill : Nat := 50
def value_ten_dollar_bill : Nat := 10

-- Define the total amount due to utility bills based on the given conditions
def total_utility_bills : Nat :=
  fifty_dollar_bills * value_fifty_dollar_bill + ten_dollar_bills * value_ten_dollar_bill

theorem utility_bills_total_correct : total_utility_bills = 170 := by
  sorry -- detailed proof skipped


end NUMINAMATH_GPT_utility_bills_total_correct_l1581_158189


namespace NUMINAMATH_GPT_total_silk_dyed_correct_l1581_158120

-- Define the conditions
def green_silk_yards : ℕ := 61921
def pink_silk_yards : ℕ := 49500
def total_silk_yards : ℕ := green_silk_yards + pink_silk_yards

-- State the theorem to be proved
theorem total_silk_dyed_correct : total_silk_yards = 111421 := by
  sorry

end NUMINAMATH_GPT_total_silk_dyed_correct_l1581_158120


namespace NUMINAMATH_GPT_selling_price_l1581_158146

noncomputable def total_cost_first_mixture : ℝ := 27 * 150
noncomputable def total_cost_second_mixture : ℝ := 36 * 125
noncomputable def total_cost_third_mixture : ℝ := 18 * 175
noncomputable def total_cost_fourth_mixture : ℝ := 24 * 120

noncomputable def total_cost : ℝ := total_cost_first_mixture + total_cost_second_mixture + total_cost_third_mixture + total_cost_fourth_mixture

noncomputable def profit_first_mixture : ℝ := 0.4 * total_cost_first_mixture
noncomputable def profit_second_mixture : ℝ := 0.3 * total_cost_second_mixture
noncomputable def profit_third_mixture : ℝ := 0.2 * total_cost_third_mixture
noncomputable def profit_fourth_mixture : ℝ := 0.25 * total_cost_fourth_mixture

noncomputable def total_profit : ℝ := profit_first_mixture + profit_second_mixture + profit_third_mixture + profit_fourth_mixture

noncomputable def total_weight : ℝ := 27 + 36 + 18 + 24
noncomputable def total_selling_price : ℝ := total_cost + total_profit

noncomputable def selling_price_per_kg : ℝ := total_selling_price / total_weight

theorem selling_price : selling_price_per_kg = 180 := by
  sorry

end NUMINAMATH_GPT_selling_price_l1581_158146


namespace NUMINAMATH_GPT_simplify_expression_l1581_158141

theorem simplify_expression (x : ℝ) :
  (2 * x + 30) + (150 * x + 45) + 5 = 152 * x + 80 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1581_158141


namespace NUMINAMATH_GPT_volleyball_teams_l1581_158197

theorem volleyball_teams (managers employees teams : ℕ) (h1 : managers = 3) (h2 : employees = 3) (h3 : teams = 3) :
  ((managers + employees) / teams) = 2 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_teams_l1581_158197


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_shifted_roots_l1581_158190

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) (h : ∀ x, x^3 - x + 2 = 0 → x = a ∨ x = b ∨ x = c) :
  1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = 11 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_shifted_roots_l1581_158190


namespace NUMINAMATH_GPT_total_towels_l1581_158116

theorem total_towels (packs : ℕ) (towels_per_pack : ℕ) (h1 : packs = 9) (h2 : towels_per_pack = 3) : packs * towels_per_pack = 27 := by
  sorry

end NUMINAMATH_GPT_total_towels_l1581_158116


namespace NUMINAMATH_GPT_jumping_contest_l1581_158163

variables (G F M K : ℤ)

-- Define the conditions
def condition_1 : Prop := G = 39
def condition_2 : Prop := G = F + 19
def condition_3 : Prop := M = F - 12
def condition_4 : Prop := K = 2 * F - 5

-- The theorem asserting the final distances
theorem jumping_contest 
    (h1 : condition_1 G)
    (h2 : condition_2 G F)
    (h3 : condition_3 F M)
    (h4 : condition_4 F K) :
    G = 39 ∧ F = 20 ∧ M = 8 ∧ K = 35 := by
  sorry

end NUMINAMATH_GPT_jumping_contest_l1581_158163


namespace NUMINAMATH_GPT_rebecca_eggs_l1581_158108

/-- Rebecca wants to split a collection of eggs into 4 groups. Each group will have 2 eggs. -/
def number_of_groups : Nat := 4

def eggs_per_group : Nat := 2

theorem rebecca_eggs : (number_of_groups * eggs_per_group) = 8 := by
  sorry

end NUMINAMATH_GPT_rebecca_eggs_l1581_158108


namespace NUMINAMATH_GPT_find_height_of_cylinder_l1581_158114

theorem find_height_of_cylinder (h r : ℝ) (π : ℝ) (SA : ℝ) (r_val : r = 3) (SA_val : SA = 36 * π) 
  (SA_formula : SA = 2 * π * r^2 + 2 * π * r * h) : h = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_height_of_cylinder_l1581_158114


namespace NUMINAMATH_GPT_value_of_a2022_l1581_158191

theorem value_of_a2022 (a : ℕ → ℤ) (h : ∀ (n k : ℕ), 1 ≤ n ∧ n ≤ 2022 ∧ 1 ≤ k ∧ k ≤ 2022 → a n - a k ≥ (n^3 : ℤ) - (k^3 : ℤ)) (ha1011 : a 1011 = 0) : 
  a 2022 = 7246031367 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a2022_l1581_158191


namespace NUMINAMATH_GPT_ratio_areas_l1581_158192

-- Define the perimeter P
variable (P : ℝ) (hP : P > 0)

-- Define the side lengths
noncomputable def side_length_square := P / 4
noncomputable def side_length_triangle := P / 3

-- Define the radius of the circumscribed circle for the square
noncomputable def radius_square := (P * Real.sqrt 2) / 8
-- Define the area of the circumscribed circle for the square
noncomputable def area_circle_square := Real.pi * (radius_square P)^2

-- Define the radius of the circumscribed circle for the equilateral triangle
noncomputable def radius_triangle := (P * Real.sqrt 3) / 9 
-- Define the area of the circumscribed circle for the equilateral triangle
noncomputable def area_circle_triangle := Real.pi * (radius_triangle P)^2

-- Prove the ratio of the areas is 27/32
theorem ratio_areas (P : ℝ) (hP : P > 0) : 
  (area_circle_square P / area_circle_triangle P) = (27 / 32) := by
  sorry

end NUMINAMATH_GPT_ratio_areas_l1581_158192


namespace NUMINAMATH_GPT_fraction_problem_l1581_158194

theorem fraction_problem (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (2 * a - b) / (a + 4 * b) = 3) : 
  (a - 4 * b) / (2 * a + b) = 17 / 25 :=
by sorry

end NUMINAMATH_GPT_fraction_problem_l1581_158194


namespace NUMINAMATH_GPT_droid_weekly_coffee_consumption_l1581_158183

noncomputable def weekly_consumption_A : ℕ :=
  (3 * 5) + 4 + 2 + 1 -- Weekdays + Saturday + Sunday + Monday increase

noncomputable def weekly_consumption_B : ℕ :=
  (2 * 5) + 3 + (1 - 1 / 2) -- Weekdays + Saturday + Sunday decrease

noncomputable def weekly_consumption_C : ℕ :=
  (1 * 5) + 2 + 1 -- Weekdays + Saturday + Sunday

theorem droid_weekly_coffee_consumption :
  weekly_consumption_A = 22 ∧ weekly_consumption_B = 14 ∧ weekly_consumption_C = 8 :=
by 
  sorry

end NUMINAMATH_GPT_droid_weekly_coffee_consumption_l1581_158183


namespace NUMINAMATH_GPT_fraction_weevils_25_percent_l1581_158101

-- Define the probabilities
def prob_good_milk : ℝ := 0.8
def prob_good_egg : ℝ := 0.4
def prob_all_good : ℝ := 0.24

-- The problem definition and statement
def fraction_weevils (F : ℝ) : Prop :=
  0.32 * (1 - F) = 0.24

theorem fraction_weevils_25_percent : fraction_weevils 0.25 :=
by sorry

end NUMINAMATH_GPT_fraction_weevils_25_percent_l1581_158101


namespace NUMINAMATH_GPT_f_increasing_on_Ioo_l1581_158165

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_increasing_on_Ioo : ∀ x y : ℝ, x < y → f x < f y :=
sorry

end NUMINAMATH_GPT_f_increasing_on_Ioo_l1581_158165


namespace NUMINAMATH_GPT_simplify_to_ap_minus_b_l1581_158144

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((7*p + 3) - 3*p * 2) * 4 + (5 - 2 / 4) * (8*p - 12)

theorem simplify_to_ap_minus_b (p : ℝ) :
  simplify_expression p = 40 * p - 42 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_simplify_to_ap_minus_b_l1581_158144


namespace NUMINAMATH_GPT_find_AB_l1581_158154

theorem find_AB 
  (A B C Q N : Point)
  (h_AQ_QC : AQ / QC = 5 / 2)
  (h_CN_NB : CN / NB = 5 / 2)
  (h_QN : QN = 5 * Real.sqrt 2) : 
  AB = 7 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_find_AB_l1581_158154


namespace NUMINAMATH_GPT_zero_a_if_square_every_n_l1581_158105

theorem zero_a_if_square_every_n (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 := 
sorry

end NUMINAMATH_GPT_zero_a_if_square_every_n_l1581_158105


namespace NUMINAMATH_GPT_minimum_cuts_for_48_rectangles_l1581_158126

theorem minimum_cuts_for_48_rectangles : 
  ∃ n : ℕ, n = 6 ∧ (∀ m < 6, 2 ^ m < 48) ∧ 2 ^ n ≥ 48 :=
by
  sorry

end NUMINAMATH_GPT_minimum_cuts_for_48_rectangles_l1581_158126


namespace NUMINAMATH_GPT_neq_zero_necessary_not_sufficient_l1581_158182

theorem neq_zero_necessary_not_sufficient (x : ℝ) (h : x ≠ 0) : 
  (¬ (x = 0) ↔ x > 0) ∧ ¬ (x > 0 → x ≠ 0) :=
by sorry

end NUMINAMATH_GPT_neq_zero_necessary_not_sufficient_l1581_158182


namespace NUMINAMATH_GPT_difference_of_cubes_not_divisible_by_19_l1581_158185

theorem difference_of_cubes_not_divisible_by_19 (a b : ℤ) : 
  ¬ (19 ∣ ((3 * a + 2) ^ 3 - (3 * b + 2) ^ 3)) := by
  sorry

end NUMINAMATH_GPT_difference_of_cubes_not_divisible_by_19_l1581_158185


namespace NUMINAMATH_GPT_students_taking_both_languages_l1581_158151

theorem students_taking_both_languages (F S B : ℕ) (hF : F = 21) (hS : S = 21) (h30 : 30 = F - B + (S - B)) : B = 6 :=
by
  rw [hF, hS] at h30
  sorry

end NUMINAMATH_GPT_students_taking_both_languages_l1581_158151


namespace NUMINAMATH_GPT_price_of_other_pieces_l1581_158106

theorem price_of_other_pieces (total_spent : ℕ) (total_pieces : ℕ) (price_piece1 : ℕ) (price_piece2 : ℕ) 
  (remaining_pieces : ℕ) (price_remaining_piece : ℕ) (h1 : total_spent = 610) (h2 : total_pieces = 7)
  (h3 : price_piece1 = 49) (h4 : price_piece2 = 81) (h5 : remaining_pieces = (total_pieces - 2))
  (h6 : total_spent - price_piece1 - price_piece2 = remaining_pieces * price_remaining_piece) :
  price_remaining_piece = 96 := 
by
  sorry

end NUMINAMATH_GPT_price_of_other_pieces_l1581_158106


namespace NUMINAMATH_GPT_min_buses_l1581_158148

theorem min_buses (n : ℕ) : (47 * n >= 625) → (n = 14) :=
by {
  -- Proof is omitted since the problem only asks for the Lean statement, not the solution steps.
  sorry
}

end NUMINAMATH_GPT_min_buses_l1581_158148


namespace NUMINAMATH_GPT_temperature_problem_l1581_158195

theorem temperature_problem (N : ℤ) (P : ℤ) (D : ℤ) (D_3_pm : ℤ) (P_3_pm : ℤ) :
  D = P + N →
  D_3_pm = D - 8 →
  P_3_pm = P + 9 →
  |D_3_pm - P_3_pm| = 1 →
  (N = 18 ∨ N = 16) →
  18 * 16 = 288 :=
by
  sorry

end NUMINAMATH_GPT_temperature_problem_l1581_158195


namespace NUMINAMATH_GPT_larger_angle_measure_l1581_158162

theorem larger_angle_measure (x : ℝ) (h : 4 * x + 5 * x = 180) : 5 * x = 100 :=
by
  sorry

end NUMINAMATH_GPT_larger_angle_measure_l1581_158162


namespace NUMINAMATH_GPT_charles_total_earnings_l1581_158117

def charles_earnings (house_rate dog_rate : ℝ) (house_hours dog_count dog_hours : ℝ) : ℝ :=
  (house_rate * house_hours) + (dog_rate * dog_count * dog_hours)

theorem charles_total_earnings :
  charles_earnings 15 22 10 3 1 = 216 := by
  sorry

end NUMINAMATH_GPT_charles_total_earnings_l1581_158117


namespace NUMINAMATH_GPT_compare_abc_l1581_158132

noncomputable def a : ℝ := (1 / 2) * Real.cos (4 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (2 * 13 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (2 * 23 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c := by
  sorry

end NUMINAMATH_GPT_compare_abc_l1581_158132


namespace NUMINAMATH_GPT_abs_eq_5_iff_l1581_158111

theorem abs_eq_5_iff (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by
  sorry

end NUMINAMATH_GPT_abs_eq_5_iff_l1581_158111


namespace NUMINAMATH_GPT_bc_sum_l1581_158127

theorem bc_sum (A B C : ℝ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 10) : B + C = 310 := by
  sorry

end NUMINAMATH_GPT_bc_sum_l1581_158127


namespace NUMINAMATH_GPT_cost_equivalence_min_sets_of_A_l1581_158130

noncomputable def cost_of_B := 120
noncomputable def cost_of_A := cost_of_B + 30

theorem cost_equivalence (x : ℕ) :
  (1200 / (x + 30) = 960 / x) → x = 120 :=
by
  sorry

theorem min_sets_of_A :
  ∀ m : ℕ, (150 * m + 120 * (20 - m) ≥ 2800) ↔ m ≥ 14 :=
by
  sorry

end NUMINAMATH_GPT_cost_equivalence_min_sets_of_A_l1581_158130


namespace NUMINAMATH_GPT_sin_alpha_at_point_l1581_158196

open Real

theorem sin_alpha_at_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (1, -2)) :
  sin α = -2 * sqrt 5 / 5 :=
sorry

end NUMINAMATH_GPT_sin_alpha_at_point_l1581_158196


namespace NUMINAMATH_GPT_equivalent_forms_l1581_158125

-- Given line equation
def given_line_eq (x y : ℝ) : Prop :=
  (3 * x - 2) / 4 - (2 * y - 1) / 2 = 1

-- General form of the line
def general_form (x y : ℝ) : Prop :=
  3 * x - 8 * y - 2 = 0

-- Slope-intercept form of the line
def slope_intercept_form (x y : ℝ) : Prop := 
  y = (3 / 8) * x - 1 / 4

-- Intercept form of the line
def intercept_form (x y : ℝ) : Prop :=
  x / (2 / 3) + y / (-1 / 4) = 1

-- Normal form of the line
def normal_form (x y : ℝ) : Prop :=
  3 / Real.sqrt 73 * x - 8 / Real.sqrt 73 * y - 2 / Real.sqrt 73 = 0

-- Proof problem: Prove that the given line equation is equivalent to the derived forms
theorem equivalent_forms (x y : ℝ) :
  given_line_eq x y ↔ (general_form x y ∧ slope_intercept_form x y ∧ intercept_form x y ∧ normal_form x y) :=
sorry

end NUMINAMATH_GPT_equivalent_forms_l1581_158125


namespace NUMINAMATH_GPT_combine_fraction_l1581_158142

variable (d : ℤ)

theorem combine_fraction : (3 + 4 * d) / 5 + 3 = (18 + 4 * d) / 5 := by
  sorry

end NUMINAMATH_GPT_combine_fraction_l1581_158142


namespace NUMINAMATH_GPT_triangle_third_side_count_l1581_158149

theorem triangle_third_side_count : 
  ∀ (x : ℕ), (3 < x ∧ x < 19) → ∃ (n : ℕ), n = 15 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_third_side_count_l1581_158149


namespace NUMINAMATH_GPT_quadrilateral_condition_l1581_158171

variable (a b c d : ℝ)

theorem quadrilateral_condition (h1 : a + b + c + d = 2) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) :
  a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ a + b + c > 1 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_condition_l1581_158171


namespace NUMINAMATH_GPT_smaller_number_l1581_158136

theorem smaller_number (x y : ℝ) (h1 : x - y = 1650) (h2 : 0.075 * x = 0.125 * y) : y = 2475 := 
sorry

end NUMINAMATH_GPT_smaller_number_l1581_158136


namespace NUMINAMATH_GPT_geometric_sequence_term_302_l1581_158129

def geometric_sequence (a r : ℤ) (n : ℕ) : ℤ := a * r ^ (n - 1)

theorem geometric_sequence_term_302 :
  let a := 8
  let r := -2
  geometric_sequence a r 302 = -2^304 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_302_l1581_158129


namespace NUMINAMATH_GPT_ratio_of_down_payment_l1581_158160

theorem ratio_of_down_payment (C D : ℕ) (daily_min : ℕ) (days : ℕ) (balance : ℕ) (total_cost : ℕ) 
  (h1 : total_cost = 120)
  (h2 : daily_min = 6)
  (h3 : days = 10)
  (h4 : balance = daily_min * days) 
  (h5 : D + balance = total_cost) : 
  D / total_cost = 1 / 2 := 
  by
  sorry

end NUMINAMATH_GPT_ratio_of_down_payment_l1581_158160


namespace NUMINAMATH_GPT_arithmetic_mean_midpoint_l1581_158188

theorem arithmetic_mean_midpoint (a b : ℝ) : ∃ m : ℝ, m = (a + b) / 2 ∧ m = a + (b - a) / 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_midpoint_l1581_158188


namespace NUMINAMATH_GPT_cos_theta_seven_l1581_158186

theorem cos_theta_seven {θ : ℝ} (h : Real.cos θ = 1 / 4) : Real.cos (7 * θ) = -8383 / 98304 :=
by
  sorry

end NUMINAMATH_GPT_cos_theta_seven_l1581_158186


namespace NUMINAMATH_GPT_proof_problem_l1581_158139

def f (a b c : ℕ) : ℕ :=
  a * 100 + b * 10 + c

def special_op (a b c : ℕ) : ℕ :=
  f (a * b) (b * c / 10) (b * c % 10)

theorem proof_problem :
  special_op 5 7 4 - special_op 7 4 5 = 708 := 
    sorry

end NUMINAMATH_GPT_proof_problem_l1581_158139


namespace NUMINAMATH_GPT_min_value_f_l1581_158164

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^9 + 2

theorem min_value_f (a b : ℝ) (h : ∀ x : ℝ, 0 < x → f a b x ≤ 5) : 
  ∀ x : ℝ, x < 0 → f a b x ≥ -1 :=
by
  -- Since this is a statement-only problem, we leave the proof to be filled in
  sorry

end NUMINAMATH_GPT_min_value_f_l1581_158164


namespace NUMINAMATH_GPT_num_ordered_triples_l1581_158181

theorem num_ordered_triples : 
  {n : ℕ // ∃ (a b c : ℤ), 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ a * b * c = (2 * (a * b + b * c + c * a)) / 3 ∧ n = 3} :=
sorry

end NUMINAMATH_GPT_num_ordered_triples_l1581_158181


namespace NUMINAMATH_GPT_sale_in_second_month_l1581_158193

theorem sale_in_second_month 
  (m1 m2 m3 m4 m5 m6 : ℕ) 
  (h1: m1 = 6335) 
  (h2: m3 = 6855) 
  (h3: m4 = 7230) 
  (h4: m5 = 6562) 
  (h5: m6 = 5091)
  (average: (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 6500) : 
  m2 = 6927 :=
sorry

end NUMINAMATH_GPT_sale_in_second_month_l1581_158193


namespace NUMINAMATH_GPT_average_a_b_l1581_158121

theorem average_a_b (a b : ℝ) (h : (4 + 6 + 8 + a + b) / 5 = 20) : (a + b) / 2 = 41 :=
by
  sorry

end NUMINAMATH_GPT_average_a_b_l1581_158121


namespace NUMINAMATH_GPT_price_of_when_you_rescind_cd_l1581_158122

variable (W : ℕ) -- Defining W as a natural number since prices can't be negative

theorem price_of_when_you_rescind_cd
  (price_life_journey : ℕ := 100)
  (price_day_life : ℕ := 50)
  (num_cds_each : ℕ := 3)
  (total_spent : ℕ := 705) :
  3 * price_life_journey + 3 * price_day_life + 3 * W = total_spent → 
  W = 85 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_price_of_when_you_rescind_cd_l1581_158122


namespace NUMINAMATH_GPT_winter_expenditure_l1581_158198

theorem winter_expenditure (exp_end_nov : Real) (exp_end_feb : Real) 
  (h_nov : exp_end_nov = 3.0) (h_feb : exp_end_feb = 5.5) : 
  (exp_end_feb - exp_end_nov) = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_winter_expenditure_l1581_158198


namespace NUMINAMATH_GPT_solve_equation_l1581_158172

theorem solve_equation :
  ∀ x : ℝ, 
    (8 / (Real.sqrt (x - 10) - 10) + 
     2 / (Real.sqrt (x - 10) - 5) + 
     9 / (Real.sqrt (x - 10) + 5) + 
     16 / (Real.sqrt (x - 10) + 10) = 0)
    ↔ 
    x = 1841 / 121 ∨ x = 190 / 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1581_158172


namespace NUMINAMATH_GPT_number_of_bird_cages_l1581_158104

-- Definitions for the problem conditions
def birds_per_cage : ℕ := 2 + 7
def total_birds : ℕ := 72

-- The theorem to prove the number of bird cages is 8
theorem number_of_bird_cages : total_birds / birds_per_cage = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_bird_cages_l1581_158104


namespace NUMINAMATH_GPT_polar_circle_l1581_158107

def is_circle (ρ θ : ℝ) : Prop :=
  ρ = Real.cos (Real.pi / 4 - θ)

theorem polar_circle : 
  ∀ ρ θ : ℝ, is_circle ρ θ ↔ ∃ (x y : ℝ), (x - 1/(2 * Real.sqrt 2))^2 + (y - 1/(2 * Real.sqrt 2))^2 = (1/(2 * Real.sqrt 2))^2 :=
by
  intro ρ θ
  sorry

end NUMINAMATH_GPT_polar_circle_l1581_158107


namespace NUMINAMATH_GPT_find_k_l1581_158118

-- Definitions based on given conditions
def ellipse_equation (x y : ℝ) (k : ℝ) : Prop :=
  5 * x^2 + k * y^2 = 5

def is_focus (x y : ℝ) : Prop :=
  x = 0 ∧ y = 2

-- Statement of the problem
theorem find_k (k : ℝ) :
  (∀ x y, ellipse_equation x y k) →
  is_focus 0 2 →
  k = 1 :=
sorry

end NUMINAMATH_GPT_find_k_l1581_158118


namespace NUMINAMATH_GPT_pitcher_fill_four_glasses_l1581_158184

variable (P G : ℚ) -- P: Volume of pitcher, G: Volume of one glass
variable (h : P / 2 = 3 * G)

theorem pitcher_fill_four_glasses : (4 * G = 2 * P / 3) :=
by
  sorry

end NUMINAMATH_GPT_pitcher_fill_four_glasses_l1581_158184


namespace NUMINAMATH_GPT_smallest_d_in_range_l1581_158128

theorem smallest_d_in_range (d : ℝ) : (∃ x : ℝ, x^2 + 5 * x + d = 5) ↔ d ≤ 45 / 4 := 
sorry

end NUMINAMATH_GPT_smallest_d_in_range_l1581_158128


namespace NUMINAMATH_GPT_remainder_division_l1581_158131

theorem remainder_division (n : ℕ) :
  n = 2345678901 →
  n % 102 = 65 :=
by sorry

end NUMINAMATH_GPT_remainder_division_l1581_158131


namespace NUMINAMATH_GPT_acres_used_for_corn_l1581_158124

noncomputable def total_acres : ℝ := 1634
noncomputable def beans_ratio : ℝ := 4.5
noncomputable def wheat_ratio : ℝ := 2.3
noncomputable def corn_ratio : ℝ := 3.8
noncomputable def barley_ratio : ℝ := 3.4

noncomputable def total_parts : ℝ := beans_ratio + wheat_ratio + corn_ratio + barley_ratio
noncomputable def acres_per_part : ℝ := total_acres / total_parts
noncomputable def corn_acres : ℝ := corn_ratio * acres_per_part

theorem acres_used_for_corn :
  corn_acres = 443.51 := by
  sorry

end NUMINAMATH_GPT_acres_used_for_corn_l1581_158124


namespace NUMINAMATH_GPT_distances_inequality_l1581_158159

theorem distances_inequality (x y z : ℝ) (h : x + y + z = 1): x^2 + y^2 + z^2 ≥ x^3 + y^3 + z^3 + 6 * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_distances_inequality_l1581_158159


namespace NUMINAMATH_GPT_aquariums_have_13_saltwater_animals_l1581_158156

theorem aquariums_have_13_saltwater_animals:
  ∀ x : ℕ, 26 * x = 52 → (∀ n : ℕ, n = 26 → (n * x = 52 ∧ x % 2 = 1 ∧ x > 1)) → x = 13 :=
by
  sorry

end NUMINAMATH_GPT_aquariums_have_13_saltwater_animals_l1581_158156


namespace NUMINAMATH_GPT_correct_average_of_20_numbers_l1581_158102

theorem correct_average_of_20_numbers 
  (incorrect_avg : ℕ) 
  (n : ℕ) 
  (incorrectly_read : ℕ) 
  (correction : ℕ) 
  (a b c d e f g h i j : ℤ) 
  (sum_a_b_c_d_e : ℤ)
  (sum_f_g_h_i_j : ℤ)
  (incorrect_sum : ℤ)
  (correction_sum : ℤ) 
  (corrected_sum : ℤ)
  (correct_avg : ℤ) : 
  incorrect_avg = 35 ∧ 
  n = 20 ∧ 
  incorrectly_read = 5 ∧ 
  correction = 136 ∧ 
  a = 90 ∧ b = 73 ∧ c = 85 ∧ d = -45 ∧ e = 64 ∧ 
  f = 45 ∧ g = 36 ∧ h = 42 ∧ i = -27 ∧ j = 35 ∧ 
  sum_a_b_c_d_e = a + b + c + d + e ∧
  sum_f_g_h_i_j = f + g + h + i + j ∧
  incorrect_sum = incorrect_avg * n ∧ 
  correction_sum = sum_a_b_c_d_e - sum_f_g_h_i_j ∧ 
  corrected_sum = incorrect_sum + correction_sum → correct_avg = corrected_sum / n := 
  by sorry

end NUMINAMATH_GPT_correct_average_of_20_numbers_l1581_158102


namespace NUMINAMATH_GPT_total_cost_food_l1581_158158

theorem total_cost_food
  (beef_pounds : ℕ)
  (beef_cost_per_pound : ℕ)
  (chicken_pounds : ℕ)
  (chicken_cost_per_pound : ℕ)
  (h_beef : beef_pounds = 1000)
  (h_beef_cost : beef_cost_per_pound = 8)
  (h_chicken : chicken_pounds = 2 * beef_pounds)
  (h_chicken_cost : chicken_cost_per_pound = 3) :
  (beef_pounds * beef_cost_per_pound + chicken_pounds * chicken_cost_per_pound = 14000) :=
by
  sorry

end NUMINAMATH_GPT_total_cost_food_l1581_158158


namespace NUMINAMATH_GPT_minimum_x2_y2_z2_l1581_158109

theorem minimum_x2_y2_z2 :
  ∀ x y z : ℝ, (x^3 + y^3 + z^3 - 3 * x * y * z = 1) → (∃ a b c : ℝ, a = x ∨ a = y ∨ a = z ∧ b = x ∨ b = y ∨ b = z ∧ c = x ∨ c = y ∨ a ≤ b ∨ a ≤ c ∧ b ≤ c) → (x^2 + y^2 + z^2 ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_minimum_x2_y2_z2_l1581_158109


namespace NUMINAMATH_GPT_interest_percentage_calculation_l1581_158137

-- Definitions based on problem conditions
def purchase_price : ℝ := 110
def down_payment : ℝ := 10
def monthly_payment : ℝ := 10
def number_of_monthly_payments : ℕ := 12

-- Theorem statement:
theorem interest_percentage_calculation :
  let total_paid := down_payment + (monthly_payment * number_of_monthly_payments)
  let interest_paid := total_paid - purchase_price
  let interest_percent := (interest_paid / purchase_price) * 100
  interest_percent = 18.2 :=
by sorry

end NUMINAMATH_GPT_interest_percentage_calculation_l1581_158137


namespace NUMINAMATH_GPT_x_plus_q_in_terms_of_q_l1581_158169

theorem x_plus_q_in_terms_of_q (x q : ℝ) (h1 : |x - 5| = q) (h2 : x > 5) : x + q = 2 * q + 5 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_q_in_terms_of_q_l1581_158169


namespace NUMINAMATH_GPT_third_speed_is_9_kmph_l1581_158119

/-- Problem Statement: Given the total travel time, total distance, and two speeds, 
    prove that the third speed is 9 km/hr when distances are equal. -/
theorem third_speed_is_9_kmph (t : ℕ) (d_total : ℕ) (v1 v2 : ℕ) (d1 d2 d3 : ℕ) 
(h_t : t = 11)
(h_d_total : d_total = 900)
(h_v1 : v1 = 3)
(h_v2 : v2 = 6)
(h_d_eq : d1 = 300 ∧ d2 = 300 ∧ d3 = 300)
(h_sum_t : d1 / (v1 * 1000 / 60) + d2 / (v2 * 1000 / 60) + d3 / (v3 * 1000 / 60) = t) 
: (v3 = 9) :=
by 
  sorry

end NUMINAMATH_GPT_third_speed_is_9_kmph_l1581_158119


namespace NUMINAMATH_GPT_probability_of_both_chinese_books_l1581_158153

def total_books := 5
def chinese_books := 3
def math_books := 2

theorem probability_of_both_chinese_books (select_books : ℕ) 
  (total_choices : ℕ) (favorable_choices : ℕ) :
  select_books = 2 →
  total_choices = (Nat.choose total_books select_books) →
  favorable_choices = (Nat.choose chinese_books select_books) →
  (favorable_choices : ℚ) / (total_choices : ℚ) = 3 / 10 := by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_probability_of_both_chinese_books_l1581_158153


namespace NUMINAMATH_GPT_number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l1581_158147

noncomputable def a (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem number_of_diagonals_pentagon : a 5 = 5 := sorry

theorem difference_hexagon_pentagon : a 6 - a 5 = 4 := sorry

theorem difference_successive_polygons (n : ℕ) (h : 4 ≤ n) : a (n + 1) - a n = n - 1 := sorry

end NUMINAMATH_GPT_number_of_diagonals_pentagon_difference_hexagon_pentagon_difference_successive_polygons_l1581_158147


namespace NUMINAMATH_GPT_quadratic_has_real_roots_find_value_of_m_l1581_158177

theorem quadratic_has_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2) ≠ 0 ∧ (x^2 - 4 * m * x + 3 * m^2 = 0) := 
by 
  sorry

theorem find_value_of_m (m : ℝ) (h1 : m > 0) (h2 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1 - x2 = 2)) :
  m = 1 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_find_value_of_m_l1581_158177


namespace NUMINAMATH_GPT_books_on_each_shelf_l1581_158167

-- Define the conditions and the problem statement
theorem books_on_each_shelf :
  ∀ (M P : ℕ), 
  -- Conditions
  (5 * M + 4 * P = 72) ∧ (M = P) ∧ (∃ B : ℕ, M = B ∧ P = B) ->
  -- Conclusion
  (∃ B : ℕ, B = 8) :=
by
  sorry

end NUMINAMATH_GPT_books_on_each_shelf_l1581_158167


namespace NUMINAMATH_GPT_fraction_sum_zero_implies_square_sum_zero_l1581_158138

theorem fraction_sum_zero_implies_square_sum_zero (a b c : ℝ) (h₀ : a ≠ b) (h₁ : b ≠ c) (h₂ : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_fraction_sum_zero_implies_square_sum_zero_l1581_158138


namespace NUMINAMATH_GPT_problem_inequality_l1581_158110

theorem problem_inequality 
  (a b c : ℝ)
  (a_pos : 0 < a) 
  (b_pos : 0 < b)
  (c_pos : 0 < c) 
  (h : a * b * c * (a + b + c) = 3) : 
  (a + b) * (b + c) * (c + a) ≥ 8 := 
sorry

end NUMINAMATH_GPT_problem_inequality_l1581_158110


namespace NUMINAMATH_GPT_no_real_solutions_l1581_158113

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (2 - x^2) / x

theorem no_real_solutions :
  (∀ x : ℝ, x ≠ 0 → (f x + 2 * f (1 / x) = 3 * x)) →
  (∀ x : ℝ, f x = f (-x) → false) :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1581_158113


namespace NUMINAMATH_GPT_actual_distance_traveled_l1581_158199

theorem actual_distance_traveled (D T : ℝ)
  (h1 : D = 10 * T)
  (h2 : D + 20 = 20 * T) : D = 20 :=
by
  sorry

end NUMINAMATH_GPT_actual_distance_traveled_l1581_158199


namespace NUMINAMATH_GPT_area_after_shortening_other_side_l1581_158155

-- Define initial dimensions of the index card
def initial_length := 5
def initial_width := 7
def initial_area := initial_length * initial_width

-- Define the area condition when one side is shortened by 2 inches
def shortened_side_length := initial_length - 2
def new_area_after_shortening_one_side := 21

-- Definition of the problem condition that results in 21 square inches area
def condition := 
  (shortened_side_length * initial_width = new_area_after_shortening_one_side)

-- Final statement
theorem area_after_shortening_other_side :
  condition →
  (initial_length * (initial_width - 2) = 25) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_area_after_shortening_other_side_l1581_158155


namespace NUMINAMATH_GPT_correct_calculation_l1581_158145

theorem correct_calculation (x : ℝ) :
  (x / 5 + 16 = 58) → (x / 15 + 74 = 88) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1581_158145


namespace NUMINAMATH_GPT_slope_of_line_l1581_158157

def point1 : (ℤ × ℤ) := (-4, 6)
def point2 : (ℤ × ℤ) := (3, -4)

def slope_formula (p1 p2 : (ℤ × ℤ)) : ℚ :=
  (p2.snd - p1.snd) / (p2.fst - p1.fst : ℚ)

theorem slope_of_line : slope_formula point1 point2 = -10 / 7 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_l1581_158157


namespace NUMINAMATH_GPT_total_area_of_region_l1581_158180

variable (a b c d : ℝ)
variable (ha : a > 0) (hb : b > 0) (hd : d > 0)

theorem total_area_of_region : (a + b) * d + (1 / 2) * Real.pi * c ^ 2 = (a + b) * d + (1 / 2) * Real.pi * c ^ 2 := by
  sorry

end NUMINAMATH_GPT_total_area_of_region_l1581_158180


namespace NUMINAMATH_GPT_height_of_oil_truck_tank_l1581_158123

/-- 
Given that a stationary oil tank is a right circular cylinder 
with a radius of 100 feet and its oil level dropped by 0.025 feet,
proving that if this oil is transferred to a right circular 
cylindrical oil truck's tank with a radius of 5 feet, then the 
height of the oil in the truck's tank will be 10 feet. 
-/
theorem height_of_oil_truck_tank
    (radius_stationary : ℝ) (height_drop_stationary : ℝ) (radius_truck : ℝ) 
    (height_truck : ℝ) (π : ℝ)
    (h1 : radius_stationary = 100)
    (h2 : height_drop_stationary = 0.025)
    (h3 : radius_truck = 5)
    (pi_approx : π = 3.14159265) :
    height_truck = 10 :=
by
    sorry

end NUMINAMATH_GPT_height_of_oil_truck_tank_l1581_158123


namespace NUMINAMATH_GPT_percent_change_range_l1581_158152

-- Define initial conditions
def initial_yes_percent : ℝ := 0.60
def initial_no_percent : ℝ := 0.40
def final_yes_percent : ℝ := 0.80
def final_no_percent : ℝ := 0.20

-- Define the key statement to prove
theorem percent_change_range : 
  ∃ y_min y_max : ℝ, 
  y_min = 0.20 ∧ 
  y_max = 0.60 ∧ 
  (y_max - y_min = 0.40) :=
sorry

end NUMINAMATH_GPT_percent_change_range_l1581_158152


namespace NUMINAMATH_GPT_find_a_of_parabola_l1581_158134

theorem find_a_of_parabola (a b c : ℤ) (h_vertex : (2, 5) = (2, 5)) (h_point : 8 = a * (3 - 2) ^ 2 + 5) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_of_parabola_l1581_158134


namespace NUMINAMATH_GPT_value_of_m_l1581_158179

theorem value_of_m (m : ℤ) (h : ∃ x : ℤ, x = 2 ∧ x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1581_158179


namespace NUMINAMATH_GPT_each_friend_should_contribute_equally_l1581_158140

-- Define the total expenses and number of friends
def total_expenses : ℝ := 35 + 9 + 9 + 6 + 2
def number_of_friends : ℕ := 5

-- Define the expected contribution per friend
def expected_contribution : ℝ := 12.20

-- Theorem statement
theorem each_friend_should_contribute_equally :
  total_expenses / number_of_friends = expected_contribution :=
by
  sorry

end NUMINAMATH_GPT_each_friend_should_contribute_equally_l1581_158140


namespace NUMINAMATH_GPT_correct_statements_count_l1581_158166

-- Definition of the statements
def statement1 := ∀ (q : ℚ), q > 0 ∨ q < 0
def statement2 (a : ℝ) := |a| = -a → a < 0
def statement3 := ∀ (x y : ℝ), 0 = 3
def statement4 := ∀ (q : ℚ), ∃ (p : ℝ), q = p
def statement5 := 7 = 7 ∧ 10 = 10 ∧ 15 = 15

-- Define what it means for each statement to be correct
def is_correct_statement1 := statement1 = false
def is_correct_statement2 := ∀ a : ℝ, statement2 a = false
def is_correct_statement3 := statement3 = false
def is_correct_statement4 := statement4 = true
def is_correct_statement5 := statement5 = true

-- Define the problem and its correct answer
def problem := is_correct_statement1 ∧ is_correct_statement2 ∧ is_correct_statement3 ∧ is_correct_statement4 ∧ is_correct_statement5

-- Prove that the number of correct statements is 2
theorem correct_statements_count : problem → (2 = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_statements_count_l1581_158166


namespace NUMINAMATH_GPT_student_A_more_stable_performance_l1581_158100

theorem student_A_more_stable_performance
    (mean : ℝ)
    (n : ℕ)
    (variance_A variance_B : ℝ)
    (h1 : mean = 1.6)
    (h2 : n = 10)
    (h3 : variance_A = 1.4)
    (h4 : variance_B = 2.5) :
    variance_A < variance_B :=
by {
    -- The proof is omitted as we are only writing the statement here.
    sorry
}

end NUMINAMATH_GPT_student_A_more_stable_performance_l1581_158100


namespace NUMINAMATH_GPT_inequality_for_a_and_b_l1581_158173

theorem inequality_for_a_and_b (a b : ℝ) : 
  (1 / 3 * a - b) ≤ 5 :=
sorry

end NUMINAMATH_GPT_inequality_for_a_and_b_l1581_158173


namespace NUMINAMATH_GPT_statement_A_statement_C_statement_D_l1581_158103

theorem statement_A (x : ℝ) :
  (¬ (∀ x ≥ 3, 2 * x - 10 ≥ 0)) ↔ (∃ x0 ≥ 3, 2 * x0 - 10 < 0) := 
sorry

theorem statement_C {a b c : ℝ} (h1 : c > a) (h2 : a > b) (h3 : b > 0) :
  (a / (c - a)) > (b / (c - b)) := 
sorry

theorem statement_D {a b m : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : m > 0) :
  (a / b) > ((a + m) / (b + m)) := 
sorry

end NUMINAMATH_GPT_statement_A_statement_C_statement_D_l1581_158103


namespace NUMINAMATH_GPT_solution_x_alcohol_percentage_l1581_158115

theorem solution_x_alcohol_percentage (P : ℝ) :
  let y_percentage := 0.30
  let mixture_percentage := 0.25
  let y_volume := 600
  let x_volume := 200
  let mixture_volume := y_volume + x_volume
  let y_alcohol_content := y_volume * y_percentage
  let mixture_alcohol_content := mixture_volume * mixture_percentage
  P * x_volume + y_alcohol_content = mixture_alcohol_content →
  P = 0.10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_solution_x_alcohol_percentage_l1581_158115


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1581_158135

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x - 2

theorem sufficient_but_not_necessary_condition 
  (a : ℝ) 
  (h : ∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) : 
  a ≤ 0 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1581_158135


namespace NUMINAMATH_GPT_traci_flour_brought_l1581_158174

-- Definitions based on the conditions
def harris_flour : ℕ := 400
def flour_per_cake : ℕ := 100
def cakes_each : ℕ := 9

-- Proving the amount of flour Traci brought
theorem traci_flour_brought :
  (cakes_each * flour_per_cake) - harris_flour = 500 :=
by
  sorry

end NUMINAMATH_GPT_traci_flour_brought_l1581_158174


namespace NUMINAMATH_GPT_new_profit_is_122_03_l1581_158178

noncomputable def new_profit_percentage (P : ℝ) (tax_rate : ℝ) (profit_rate : ℝ) (market_increase_rate : ℝ) (months : ℕ) : ℝ :=
  let total_cost := P * (1 + tax_rate)
  let initial_selling_price := total_cost * (1 + profit_rate)
  let market_price_after_months := initial_selling_price * (1 + market_increase_rate) ^ months
  let final_selling_price := 2 * initial_selling_price
  let profit := final_selling_price - total_cost
  (profit / total_cost) * 100

theorem new_profit_is_122_03 :
  new_profit_percentage (P : ℝ) 0.18 0.40 0.05 3 = 122.03 := 
by
  sorry

end NUMINAMATH_GPT_new_profit_is_122_03_l1581_158178


namespace NUMINAMATH_GPT_fewer_people_correct_l1581_158176

def pop_Springfield : ℕ := 482653
def pop_total : ℕ := 845640
def pop_new_city : ℕ := pop_total - pop_Springfield
def fewer_people : ℕ := pop_Springfield - pop_new_city

theorem fewer_people_correct : fewer_people = 119666 :=
by
  unfold fewer_people
  unfold pop_new_city
  unfold pop_total
  unfold pop_Springfield
  sorry

end NUMINAMATH_GPT_fewer_people_correct_l1581_158176


namespace NUMINAMATH_GPT_secret_code_count_l1581_158150

noncomputable def number_of_secret_codes (colors slots : ℕ) : ℕ :=
  colors ^ slots

theorem secret_code_count : number_of_secret_codes 9 5 = 59049 := by
  sorry

end NUMINAMATH_GPT_secret_code_count_l1581_158150


namespace NUMINAMATH_GPT_twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l1581_158170

theorem twenty_percent_less_than_sixty_equals_one_third_more_than_what_number :
  (4 / 3) * n = 48 → n = 36 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_twenty_percent_less_than_sixty_equals_one_third_more_than_what_number_l1581_158170


namespace NUMINAMATH_GPT_budget_per_friend_l1581_158161

-- Definitions for conditions
def total_budget : ℕ := 100
def parents_gift_cost : ℕ := 14
def number_of_parents : ℕ := 2
def number_of_friends : ℕ := 8

-- Statement to prove
theorem budget_per_friend :
  (total_budget - number_of_parents * parents_gift_cost) / number_of_friends = 9 :=
by
  sorry

end NUMINAMATH_GPT_budget_per_friend_l1581_158161


namespace NUMINAMATH_GPT_find_angle_D_l1581_158175

theorem find_angle_D (A B C D : ℝ) (h1 : A + B = 180) (h2 : C = D) (h3 : A = 50) (h4 : ∃ B_adj, B_adj = 60 ∧ A + B_adj + B = 180) : D = 25 :=
sorry

end NUMINAMATH_GPT_find_angle_D_l1581_158175


namespace NUMINAMATH_GPT_regression_estimate_l1581_158168

theorem regression_estimate :
  ∀ (x y : ℝ), (y = 0.50 * x - 0.81) → x = 25 → y = 11.69 :=
by
  intros x y h_eq h_x_val
  sorry

end NUMINAMATH_GPT_regression_estimate_l1581_158168


namespace NUMINAMATH_GPT_is_quadratic_function_l1581_158143

theorem is_quadratic_function (x : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + 3) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 / x) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = (x - 1)^2 - x^2) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 * x^2 - 1) ∧ (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c))) :=
by
  sorry

end NUMINAMATH_GPT_is_quadratic_function_l1581_158143


namespace NUMINAMATH_GPT_simplified_evaluated_expression_l1581_158112

noncomputable def a : ℚ := 1 / 3
noncomputable def b : ℚ := 1 / 2
noncomputable def c : ℚ := 1

def expression (a b c : ℚ) : ℚ := a^2 + 2 * b - c

theorem simplified_evaluated_expression :
  expression a b c = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_simplified_evaluated_expression_l1581_158112
