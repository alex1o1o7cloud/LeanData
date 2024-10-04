import Mathlib

namespace smallest_positive_debt_l129_129584

noncomputable def pigs_value : ℤ := 300
noncomputable def goats_value : ℤ := 210

theorem smallest_positive_debt : ∃ D p g : ℤ, (D = pigs_value * p + goats_value * g) ∧ D > 0 ∧ ∀ D' p' g' : ℤ, (D' = pigs_value * p' + goats_value * g' ∧ D' > 0) → D ≤ D' :=
by
  sorry

end smallest_positive_debt_l129_129584


namespace marbles_count_l129_129672

variables {g y : ℕ}

theorem marbles_count (h1 : (g - 1)/(g + y - 1) = 1/8)
                      (h2 : g/(g + y - 3) = 1/6) :
                      g + y = 9 :=
by
-- This is just setting up the statements we need to prove the theorem. The actual proof is to be completed.
sorry

end marbles_count_l129_129672


namespace first_bell_weight_l129_129545

-- Given conditions from the problem
variable (x : ℕ) -- weight of the first bell in pounds
variable (total_weight : ℕ)

-- The condition as the sum of the weights
def bronze_weights (x total_weight : ℕ) : Prop :=
  x + 2 * x + 8 * 2 * x = total_weight

-- Prove that the weight of the first bell is 50 pounds given the total weight is 550 pounds
theorem first_bell_weight : bronze_weights x 550 → x = 50 := by
  intro h
  sorry

end first_bell_weight_l129_129545


namespace cos_135_eq_neg_inv_sqrt_2_l129_129015

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129015


namespace largest_sum_of_three_largest_angles_l129_129832

-- Definitions and main theorem statement
theorem largest_sum_of_three_largest_angles (EFGH : Type*)
    (a b c d : ℝ) 
    (h1 : a + b + c + d = 360)
    (h2 : b = 3 * c)
    (h3 : ∃ (common_diff : ℝ), (c - a = common_diff) ∧ (b - c = common_diff) ∧ (d - b = common_diff))
    (h4 : ∀ (x y z : ℝ), (x = y + z) ↔ (∃ (progression_diff : ℝ), x - y = y - z ∧ y - z = z - x)) :
    (∃ (A B C D : ℝ), A = a ∧ B = b ∧ C = c ∧ D = d ∧ A + B + C + D = 360 ∧ A = max a (max b (max c d)) ∧ B = 2 * D ∧ A + B + C = 330) :=
sorry

end largest_sum_of_three_largest_angles_l129_129832


namespace select_3_from_5_prob_l129_129351

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129351


namespace fabric_difference_fabric_total_l129_129993

noncomputable def fabric_used_coat : ℝ := 1.55
noncomputable def fabric_used_pants : ℝ := 1.05

theorem fabric_difference : fabric_used_coat - fabric_used_pants = 0.5 :=
by
  sorry

theorem fabric_total : fabric_used_coat + fabric_used_ppants = 2.6 :=
by
  sorry

end fabric_difference_fabric_total_l129_129993


namespace pairs_satisfying_int_l129_129064

theorem pairs_satisfying_int (a b : ℕ) :
  ∃ n : ℕ, a = 2 * n^2 + 1 ∧ b = n ↔ (2 * a * b^2 + 1) ∣ (a^3 + 1) := by
  sorry

end pairs_satisfying_int_l129_129064


namespace jake_pure_alcohol_l129_129106

-- Definitions based on the conditions
def shots : ℕ := 8
def ounces_per_shot : ℝ := 1.5
def vodka_purity : ℝ := 0.5
def friends : ℕ := 2

-- Statement to prove the amount of pure alcohol Jake drank
theorem jake_pure_alcohol : (shots * ounces_per_shot * vodka_purity) / friends = 3 := by
  sorry

end jake_pure_alcohol_l129_129106


namespace probability_A_wins_championship_expectation_X_is_13_l129_129710

/-
Definitions corresponding to the conditions in the problem
-/
def prob_event1_A_win : ℝ := 0.5
def prob_event2_A_win : ℝ := 0.4
def prob_event3_A_win : ℝ := 0.8

def prob_event1_B_win : ℝ := 1 - prob_event1_A_win
def prob_event2_B_win : ℝ := 1 - prob_event2_A_win
def prob_event3_B_win : ℝ := 1 - prob_event3_A_win

/-
Proof problems corresponding to the questions and correct answers
-/

theorem probability_A_wins_championship : prob_event1_A_win * prob_event2_A_win * prob_event3_A_win
    + prob_event1_A_win * prob_event2_A_win * prob_event3_B_win
    + prob_event1_A_win * prob_event2_B_win * prob_event3_A_win 
    + prob_event1_B_win * prob_event2_A_win * prob_event3_A_win = 0.6 := 
sorry

noncomputable def X_distribution_table : list (ℝ × ℝ) := 
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

noncomputable def expected_value_X : ℝ := 
  ∑ x in X_distribution_table, x.1 * x.2

theorem expectation_X_is_13 : expected_value_X = 13 := sorry

end probability_A_wins_championship_expectation_X_is_13_l129_129710


namespace table_height_l129_129218

variable (l h w : ℝ)

-- Given conditions:
def conditionA := l + h - w = 36
def conditionB := w + h - l = 30

-- Proof that height of the table h is 33 inches
theorem table_height {l h w : ℝ} 
  (h1 : l + h - w = 36) 
  (h2 : w + h - l = 30) : 
  h = 33 := 
by
  sorry

end table_height_l129_129218


namespace find_s_l129_129091

theorem find_s (n : ℤ) (hn : n ≠ 0) (s : ℝ)
  (hs : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1 / n)) :
  s = 1 / 4 :=
by
  sorry

end find_s_l129_129091


namespace range_of_k_for_distinct_real_roots_l129_129635

theorem range_of_k_for_distinct_real_roots (k : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ k*x1^2 - 2*x1 - 1 = 0 ∧ k*x2^2 - 2*x2 - 1 = 0) → k > -1 ∧ k ≠ 0 :=
by
  sorry

end range_of_k_for_distinct_real_roots_l129_129635


namespace perfectCubesBetween200and1200_l129_129518

theorem perfectCubesBetween200and1200 : ∃ n m : ℕ, (n = 6) ∧ (m = 10) ∧ (m - n + 1 = 5) ∧ (n^3 ≥ 200) ∧ (m^3 ≤ 1200) := 
by
  have h1 : 6^3 ≥ 200 := by norm_num
  have h2 : 10^3 ≤ 1200 := by norm_num
  use [6, 10]
  constructor; {refl} -- n = 6
  constructor; {refl} -- m = 10
  constructor;
  { norm_num },
  constructor; 
  { exact h1 },
  { exact h2 }
  sorry

end perfectCubesBetween200and1200_l129_129518


namespace multiplication_identity_l129_129893

theorem multiplication_identity : 32519 * 9999 = 324857481 := by
  sorry

end multiplication_identity_l129_129893


namespace last_four_digits_of_5_pow_15000_l129_129889

theorem last_four_digits_of_5_pow_15000 (h : 5^500 ≡ 1 [MOD 2000]) : 
  5^15000 ≡ 1 [MOD 2000] :=
sorry

end last_four_digits_of_5_pow_15000_l129_129889


namespace sequence_inequality_for_k_l129_129949

theorem sequence_inequality_for_k (k : ℝ) : 
  (∀ n : ℕ, 0 < n → (n + 1)^2 + k * (n + 1) + 2 > n^2 + k * n + 2) ↔ k > -3 :=
sorry

end sequence_inequality_for_k_l129_129949


namespace greatest_gcd_of_rope_lengths_l129_129105

theorem greatest_gcd_of_rope_lengths : Nat.gcd (Nat.gcd 39 52) 65 = 13 := by
  sorry

end greatest_gcd_of_rope_lengths_l129_129105


namespace new_mean_of_five_numbers_l129_129570

theorem new_mean_of_five_numbers (a b c d e : ℝ) 
  (h_mean : (a + b + c + d + e) / 5 = 25) :
  ((a + 5) + (b + 10) + (c + 15) + (d + 20) + (e + 25)) / 5 = 40 :=
by
  sorry

end new_mean_of_five_numbers_l129_129570


namespace school_A_win_prob_expectation_X_is_13_l129_129703

-- Define the probabilities of school A winning individual events
def pA_event1 : ℝ := 0.5
def pA_event2 : ℝ := 0.4
def pA_event3 : ℝ := 0.8

-- Define the probability of school A winning the championship
def pA_win_championship : ℝ :=
  (pA_event1 * pA_event2 * pA_event3) +
  (pA_event1 * (1 - pA_event2) * pA_event3) +
  (pA_event1 * pA_event2 * (1 - pA_event3)) +
  ((1 - pA_event1) * pA_event2 * pA_event3)

-- Proof statement for the probability of school A winning the championship
theorem school_A_win_prob : pA_win_championship = 0.6 := sorry

-- Define the distribution and expectation for school B's total score
def X_prob : ℝ → ℝ
| 0  := (1 - pA_event1) * (1 - pA_event2) * (1 - pA_event3)
| 10 := pA_event1 * (1 - pA_event2) * (1 - pA_event3) +
        (1 - pA_event1) * pA_event2 * (1 - pA_event3) +
        (1 - pA_event1) * (1 - pA_event2) * pA_event3
| 20 := pA_event1 * pA_event2 * (1 - pA_event3) +
        pA_event1 * (1 - pA_event2) * pA_event3 +
        (1 - pA_event1) * pA_event2 * pA_event3
| 30 := pA_event1 * pA_event2 * pA_event3
| _  := 0

def expected_X : ℝ :=
  0 * X_prob 0 +
  10 * X_prob 10 +
  20 * X_prob 20 +
  30 * X_prob 30

-- Proof statement for the expectation of school B's total score
theorem expectation_X_is_13 : expected_X = 13 := sorry

end school_A_win_prob_expectation_X_is_13_l129_129703


namespace total_yards_run_l129_129691

theorem total_yards_run (Malik_yards_per_game : ℕ) (Josiah_yards_per_game : ℕ) (Darnell_yards_per_game : ℕ) (games : ℕ) 
  (hM : Malik_yards_per_game = 18) (hJ : Josiah_yards_per_game = 22) (hD : Darnell_yards_per_game = 11) (hG : games = 4) : 
  Malik_yards_per_game * games + Josiah_yards_per_game * games + Darnell_yards_per_game * games = 204 := by
  sorry

end total_yards_run_l129_129691


namespace positive_difference_l129_129208

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l129_129208


namespace triangle_area_l129_129740

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area : 
  ∀ (A B C : (ℝ × ℝ)),
  (A = (3, 3)) →
  (B = (4.5, 7.5)) →
  (C = (7.5, 4.5)) →
  area_of_triangle A B C = 8.625 :=
by
  intros A B C hA hB hC
  rw [hA, hB, hC]
  unfold area_of_triangle
  norm_num
  sorry

end triangle_area_l129_129740


namespace karen_total_nuts_l129_129108

variable (x y : ℝ)
variable (hx : x = 0.25)
variable (hy : y = 0.25)

theorem karen_total_nuts : x + y = 0.50 := by
  rw [hx, hy]
  norm_num

end karen_total_nuts_l129_129108


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129456

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129456


namespace first_day_of_month_l129_129562

theorem first_day_of_month 
  (d_24: ℕ) (mod_7: d_24 % 7 = 6) : 
  (d_24 - 23) % 7 = 4 :=
by 
  -- denotes the 24th day is a Saturday (Saturday is the 6th day in a 0-6 index)
  -- hence mod_7: d_24 % 7 = 6 means d_24 falls on a Saturday
  sorry

end first_day_of_month_l129_129562


namespace positive_difference_l129_129177

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l129_129177


namespace probability_A_and_B_selected_l129_129501

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129501


namespace math_proof_problem_l129_129538

noncomputable def f (a b : ℚ) : ℝ := sorry

axiom f_cond1 (a b c : ℚ) : f (a * b) c = f a c * f b c ∧ f c (a * b) = f c a * f c b
axiom f_cond2 (a : ℚ) : f a (1 - a) = 1

theorem math_proof_problem (a b : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (f a a = 1) ∧ 
  (f a (-a) = 1) ∧
  (f a b * f b a = 1) := 
by 
  sorry

end math_proof_problem_l129_129538


namespace quadratic_polynomial_inequality_l129_129877

variable {a b c : ℝ}

theorem quadratic_polynomial_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0)
    (h2 : a < 0)
    (h3 : b^2 - 4 * a * c < 0) :
    b / a < c / a + 1 := 
by 
  sorry

end quadratic_polynomial_inequality_l129_129877


namespace actual_diameter_of_tissue_l129_129887

theorem actual_diameter_of_tissue (magnification_factor : ℝ) (magnified_diameter : ℝ) (image_magnified : magnification_factor = 1000 ∧ magnified_diameter = 2) : (1 / magnification_factor) * magnified_diameter = 0.002 :=
by
  sorry

end actual_diameter_of_tissue_l129_129887


namespace solution_to_equation_l129_129129

theorem solution_to_equation (x y : ℕ → ℕ) (h1 : x 1 = 2) (h2 : y 1 = 3)
  (h3 : ∀ k, x (k + 1) = 3 * x k + 2 * y k)
  (h4 : ∀ k, y (k + 1) = 4 * x k + 3 * y k) :
  ∀ n, 2 * (x n)^2 + 1 = (y n)^2 := 
by
  sorry

end solution_to_equation_l129_129129


namespace right_angled_triangle_hypotenuse_and_altitude_relation_l129_129251

variables (a b c m : ℝ)

theorem right_angled_triangle_hypotenuse_and_altitude_relation
  (h1 : b^2 + c^2 = a^2)
  (h2 : m^2 = (b - c)^2)
  (h3 : b * c = a * m) :
  m = (a * (Real.sqrt 5 - 1)) / 2 := 
sorry

end right_angled_triangle_hypotenuse_and_altitude_relation_l129_129251


namespace min_c_plus_3d_l129_129109

theorem min_c_plus_3d (c d : ℝ) (hc : 0 < c) (hd : 0 < d) 
    (h1 : c^2 ≥ 12 * d) (h2 : 9 * d^2 ≥ 4 * c) : 
  c + 3 * d ≥ 8 :=
  sorry

end min_c_plus_3d_l129_129109


namespace john_horizontal_distance_l129_129843

theorem john_horizontal_distance (v_increase : ℕ)
  (elevation_start : ℕ) (elevation_end : ℕ) (h_ratio : ℕ) :
  (elevation_end - elevation_start) * h_ratio = 1350 * 2 :=
begin
  -- Let elevation_start be 100 feet
  let elevation_start := 100,
  -- Let elevation_end be 1450 feet
  let elevation_end := 1450,
  -- The steepened ratio, height per step
  let v_increase := elevation_end - elevation_start,
  -- John travels 1 foot vertically for every 2 feet horizontally
  let h_ratio := 2,
  -- Hence the horizontal distance theorem
  -- s_foot = h_ratio * t_foot = 1350 * 2 = 2700 feet.
  sorry
end

end john_horizontal_distance_l129_129843


namespace solve_and_sum_solutions_l129_129137

theorem solve_and_sum_solutions :
  (∀ x : ℝ, x^2 > 9 → 
  (∃ S : ℝ, (∀ x : ℝ, (x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12) → S = 8.75))) :=
sorry

end solve_and_sum_solutions_l129_129137


namespace probability_A_and_B_selected_l129_129493

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129493


namespace probability_A_and_B_selected_l129_129496

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129496


namespace right_angle_triangle_XY_length_l129_129673

noncomputable def length_XY {X Y Z : Type} [MetricSpace X] [MetricSpace Y] [MetricSpace Z] 
  (triangleXYZ : IsRightTriangle X Y Z) 
  (cosZ : real.cos Z = 3 / 5) 
  (YZ : dist Y Z = 10) 
  : real :=
sqrt (YZ^2 - (cosZ * YZ)^2)

theorem right_angle_triangle_XY_length (XYZ : Triangle)
  (angleY : XYZ.angled_y = π / 2) 
  (cosZ : ∀ Z, real.cos XYZ.angled_z = 3 / 5) 
  (YZ : ∀ Y Z, dist Y Z = 10) 
  : dist XYZ.side_x Y = 8 :=
sorry

end right_angle_triangle_XY_length_l129_129673


namespace min_value_of_f_l129_129075

-- Define the function f
def f (a b c x y z : ℤ) : ℤ := a * x + b * y + c * z

-- Define the gcd function for three integers
def gcd3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- Define the main theorem to prove
theorem min_value_of_f (a b c : ℕ) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) :
  ∃ (x y z : ℤ), f a b c x y z = gcd3 a b c := 
by
  sorry

end min_value_of_f_l129_129075


namespace problem_1_problem_2_l129_129611

theorem problem_1 : ((1 / 3 - 3 / 4 + 5 / 6) / (1 / 12)) = 5 := 
  sorry

theorem problem_2 : ((-1 : ℤ) ^ 2023 + |(1 : ℝ) - 0.5| * (-4 : ℝ) ^ 2) = 7 := 
  sorry

end problem_1_problem_2_l129_129611


namespace function_conditions_satisfied_l129_129524

noncomputable def function_satisfying_conditions : ℝ → ℝ := fun x => -2 * x^2 + 3 * x

theorem function_conditions_satisfied :
  (function_satisfying_conditions 1 = 1) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ function_satisfying_conditions x = y) ∧
  (∀ x y : ℝ, x > 1 ∧ y = function_satisfying_conditions x → ∃ ε > 0, ∀ δ > 0, (x + δ > 1 → function_satisfying_conditions (x + δ) < y)) :=
by
  sorry

end function_conditions_satisfied_l129_129524


namespace latest_time_temp_decreasing_l129_129995

theorem latest_time_temp_decreasing (t : ℝ) 
  (h1 : -t^2 + 12 * t + 55 = 82) 
  (h2 : ∀ t0 : ℝ, -2 * t0 + 12 < 0 → t > t0) : 
  t = 6 + (3 * Real.sqrt 28 / 2) :=
sorry

end latest_time_temp_decreasing_l129_129995


namespace total_earnings_after_six_months_l129_129132

def area_of_farm : ℕ := 20
def trees_per_square_meter : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval_months : ℕ := 3
def price_per_coconut : ℕ := 0.5 -- Note: Lean requires specific handling for non-integer values, so we can use a fractional form instead

theorem total_earnings_after_six_months :
  let total_trees := area_of_farm * trees_per_square_meter in
  let total_coconuts := total_trees * coconuts_per_tree in
  let number_of_harvests := 6 / harvest_interval_months in
  let total_coconuts_six_months := total_coconuts * number_of_harvests in
  let earnings := total_coconuts_six_months * price_per_coconut in
  earnings = 240 :=
by {
  sorry
}

end total_earnings_after_six_months_l129_129132


namespace find_f_six_l129_129724

noncomputable def f : ℝ → ℝ := sorry -- placeholder for the function definition

axiom f_property : ∀ x y : ℝ, f (x - y) = f x * f y
axiom f_nonzero : ∀ x : ℝ, f x ≠ 0
axiom f_two : f 2 = 5

theorem find_f_six : f 6 = 1 / 5 :=
sorry

end find_f_six_l129_129724


namespace positive_difference_of_two_numbers_l129_129173

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l129_129173


namespace cos_135_eq_neg_sqrt2_div_2_l129_129038

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129038


namespace calculate_expression_l129_129609

theorem calculate_expression (m n : ℝ) : 9 * m^2 - (m - 2 * n)^2 = 4 * (2 * m - n) * (m + n) :=
by
  sorry

end calculate_expression_l129_129609


namespace problem_statement_l129_129539

noncomputable def nonreal_omega_root (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω^2 + ω + 1 = 0

theorem problem_statement (ω : ℂ) (h : nonreal_omega_root ω) :
  (1 - 2 * ω + ω^2)^6 + (1 + 2 * ω - ω^2)^6 = 1458 :=
sorry

end problem_statement_l129_129539


namespace jo_thinking_greatest_integer_l129_129682

theorem jo_thinking_greatest_integer :
  ∃ n : ℕ, n < 150 ∧ 
           (∃ k : ℤ, n = 9 * k - 2) ∧ 
           (∃ m : ℤ, n = 11 * m - 4) ∧ 
           (∀ N : ℕ, (N < 150 ∧ 
                      (∃ K : ℤ, N = 9 * K - 2) ∧ 
                      (∃ M : ℤ, N = 11 * M - 4)) → N ≤ n) 
:= by
  sorry

end jo_thinking_greatest_integer_l129_129682


namespace liars_positions_l129_129125

structure Islander :=
  (position : Nat)
  (statement : String)

-- Define our islanders
def A : Islander := { position := 1, statement := "My closest tribesman in this line is 3 meters away from me." }
def D : Islander := { position := 4, statement := "My closest tribesman in this line is 1 meter away from me." }
def E : Islander := { position := 5, statement := "My closest tribesman in this line is 2 meters away from me." }

-- Define the other islanders with dummy statements
def B : Islander := { position := 2, statement := "" }
def C : Islander := { position := 3, statement := "" }
def F : Islander := { position := 6, statement := "" }

-- Define the main theorem
theorem liars_positions (knights_count : Nat) (liars_count : Nat) (is_knight : Islander → Bool)
  (is_lair : Islander → Bool) : 
  ( ∀ x, is_knight x ↔ ¬is_lair x ) → -- Knight and liar are mutually exclusive
  knights_count = 3 → 
  liars_count = 3 →
  is_knight A = false → 
  is_knight D = false → 
  is_knight E = false → 
  is_lair A = true ∧
  is_lair D = true ∧
  is_lair E = true := by
  sorry

end liars_positions_l129_129125


namespace probability_A_B_selected_l129_129422

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129422


namespace min_m_n_sum_l129_129867

theorem min_m_n_sum (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 108 * m = n^3) : m + n = 8 :=
  sorry

end min_m_n_sum_l129_129867


namespace balance_problem_l129_129556

variable {G B Y W : ℝ}

theorem balance_problem
  (h1 : 4 * G = 8 * B)
  (h2 : 3 * Y = 7.5 * B)
  (h3 : 5 * B = 3.5 * W) :
  5 * G + 4 * Y + 3 * W = (170 / 7) * B := by
  sorry

end balance_problem_l129_129556


namespace math_problem_proof_l129_129883

-- Define the base conversion functions
def base11_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2471 => 1 * 11^0 + 7 * 11^1 + 4 * 11^2 + 2 * 11^3
  | _    => 0

def base5_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 1 * 5^0 + 2 * 5^1 + 1 * 5^2
  | _   => 0

def base7_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 3654 => 4 * 7^0 + 5 * 7^1 + 6 * 7^2 + 3 * 7^3
  | _    => 0

def base8_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 5680 => 0 * 8^0 + 8 * 8^1 + 6 * 8^2 + 5 * 8^3
  | _    => 0

theorem math_problem_proof :
  let x := base11_to_base10 2471
  let y := base5_to_base10 121
  let z := base7_to_base10 3654
  let w := base8_to_base10 5680
  x / y - z + w = 1736 :=
by
  sorry

end math_problem_proof_l129_129883


namespace greatest_radius_l129_129961

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l129_129961


namespace min_w_value_l129_129616

open Real

noncomputable def w (x y : ℝ) : ℝ := 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27

theorem min_w_value : ∃ x y : ℝ, w x y = 81 / 4 :=
by
  use [-3/2, 1]
  dsimp [w]
  norm_num
  done

end min_w_value_l129_129616


namespace right_triangle_acute_angle_30_l129_129907

theorem right_triangle_acute_angle_30 (α β : ℝ) (h1 : α = 60) (h2 : α + β + 90 = 180) : β = 30 :=
by
  sorry

end right_triangle_acute_angle_30_l129_129907


namespace probability_AB_selected_l129_129378

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129378


namespace find_digit_sum_l129_129066

theorem find_digit_sum (A B X D C Y : ℕ) :
  (A * 100 + B * 10 + X) + (C * 100 + D * 10 + Y) = Y * 1010 + X * 1010 →
  A + D = 6 :=
by
  sorry

end find_digit_sum_l129_129066


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129460

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129460


namespace john_horizontal_distance_l129_129844

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end john_horizontal_distance_l129_129844


namespace buckets_required_l129_129582

theorem buckets_required (C : ℝ) (N : ℝ):
  (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  sorry

end buckets_required_l129_129582


namespace probability_A_and_B_selected_l129_129321

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129321


namespace find_second_number_l129_129730

theorem find_second_number (x y z : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x = (3/4) * y) 
  (h3 : z = (9/7) * y) 
  : y = 40 :=
sorry

end find_second_number_l129_129730


namespace vector_addition_proof_l129_129661

def vector_add (a b : ℤ × ℤ) : ℤ × ℤ :=
  (a.1 + b.1, a.2 + b.2)

theorem vector_addition_proof :
  let a := (2, 0)
  let b := (-1, -2)
  vector_add a b = (1, -2) :=
by
  sorry

end vector_addition_proof_l129_129661


namespace pascal_probability_first_twenty_rows_l129_129773

theorem pascal_probability_first_twenty_rows :
  let total_elements := (20 * (20 + 1)) / 2
  let num_ones := 1 + 2 * (20 - 1)
  let num_twos := 2 * (20 - 3)
  let num_ones_or_twos := num_ones + num_twos
  (num_ones_or_twos : ℚ) / total_elements = 73 / 210 := by
sorry

end pascal_probability_first_twenty_rows_l129_129773


namespace price_per_unit_max_profit_l129_129901

-- Part 1: Finding the Prices

theorem price_per_unit (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 690) 
  (h2 : x + 4 * y = 720) : 
  x = 120 ∧ y = 150 :=
by
  sorry

-- Part 2: Maximizing Profit

theorem max_profit (m : ℕ) 
  (h1 : m ≤ 3 * (40 - m)) 
  (h2 : 120 * m + 150 * (40 - m) ≤ 5400) : 
  (m = 20) ∧ (40 - m = 20) :=
by
  sorry

end price_per_unit_max_profit_l129_129901


namespace probability_A_B_selected_l129_129431

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129431


namespace select_3_from_5_prob_l129_129348

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129348


namespace probability_A_and_B_selected_l129_129333

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129333


namespace probability_A_and_B_selected_l129_129339

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129339


namespace mod_exp_equivalence_l129_129248

theorem mod_exp_equivalence :
  (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end mod_exp_equivalence_l129_129248


namespace problem_proof_equality_cases_l129_129112

theorem problem_proof (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (x * y - 10) ^ 2 ≥ 64 := sorry

theorem equality_cases (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : 
  (x * y - 10) ^ 2 = 64 ↔ ((x,y) = (1, 2) ∨ (x,y) = (-3, -6)) := sorry

end problem_proof_equality_cases_l129_129112


namespace probability_A_and_B_selected_l129_129410

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129410


namespace domain_of_f_parity_of_f_range_of_f_l129_129069

noncomputable def f (a x : ℝ) := Real.log (1 + x) / Real.log a - Real.log (1 - x) / Real.log a

variables {a x : ℝ}

-- The properties derived:
theorem domain_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (-1 < x ∧ x < 1) ↔ ∃ y, f a x = y :=
sorry

theorem parity_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  f a (-x) = -f a x :=
sorry

theorem range_of_f (ha : a > 0) (ha1 : a ≠ 1) : 
  (f a x > 0 ↔ (a > 1 ∧ 0 < x ∧ x < 1) ∨ (0 < a ∧ a < 1 ∧ -1 < x ∧ x < 0)) :=
sorry

end domain_of_f_parity_of_f_range_of_f_l129_129069


namespace positive_difference_of_two_numbers_l129_129170

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l129_129170


namespace prob_select_A_and_B_l129_129324

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129324


namespace probability_A_and_B_selected_l129_129418

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129418


namespace probability_A_and_B_selected_l129_129392

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129392


namespace john_total_spent_l129_129981

/-- John's expenditure calculations -/
theorem john_total_spent :
  let computer_cost := 1500
  let peripherals_cost := computer_cost / 5
  let original_video_card_cost := 300
  let upgraded_video_card_cost := original_video_card_cost * 2
  let additional_upgrade_cost := upgraded_video_card_cost - original_video_card_cost
  let total_spent := computer_cost + peripherals_cost + additional_upgrade_cost
  total_spent = 2100 :=
by
  sorry

end john_total_spent_l129_129981


namespace cos_135_eq_neg_sqrt2_div_2_l129_129034

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129034


namespace quadratic_inequality_l129_129875

variable {a b c : ℝ}

noncomputable def quadratic_polynomial (x : ℝ) := a * x^2 + b * x + c

theorem quadratic_inequality (h1 : ∀ x : ℝ, quadratic_polynomial x < 0)
    (h2 : a < 0) (h3 : b^2 - 4*a*c < 0) : (b / a) < (c / a + 1) := 
sorry

end quadratic_inequality_l129_129875


namespace borel_functions_l129_129560

open MeasureTheory

noncomputable def is_borel_function {α β : Type*} [MeasurableSpace α] [MeasurableSpace β] (f : α → β) : Prop :=
measurable f

theorem borel_functions :
  (∀ n : ℕ, is_borel_function (λ (x : ℝ), x ^ n)) ∧
  is_borel_function (λ (x : ℝ), max x 0) ∧
  is_borel_function (λ (x : ℝ), max (-x) 0) ∧
  is_borel_function (λ (x : ℝ), max x 0 + max (-x) 0) ∧
  (∀ {n : ℕ}, n ≥ 1 → ∀ f : ℝ^n → ℝ, continuous f → is_borel_function f) :=
by sorry

end borel_functions_l129_129560


namespace unique_positive_integer_divisibility_l129_129291

theorem unique_positive_integer_divisibility (n : ℕ) (h : n > 0) : 
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 :=
by
  sorry

end unique_positive_integer_divisibility_l129_129291


namespace right_triangle_height_l129_129906

theorem right_triangle_height
  (h : ℕ)
  (base : ℕ)
  (rectangle_area : ℕ)
  (same_area : (1 / 2 : ℚ) * base * h = rectangle_area)
  (base_eq_width : base = 5)
  (rectangle_area_eq : rectangle_area = 45) :
  h = 18 :=
by
  sorry

end right_triangle_height_l129_129906


namespace kiley_slices_eaten_l129_129261

def slices_of_cheesecake (total_calories_per_cheesecake calories_per_slice : ℕ) : ℕ :=
  total_calories_per_cheesecake / calories_per_slice

def slices_eaten (total_slices percentage_ate : ℚ) : ℚ :=
  total_slices * percentage_ate

theorem kiley_slices_eaten :
  ∀ (total_calories_per_cheesecake calories_per_slice : ℕ) (percentage_ate : ℚ),
  total_calories_per_cheesecake = 2800 →
  calories_per_slice = 350 →
  percentage_ate = (25 / 100 : ℚ) →
  slices_eaten (slices_of_cheesecake total_calories_per_cheesecake calories_per_slice) percentage_ate = 2 :=
by
  intros total_calories_per_cheesecake calories_per_slice percentage_ate h1 h2 h3
  rw [h1, h2, h3]
  sorry

end kiley_slices_eaten_l129_129261


namespace water_usage_difference_l129_129923

variable (a b : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (ha_plus_4 : a + 4 ≠ 0)

theorem water_usage_difference :
  b / a - b / (a + 4) = 4 * b / (a * (a + 4)) :=
by
  sorry

end water_usage_difference_l129_129923


namespace probability_both_A_B_selected_l129_129298

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129298


namespace probability_A_and_B_selected_l129_129403

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129403


namespace projection_identity_l129_129074

variables (P : ℝ × ℝ × ℝ) (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ)

-- Define point P as (-1, 3, -4)
def point_P := (-1, 3, -4) = P

-- Define projections on the coordinate planes
def projection_yoz := (x1, y1, z1) = (0, 3, -4)
def projection_zox := (x2, y2, z2) = (-1, 0, -4)
def projection_xoy := (x3, y3, z3) = (-1, 3, 0)

-- Prove that x1^2 + y2^2 + z3^2 = 0 under the given conditions
theorem projection_identity :
  point_P P ∧ projection_yoz x1 y1 z1 ∧ projection_zox x2 y2 z2 ∧ projection_xoy x3 y3 z3 →
  (x1^2 + y2^2 + z3^2 = 0) :=
by
  sorry

end projection_identity_l129_129074


namespace combined_value_of_cookies_l129_129761

theorem combined_value_of_cookies
  (total_boxes_sold : ℝ)
  (plain_boxes_sold : ℝ)
  (price_chocolate_chip : ℝ)
  (price_plain : ℝ)
  (h1 : total_boxes_sold = 1585)
  (h2 : plain_boxes_sold = 793.375)
  (h3 : price_chocolate_chip = 1.25)
  (h4 : price_plain = 0.75) :
  (plain_boxes_sold * price_plain) + ((total_boxes_sold - plain_boxes_sold) * price_chocolate_chip) = 1584.5625 :=
by
  sorry

end combined_value_of_cookies_l129_129761


namespace factorization_eq_l129_129288

theorem factorization_eq (x : ℝ) : 
  -3 * x^3 + 12 * x^2 - 12 * x = -3 * x * (x - 2)^2 :=
by
  sorry

end factorization_eq_l129_129288


namespace probability_AB_selected_l129_129373

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129373


namespace greatest_int_radius_lt_75pi_l129_129959

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l129_129959


namespace probability_A_B_selected_l129_129428

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129428


namespace math_problem_proof_l129_129956

theorem math_problem_proof (a b x y : ℝ) 
  (h1: x = a) 
  (h2: y = b)
  (h3: a + a = b * a)
  (h4: y = a)
  (h5: a * a = a + a)
  (h6: b = 3) : 
  x * y = 4 := 
by 
  sorry

end math_problem_proof_l129_129956


namespace spider_final_position_l129_129859

def circle_points : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def next_position (current : ℕ) : ℕ :=
  if current % 2 = 0 
  then (current + 3 - 1) % 7 + 1 -- Clockwise modulo operation for even
  else (current + 1 - 1) % 7 + 1 -- Clockwise modulo operation for odd

def spider_position_after_jumps (start : ℕ) (jumps : ℕ) : ℕ :=
  (Nat.iterate next_position jumps start)

theorem spider_final_position : spider_position_after_jumps 6 2055 = 2 := 
  by
  sorry

end spider_final_position_l129_129859


namespace range_of_reciprocals_l129_129667

theorem range_of_reciprocals (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) :
  ∃ c ∈ Set.Ici (9 : ℝ), (c = (1/a + 4/b)) :=
by
  sorry

end range_of_reciprocals_l129_129667


namespace probability_of_selecting_A_and_B_l129_129434

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129434


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129044

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129044


namespace positive_difference_of_numbers_l129_129197

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l129_129197


namespace infinite_points_with_sum_of_squares_condition_l129_129921

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle centered at origin with given radius
def isWithinCircle (P : Point2D) (r : ℝ) :=
  P.x^2 + P.y^2 ≤ r^2

-- Define the distance squared from a point to another point
def dist2 (P Q : Point2D) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the problem
theorem infinite_points_with_sum_of_squares_condition :
  ∃ P : Point2D, isWithinCircle P 1 → (dist2 P ⟨-1, 0⟩ + dist2 P ⟨1, 0⟩ = 3) :=
by  
  sorry

end infinite_points_with_sum_of_squares_condition_l129_129921


namespace probability_red_odd_green_special_l129_129583

theorem probability_red_odd_green_special : 
  let red_die := {1, 2, 3, 4, 5, 6, 7, 8}
  let green_die := {1, 2, 3, 4, 5, 6, 7, 8}
  let is_odd (x : Nat) : Prop := x % 2 = 1
  let is_perfect_square (x : Nat) : Prop := x = 1 ∨ x = 4
  let is_prime (x : Nat) : Prop := x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7
  let is_special (x : Nat) : Prop := is_perfect_square x ∨ is_prime x
  let outcomes := red_die.Product green_die
  let successful_outcomes := {(r, g) | r ∈ red_die ∧ g ∈ green_die ∧ is_odd r ∧ is_special g}
  let total_outcomes := 8 * 8
  let successful_outcomes_count := 4 * 6
  let probability := successful_outcomes_count / total_outcomes
  in probability = 3 / 8 :=
by {
  sorry
}

end probability_red_odd_green_special_l129_129583


namespace arithmetic_series_sum_l129_129729

theorem arithmetic_series_sum (n P q S₃n : ℕ) (h₁ : 2 * S₃n = 3 * P - q) : S₃n = 3 * P - q :=
by
  sorry

end arithmetic_series_sum_l129_129729


namespace tangent_line_to_curve_at_point_l129_129722

-- Define the function
def f (x : ℝ) : ℝ := x * (3 * Real.log x + 1)

-- The definition of the point
def point : (ℝ × ℝ) := (1, 1)

-- The equation of the tangent line at the given point
def tangent_line_eq (x : ℝ) : ℝ := 4 * x - 3

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ), (x = 1 ∧ y = 1) → (f x = y) → ∀ (t : ℝ), tangent_line_eq t = 4 * t - 3 := by
  assume x y hxy hfx t
  sorry

end tangent_line_to_curve_at_point_l129_129722


namespace probability_A_and_B_selected_l129_129499

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129499


namespace solve_for_n_l129_129712

theorem solve_for_n (n : ℝ) : 
  (0.05 * n + 0.06 * (30 + n)^2 = 45) ↔ 
  (n = -2.5833333333333335 ∨ n = -58.25) :=
sorry

end solve_for_n_l129_129712


namespace flower_problem_l129_129228

theorem flower_problem
  (O : ℕ) 
  (total : ℕ := 105)
  (pink_purple : ℕ := 30)
  (red := 2 * O)
  (yellow := 2 * O - 5)
  (pink := pink_purple / 2)
  (purple := pink)
  (H1 : pink + purple = pink_purple)
  (H2 : pink_purple = 30)
  (H3 : pink = purple)
  (H4 : O + red + yellow + pink + purple = total)
  (H5 : total = 105):
  O = 16 := 
by 
  sorry

end flower_problem_l129_129228


namespace find_a_l129_129849

theorem find_a (a : ℤ) (h1 : 0 < a) (h2 : ∀ (x : ℝ), |x - a| < 1 → x ∈ {x | x = 2}) : a = 2 :=
sorry

end find_a_l129_129849


namespace smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l129_129290

theorem smallest_k_repr_19_pow_n_sub_5_pow_m_exists :
  ∃ (k n m : ℕ), k > 0 ∧ n > 0 ∧ m > 0 ∧ k = 19 ^ n - 5 ^ m ∧ k = 14 :=
by
  sorry

end smallest_k_repr_19_pow_n_sub_5_pow_m_exists_l129_129290


namespace material_needed_for_second_type_l129_129922

namespace CherylProject

def first_material := 5 / 9
def leftover_material := 1 / 3
def total_material_used := 5 / 9

theorem material_needed_for_second_type :
  0.8888888888888889 - (5 / 9 : ℝ) = 0.3333333333333333 := by
  sorry

end CherylProject

end material_needed_for_second_type_l129_129922


namespace minimum_value_of_w_l129_129619

noncomputable def w (x y : ℝ) : ℝ := 3 * x ^ 2 + 3 * y ^ 2 + 9 * x - 6 * y + 27

theorem minimum_value_of_w : (∃ x y : ℝ, w x y = 20.25) := sorry

end minimum_value_of_w_l129_129619


namespace max_horizontal_segment_length_l129_129631

theorem max_horizontal_segment_length (y : ℝ → ℝ) (h : ∀ x, y x = x^3 - x) :
  ∃ a, (∀ x₁, y x₁ = y (x₁ + a)) ∧ a = 2 :=
by
  sorry

end max_horizontal_segment_length_l129_129631


namespace product_of_terms_l129_129837

variable {α : Type*} [LinearOrderedField α]

namespace GeometricSequence

def is_geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_of_terms (a : ℕ → α) (r : α) (h_geo : is_geometric_sequence a) :
  (a 4) * (a 8) = 16 → (a 2) * (a 10) = 16 :=
by
  intro h1
  sorry

end GeometricSequence

end product_of_terms_l129_129837


namespace fraction_of_mothers_with_full_time_jobs_l129_129529

theorem fraction_of_mothers_with_full_time_jobs :
  (0.4 : ℝ) * M = 0.3 →
  (9 / 10 : ℝ) * 0.6 = 0.54 →
  1 - 0.16 = 0.84 →
  0.84 - 0.54 = 0.3 →
  M = 3 / 4 :=
by
  intros h1 h2 h3 h4
  -- The proof steps would go here.
  sorry

end fraction_of_mothers_with_full_time_jobs_l129_129529


namespace number_of_girls_l129_129154

-- Define the number of girls and boys
variables (G B : ℕ)

-- Define the conditions
def condition1 : Prop := B = 2 * G - 16
def condition2 : Prop := G + B = 68

-- The theorem we want to prove
theorem number_of_girls (h1 : condition1 G B) (h2 : condition2 G B) : G = 28 :=
by
  sorry

end number_of_girls_l129_129154


namespace ants_square_paths_l129_129880

theorem ants_square_paths (a : ℝ) :
  (∃ a, a = 4 ∧ a + 2 = 6 ∧ a + 4 = 8) →
  (∀ (Mu Ra Vey : ℝ), 
    (Mu = (a + 4) / 2) ∧ 
    (Ra = (a + 2) / 2 + 1) ∧ 
    (Vey = 6) →
    (Mu + Ra + Vey = 2 * (a + 4) + 2)) :=
sorry

end ants_square_paths_l129_129880


namespace positive_difference_of_two_numbers_l129_129166

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129166


namespace result_number_of_edges_l129_129557

-- Define the conditions
def hexagon (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 6 ∧ edges = 6 * side_length)
def triangle (side_length : ℕ) : Prop := side_length = 1 ∧ (∃ edges, edges = 3 ∧ edges = 3 * side_length)

-- State the theorem
theorem result_number_of_edges (side_length_hex : ℕ) (side_length_tri : ℕ)
  (h_h : hexagon side_length_hex) (h_t : triangle side_length_tri)
  (aligned_edge_to_edge : side_length_hex = side_length_tri ∧ side_length_hex = 1 ∧ side_length_tri = 1) :
  ∃ edges, edges = 5 :=
by
  -- Proof is not provided, it is marked with sorry
  sorry

end result_number_of_edges_l129_129557


namespace positive_difference_of_two_numbers_l129_129194

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l129_129194


namespace pure_imaginary_solution_second_quadrant_solution_l129_129848

def isPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

def isSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

def complexNumber (m : ℝ) : ℂ :=
  ⟨m^2 - 2*m - 3, m^2 + 3*m + 2⟩

theorem pure_imaginary_solution (m : ℝ) : isPureImaginary (complexNumber m) ↔ m = 3 :=
by sorry

theorem second_quadrant_solution (m : ℝ) : isSecondQuadrant (complexNumber m) ↔ (-1 < m ∧ m < 3) :=
by sorry

end pure_imaginary_solution_second_quadrant_solution_l129_129848


namespace cos_135_eq_correct_l129_129020

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129020


namespace find_x_l129_129927

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉ * x = 220) : x = 14.67 :=
sorry

end find_x_l129_129927


namespace cos_135_eq_neg_sqrt2_div_2_l129_129036

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129036


namespace daniel_original_noodles_l129_129615

-- Define the total number of noodles Daniel had originally
def original_noodles : ℕ := 81

-- Define the remaining noodles after giving 1/3 to William
def remaining_noodles (n : ℕ) : ℕ := (2 * n) / 3

-- State the theorem
theorem daniel_original_noodles (n : ℕ) (h : remaining_noodles n = 54) : n = original_noodles := by sorry

end daniel_original_noodles_l129_129615


namespace probability_A_and_B_selected_l129_129336

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129336


namespace problem_statement_l129_129957

theorem problem_statement
  (m : ℝ) 
  (h : m + (1/m) = 5) :
  m^2 + (1 / m^2) + 4 = 27 :=
by
  -- Parameter types are chosen based on the context and problem description.
  sorry

end problem_statement_l129_129957


namespace prob_select_A_and_B_l129_129306

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129306


namespace combinatorial_sum_identity_l129_129786

theorem combinatorial_sum_identity :
  ∑ k in Finset.range 51, (-1: ℤ)^k * (k + 1) * Nat.choose 50 k = 0 := 
by
  sorry

end combinatorial_sum_identity_l129_129786


namespace john_trip_time_30_min_l129_129684

-- Definitions of the given conditions
variables {D : ℝ} -- Distance John traveled
variables {T : ℝ} -- Time John took
variable (T_john : ℝ) -- Time it took John (in hours)
variable (T_beth : ℝ) -- Time it took Beth (in hours)
variable (D_john : ℝ) -- Distance John traveled (in miles)
variable (D_beth : ℝ) -- Distance Beth traveled (in miles)

-- Given conditions
def john_speed := 40 -- John's speed in mph
def beth_speed := 30 -- Beth's speed in mph
def additional_distance := 5 -- Additional distance Beth traveled in miles
def additional_time := 1 / 3 -- Additional time Beth took in hours

-- Proving the time it took John to complete the trip is 30 minutes (0.5 hours)
theorem john_trip_time_30_min : 
  ∀ (T_john T_beth : ℝ), 
    T_john = (D) / john_speed →
    T_beth = (D + additional_distance) / beth_speed →
    (T_beth = T_john + additional_time) →
    T_john = 1 / 2 :=
by
  intro T_john T_beth
  sorry

end john_trip_time_30_min_l129_129684


namespace min_width_l129_129689

theorem min_width (w : ℝ) (h : w * (w + 20) ≥ 150) : w ≥ 10 := by
  sorry

end min_width_l129_129689


namespace denote_below_warning_level_l129_129156

-- Conditions
def warning_water_level : ℝ := 905.7
def exceed_by_10 : ℝ := 10
def below_by_5 : ℝ := -5

-- Problem statement
theorem denote_below_warning_level : below_by_5 = -5 := 
by
  sorry

end denote_below_warning_level_l129_129156


namespace share_of_C_l129_129134

variable (A B C x : ℝ)

theorem share_of_C (hA : A = (2/3) * B) 
(hB : B = (1/4) * C) 
(hTotal : A + B + C = 595) 
(hC : C = x) : x = 420 :=
by
  -- Proof will follow here
  sorry

end share_of_C_l129_129134


namespace adi_change_l129_129606

theorem adi_change : 
  let pencil := 0.35
  let notebook := 1.50
  let colored_pencils := 2.75
  let discount := 0.05
  let tax := 0.10
  let payment := 20.00
  let total_cost_before_discount := pencil + notebook + colored_pencils
  let discount_amount := discount * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  let tax_amount := tax * total_cost_after_discount
  let total_cost := total_cost_after_discount + tax_amount
  let change := payment - total_cost
  change = 15.19 :=
by
  sorry

end adi_change_l129_129606


namespace diagonal_of_larger_screen_l129_129727

theorem diagonal_of_larger_screen (d : ℝ) 
  (h1 : ∃ s : ℝ, s^2 = 20^2 + 42) 
  (h2 : ∀ s, d = s * Real.sqrt 2) : 
  d = Real.sqrt 884 :=
by
  sorry

end diagonal_of_larger_screen_l129_129727


namespace probability_A_and_B_selected_l129_129477

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129477


namespace determine_s_plus_u_l129_129250

theorem determine_s_plus_u (p r s u : ℂ) (q t : ℂ) (h₁ : q = 5)
    (h₂ : t = -p - r) (h₃ : p + q * I + r + s * I + t + u * I = 4 * I) : s + u = -1 :=
by
  sorry

end determine_s_plus_u_l129_129250


namespace max_radius_of_circle_l129_129967

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l129_129967


namespace cos_135_eq_neg_sqrt2_div_2_l129_129027

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129027


namespace percentage_girls_l129_129578

theorem percentage_girls (initial_boys : ℕ) (initial_girls : ℕ) (added_boys : ℕ) :
  initial_boys = 11 → initial_girls = 13 → added_boys = 1 → 
  100 * initial_girls / (initial_boys + added_boys + initial_girls) = 52 :=
by
  intros h_boys h_girls h_added
  sorry

end percentage_girls_l129_129578


namespace committee_selections_with_at_least_one_prev_served_l129_129774

-- Define the conditions
def total_candidates := 20
def previously_served := 8
def committee_size := 4
def never_served := total_candidates - previously_served

-- The proof problem statement
theorem committee_selections_with_at_least_one_prev_served : 
  (Nat.choose total_candidates committee_size - Nat.choose never_served committee_size) = 4350 :=
by
  sorry

end committee_selections_with_at_least_one_prev_served_l129_129774


namespace positive_difference_l129_129175

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l129_129175


namespace probability_AB_selected_l129_129379

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129379


namespace cos_135_eq_neg_sqrt2_div_2_l129_129040

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129040


namespace water_content_in_boxes_l129_129225

noncomputable def totalWaterInBoxes (num_boxes : ℕ) (bottles_per_box : ℕ) (capacity_per_bottle : ℚ) (fill_fraction : ℚ) : ℚ :=
  num_boxes * bottles_per_box * capacity_per_bottle * fill_fraction

theorem water_content_in_boxes :
  totalWaterInBoxes 10 50 12 (3 / 4) = 4500 := 
by
  sorry

end water_content_in_boxes_l129_129225


namespace find_ordered_pair_l129_129569

theorem find_ordered_pair (s m : ℚ) :
  (∃ t : ℚ, (5 * s - 7 = 2) ∧ 
           ((∃ (t1 : ℚ), (x = s + 3 * t1) ∧  (y = 2 + m * t1)) 
           → (x = 24 / 5) → (y = 5))) →
  (s = 9 / 5 ∧ m = 3) :=
by
  sorry

end find_ordered_pair_l129_129569


namespace symmetric_point_origin_l129_129811

theorem symmetric_point_origin (A : ℝ × ℝ) (A_sym : ℝ × ℝ) (h : A = (3, -2)) (h_sym : A_sym = (-A.1, -A.2)) : A_sym = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l129_129811


namespace gavin_shirts_l129_129796

theorem gavin_shirts (t g b : ℕ) (h_total : t = 23) (h_green : g = 17) (h_blue : b = t - g) : b = 6 :=
by sorry

end gavin_shirts_l129_129796


namespace prob_select_A_and_B_l129_129323

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129323


namespace train_speed_l129_129219

def length_of_train : ℝ := 160
def time_to_cross : ℝ := 18
def speed_in_kmh : ℝ := 32

theorem train_speed :
  (length_of_train / time_to_cross) * 3.6 = speed_in_kmh :=
by
  sorry

end train_speed_l129_129219


namespace probability_AB_selected_l129_129377

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129377


namespace cos_135_eq_neg_sqrt2_div_2_l129_129028

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129028


namespace cos_135_eq_correct_l129_129023

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129023


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129041

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129041


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129046

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129046


namespace factorization_identity_l129_129287

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l129_129287


namespace div_1988_form_1989_div_1989_form_1988_l129_129892

/-- There exists a number of the form 1989...19890... (1989 repeated several times followed by several zeros), which is divisible by 1988. -/
theorem div_1988_form_1989 (k : ℕ) : ∃ n : ℕ, (n = 1989 * 10^(4*k) ∧ n % 1988 = 0) := sorry

/-- There exists a number of the form 1988...1988 (1988 repeated several times), which is divisible by 1989. -/
theorem div_1989_form_1988 (k : ℕ) : ∃ n : ℕ, (n = 1988 * ((10^(4*k)) - 1) ∧ n % 1989 = 0) := sorry

end div_1988_form_1989_div_1989_form_1988_l129_129892


namespace cos_135_eq_neg_inv_sqrt_2_l129_129008

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129008


namespace max_edges_of_colored_graph_l129_129233

noncomputable def max_edges_colored_graph (G : SimpleGraph (Fin 2020)) (edges_colored : Fin 2020 → Fin 2020 → Prop) : ℕ :=
  sorry

theorem max_edges_of_colored_graph (G : SimpleGraph (Fin 2020)) (edges_colored : Fin 2020 → Fin 2020 → Prop) (h1 : ∀ u v, edges_colored u v = (edges_colored v u)) (h2 : ∀ cycle, is_monochromatic cycle -> even_length cycle) :
  max_edges_colored_graph G edges_colored = 1530150 :=
begin
  sorry,
end

end max_edges_of_colored_graph_l129_129233


namespace find_sum_of_terms_l129_129073

noncomputable def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d
noncomputable def sum_of_first_n_terms (a₁ d n : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem find_sum_of_terms (a₁ d : ℕ) (S : ℕ → ℕ) (h1 : S 4 = 8) (h2 : S 8 = 20) :
    S 4 = 4 * (2 * a₁ + 3 * d) / 2 → S 8 = 8 * (2 * a₁ + 7 * d) / 2 →
    a₁ = 13 / 8 ∧ d = 1 / 4 →
    a₁ + 10 * d + a₁ + 11 * d + a₁ + 12 * d + a₁ + 13 * d = 18 :=
by 
  sorry

end find_sum_of_terms_l129_129073


namespace probability_A_and_B_selected_l129_129393

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129393


namespace three_digit_number_problem_l129_129874

theorem three_digit_number_problem (c d : ℕ) (h1 : 400 + c*10 + 1 = 786 - (300 + d*10 + 5)) (h2 : (300 + d*10 + 5) % 7 = 0) : c + d = 8 := 
sorry

end three_digit_number_problem_l129_129874


namespace percent_employed_females_l129_129533

theorem percent_employed_females (total_population employed_population employed_males : ℝ)
  (h1 : employed_population = 0.6 * total_population)
  (h2 : employed_males = 0.48 * total_population) :
  ((employed_population - employed_males) / employed_population) * 100 = 20 := 
by
  sorry

end percent_employed_females_l129_129533


namespace cost_of_4_stamps_l129_129596

theorem cost_of_4_stamps (cost_per_stamp : ℕ) (h : cost_per_stamp = 34) : 4 * cost_per_stamp = 136 :=
by
  sorry

end cost_of_4_stamps_l129_129596


namespace cos_135_eq_neg_sqrt2_div_2_l129_129030

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129030


namespace setC_not_pythagorean_l129_129771

/-- Defining sets of numbers as options -/
def SetA := (3, 4, 5)
def SetB := (5, 12, 13)
def SetC := (7, 25, 26)
def SetD := (6, 8, 10)

/-- Function to check if a set is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem stating set C is not a Pythagorean triple -/
theorem setC_not_pythagorean :
  ¬isPythagoreanTriple 7 25 26 :=
by {
  -- This slot will be filled with the concrete proof steps in Lean.
  sorry
}

end setC_not_pythagorean_l129_129771


namespace water_added_l129_129226

theorem water_added (initial_fullness : ℝ) (fullness_after : ℝ) (capacity : ℝ) 
  (h_initial : initial_fullness = 0.30) (h_after : fullness_after = 3/4) (h_capacity : capacity = 100) : 
  fullness_after * capacity - initial_fullness * capacity = 45 := 
by 
  sorry

end water_added_l129_129226


namespace cos_135_eq_neg_inv_sqrt_2_l129_129007

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129007


namespace profit_per_meter_l129_129913

theorem profit_per_meter 
  (total_meters : ℕ)
  (cost_price_per_meter : ℝ)
  (total_selling_price : ℝ)
  (h1 : total_meters = 92)
  (h2 : cost_price_per_meter = 83.5)
  (h3 : total_selling_price = 9890) : 
  (total_selling_price - total_meters * cost_price_per_meter) / total_meters = 24.1 :=
by
  sorry

end profit_per_meter_l129_129913


namespace min_distance_squared_l129_129642

theorem min_distance_squared (a b c d : ℝ) (e : ℝ) (h₀ : e = Real.exp 1) 
  (h₁ : (a - 2 * Real.exp a) / b = 1) (h₂ : (2 - c) / d = 1) :
  (a - c)^2 + (b - d)^2 = 8 := by
  sorry

end min_distance_squared_l129_129642


namespace cubic_vs_square_ratio_l129_129646

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l129_129646


namespace train_cross_time_l129_129605

-- Define the conditions
def train_speed_kmhr := 52
def train_length_meters := 130

-- Conversion factor from km/hr to m/s
def kmhr_to_ms (speed_kmhr : ℕ) : ℕ := (speed_kmhr * 1000) / 3600

-- Speed of the train in m/s
def train_speed_ms := kmhr_to_ms train_speed_kmhr

-- Calculate time to cross the pole
def time_to_cross_pole (distance_m : ℕ) (speed_ms : ℕ) : ℕ := distance_m / speed_ms

-- The theorem to prove
theorem train_cross_time : time_to_cross_pole train_length_meters train_speed_ms = 9 := by sorry

end train_cross_time_l129_129605


namespace even_function_and_monotonicity_l129_129254

noncomputable def f (x : ℝ) : ℝ := sorry

theorem even_function_and_monotonicity (f_symm : ∀ x : ℝ, f x = f (-x))
  (f_inc_neg : ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → x1 ≤ 0 → x2 ≤ 0 → f x1 < f x2)
  (n : ℕ) (hn : n > 0) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) := 
sorry

end even_function_and_monotonicity_l129_129254


namespace sqrt_of_mixed_fraction_simplified_l129_129791

theorem sqrt_of_mixed_fraction_simplified :
  let x := 8 + (9 / 16) in
  sqrt x = (sqrt 137) / 4 := by
  sorry

end sqrt_of_mixed_fraction_simplified_l129_129791


namespace min_n_value_l129_129510

theorem min_n_value (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, k * k = 12 * n) : n ≥ 3 :=
by
  sorry

example : min_n_value 3 (by norm_num) (by norm_num; use 6; norm_num) := 
by
  sorry

end min_n_value_l129_129510


namespace greatest_integer_func_l129_129625

noncomputable def pi_approx : ℝ := 3.14159

theorem greatest_integer_func : (⌊2 * pi_approx - 6⌋ : ℝ) = 0 := 
by
  sorry

end greatest_integer_func_l129_129625


namespace conic_section_is_ellipse_l129_129589

/-- Given two fixed points (0, 2) and (4, -1) and the equation 
    sqrt(x^2 + (y - 2)^2) + sqrt((x - 4)^2 + (y + 1)^2) = 12, 
    prove that the conic section is an ellipse. -/
theorem conic_section_is_ellipse 
  (x y : ℝ)
  (h : Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 4)^2 + (y + 1)^2) = 12) :
  ∃ (F1 F2 : ℝ × ℝ), 
    F1 = (0, 2) ∧ 
    F2 = (4, -1) ∧ 
    ∀ (P : ℝ × ℝ), P = (x, y) → 
      Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2) + 
      Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2) = 12 := 
sorry

end conic_section_is_ellipse_l129_129589


namespace evaluate_fraction_l129_129061

theorem evaluate_fraction:
  (125 : ℝ)^(1/3) / (64 : ℝ)^(1/2) * (81 : ℝ)^(1/4) = 15 / 8 := 
by
  sorry

end evaluate_fraction_l129_129061


namespace additional_height_last_two_floors_l129_129898

-- Definitions of the problem conditions
def num_floors : ℕ := 20
def height_per_floor : ℕ := 3
def building_total_height : ℤ := 61

-- Condition on the height of first 18 floors
def height_first_18_floors : ℤ := 18 * 3

-- Height of the last two floors
def height_last_two_floors : ℤ := building_total_height - height_first_18_floors
def height_each_last_two_floor : ℤ := height_last_two_floors / 2

-- Height difference between the last two floors and the first 18 floors
def additional_height : ℤ := height_each_last_two_floor - height_per_floor

-- Theorem to prove
theorem additional_height_last_two_floors :
  additional_height = 1 / 2 := 
sorry

end additional_height_last_two_floors_l129_129898


namespace inequality_solution_l129_129801

theorem inequality_solution (a : ℝ) (h : 1 < a) : ∀ x : ℝ, a ^ (2 * x + 1) > (1 / a) ^ (2 * x) ↔ x > -1 / 4 :=
by
  sorry

end inequality_solution_l129_129801


namespace total_fencing_cost_is_correct_l129_129525

-- Defining the lengths of each side
def length1 : ℝ := 50
def length2 : ℝ := 75
def length3 : ℝ := 60
def length4 : ℝ := 80
def length5 : ℝ := 65

-- Defining the cost per unit length for each side
def cost_per_meter1 : ℝ := 2
def cost_per_meter2 : ℝ := 3
def cost_per_meter3 : ℝ := 4
def cost_per_meter4 : ℝ := 3.5
def cost_per_meter5 : ℝ := 5

-- Calculating the total cost for each side
def cost1 : ℝ := length1 * cost_per_meter1
def cost2 : ℝ := length2 * cost_per_meter2
def cost3 : ℝ := length3 * cost_per_meter3
def cost4 : ℝ := length4 * cost_per_meter4
def cost5 : ℝ := length5 * cost_per_meter5

-- Summing up the total cost for all sides
def total_cost : ℝ := cost1 + cost2 + cost3 + cost4 + cost5

-- The theorem to be proven
theorem total_fencing_cost_is_correct : total_cost = 1170 := by
  sorry

end total_fencing_cost_is_correct_l129_129525


namespace probability_of_A_and_B_selected_l129_129490

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129490


namespace arlene_hike_distance_l129_129240

-- Define the conditions: Arlene's pace and the time she spent hiking
def arlene_pace : ℝ := 4 -- miles per hour
def arlene_time_hiking : ℝ := 6 -- hours

-- Define the problem statement and provide the mathematical proof
theorem arlene_hike_distance :
  arlene_pace * arlene_time_hiking = 24 :=
by
  -- This is where the proof would go
  sorry

end arlene_hike_distance_l129_129240


namespace weekly_allowance_l129_129087

variable (A : ℝ)   -- declaring A as a real number

theorem weekly_allowance (h1 : (3/5 * A) + 1/3 * (2/5 * A) + 1 = A) : 
  A = 3.75 :=
sorry

end weekly_allowance_l129_129087


namespace cubic_vs_square_ratio_l129_129644

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l129_129644


namespace hyperbola_standard_equation_l129_129657

theorem hyperbola_standard_equation (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h_real_axis : 2 * a = 4 * Real.sqrt 2) (h_eccentricity : a / Real.sqrt (a^2 + b^2) = Real.sqrt 6 / 2) :
    (a = 2 * Real.sqrt 2) ∧ (b = 2) → ∀ x y : ℝ, (x^2)/8 - (y^2)/4 = 1 :=
sorry

end hyperbola_standard_equation_l129_129657


namespace cos_135_eq_neg_sqrt2_div_2_l129_129026

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129026


namespace fraction_white_surface_area_l129_129600

-- Definitions for conditions
def larger_cube_side : ℕ := 3
def smaller_cube_count : ℕ := 27
def white_cube_count : ℕ := 19
def black_cube_count : ℕ := 8
def black_corners : Nat := 8
def faces_per_cube : ℕ := 6
def exposed_faces_per_corner : ℕ := 3

-- Theorem statement for proving the fraction of the white surface area
theorem fraction_white_surface_area : (30 : ℚ) / 54 = 5 / 9 :=
by 
  -- Add the proof steps here if necessary
  sorry

end fraction_white_surface_area_l129_129600


namespace john_horizontal_distance_l129_129845

-- Define the conditions and the question
def elevation_initial : ℕ := 100
def elevation_final : ℕ := 1450
def vertical_to_horizontal_ratio (v h : ℕ) : Prop := v * 2 = h

-- Define the proof problem: the horizontal distance John moves
theorem john_horizontal_distance : ∃ h, vertical_to_horizontal_ratio (elevation_final - elevation_initial) h ∧ h = 2700 := 
by 
  sorry

end john_horizontal_distance_l129_129845


namespace probability_AB_selected_l129_129375

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129375


namespace probability_A_B_l129_129352

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129352


namespace probability_both_A_B_selected_l129_129292

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129292


namespace number_of_buses_l129_129757

theorem number_of_buses (total_people : ℕ) (bus_capacity : ℕ) (h1 : total_people = 1230) (h2 : bus_capacity = 48) : 
  Nat.ceil (total_people / bus_capacity : ℝ) = 26 := 
by 
  unfold Nat.ceil 
  sorry

end number_of_buses_l129_129757


namespace time_for_type_Q_machine_l129_129766

theorem time_for_type_Q_machine (Q : ℝ) (h1 : Q > 0)
  (h2 : 2 * (1 / Q) + 3 * (1 / 7) = 5 / 6) :
  Q = 84 / 17 :=
sorry

end time_for_type_Q_machine_l129_129766


namespace cube_edge_length_l129_129527

theorem cube_edge_length (a : ℝ) (h : 6 * a^2 = 24) : a = 2 :=
by sorry

end cube_edge_length_l129_129527


namespace exist_coprime_integers_l129_129633

theorem exist_coprime_integers:
  ∀ (a b p : ℤ), ∃ (k l : ℤ), Int.gcd k l = 1 ∧ p ∣ (a * k + b * l) :=
by
  sorry

end exist_coprime_integers_l129_129633


namespace probability_of_selecting_A_and_B_l129_129365

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129365


namespace prob_first_class_individual_prob_at_least_one_first_class_part_l129_129544

open ProbabilityTheory

variable {Ω : Type*} {𝒜 : MeasurableSpace Ω} [ProbabilitySpace Ω]

-- Conditions as probabilities
variable (A B C : Set Ω)
variable (probAovrB : ℙ(A ∩ Bᶜ) = 1/4)
variable (probBovrC : ℙ(B ∩ Cᶜ) = 1/12)
variable (probAC : ℙ(A ∩ C) = 2/9)

-- Prove the individual probabilities
theorem prob_first_class_individual :
  ℙ(A) = 1/3 ∧
  ℙ(B) = 1/4 ∧
  ℙ(C) = 2/3 :=
sorry

-- Define event D for getting at least one first-class part
def D := (A ∪ B ∪ C)

-- Prove the probability of at least one first-class part
theorem prob_at_least_one_first_class_part :
  ℙ(D) = 5/6 :=
sorry

end prob_first_class_individual_prob_at_least_one_first_class_part_l129_129544


namespace probability_of_selecting_A_and_B_l129_129470

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129470


namespace cos_135_eq_correct_l129_129021

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129021


namespace problem1_problem2_l129_129612

-- Problem 1: Prove the expression equals 5
theorem problem1 : (1 : ℚ) * ((1/3 : ℚ) - (3/4) + (5/6)) / (1/12) = 5 := by
  sorry

-- Problem 2: Prove the expression equals 7
theorem problem2 : ((-1 : ℤ)^2023 + |(1 - 0.5 : ℚ)| * ((-4)^2)) = 7 := by
  sorry

end problem1_problem2_l129_129612


namespace point_of_tangent_parallel_x_axis_l129_129819

theorem point_of_tangent_parallel_x_axis :
  ∃ M : ℝ × ℝ, (M.1 = -1 ∧ M.2 = -3) ∧
    (∃ y : ℝ, y = M.1^2 + 2 * M.1 - 2 ∧
    (∃ y' : ℝ, y' = 2 * M.1 + 2 ∧ y' = 0)) :=
sorry

end point_of_tangent_parallel_x_axis_l129_129819


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l129_129705

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l129_129705


namespace setC_is_not_pythagorean_triple_l129_129770

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of numbers
def setA := (3, 4, 5)
def setB := (5, 12, 13)
def setC := (7, 25, 26)
def setD := (6, 8, 10)

-- The theorem stating that setC is not a Pythagorean triple
theorem setC_is_not_pythagorean_triple : ¬isPythagoreanTriple 7 25 26 := 
by sorry

end setC_is_not_pythagorean_triple_l129_129770


namespace jordan_width_45_l129_129246

noncomputable def carolRectangleLength : ℕ := 15
noncomputable def carolRectangleWidth : ℕ := 24
noncomputable def jordanRectangleLength : ℕ := 8
noncomputable def carolRectangleArea : ℕ := carolRectangleLength * carolRectangleWidth
noncomputable def jordanRectangleWidth (area : ℕ) : ℕ := area / jordanRectangleLength

theorem jordan_width_45 : jordanRectangleWidth carolRectangleArea = 45 :=
by sorry

end jordan_width_45_l129_129246


namespace factor_quadratic_l129_129281

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l129_129281


namespace christmas_bonus_remainder_l129_129552

theorem christmas_bonus_remainder (P : ℕ) (h : P % 5 = 2) : (3 * P) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l129_129552


namespace probability_of_selecting_A_and_B_l129_129366

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129366


namespace probability_A_and_B_selected_l129_129415

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129415


namespace number_of_allowed_pairs_l129_129521

theorem number_of_allowed_pairs (total_books : ℕ) (prohibited_books : ℕ) : ℕ :=
  let total_pairs := (total_books * (total_books - 1)) / 2
  let prohibited_pairs := (prohibited_books * (prohibited_books - 1)) / 2
  total_pairs - prohibited_pairs

example : number_of_allowed_pairs 15 3 = 102 :=
by
  sorry

end number_of_allowed_pairs_l129_129521


namespace inequality_proof_l129_129651

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l129_129651


namespace cos_135_eq_neg_inv_sqrt_2_l129_129003

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129003


namespace probability_both_A_and_B_selected_l129_129444

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129444


namespace tank_capacity_l129_129891

theorem tank_capacity :
  (∃ (C : ℕ), ∀ (leak_rate inlet_rate net_rate : ℕ),
    leak_rate = C / 6 ∧
    inlet_rate = 6 * 60 ∧
    net_rate = C / 12 ∧
    inlet_rate - leak_rate = net_rate → C = 1440) :=
sorry

end tank_capacity_l129_129891


namespace choir_average_age_l129_129144

theorem choir_average_age 
  (avg_f : ℝ) (n_f : ℕ)
  (avg_m : ℝ) (n_m : ℕ)
  (h_f : avg_f = 28) 
  (h_nf : n_f = 12) 
  (h_m : avg_m = 40) 
  (h_nm : n_m = 18) 
  : (n_f * avg_f + n_m * avg_m) / (n_f + n_m) = 35.2 := 
by 
  sorry

end choir_average_age_l129_129144


namespace find_real_numbers_l129_129628

theorem find_real_numbers (x y : ℝ) (h₁ : x + y = 3) (h₂ : x^5 + y^5 = 33) :
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end find_real_numbers_l129_129628


namespace sin_cos_pow_eq_l129_129935

theorem sin_cos_pow_eq (sin cos : ℝ → ℝ) (x : ℝ) (h₀ : sin x + cos x = -1) (n : ℕ) : 
  sin x ^ n + cos x ^ n = (-1) ^ n :=
by
  sorry

end sin_cos_pow_eq_l129_129935


namespace probability_A_and_B_selected_l129_129479

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129479


namespace find_blue_shirts_l129_129799

-- Statements of the problem conditions
def total_shirts : ℕ := 23
def green_shirts : ℕ := 17

-- Definition that we want to prove
def blue_shirts : ℕ := total_shirts - green_shirts

-- Proof statement (no need to include the proof itself)
theorem find_blue_shirts : blue_shirts = 6 := by
  sorry

end find_blue_shirts_l129_129799


namespace not_cube_of_sum_l129_129559

theorem not_cube_of_sum (a b : ℕ) : ¬ ∃ (k : ℤ), a^3 + b^3 + 4 = k^3 :=
by
  sorry

end not_cube_of_sum_l129_129559


namespace union_condition_implies_l129_129940

-- Define set A as per the given condition
def setA : Set ℝ := { x | x * (x - 1) ≤ 0 }

-- Define set B as per the given condition with parameter a
def setB (a : ℝ) : Set ℝ := { x | Real.log x ≤ a }

-- Given condition A ∪ B = A, we need to prove that a ≤ 0
theorem union_condition_implies (a : ℝ) (h : setA ∪ setB a = setA) : a ≤ 0 := 
by
  sorry

end union_condition_implies_l129_129940


namespace john_horizontal_distance_l129_129842

theorem john_horizontal_distance (v_increase : ℕ)
  (elevation_start : ℕ) (elevation_end : ℕ) (h_ratio : ℕ) :
  (elevation_end - elevation_start) * h_ratio = 1350 * 2 :=
begin
  -- Let elevation_start be 100 feet
  let elevation_start := 100,
  -- Let elevation_end be 1450 feet
  let elevation_end := 1450,
  -- The steepened ratio, height per step
  let v_increase := elevation_end - elevation_start,
  -- John travels 1 foot vertically for every 2 feet horizontally
  let h_ratio := 2,
  -- Hence the horizontal distance theorem
  -- s_foot = h_ratio * t_foot = 1350 * 2 = 2700 feet.
  sorry
end

end john_horizontal_distance_l129_129842


namespace probability_A_wins_championship_expectation_X_is_13_l129_129709

/-
Definitions corresponding to the conditions in the problem
-/
def prob_event1_A_win : ℝ := 0.5
def prob_event2_A_win : ℝ := 0.4
def prob_event3_A_win : ℝ := 0.8

def prob_event1_B_win : ℝ := 1 - prob_event1_A_win
def prob_event2_B_win : ℝ := 1 - prob_event2_A_win
def prob_event3_B_win : ℝ := 1 - prob_event3_A_win

/-
Proof problems corresponding to the questions and correct answers
-/

theorem probability_A_wins_championship : prob_event1_A_win * prob_event2_A_win * prob_event3_A_win
    + prob_event1_A_win * prob_event2_A_win * prob_event3_B_win
    + prob_event1_A_win * prob_event2_B_win * prob_event3_A_win 
    + prob_event1_B_win * prob_event2_A_win * prob_event3_A_win = 0.6 := 
sorry

noncomputable def X_distribution_table : list (ℝ × ℝ) := 
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

noncomputable def expected_value_X : ℝ := 
  ∑ x in X_distribution_table, x.1 * x.2

theorem expectation_X_is_13 : expected_value_X = 13 := sorry

end probability_A_wins_championship_expectation_X_is_13_l129_129709


namespace cube_fraction_inequality_l129_129650

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l129_129650


namespace factorization_identity_l129_129285

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l129_129285


namespace largest_of_numbers_l129_129916

theorem largest_of_numbers (a b c d : ℝ) 
  (ha : a = 0) (hb : b = -1) (hc : c = 3.5) (hd : d = Real.sqrt 13) : 
  ∃ x, x = Real.sqrt 13 ∧ (x > a) ∧ (x > b) ∧ (x > c) ∧ (x > d) :=
by
  sorry

end largest_of_numbers_l129_129916


namespace b_100_is_15001_5_l129_129924

def sequence_b : ℕ → ℝ
| 0       => 0  -- We will define b_1 as sequence_b 1, so b_0 is irrelevant.
| 1       => 3
| (n + 1) => sequence_b n + 3 * n

theorem b_100_is_15001_5 : sequence_b 100 = 15001.5 :=
  sorry

end b_100_is_15001_5_l129_129924


namespace cos_135_eq_neg_inv_sqrt_2_l129_129053

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129053


namespace students_arrangement_l129_129917

def num_students := 5
def num_females := 2
def num_males := 3
def female_A_cannot_end := true
def only_two_males_next_to_each_other := true

theorem students_arrangement (h1: num_students = 5)
                             (h2: num_females = 2)
                             (h3: num_males = 3)
                             (h4: female_A_cannot_end = true)
                             (h5: only_two_males_next_to_each_other = true) :
    ∃ n, n = 48 :=
by
  sorry

end students_arrangement_l129_129917


namespace probability_A_and_B_selected_l129_129338

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129338


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129045

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129045


namespace units_digit_of_five_consecutive_product_is_zero_l129_129210

theorem units_digit_of_five_consecutive_product_is_zero (n : ℕ) :
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10 = 0 :=
by
  sorry

end units_digit_of_five_consecutive_product_is_zero_l129_129210


namespace intersection_eq_l129_129939

def setA : Set ℕ := {0, 1, 2, 3, 4, 5 }
def setB : Set ℕ := { x | |(x : ℤ) - 2| ≤ 1 }

theorem intersection_eq :
  setA ∩ setB = {1, 2, 3} := by
  sorry

end intersection_eq_l129_129939


namespace computation_result_l129_129779

def a : ℕ := 3
def b : ℕ := 5
def c : ℕ := 7

theorem computation_result :
  (a + b + c) ^ 2 + (a ^ 2 + b ^ 2 + c ^ 2) = 308 := by
  sorry

end computation_result_l129_129779


namespace probability_of_A_and_B_selected_l129_129491

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129491


namespace factorization_identity_l129_129283

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l129_129283


namespace remainder_of_3_to_40_plus_5_mod_5_l129_129215

theorem remainder_of_3_to_40_plus_5_mod_5 : (3^40 + 5) % 5 = 1 :=
by
  sorry

end remainder_of_3_to_40_plus_5_mod_5_l129_129215


namespace greatest_integer_radius_l129_129964

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l129_129964


namespace boar_sausages_left_l129_129551

def boar_sausages_final_count(sausages_initial : ℕ) : ℕ :=
  let after_monday := sausages_initial - (2 / 5 * sausages_initial)
  let after_tuesday := after_monday - (1 / 2 * after_monday)
  let after_wednesday := after_tuesday - (1 / 4 * after_tuesday)
  let after_thursday := after_wednesday - (1 / 3 * after_wednesday)
  let after_sharing := after_thursday - (1 / 5 * after_thursday)
  let after_eating := after_sharing - (3 / 5 * after_sharing)
  after_eating

theorem boar_sausages_left : boar_sausages_final_count 1200 = 58 := 
  sorry

end boar_sausages_left_l129_129551


namespace sum_of_arithmetic_sequence_l129_129655

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : a 5 + a 4 = 18) (hS_def : ∀ n, S n = n * (a 1 + a n) / 2) : S 8 = 72 := 
sorry

end sum_of_arithmetic_sequence_l129_129655


namespace number_of_persimmons_l129_129732

theorem number_of_persimmons (t p total : ℕ) (h1 : t = 19) (h2 : total = 37) (h3 : t + p = total) : p = 18 := 
by 
  rw [h1, h2] at h3
  linarith

end number_of_persimmons_l129_129732


namespace parallel_planes_transitivity_l129_129078

-- Define different planes α, β, γ
variables (α β γ : Plane)

-- Define the parallel relation between planes
axiom parallel : Plane → Plane → Prop

-- Conditions
axiom diff_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ
axiom β_parallel_α : parallel β α
axiom γ_parallel_α : parallel γ α

-- Statement to prove
theorem parallel_planes_transitivity (α β γ : Plane) 
  (h1 : parallel β α) 
  (h2 : parallel γ α) 
  (h3 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) : parallel β γ :=
sorry

end parallel_planes_transitivity_l129_129078


namespace shelby_stars_yesterday_l129_129711

-- Define the number of stars earned yesterday
def stars_yesterday : ℕ := sorry

-- Condition 1: In all, Shelby earned 7 gold stars
def stars_total : ℕ := 7

-- Condition 2: Today, she earned 3 more gold stars
def stars_today : ℕ := 3

-- The proof statement that combines the conditions 
-- and question to the correct answer
theorem shelby_stars_yesterday (y : ℕ) (h1 : y + stars_today = stars_total) : y = 4 := 
by
  -- Placeholder for the actual proof
  sorry

end shelby_stars_yesterday_l129_129711


namespace solution_set_l129_129058

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ) -- Function for the derivative of f

axiom f_deriv : ∀ x, f' x = (deriv f) x

axiom f_condition1 : ∀ x, f x > 1 - f' x
axiom f_condition2 : f 0 = 0
  
theorem solution_set (x : ℝ) : (e^x * f x > e^x - 1) ↔ (x > 0) := 
  sorry

end solution_set_l129_129058


namespace quadratic_equation_solution_l129_129826

theorem quadratic_equation_solution (m : ℝ) :
  (m - 3) * x ^ (m^2 - 7) - x + 3 = 0 → m^2 - 7 = 2 → m ≠ 3 → m = -3 :=
by
  intros h_eq h_power h_nonzero
  sorry

end quadratic_equation_solution_l129_129826


namespace range_of_m_l129_129070

noncomputable def f (x : ℝ) : ℝ := -x^2
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := 2^x - m

theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc (-1 : ℝ) 3, ∃ x2 ∈ Set.Icc (0 : ℝ) 2, f x1 ≥ g x2 m) ↔ m ≥ 10 := 
by
  sorry

end range_of_m_l129_129070


namespace parallelogram_area_leq_half_triangle_area_l129_129104

-- Definition of a triangle and a parallelogram inside it.
structure Triangle (α : Type) [LinearOrderedField α] :=
(A B C : α × α)

structure Parallelogram (α : Type) [LinearOrderedField α] :=
(P Q R S : α × α)

-- Function to calculate the area of a triangle
def triangle_area {α : Type} [LinearOrderedField α] (T : Triangle α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Function to calculate the area of a parallelogram
def parallelogram_area {α : Type} [LinearOrderedField α] (P : Parallelogram α) : α :=
-- Placeholder for the actual area calculation formula
sorry

-- Statement of the problem
theorem parallelogram_area_leq_half_triangle_area {α : Type} [LinearOrderedField α]
(T : Triangle α) (P : Parallelogram α) (inside : P.P.1 < T.A.1 ∧ P.P.2 < T.C.1) : 
  parallelogram_area P ≤ 1 / 2 * triangle_area T :=
sorry

end parallelogram_area_leq_half_triangle_area_l129_129104


namespace minimum_AP_BP_CP_DP_EP_l129_129890

-- Define points and constants
def A : ℝ := 0
def B : ℝ := 3
def C : ℝ := 4
def D : ℝ := 9
def E : ℝ := 13

-- Define the function S(x)
def S (x : ℝ) : ℝ := (x - A)^2 + (x - B)^2 + (x - C)^2 + (x - D)^2 + (x - E)^2

-- Derivative of S(x)
def dS_dx (x : ℝ) : ℝ := 2 * (x - A) + 2 * (x - B) + 2 * (x - C) + 2 * (x - D) + 2 * (x - E)

-- Statement of the theorem
theorem minimum_AP_BP_CP_DP_EP :
  (∃ x : ℝ, (S x = 170.24 ∧ dS_dx x = 0)) :=
sorry

end minimum_AP_BP_CP_DP_EP_l129_129890


namespace probability_of_selecting_A_and_B_l129_129362

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129362


namespace probability_of_selecting_A_and_B_l129_129440

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129440


namespace sufficient_conditions_for_quadratic_l129_129821

theorem sufficient_conditions_for_quadratic (x : ℝ) : 
  (0 < x ∧ x < 4) ∨ (-2 < x ∧ x < 4) ∨ (-2 < x ∧ x < 3) → x^2 - 2*x - 8 < 0 :=
by
  sorry

end sufficient_conditions_for_quadratic_l129_129821


namespace factorization_of_polynomial_l129_129265

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l129_129265


namespace probability_A_and_B_selected_l129_129497

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129497


namespace find_a_from_complex_condition_l129_129513

theorem find_a_from_complex_condition (a : ℝ) (x y : ℝ) 
  (h : x = -1 ∧ y = -2 * a)
  (h_line : x - y = 0) : a = 1 / 2 :=
by
  sorry

end find_a_from_complex_condition_l129_129513


namespace journey_total_time_l129_129881

def journey_time (d1 d2 : ℕ) (total_distance : ℕ) (car_speed walk_speed : ℕ) : ℕ :=
  d1 / car_speed + (total_distance - d1) / walk_speed

theorem journey_total_time :
  let total_distance := 150
  let car_speed := 30
  let walk_speed := 3
  let d1 := 50
  let d2 := 15
  
  journey_time d1 d2 total_distance car_speed walk_speed =
  max (journey_time d1 0 total_distance car_speed walk_speed / car_speed + 
       (total_distance - d1) / walk_speed)
      ((d1 / car_speed + (d1 - d2) / car_speed + (total_distance - d1 + d2) / car_speed)) :=
by
  sorry

end journey_total_time_l129_129881


namespace probability_of_A_and_B_selected_l129_129482

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129482


namespace cos_135_eq_neg_inv_sqrt_2_l129_129055

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129055


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129043

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129043


namespace probability_both_A_and_B_selected_l129_129448

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129448


namespace probability_A_and_B_selected_l129_129409

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129409


namespace find_price_max_profit_l129_129899

/-
Part 1: Prove the price per unit of type A and B
-/

def price_per_unit (x y : ℕ) : Prop :=
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720)

theorem find_price :
  ∃ x y : ℕ, price_per_unit x y ∧ x = 120 ∧ y = 150 :=
by
  sorry

/-
Part 2: Prove the maximum profit with constraints
-/

def conditions (m : ℕ) : Prop :=
  m ≤ 3 * (40 - m) ∧ 120 * m + 150 * (40 - m) ≤ 5400

def profit (m : ℕ) : ℕ :=
  (160 - 120) * m + (200 - 150) * (40 - m)

theorem max_profit :
  ∃ m : ℕ, 20 ≤ m ∧ m ≤ 30 ∧ conditions m ∧ profit m = profit 20 :=
by
  sorry

end find_price_max_profit_l129_129899


namespace household_A_bill_bill_formula_household_B_usage_household_C_usage_l129_129736

-- Definition of the tiered water price system
def water_bill (x : ℕ) : ℕ :=
if x <= 22 then 3 * x
else if x <= 30 then 3 * 22 + 5 * (x - 22)
else 3 * 22 + 5 * 8 + 7 * (x - 30)

-- Prove that if a household uses 25m^3 of water, the water bill is 81 yuan.
theorem household_A_bill : water_bill 25 = 81 := by 
  sorry

-- Prove that the formula for the water bill when x > 30 is y = 7x - 104.
theorem bill_formula (x : ℕ) (hx : x > 30) : water_bill x = 7 * x - 104 := by 
  sorry

-- Prove that if a household paid 120 yuan for water, their usage was 32m^3.
theorem household_B_usage : ∃ x : ℕ, water_bill x = 120 ∧ x = 32 := by 
  sorry

-- Prove that if household C uses a total of 50m^3 over May and June with a total bill of 174 yuan, their usage was 18m^3 in May and 32m^3 in June.
theorem household_C_usage (a b : ℕ) (ha : a + b = 50) (hb : a < b) (total_bill : water_bill a + water_bill b = 174) :
  a = 18 ∧ b = 32 := by
  sorry

end household_A_bill_bill_formula_household_B_usage_household_C_usage_l129_129736


namespace seq_solution_l129_129072

-- Definitions: Define the sequence {a_n} according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 0 ∧ ∀ n ≥ 2, a n - 2 * a (n - 1) = n ^ 2 - 3

-- Main statement: Prove that for all n, the sequence satisfies the derived formula
theorem seq_solution (a : ℕ → ℤ) (h : seq a) : ∀ n, a n = 2 ^ (n + 2) - n ^ 2 - 4 * n - 3 :=
sorry

end seq_solution_l129_129072


namespace period_start_time_l129_129795

/-- A period of time had 4 hours of rain and 5 hours without rain, ending at 5 pm. 
Prove that the period started at 8 am. -/
theorem period_start_time :
  let end_time := 17 -- 5 pm in 24-hour format
  let rainy_hours := 4
  let non_rainy_hours := 5
  let total_hours := rainy_hours + non_rainy_hours
  let start_time := end_time - total_hours
  start_time = 8 :=
by
  sorry

end period_start_time_l129_129795


namespace sum_abc_is_eight_l129_129141

theorem sum_abc_is_eight (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
by
  sorry

end sum_abc_is_eight_l129_129141


namespace area_of_one_trapezoid_l129_129836

theorem area_of_one_trapezoid (outer_area inner_area : ℝ) (num_trapezoids : ℕ) (h_outer : outer_area = 36) (h_inner : inner_area = 4) (h_num_trapezoids : num_trapezoids = 3) : (outer_area - inner_area) / num_trapezoids = 32 / 3 :=
by
  rw [h_outer, h_inner, h_num_trapezoids]
  norm_num

end area_of_one_trapezoid_l129_129836


namespace quadratic_inequality_solution_l129_129065

theorem quadratic_inequality_solution (k : ℝ) :
  (-1 < k ∧ k < 7) ↔ ∀ x : ℝ, x^2 - (k - 5) * x - k + 8 > 0 :=
by
  sorry

end quadratic_inequality_solution_l129_129065


namespace problem_solution_l129_129803

theorem problem_solution (a b c d : ℝ) (h1 : ab + bc + cd + da = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end problem_solution_l129_129803


namespace exactly_one_even_needs_assumption_l129_129697

open Nat

theorem exactly_one_even_needs_assumption 
  {a b c : ℕ} 
  (h : (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) ∧ (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) ∧ (a % 2 = 0 → b % 2 = 1) ∧ (a % 2 = 0 → c % 2 = 1) ∧ (b % 2 = 0 → c % 2 = 1)) :
  (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) → (a % 2 = 1 ∨ b % 2 = 1 ∨ c % 2 = 1) → (¬(a % 2 = 0 ∧ b % 2 = 0) ∧ ¬(b % 2 = 0 ∧ c % 2 = 0) ∧ ¬(a % 2 = 0 ∧ c % 2 = 0)) := 
by
  sorry

end exactly_one_even_needs_assumption_l129_129697


namespace kamal_marks_in_mathematics_l129_129685

def kamal_marks_english : ℕ := 96
def kamal_marks_physics : ℕ := 82
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 79
def kamal_number_of_subjects : ℕ := 5

theorem kamal_marks_in_mathematics :
  let total_marks := kamal_average_marks * kamal_number_of_subjects
  let total_known_marks := kamal_marks_english + kamal_marks_physics + kamal_marks_chemistry + kamal_marks_biology
  total_marks - total_known_marks = 65 :=
by
  sorry

end kamal_marks_in_mathematics_l129_129685


namespace number_of_girls_l129_129153

-- Define the number of girls and boys
variables (G B : ℕ)

-- Define the conditions
def condition1 : Prop := B = 2 * G - 16
def condition2 : Prop := G + B = 68

-- The theorem we want to prove
theorem number_of_girls (h1 : condition1 G B) (h2 : condition2 G B) : G = 28 :=
by
  sorry

end number_of_girls_l129_129153


namespace transport_load_with_trucks_l129_129230

theorem transport_load_with_trucks
  (total_weight : ℕ)
  (box_max_weight : ℕ)
  (truck_capacity : ℕ)
  (num_trucks : ℕ)
  (H_weight : total_weight = 13500)
  (H_box : box_max_weight = 350)
  (H_truck : truck_capacity = 1500)
  (H_num_trucks : num_trucks = 11) :
  ∃ (boxes : ℕ), boxes * box_max_weight >= total_weight ∧ num_trucks * truck_capacity >= total_weight := 
sorry

end transport_load_with_trucks_l129_129230


namespace proof_cos_135_degree_l129_129000

noncomputable def cos_135_degree_is_negative_sqrt2_over_2 : Prop :=
  real.cos (135 * real.pi / 180) = - real.sqrt 2 / 2

theorem proof_cos_135_degree : cos_135_degree_is_negative_sqrt2_over_2 :=
by
  sorry

end proof_cos_135_degree_l129_129000


namespace probability_of_selecting_A_and_B_l129_129387

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129387


namespace find_two_heaviest_l129_129211

theorem find_two_heaviest (a b c d : ℝ) : 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  ∃ x y : ℝ, (x ≠ y) ∧ (x = max (max (max a b) c) d) ∧ (y = max (max (min (max a b) c) d) d) :=
by sorry

end find_two_heaviest_l129_129211


namespace cos_135_eq_neg_inv_sqrt_2_l129_129002

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129002


namespace positive_difference_l129_129205

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l129_129205


namespace two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l129_129595

theorem two_pow_n_minus_one_div_by_seven_iff (n : ℕ) : (7 ∣ 2^n - 1) ↔ ∃ k : ℕ, n = 3 * k :=
by sorry

theorem two_pow_n_plus_one_not_div_by_seven (n : ℕ) : n > 0 → ¬(7 ∣ 2^n + 1) :=
by sorry

end two_pow_n_minus_one_div_by_seven_iff_two_pow_n_plus_one_not_div_by_seven_l129_129595


namespace learning_hours_difference_l129_129062

/-- Define the hours Ryan spends on each language. -/
def hours_learned (lang : String) : ℝ :=
  if lang = "English" then 2 else
  if lang = "Chinese" then 5 else
  if lang = "Spanish" then 4 else
  if lang = "French" then 3 else
  if lang = "German" then 1.5 else 0

/-- Prove that Ryan spends 2.5 more hours learning Chinese and French combined
    than he does learning German and Spanish combined. -/
theorem learning_hours_difference :
  hours_learned "Chinese" + hours_learned "French" - (hours_learned "German" + hours_learned "Spanish") = 2.5 :=
by
  sorry

end learning_hours_difference_l129_129062


namespace greatest_radius_l129_129963

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l129_129963


namespace class_percentage_of_girls_l129_129577

/-
Given:
- Initial number of boys in the class: 11
- Number of girls in the class: 13
- 1 boy is added to the class, resulting in the new total number of boys being 12

Prove:
- The percentage of the class that are girls is 52%.
-/
theorem class_percentage_of_girls (initial_boys : ℕ) (girls : ℕ) (added_boy : ℕ)
  (new_boy_total : ℕ) (total_students : ℕ) (percent_girls : ℕ) (h1 : initial_boys = 11) 
  (h2 : girls = 13) (h3 : added_boy = 1) (h4 : new_boy_total = initial_boys + added_boy) 
  (h5 : total_students = new_boy_total + girls) 
  (h6 : percent_girls = (girls * 100) / total_students) : percent_girls = 52 :=
sorry

end class_percentage_of_girls_l129_129577


namespace sequence_length_l129_129517

theorem sequence_length (a d n : ℕ) (h1 : a = 3) (h2 : d = 5) (h3: 3 + (n-1) * d = 3008) : n = 602 := 
by
  sorry

end sequence_length_l129_129517


namespace prob_select_A_and_B_l129_129305

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129305


namespace cubic_vs_square_ratio_l129_129645

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l129_129645


namespace probability_A_wins_championship_distribution_and_expectation_B_l129_129701

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l129_129701


namespace intersecting_rectangles_area_l129_129741

-- Define the dimensions of the rectangles
def rect1_length : ℝ := 12
def rect1_width : ℝ := 4
def rect2_length : ℝ := 7
def rect2_width : ℝ := 5

-- Define the areas of the individual rectangles
def area_rect1 : ℝ := rect1_length * rect1_width
def area_rect2 : ℝ := rect2_length * rect2_width

-- Assume overlapping region area
def area_overlap : ℝ := rect1_width * rect2_width

-- Define the total shaded area
def shaded_area : ℝ := area_rect1 + area_rect2 - area_overlap

-- Prove the shaded area is 63 square units
theorem intersecting_rectangles_area : shaded_area = 63 :=
by 
  -- Insert proof steps here, we only provide the theorem statement and leave the proof unfinished
  sorry

end intersecting_rectangles_area_l129_129741


namespace probability_of_selecting_A_and_B_l129_129389

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129389


namespace eighth_grade_girls_l129_129151

theorem eighth_grade_girls
  (G : ℕ) 
  (boys : ℕ) 
  (h1 : boys = 2 * G - 16) 
  (h2 : G + boys = 68) : 
  G = 28 :=
by
  sorry

end eighth_grade_girls_l129_129151


namespace journey_total_distance_l129_129092

theorem journey_total_distance (D : ℝ) 
  (train_fraction : ℝ := 3/5) 
  (bus_fraction : ℝ := 7/20) 
  (walk_distance : ℝ := 6.5) 
  (total_fraction : ℝ := 1) : 
  (1 - (train_fraction + bus_fraction)) * D = walk_distance → D = 130 := 
by
  sorry

end journey_total_distance_l129_129092


namespace product_consecutive_natural_number_square_l129_129839

theorem product_consecutive_natural_number_square (n : ℕ) : 
  ∃ k : ℕ, 100 * (n^2 + n) + 25 = k^2 :=
by
  sorry

end product_consecutive_natural_number_square_l129_129839


namespace problem_solution_l129_129802

theorem problem_solution (a b c d : ℝ) (h1 : ab + bc + cd + da = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end problem_solution_l129_129802


namespace probability_A_and_B_selected_l129_129408

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129408


namespace fraction_days_passed_l129_129124

-- Conditions
def total_days : ℕ := 30
def pills_per_day : ℕ := 2
def total_pills : ℕ := total_days * pills_per_day -- 60 pills
def pills_left : ℕ := 12
def pills_taken : ℕ := total_pills - pills_left -- 48 pills
def days_taken : ℕ := pills_taken / pills_per_day -- 24 days

-- Question and answer
theorem fraction_days_passed :
  (days_taken : ℚ) / (total_days : ℚ) = 4 / 5 := 
by
  sorry

end fraction_days_passed_l129_129124


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l129_129707

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l129_129707


namespace find_AB_l129_129979

-- Definitions based on conditions
variables (AB CD : ℝ)

-- Given conditions
def area_ratio_condition : Prop :=
  AB / CD = 5 / 3

def sum_condition : Prop :=
  AB + CD = 160

-- The main statement to be proven
theorem find_AB (h_ratio : area_ratio_condition AB CD) (h_sum : sum_condition AB CD) :
  AB = 100 :=
by
  sorry

end find_AB_l129_129979


namespace probability_A_and_B_selected_l129_129332

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129332


namespace cos_135_eq_neg_sqrt2_div_2_l129_129037

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129037


namespace probability_A_and_B_selected_l129_129494

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129494


namespace probability_of_selecting_A_and_B_l129_129368

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129368


namespace probability_both_A_and_B_selected_l129_129450

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129450


namespace decorate_eggs_time_calculation_l129_129991

/-- Definition of Mia's and Billy's egg decorating rates, total number of eggs to be decorated, and the calculated time when working together --/
def MiaRate : ℕ := 24
def BillyRate : ℕ := 10
def totalEggs : ℕ := 170
def combinedRate : ℕ := MiaRate + BillyRate

theorem decorate_eggs_time_calculation :
  (totalEggs / combinedRate) = 5 := by
  sorry

end decorate_eggs_time_calculation_l129_129991


namespace select_3_from_5_prob_l129_129349

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129349


namespace probability_of_selecting_A_and_B_l129_129388

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129388


namespace smallest_positive_integer_remainder_l129_129747

theorem smallest_positive_integer_remainder :
  ∃ (a : ℕ), a % 6 = 3 ∧ a % 8 = 5 ∧ ∀ b : ℕ, (b % 6 = 3 ∧ b % 8 = 5) → a ≤ b :=
begin
  use 21,
  split,
  { norm_num, },
  split,
  { norm_num, },
  intro b,
  intro hb,
  rcases hb with ⟨hb1, hb2⟩,
  sorry
end

end smallest_positive_integer_remainder_l129_129747


namespace find_omega_l129_129816

theorem find_omega 
  (w : ℝ) 
  (h₁ : 0 < w)
  (h₂ : (π / w) = (π / 2)) : w = 2 :=
by
  sorry

end find_omega_l129_129816


namespace sum_of_angles_l129_129068

theorem sum_of_angles (k : ℝ) :
  (k = 1) →
  (∀ x ∈ set.Icc 0 360, 
    sin x ^ 3 - cos x ^ 3 = k * ((1 / cos x) - (1 / sin x))) →
  ∃ (angles : list ℝ), 
    (∀ x ∈ angles, x ∈ set.Icc 0 360 ∧ sin x ^ 3 - cos x ^ 3 = k * ((1 / cos x) - (1 / sin x))) ∧ 
    angles.sum = 270 :=
begin
  intros hk h,
  -- sorry, the proof should be placed here.
  sorry,
end

end sum_of_angles_l129_129068


namespace average_of_remaining_two_numbers_l129_129753

theorem average_of_remaining_two_numbers :
  ∀ (a b c d e f : ℝ),
    (a + b + c + d + e + f) / 6 = 3.95 →
    (a + b) / 2 = 3.6 →
    (c + d) / 2 = 3.85 →
    ((e + f) / 2 = 4.4) :=
by
  intros a b c d e f h1 h2 h3
  have h4 : a + b + c + d + e + f = 23.7 := sorry
  have h5 : a + b = 7.2 := sorry
  have h6 : c + d = 7.7 := sorry
  have h7 : e + f = 8.8 := sorry
  exact sorry

end average_of_remaining_two_numbers_l129_129753


namespace a_must_be_negative_l129_129508

variable (a b c d e : ℝ)

theorem a_must_be_negative
  (h1 : a / b < -c / d)
  (hb : b > 0)
  (hd : d > 0)
  (he : e > 0)
  (h2 : a + e > 0) : a < 0 := by
  sorry

end a_must_be_negative_l129_129508


namespace ratio_of_diagonals_l129_129725

theorem ratio_of_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (4 * b) / (4 * a) = 11) : (b * Real.sqrt 2) / (a * Real.sqrt 2) = 11 := 
by 
  sorry

end ratio_of_diagonals_l129_129725


namespace roots_quadratic_identity_l129_129987

theorem roots_quadratic_identity :
  ∀ (r s : ℝ), (r^2 - 5 * r + 3 = 0) ∧ (s^2 - 5 * s + 3 = 0) → r^2 + s^2 = 19 :=
by
  intros r s h
  sorry

end roots_quadratic_identity_l129_129987


namespace probability_of_selecting_A_and_B_l129_129391

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129391


namespace probability_A_and_B_selected_l129_129476

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129476


namespace factorization_identity_l129_129284

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l129_129284


namespace factorize_quadratic_l129_129273

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l129_129273


namespace proposition_1_proposition_3_l129_129947

variables {Line Plane : Type}
variables (m n : Line) (α β γ : Plane)

-- Condition predicates
def parallel (p q : Plane) : Prop := sorry -- parallelism of p and q
def perpendicular (p q : Plane) : Prop := sorry -- perpendicularly of p and q
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry -- parallelism of line and plane
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry -- perpendicularity of line and plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry -- line is in the plane

-- Proposition ①
theorem proposition_1 (h1 : parallel α β) (h2 : parallel α γ) : parallel β γ := sorry

-- Proposition ③
theorem proposition_3 (h1 : line_perpendicular_plane m α) (h2 : line_parallel_plane m β) : perpendicular α β := sorry

end proposition_1_proposition_3_l129_129947


namespace longest_chord_length_of_circle_l129_129872

theorem longest_chord_length_of_circle (r : ℝ) (h : r = 5) : ∃ d, d = 10 :=
by
  sorry

end longest_chord_length_of_circle_l129_129872


namespace factor_quadratic_l129_129276

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l129_129276


namespace largest_p_q_sum_l129_129212

theorem largest_p_q_sum 
  (p q : ℝ)
  (A := (p, q))
  (B := (12, 19))
  (C := (23, 20))
  (area_ABC : ℝ := 70)
  (slope_median : ℝ := -5)
  (midpoint_BC := ((12 + 23) / 2, (19 + 20) / 2))
  (eq_median : (q - midpoint_BC.2) = slope_median * (p - midpoint_BC.1))
  (area_eq : 140 = 240 - 437 - 20 * p + 23 * q + 19 * p - 12 * q) :
  p + q ≤ 47 :=
sorry

end largest_p_q_sum_l129_129212


namespace visibility_time_correct_l129_129242

noncomputable def visibility_time (r : ℝ) (d : ℝ) (v_j : ℝ) (v_k : ℝ) : ℝ :=
  (d / (v_j + v_k)) * (r / (r * (v_j / v_k + 1)))

theorem visibility_time_correct :
  visibility_time 60 240 4 2 = 120 :=
by
  sorry

end visibility_time_correct_l129_129242


namespace probability_A_and_B_selected_l129_129399

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129399


namespace probability_A_and_B_selected_l129_129313

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129313


namespace min_value_expression_l129_129113

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^3 + 6 * b^3 + 27 * c^3 + 9 / (8 * a * b * c) ≥ 18 :=
by
  sorry

end min_value_expression_l129_129113


namespace ratio_ramesh_xyz_l129_129998

theorem ratio_ramesh_xyz 
(total_profit : ℝ) 
(ratio_xyz_rajeev : ℚ) 
(rajeev_share : ℝ) 
(h1 : total_profit = 36000) 
(h2 : ratio_xyz_rajeev = 8 / 9) 
(h3 : rajeev_share = 12000) 
: ∃ ratio_ramesh_xyz : ℚ, ratio_ramesh_xyz = 5 / 4 :=
by
  -- Definitions of shares based on conditions
  let X : ℝ := (8 / 9 : ℚ) * rajeev_share
  let R : ℝ := total_profit - (X + rajeev_share)

  -- Simple conditions for R and X
  have hX : X = (8 / 9 : ℚ) * rajeev_share := by sorry
  have hR : R = total_profit - (X + rajeev_share) := by sorry

  -- Ratio calculation step
  let ratio_ramesh_xyz : ℚ := (R / X : ℝ).to_rat simpl

  -- Prove the required ratio
  use ratio_ramesh_xyz
  rw [←Rat.to_rat_of_int.div_eq_div_of_int]
  sorry

end ratio_ramesh_xyz_l129_129998


namespace cos_135_eq_correct_l129_129022

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129022


namespace stewarts_theorem_l129_129856

theorem stewarts_theorem
  (A B C D : ℝ)
  (AB AC AD : ℝ)
  (BD CD BC : ℝ)
  (hD_on_BC : BD + CD = BC) :
  AB^2 * CD + AC^2 * BD - AD^2 * BC = BD * CD * BC := 
sorry

end stewarts_theorem_l129_129856


namespace second_term_arithmetic_sequence_l129_129097

theorem second_term_arithmetic_sequence (a d : ℝ) (h : a + (a + 2 * d) = 10) : 
  a + d = 5 :=
by
  sorry

end second_term_arithmetic_sequence_l129_129097


namespace probability_A_and_B_selected_l129_129312

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129312


namespace solution_set_of_f_prime_gt_zero_l129_129523

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4 * Real.log x

theorem solution_set_of_f_prime_gt_zero :
  {x : ℝ | 0 < x ∧ 2*x - 2 - (4 / x) > 0} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_of_f_prime_gt_zero_l129_129523


namespace prob_select_A_and_B_l129_129331

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129331


namespace find_largest_p_plus_q_l129_129213

noncomputable def largest_possible_p_plus_q : ℝ :=
  let B := (12 : ℝ, 19 : ℝ)
  let C := (23 : ℝ, 20 : ℝ)
  let area_ABC : ℝ := 70
  let slope_median : ℝ := -5 in
  47

theorem find_largest_p_plus_q (p q : ℝ) (hA : (p, q) = A) (h_area : 2 * area_ABC = abs (p * (B.2 - C.2) + B.1 * (C.2 - q) + C.1 * (q - B.2)))
(h_slope : q = -5 * p + 107) : p + q = largest_possible_p_plus_q := 
  sorry

end find_largest_p_plus_q_l129_129213


namespace sum_of_roots_of_quadratic_l129_129090

open Polynomial

theorem sum_of_roots_of_quadratic :
  ∀ (m n : ℝ), (m ≠ n ∧ (∀ x, x^2 + 2*x - 1 = 0 → x = m ∨ x = n)) → m + n = -2 :=
by
  sorry

end sum_of_roots_of_quadratic_l129_129090


namespace obtuse_triangle_contradiction_l129_129217

theorem obtuse_triangle_contradiction (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A) (h3 : 0 < B) (h4 : 0 < C) : 
  (A > 90 ∧ B > 90) → false :=
by
  sorry

end obtuse_triangle_contradiction_l129_129217


namespace inequality_proof_l129_129652

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l129_129652


namespace inequality_proof_l129_129654

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l129_129654


namespace equation_of_line_l129_129080

theorem equation_of_line (l : ℝ → ℝ) :
  (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x : ℝ, l x = (2 * l a / a) * x))
  ∨ (l 1 = 2 ∧ (∃ a : ℝ, l 0 = 2 * a ∧ a ≠ 0 ∧ ∀ x y : ℝ, 2 * x + y - 4 = 0)) := sorry

end equation_of_line_l129_129080


namespace smallest_positive_integer_l129_129746

theorem smallest_positive_integer {x : ℕ} (h1 : x % 6 = 3) (h2 : x % 8 = 5) : x = 21 :=
sorry

end smallest_positive_integer_l129_129746


namespace probability_of_selecting_A_and_B_l129_129370

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129370


namespace harmonic_mean_closest_integer_l129_129568

theorem harmonic_mean_closest_integer (a b : ℝ) (ha : a = 1) (hb : b = 2016) :
  abs ((2 * a * b) / (a + b) - 2) < 1 :=
by
  sorry

end harmonic_mean_closest_integer_l129_129568


namespace find_base_s_l129_129530

-- Definitions based on the conditions.
def five_hundred_thirty_base (s : ℕ) : ℕ := 5 * s^2 + 3 * s
def four_hundred_fifty_base (s : ℕ) : ℕ := 4 * s^2 + 5 * s
def one_thousand_one_hundred_base (s : ℕ) : ℕ := s^3 + s^2

-- The theorem to prove.
theorem find_base_s : (∃ s : ℕ, five_hundred_thirty_base s + four_hundred_fifty_base s = one_thousand_one_hundred_base s) → s = 8 :=
by
  sorry

end find_base_s_l129_129530


namespace purely_periodic_denominator_l129_129953

theorem purely_periodic_denominator :
  ∀ q : ℕ, (∃ a : ℕ, (∃ b : ℕ, q = 99 ∧ (a < 10) ∧ (b < 10) ∧ (∃ f : ℝ, f = ↑a / (10 * q) ∧ ∃ g : ℝ, g = (0.01 * ↑b / (10 * (99 / q))))) → q = 11 ∨ q = 33 ∨ q = 99) :=
by sorry

end purely_periodic_denominator_l129_129953


namespace probability_both_A_and_B_selected_l129_129445

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129445


namespace positive_difference_of_two_numbers_l129_129185

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129185


namespace sum_of_coefficients_is_7_l129_129114

noncomputable def v (n : ℕ) : ℕ := sorry

theorem sum_of_coefficients_is_7 : 
  (∀ n : ℕ, v (n + 1) - v n = 3 * n + 2) → (v 1 = 7) → (∃ a b c : ℝ, (a * n^2 + b * n + c = v n) ∧ (a + b + c = 7)) := 
by
  intros H1 H2
  sorry

end sum_of_coefficients_is_7_l129_129114


namespace min_english_score_l129_129608

theorem min_english_score (A B : ℕ) (h_avg_AB : (A + B) / 2 = 90) : 
  ∀ E : ℕ, ((A + B + E) / 3 ≥ 92) ↔ E ≥ 96 := by
  sorry

end min_english_score_l129_129608


namespace probability_A_B_selected_l129_129427

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129427


namespace probability_of_selecting_A_and_B_l129_129382

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129382


namespace total_students_mrs_mcgillicuddy_l129_129127

-- Define the conditions as variables
def students_registered_morning : ℕ := 25
def students_absent_morning : ℕ := 3
def students_registered_afternoon : ℕ := 24
def students_absent_afternoon : ℕ := 4

-- Prove the total number of students present over the two sessions
theorem total_students_mrs_mcgillicuddy : 
  students_registered_morning - students_absent_morning + students_registered_afternoon - students_absent_afternoon = 42 :=
by
  sorry

end total_students_mrs_mcgillicuddy_l129_129127


namespace total_courses_l129_129549

-- Define the conditions as variables
def max_courses : Nat := 40
def sid_courses : Nat := 4 * max_courses

-- State the theorem we want to prove
theorem total_courses : max_courses + sid_courses = 200 := 
  by
    -- This is where the actual proof would go
    sorry

end total_courses_l129_129549


namespace positive_difference_of_two_numbers_l129_129183

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129183


namespace sum_coordinates_of_k_l129_129668

theorem sum_coordinates_of_k :
  ∀ (f k : ℕ → ℕ), (f 4 = 8) → (∀ x, k x = (f x) ^ 3) → (4 + k 4) = 516 :=
by
  intros f k h1 h2
  sorry

end sum_coordinates_of_k_l129_129668


namespace simplify_proof_l129_129999

noncomputable def simplify_expression (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : ℝ :=
  (1 - 1/x) / ((1 - x^2) / x)

theorem simplify_proof (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) (hx_1 : x ≠ -1) : 
  simplify_expression x hx hx1 hx_1 = -1 / (1 + x) := by 
  sorry

end simplify_proof_l129_129999


namespace find_a_in_triangle_l129_129830

theorem find_a_in_triangle (C : ℝ) (b c : ℝ) (hC : C = 60) (hb : b = 1) (hc : c = Real.sqrt 3) :
  ∃ (a : ℝ), a = 2 := 
by
  sorry

end find_a_in_triangle_l129_129830


namespace range_of_fraction_l129_129088

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∀ z, z = x / y → (1 / 6 ≤ z ∧ z ≤ 4 / 3) :=
sorry

end range_of_fraction_l129_129088


namespace expected_number_of_socks_l129_129854

noncomputable def expected_socks_to_pick (n : ℕ) : ℚ := (2 * (n + 1)) / 3

theorem expected_number_of_socks (n : ℕ) (h : n ≥ 2) : 
  (expected_socks_to_pick n) = (2 * (n + 1)) / 3 := 
by
  sorry

end expected_number_of_socks_l129_129854


namespace sequence_property_l129_129540

theorem sequence_property (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h_rec : ∀ m n : ℕ, a (m + n) = a m + a n + m * n) :
  a 10 = 55 :=
sorry

end sequence_property_l129_129540


namespace greatest_integer_radius_l129_129970

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l129_129970


namespace ratio_height_radius_l129_129506

variable (V r h : ℝ)

theorem ratio_height_radius (h_eq_2r : h = 2 * r) (volume_eq : π * r^2 * h = V) : h / r = 2 :=
by
  sorry

end ratio_height_radius_l129_129506


namespace mateo_orange_bottles_is_1_l129_129982

def number_of_orange_bottles_mateo_has (mateo_orange : ℕ) : Prop :=
  let julios_orange_bottles := 4
  let julios_grape_bottles := 7
  let mateos_grape_bottles := 3
  let liters_per_bottle := 2
  let julios_total_liters := (julios_orange_bottles + julios_grape_bottles) * liters_per_bottle
  let mateos_grape_liters := mateos_grape_bottles * liters_per_bottle
  let mateos_total_liters := (mateo_orange * liters_per_bottle) + mateos_grape_liters
  let additional_liters_to_julio := 14
  julios_total_liters = mateos_total_liters + additional_liters_to_julio

/-
Prove that Mateo has exactly 1 bottle of orange soda (assuming the problem above)
-/
theorem mateo_orange_bottles_is_1 : number_of_orange_bottles_mateo_has 1 :=
sorry

end mateo_orange_bottles_is_1_l129_129982


namespace model_y_completion_time_l129_129759

theorem model_y_completion_time :
  ∀ (T : ℝ), (∃ k ≥ 0, k = 20) →
  (∀ (task_completed_x_per_minute : ℝ), task_completed_x_per_minute = 1 / 60) →
  (∀ (task_completed_y_per_minute : ℝ), task_completed_y_per_minute = 1 / T) →
  (20 * (1 / 60) + 20 * (1 / T) = 1) →
  T = 30 :=
by
  sorry

end model_y_completion_time_l129_129759


namespace endpoint_correctness_l129_129762

-- Define two points in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define start point (2, 2)
def startPoint : Point := ⟨2, 2⟩

-- Define the endpoint's conditions
def endPoint (x y : ℝ) : Prop :=
  y = 2 * x + 1 ∧ (x > 0) ∧ (Real.sqrt ((x - startPoint.x) ^ 2 + (y - startPoint.y) ^ 2) = 6)

-- The solution to the problem proving (3.4213, 7.8426) satisfies the conditions
theorem endpoint_correctness : ∃ (x y : ℝ), endPoint x y ∧ x = 3.4213 ∧ y = 7.8426 := by
  use 3.4213
  use 7.8426
  sorry

end endpoint_correctness_l129_129762


namespace cos_135_eq_neg_sqrt2_div_2_l129_129033

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129033


namespace triangle_area_l129_129670

theorem triangle_area {a b : ℝ} (h₁ : a = 3) (h₂ : b = 4) (h₃ : Real.sin (C : ℝ) = 1/2) :
  let area := (1 / 2) * a * b * (Real.sin C) 
  area = 3 := 
by
  rw [h₁, h₂, h₃]
  simp [Real.sin, mul_assoc]
  sorry

end triangle_area_l129_129670


namespace sum_ac_equals_seven_l129_129814

theorem sum_ac_equals_seven 
  (a b c d : ℝ)
  (h1 : ab + bc + cd + da = 42)
  (h2 : b + d = 6) :
  a + c = 7 := 
sorry

end sum_ac_equals_seven_l129_129814


namespace probability_A_and_B_selected_l129_129317

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129317


namespace vector_operation_l129_129059

open Matrix

def u : Matrix (Fin 2) (Fin 1) ℝ := ![![3], ![-6]]
def v : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![-9]]
def w : Matrix (Fin 2) (Fin 1) ℝ := ![![-1], ![4]]

--\mathbf{u} - 5\mathbf{v} + \mathbf{w} = \begin{pmatrix} = \begin{pmatrix} -3 \\ 43 \end{pmatrix}
theorem vector_operation : u - (5 : ℝ) • v + w = ![![-3], ![43]] :=
by
  sorry

end vector_operation_l129_129059


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129457

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129457


namespace find_box_value_l129_129942

theorem find_box_value (r x : ℕ) 
  (h1 : x + r = 75)
  (h2 : (x + r) + 2 * r = 143) : 
  x = 41 := 
by
  sorry

end find_box_value_l129_129942


namespace price_per_unit_max_profit_l129_129902

-- Part 1: Finding the Prices

theorem price_per_unit (x y : ℕ) 
  (h1 : 2 * x + 3 * y = 690) 
  (h2 : x + 4 * y = 720) : 
  x = 120 ∧ y = 150 :=
by
  sorry

-- Part 2: Maximizing Profit

theorem max_profit (m : ℕ) 
  (h1 : m ≤ 3 * (40 - m)) 
  (h2 : 120 * m + 150 * (40 - m) ≤ 5400) : 
  (m = 20) ∧ (40 - m = 20) :=
by
  sorry

end price_per_unit_max_profit_l129_129902


namespace fraction_of_students_paired_l129_129674

theorem fraction_of_students_paired {t s : ℕ} 
  (h1 : t / 4 = s / 3) : 
  (t / 4 + s / 3) / (t + s) = 2 / 7 := by sorry

end fraction_of_students_paired_l129_129674


namespace find_a_l129_129783

def F (a b c : ℝ) : ℝ := a * b^3 + c

theorem find_a (a : ℝ) (h : F a 3 8 = F a 5 12) : a = -2 / 49 := by
  sorry

end find_a_l129_129783


namespace value_of_a_l129_129658

def hyperbolaFociSharedEllipse : Prop :=
  ∃ a > 0, 
    (∃ c h k : ℝ, c = 3 ∧ (h, k) = (3, 0) ∨ (h, k) = (-3, 0)) ∧ 
    ∃ x y : ℝ, ((x^2) / 4) - ((y^2) / 5) = 1 ∧ ((x^2) / (a^2)) + ((y^2) / 16) = 1

theorem value_of_a : ∃ a > 0, hyperbolaFociSharedEllipse ∧ a = 5 :=
by
  sorry

end value_of_a_l129_129658


namespace greatest_integer_radius_l129_129972

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l129_129972


namespace exists_perfect_square_sum_l129_129136

theorem exists_perfect_square_sum (n : ℕ) (h : n > 2) : ∃ m : ℕ, ∃ k : ℕ, n^2 + m^2 = k^2 :=
by
  sorry

end exists_perfect_square_sum_l129_129136


namespace positive_difference_l129_129204

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l129_129204


namespace muffins_baked_by_James_correct_l129_129607

noncomputable def muffins_baked_by_James (muffins_baked_by_Arthur : ℝ) (ratio : ℝ) : ℝ :=
  muffins_baked_by_Arthur / ratio

theorem muffins_baked_by_James_correct :
  muffins_baked_by_James 115.0 12.0 = 9.5833 :=
by
  -- Add the proof here
  sorry

end muffins_baked_by_James_correct_l129_129607


namespace find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l129_129096

-- Define what it means to be a "magical point"
def is_magical_point (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, 2 * m)

-- Specialize for the specific quadratic function y = x^2 - x - 4
def on_specific_quadratic (m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, m^2 - m - 4)

-- Theorem for part 1: Find the magical points on y = x^2 - x - 4
theorem find_magical_points_on_specific_quad (m : ℝ) (A : ℝ × ℝ) :
  is_magical_point m A ∧ on_specific_quadratic m A →
  (A = (4, 8) ∨ A = (-1, -2)) :=
sorry

-- Define the quadratic function for part 2
def on_general_quadratic (t m : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (m, t * m^2 + (t-2) * m - 4)

-- Theorem for part 2: Find the t values for unique magical points
theorem find_t_for_unique_magical_point (t m : ℝ) (A : ℝ × ℝ) :
  ( ∀ m, is_magical_point m A ∧ on_general_quadratic t m A → 
    (t * m^2 + (t-4) * m - 4 = 0) ) → 
  ( ∃! m, is_magical_point m A ∧ on_general_quadratic t m A ) →
  t = -4 :=
sorry

end find_magical_points_on_specific_quad_find_t_for_unique_magical_point_l129_129096


namespace carbon_copies_after_folding_l129_129591

-- Define the initial condition of sheets and carbon papers
def initial_sheets : ℕ := 3
def initial_carbons : ℕ := 2

-- Define the condition of folding the paper
def fold_paper (sheets carbons : ℕ) : ℕ := sheets * 2

-- Statement of the problem
theorem carbon_copies_after_folding : (fold_paper initial_sheets initial_carbons - initial_sheets + initial_carbons) = 4 :=
by
  sorry

end carbon_copies_after_folding_l129_129591


namespace solve_for_a_l129_129944

theorem solve_for_a (a : Real) (h_pos : a > 0) (h_eq : (fun x => x^2 + 4) ((fun x => x^2 - 2) a) = 18) : 
  a = Real.sqrt (Real.sqrt 14 + 2) := by 
  sorry

end solve_for_a_l129_129944


namespace value_of_f_at_neg_one_l129_129516

noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x^2

noncomputable def f (x : ℝ) (h : x ≠ 0) : ℝ := (2 - 3 * x^2) / x^2

theorem value_of_f_at_neg_one : f (-1) (by norm_num) = -1 := 
sorry

end value_of_f_at_neg_one_l129_129516


namespace calculate_expression_l129_129755

theorem calculate_expression : (0.0088 * 4.5) / (0.05 * 0.1 * 0.008) = 990 := by
  sorry

end calculate_expression_l129_129755


namespace fg_3_eq_123_l129_129823

def f (x : ℤ) : ℤ := x^2 + 2
def g (x : ℤ) : ℤ := 3 * x + 2

theorem fg_3_eq_123 : f (g 3) = 123 := by
  sorry

end fg_3_eq_123_l129_129823


namespace melissa_total_score_l129_129122

theorem melissa_total_score (games : ℕ) (points_per_game : ℕ) 
  (h_games : games = 3) (h_points_per_game : points_per_game = 27) : 
  points_per_game * games = 81 := 
by 
  sorry

end melissa_total_score_l129_129122


namespace count_perfect_cubes_l129_129519

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1200) :
  ∃ n, n = 5 ∧ ∀ x, (x^3 > a) ∧ (x^3 < b) → (x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10) := 
sorry

end count_perfect_cubes_l129_129519


namespace area_of_triangle_l129_129586

def line1 (x : ℝ) : ℝ := 3 * x + 6
def line2 (x : ℝ) : ℝ := -2 * x + 10

theorem area_of_triangle : 
  let inter_x := (10 - 6) / (3 + 2)
  let inter_y := line1 inter_x
  let base := (10 - 6 : ℝ)
  let height := inter_x
  base * height / 2 = 8 / 5 := 
by
  sorry

end area_of_triangle_l129_129586


namespace allocation_schemes_for_5_teachers_to_3_buses_l129_129229

noncomputable def number_of_allocation_schemes (teachers : ℕ) (buses : ℕ) : ℕ :=
  if buses = 3 ∧ teachers = 5 then 150 else 0

theorem allocation_schemes_for_5_teachers_to_3_buses : 
  number_of_allocation_schemes 5 3 = 150 := 
by
  sorry

end allocation_schemes_for_5_teachers_to_3_buses_l129_129229


namespace positive_difference_of_two_numbers_l129_129162

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129162


namespace probability_AB_selected_l129_129376

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129376


namespace probability_of_selecting_A_and_B_l129_129441

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129441


namespace exists_consecutive_set_divisor_lcm_l129_129224

theorem exists_consecutive_set_divisor_lcm (n : ℕ) (h : n ≥ 4) :
  ∃ (A : Finset ℕ), A.card = n ∧
  ∀ (m : ℕ), m ∈ A ∧ m = A.max' (Finset.card_pos.2 (Nat.pos_of_ne_zero (ne_of_gt h))) → 
  m ∣ ∏ x in A.erase m, x :=
by
  sorry

end exists_consecutive_set_divisor_lcm_l129_129224


namespace total_students_in_school_l129_129221

theorem total_students_in_school 
  (below_8_percent : ℝ) (above_8_ratio : ℝ) (students_8 : ℕ) : 
  below_8_percent = 0.20 → above_8_ratio = 2/3 → students_8 = 12 → 
  (∃ T : ℕ, T = 25) :=
by
  sorry

end total_students_in_school_l129_129221


namespace positive_difference_of_two_numbers_l129_129195

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l129_129195


namespace proof_problem_l129_129817

-- Definitions for the solution sets
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | -3 < x ∧ x < 2}
def intersection : Set ℝ := {x | -1 < x ∧ x < 2}

-- The quadratic inequality solution sets
def solution_set (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b < 0}

-- The main theorem statement
theorem proof_problem (a b : ℝ) (h : solution_set a b = intersection) : a + b = -3 :=
sorry

end proof_problem_l129_129817


namespace arithmetic_sequence_property_l129_129978

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
  (h2 : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 :=
sorry

end arithmetic_sequence_property_l129_129978


namespace positive_difference_of_two_numbers_l129_129192

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l129_129192


namespace symmetric_point_origin_l129_129810

def symmetric_point (p: ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_point_origin : 
  (symmetric_point (3, -2)) = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l129_129810


namespace line_through_origin_and_intersection_eq_x_y_l129_129067

theorem line_through_origin_and_intersection_eq_x_y :
  ∀ (x y : ℝ), (x - 2 * y + 2 = 0) ∧ (2 * x - y - 2 = 0) →
  ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ (y = m * x + b) :=
by
  sorry

end line_through_origin_and_intersection_eq_x_y_l129_129067


namespace probability_of_selecting_A_and_B_l129_129471

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129471


namespace bowling_average_l129_129951

theorem bowling_average (gretchen_score mitzi_score beth_score : ℤ) (h1 : gretchen_score = 120) (h2 : mitzi_score = 113) (h3 : beth_score = 85) :
  (gretchen_score + mitzi_score + beth_score) / 3 = 106 :=
by
  sorry

end bowling_average_l129_129951


namespace probability_A_wins_championship_expectation_X_is_13_l129_129708

/-
Definitions corresponding to the conditions in the problem
-/
def prob_event1_A_win : ℝ := 0.5
def prob_event2_A_win : ℝ := 0.4
def prob_event3_A_win : ℝ := 0.8

def prob_event1_B_win : ℝ := 1 - prob_event1_A_win
def prob_event2_B_win : ℝ := 1 - prob_event2_A_win
def prob_event3_B_win : ℝ := 1 - prob_event3_A_win

/-
Proof problems corresponding to the questions and correct answers
-/

theorem probability_A_wins_championship : prob_event1_A_win * prob_event2_A_win * prob_event3_A_win
    + prob_event1_A_win * prob_event2_A_win * prob_event3_B_win
    + prob_event1_A_win * prob_event2_B_win * prob_event3_A_win 
    + prob_event1_B_win * prob_event2_A_win * prob_event3_A_win = 0.6 := 
sorry

noncomputable def X_distribution_table : list (ℝ × ℝ) := 
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

noncomputable def expected_value_X : ℝ := 
  ∑ x in X_distribution_table, x.1 * x.2

theorem expectation_X_is_13 : expected_value_X = 13 := sorry

end probability_A_wins_championship_expectation_X_is_13_l129_129708


namespace total_selling_price_l129_129912

theorem total_selling_price 
  (n : ℕ) (p : ℕ) (c : ℕ) 
  (h_n : n = 85) (h_p : p = 15) (h_c : c = 85) : 
  (c + p) * n = 8500 :=
by
  sorry

end total_selling_price_l129_129912


namespace positive_difference_of_numbers_l129_129200

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l129_129200


namespace probability_A_and_B_selected_l129_129319

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129319


namespace class_mean_correct_l129_129975

noncomputable def new_class_mean (number_students_midterm : ℕ) (avg_score_midterm : ℚ)
                                 (number_students_next_day : ℕ) (avg_score_next_day : ℚ)
                                 (number_students_final_day : ℕ) (avg_score_final_day : ℚ)
                                 (total_students : ℕ) : ℚ :=
  let total_score_midterm := number_students_midterm * avg_score_midterm
  let total_score_next_day := number_students_next_day * avg_score_next_day
  let total_score_final_day := number_students_final_day * avg_score_final_day
  let total_score := total_score_midterm + total_score_next_day + total_score_final_day
  total_score / total_students

theorem class_mean_correct :
  new_class_mean 50 65 8 85 2 55 60 = 67 :=
by
  sorry

end class_mean_correct_l129_129975


namespace condition_sufficiency_l129_129938

theorem condition_sufficiency (x : ℝ) :
  (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1) ∧ (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by
  sorry

end condition_sufficiency_l129_129938


namespace negation_of_universal_proposition_l129_129149

def int_divisible_by_5 (n : ℤ) := ∃ k : ℤ, n = 5 * k
def int_odd (n : ℤ) := ∃ k : ℤ, n = 2 * k + 1

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℤ, int_divisible_by_5 n → int_odd n) ↔ (∃ n : ℤ, int_divisible_by_5 n ∧ ¬ int_odd n) :=
by
  sorry

end negation_of_universal_proposition_l129_129149


namespace probability_A_B_l129_129359

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129359


namespace probability_A_B_l129_129361

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129361


namespace transforming_sin_curve_l129_129864

theorem transforming_sin_curve :
  ∀ x : ℝ, (2 * Real.sin (x + (Real.pi / 3))) = (2 * Real.sin ((1/3) * x + (Real.pi / 3))) :=
by
  sorry

end transforming_sin_curve_l129_129864


namespace factorize_quadratic_l129_129272

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l129_129272


namespace julie_monthly_salary_l129_129534

def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 6
def missed_days : ℕ := 1
def weeks_per_month : ℕ := 4

theorem julie_monthly_salary :
  (hourly_rate * hours_per_day * (days_per_week - missed_days) * weeks_per_month) = 920 :=
by
  sorry

end julie_monthly_salary_l129_129534


namespace probability_of_selecting_A_and_B_l129_129363

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129363


namespace max_value_m_l129_129522

theorem max_value_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0) -> (x < m)) -> m = -2 :=
by
  sorry

end max_value_m_l129_129522


namespace sum_of_two_longest_altitudes_l129_129664

theorem sum_of_two_longest_altitudes (a b c : ℕ) (h : a^2 + b^2 = c^2) (h1: a = 7) (h2: b = 24) (h3: c = 25) : 
  (a + b = 31) :=
by {
  sorry
}

end sum_of_two_longest_altitudes_l129_129664


namespace select_3_from_5_prob_l129_129344

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129344


namespace hyperbola_equation_l129_129512

theorem hyperbola_equation (a b k : ℝ) (p : ℝ × ℝ) (h_asymptotes : b = 3 * a)
  (h_hyperbola_passes_point : p = (2, -3 * Real.sqrt 3)) (h_hyperbola : ∀ x y, x^2 - (y^2 / (3 * a)^2) = k) :
  ∃ k, k = 1 :=
by
  -- Given the point p and asymptotes, we should prove k = 1.
  sorry

end hyperbola_equation_l129_129512


namespace probability_of_selecting_A_and_B_l129_129364

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129364


namespace probability_A_and_B_selected_l129_129341

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129341


namespace prob_select_A_and_B_l129_129309

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129309


namespace probability_square_not_touching_vertex_l129_129996

theorem probability_square_not_touching_vertex :
  let total_squares := 64
  let squares_touching_vertices := 16
  let squares_not_touching_vertices := total_squares - squares_touching_vertices
  let probability := (squares_not_touching_vertices : ℚ) / total_squares
  probability = 3 / 4 :=
by
  sorry

end probability_square_not_touching_vertex_l129_129996


namespace problem1_problem2_l129_129613

-- Problem 1: Prove the expression equals 5
theorem problem1 : (1 : ℚ) * ((1/3 : ℚ) - (3/4) + (5/6)) / (1/12) = 5 := by
  sorry

-- Problem 2: Prove the expression equals 7
theorem problem2 : ((-1 : ℤ)^2023 + |(1 - 0.5 : ℚ)| * ((-4)^2)) = 7 := by
  sorry

end problem1_problem2_l129_129613


namespace correct_sum_after_digit_change_l129_129869

theorem correct_sum_after_digit_change :
  let d := 7
  let e := 8
  let num1 := 935641
  let num2 := 471850
  let correct_sum := num1 + num2
  let new_sum := correct_sum + 10000
  new_sum = 1417491 := 
sorry

end correct_sum_after_digit_change_l129_129869


namespace lcm_105_360_eq_2520_l129_129214

theorem lcm_105_360_eq_2520 :
  Nat.lcm 105 360 = 2520 :=
by
  have h1 : 105 = 3 * 5 * 7 := by norm_num
  have h2 : 360 = 2^3 * 3^2 * 5 := by norm_num
  rw [h1, h2]
  sorry

end lcm_105_360_eq_2520_l129_129214


namespace distance_min_value_l129_129077

theorem distance_min_value (a b c d : ℝ) 
  (h₁ : |b - (Real.log a) / a| + |c - d + 2| = 0) : 
  (a - c)^2 + (b - d)^2 = 9 / 2 :=
by {
  sorry
}

end distance_min_value_l129_129077


namespace Nikka_stamp_collection_l129_129852

theorem Nikka_stamp_collection (S : ℝ) 
  (h1 : 0.35 * S ≥ 0) 
  (h2 : 0.2 * S ≥ 0) 
  (h3 : 0 < S) 
  (h4 : 0.45 * S = 45) : S = 100 :=
sorry

end Nikka_stamp_collection_l129_129852


namespace triangle_area_example_l129_129669

noncomputable def area_triangle (BC AB : ℝ) (B : ℝ) : ℝ :=
  (1 / 2) * BC * AB * Real.sin B

theorem triangle_area_example
  (BC AB : ℝ) (B : ℝ)
  (hBC : BC = 2)
  (hAB : AB = 3)
  (hB : B = Real.pi / 3) :
  area_triangle BC AB B = (3 * Real.sqrt 3) / 2 :=
by
  sorry

end triangle_area_example_l129_129669


namespace cos_135_eq_neg_inv_sqrt_2_l129_129013

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129013


namespace math_contest_students_l129_129719

theorem math_contest_students (n : ℝ) (h : n / 3 + n / 4 + n / 5 + 26 = n) : n = 120 :=
by {
    sorry
}

end math_contest_students_l129_129719


namespace greatest_integer_radius_l129_129966

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l129_129966


namespace probability_both_A_and_B_selected_l129_129451

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129451


namespace number_of_cards_above_1999_l129_129910

def numberOfCardsAbove1999 (n : ℕ) : ℕ :=
  if n < 2 then 0
  else if numberOfCardsAbove1999 (n-1) = n-2 then 1
  else numberOfCardsAbove1999 (n-1) + 2

theorem number_of_cards_above_1999 : numberOfCardsAbove1999 2000 = 927 := by
  sorry

end number_of_cards_above_1999_l129_129910


namespace probability_A_and_B_selected_l129_129400

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129400


namespace Haleigh_can_make_3_candles_l129_129662

variable (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ)

def wax_leftover (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ) : ℝ := 
  n20 * w20 + n5 * w5 + n1 * w1 

theorem Haleigh_can_make_3_candles :
  ∀ (n20 n5 n1 : ℕ) (w20 w5 w1 oz10 : ℝ), 
  n20 = 5 →
  w20 = 2 →
  n5 = 5 →
  w5 = 0.5 →
  n1 = 25 →
  w1 = 0.1 →
  oz10 = 10 →
  (wax_leftover n20 n5 n1 w20 w5 w1 oz10) / 5 = 3 := 
by
  intros n20 n5 n1 w20 w5 w1 oz10 hn20 hw20 hn5 hw5 hn1 hw1 hoz10
  rw [hn20, hw20, hn5, hw5, hn1, hw1, hoz10]
  sorry

end Haleigh_can_make_3_candles_l129_129662


namespace positive_difference_of_numbers_l129_129196

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l129_129196


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129461

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129461


namespace max_radius_of_circle_l129_129968

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l129_129968


namespace range_of_m_l129_129116

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 :=
sorry

end range_of_m_l129_129116


namespace randy_initial_money_l129_129838

/--
Initially, Randy had an unknown amount of money. He was given $2000 by Smith and $900 by Michelle.
After that, Randy gave Sally a 1/4th of his total money after which he gave Jake and Harry $800 and $500 respectively.
If Randy is left with $5500 after all the transactions, prove that Randy initially had $6166.67.
-/
theorem randy_initial_money (X : ℝ) :
  (3/4 * (X + 2000 + 900) - 1300 = 5500) -> (X = 6166.67) :=
by
  sorry

end randy_initial_money_l129_129838


namespace probability_of_selecting_A_and_B_l129_129384

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129384


namespace train_john_arrival_probability_l129_129731

-- Define the probability of independent uniform distributions on the interval [0, 120]
noncomputable def probability_train_present_when_john_arrives : ℝ :=
  let total_square_area := (120 : ℝ) * 120
  let triangle_area := (1 / 2) * 90 * 30
  let trapezoid_area := (1 / 2) * (30 + 0) * 30
  let total_shaded_area := triangle_area + trapezoid_area
  total_shaded_area / total_square_area

theorem train_john_arrival_probability :
  probability_train_present_when_john_arrives = 1 / 8 :=
by {
  sorry
}

end train_john_arrival_probability_l129_129731


namespace greatest_int_radius_lt_75pi_l129_129958

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l129_129958


namespace quadratic_interval_inequality_l129_129948

theorem quadratic_interval_inequality (a b c : ℝ) :
  (∀ x : ℝ, -1 / 2 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  a < 0 ∧ c > 0 :=
sorry

end quadratic_interval_inequality_l129_129948


namespace difference_of_squares_example_l129_129781

theorem difference_of_squares_example :
  262^2 - 258^2 = 2080 := by
sorry

end difference_of_squares_example_l129_129781


namespace exinscribed_sphere_inequality_l129_129103

variable (r r_A r_B r_C r_D : ℝ)

theorem exinscribed_sphere_inequality 
  (hr : 0 < r) 
  (hrA : 0 < r_A) 
  (hrB : 0 < r_B) 
  (hrC : 0 < r_C) 
  (hrD : 0 < r_D) :
  1 / Real.sqrt (r_A^2 - r_A * r_B + r_B^2) +
  1 / Real.sqrt (r_B^2 - r_B * r_C + r_C^2) +
  1 / Real.sqrt (r_C^2 - r_C * r_D + r_D^2) +
  1 / Real.sqrt (r_D^2 - r_D * r_A + r_A^2) ≤
  2 / r := by
  sorry

end exinscribed_sphere_inequality_l129_129103


namespace clock_correction_time_l129_129903

theorem clock_correction_time :
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  correction = 138.75 :=
by
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  have : correction = 138.75 := sorry
  exact this

end clock_correction_time_l129_129903


namespace combined_area_ratio_l129_129909

theorem combined_area_ratio (s : ℝ) (h₁ : s > 0) : 
  let r := s / 2
  let area_semicircle := (1/2) * π * r^2
  let area_quarter_circle := (1/4) * π * r^2
  let area_square := s^2
  let combined_area := area_semicircle + area_quarter_circle
  let ratio := combined_area / area_square
  ratio = 3 * π / 16 :=
by
  sorry

end combined_area_ratio_l129_129909


namespace cos_135_eq_neg_sqrt2_div_2_l129_129031

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129031


namespace greatest_int_radius_lt_75pi_l129_129960

noncomputable def circle_radius_max (A : ℝ) (π : ℝ) : ℕ :=
  Nat.floor (Real.sqrt (A / π))

theorem greatest_int_radius_lt_75pi :
  circle_radius_max 75 Real.pi = 8 := by
  sorry

end greatest_int_radius_lt_75pi_l129_129960


namespace fraction_greater_than_decimal_l129_129255

/-- 
  Prove that the fraction 1/3 is greater than the decimal 0.333 by the amount 1/(3 * 10^3)
-/
theorem fraction_greater_than_decimal :
  (1 / 3 : ℚ) = (333 / 1000 : ℚ) + (1 / (3 * 1000) : ℚ) :=
by
  sorry

end fraction_greater_than_decimal_l129_129255


namespace coin_combinations_l129_129820

-- Define the coins and their counts
def one_cent_count := 1
def two_cent_count := 1
def five_cent_count := 1
def ten_cent_count := 4
def fifty_cent_count := 2

-- Define the expected number of different possible amounts
def expected_amounts := 119

-- Prove that the expected number of possible amounts can be achieved given the coins
theorem coin_combinations : 
  (∃ sums : Finset ℕ, 
    sums.card = expected_amounts ∧ 
    (∀ n ∈ sums, n = one_cent_count * 1 + 
                          two_cent_count * 2 + 
                          five_cent_count * 5 + 
                          ten_cent_count * 10 + 
                          fifty_cent_count * 50)) :=
sorry

end coin_combinations_l129_129820


namespace cos_135_eq_neg_sqrt2_div_2_l129_129025

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129025


namespace probability_both_A_B_selected_l129_129301

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129301


namespace prob_select_A_and_B_l129_129304

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129304


namespace black_pieces_count_l129_129555

theorem black_pieces_count :
  ∃ (grid : Fin 6 → Fin 6 → bool), -- grid configuration
  (∀ i : Fin 6, ∃! w_i, ∀ j : Fin 6, grid i j = (w_i > 0)) ∧ -- distinct number of white pieces per row
  (∃ c : Fin 6 → ℕ, ∀ j : Fin 6, ∀ i : Fin 6, grid i j = (c j > 0) ∧ (c 0 = c j)) → -- same number of white pieces per column
  ∑ i, ∑ j, if grid i j then 0 else 1 = 18 := -- total number of black pieces

begin
  sorry
end

end black_pieces_count_l129_129555


namespace arithmetic_sequence_sum_l129_129095

theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by
  sorry

end arithmetic_sequence_sum_l129_129095


namespace correct_operation_l129_129588

theorem correct_operation :
  (∀ a : ℝ, (a^5 * a^3 = a^15) = false) ∧
  (∀ a : ℝ, (a^5 - a^3 = a^2) = false) ∧
  (∀ a : ℝ, ((-a^5)^2 = a^10) = true) ∧
  (∀ a : ℝ, (a^6 / a^3 = a^2) = false) :=
by
  sorry

end correct_operation_l129_129588


namespace positive_difference_l129_129203

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l129_129203


namespace probability_of_A_and_B_selected_l129_129488

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129488


namespace time_to_decorate_l129_129988

variable (mia_rate billy_rate total_eggs : ℕ)

theorem time_to_decorate (h_mia : mia_rate = 24) (h_billy : billy_rate = 10) (h_total : total_eggs = 170) :
  total_eggs / (mia_rate + billy_rate) = 5 :=
by
  sorry

end time_to_decorate_l129_129988


namespace probability_both_A_B_selected_l129_129296

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129296


namespace probability_A_and_B_selected_l129_129500

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129500


namespace probability_A_and_B_selected_l129_129396

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129396


namespace brenda_peaches_left_brenda_peaches_left_correct_l129_129244

theorem brenda_peaches_left (total_peaches : ℕ) (fresh_pct : ℝ) (too_small_peaches : ℕ)
  (h1 : total_peaches = 250)
  (h2 : fresh_pct = 0.60)
  (h3 : too_small_peaches = 15) : ℕ :=
sorry

theorem brenda_peaches_left_correct : brenda_peaches_left 250 0.60 15 = 135 :=
by {
  rw brenda_peaches_left,
  exact sorry
}

end brenda_peaches_left_brenda_peaches_left_correct_l129_129244


namespace smallest_number_divisible_l129_129216

   theorem smallest_number_divisible (d n : ℕ) (h₁ : (n + 7) % 11 = 0) (h₂ : (n + 7) % 24 = 0) (h₃ : (n + 7) % d = 0) (h₄ : (n + 7) = 257) : n = 250 :=
   by
     sorry
   
end smallest_number_divisible_l129_129216


namespace probability_A_and_B_selected_l129_129407

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129407


namespace positive_difference_of_two_numbers_l129_129169

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l129_129169


namespace student_A_incorrect_l129_129531

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  let (cx, cy) := center
  let (px, py) := point
  (px - cx)^2 + (py - cy)^2 = radius^2

def center : ℝ × ℝ := (2, -3)
def radius : ℝ := 5
def point_A : ℝ × ℝ := (-2, -1)
def point_D : ℝ × ℝ := (5, 1)

theorem student_A_incorrect :
  ¬ is_on_circle center radius point_A ∧ is_on_circle center radius point_D :=
by
  sorry

end student_A_incorrect_l129_129531


namespace part1_part2_l129_129223

theorem part1 (x : ℝ) : 3 + 2 * x > - x - 6 ↔ x > -3 := by
  sorry

theorem part2 (x : ℝ) : 2 * x + 1 ≤ x + 3 ∧ (2 * x + 1) / 3 > 1 ↔ 1 < x ∧ x ≤ 2 := by
  sorry

end part1_part2_l129_129223


namespace validate_shots_statistics_l129_129155

-- Define the scores and their frequencies
def scores : List ℕ := [6, 7, 8, 9, 10]
def times : List ℕ := [4, 10, 11, 9, 6]

-- Condition 1: Calculate the mode
def mode := 8

-- Condition 2: Calculate the median
def median := 8

-- Condition 3: Calculate the 35th percentile
def percentile_35 := ¬(35 * 40 / 100 = 7)

-- Condition 4: Calculate the average
def average := 8.075

theorem validate_shots_statistics :
  mode = 8
  ∧ median = 8
  ∧ percentile_35
  ∧ average = 8.075 :=
by
  sorry

end validate_shots_statistics_l129_129155


namespace evaluate_polynomial_l129_129641

variable {x y : ℚ}

theorem evaluate_polynomial (h : x - 2 * y - 3 = -5) : 2 * y - x = 2 :=
by
  sorry

end evaluate_polynomial_l129_129641


namespace solve_absolute_value_eq_l129_129629

theorem solve_absolute_value_eq (x : ℝ) : |x - 5| = 3 * x - 2 ↔ x = 7 / 4 :=
sorry

end solve_absolute_value_eq_l129_129629


namespace manuscript_pages_l129_129857

theorem manuscript_pages (P : ℕ)
  (h1 : 30 = 30)
  (h2 : 20 = 20)
  (h3 : 50 = 30 + 20)
  (h4 : 710 = 5 * (P - 50) + 30 * 8 + 20 * 11) :
  P = 100 :=
by
  sorry

end manuscript_pages_l129_129857


namespace smallest_integer_satisfies_inequality_l129_129745

theorem smallest_integer_satisfies_inequality :
  ∃ (x : ℤ), (x^2 < 2 * x + 3) ∧ ∀ (y : ℤ), (y^2 < 2 * y + 3) → x ≤ y ∧ x = 0 :=
sorry

end smallest_integer_satisfies_inequality_l129_129745


namespace cos_135_eq_neg_inv_sqrt_2_l129_129012

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129012


namespace inequality_not_satisfied_integer_values_count_l129_129634

theorem inequality_not_satisfied_integer_values_count :
  ∃ (n : ℕ), n = 5 ∧ ∀ (x : ℤ), 3 * x^2 + 17 * x + 20 ≤ 25 → x ∈ [-4, -3, -2, -1, 0] :=
  sorry

end inequality_not_satisfied_integer_values_count_l129_129634


namespace value_of_7_prime_prime_l129_129083

-- Define the function q' (written as q_prime in Lean)
def q_prime (q : ℕ) : ℕ := 3 * q - 3

-- Define the specific value problem
theorem value_of_7_prime_prime : q_prime (q_prime 7) = 51 := by
  sorry

end value_of_7_prime_prime_l129_129083


namespace sum_of_two_longest_altitudes_l129_129663

-- Define what it means for a triangle to have sides 7, 24, 25
def is_triangle_7_24_25 (a b c : ℝ) : Prop :=
  (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 7 ∧ b = 25 ∧ c = 24) ∨ (a = 24 ∧ b = 7 ∧ c = 25) ∨ 
  (a = 24 ∧ b = 25 ∧ c = 7) ∨ (a = 25 ∧ b = 7 ∧ c = 24) ∨ (a = 25 ∧ b = 24 ∧ c = 7)

-- Prove the sum of the two longest altitudes in such a triangle is 31
theorem sum_of_two_longest_altitudes (a b c : ℝ) (h : is_triangle_7_24_25 a b c) :
  let h_altitude (c : ℝ) := (a * b) / c in
  (a + b) - (a * b > c ∨ b * c > a ∨ c * a > b) → ℝ :=
by
  sorry

end sum_of_two_longest_altitudes_l129_129663


namespace probability_A_B_selected_l129_129423

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129423


namespace positive_difference_of_numbers_l129_129198

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l129_129198


namespace symmetric_point_origin_l129_129812

theorem symmetric_point_origin (A : ℝ × ℝ) (A_sym : ℝ × ℝ) (h : A = (3, -2)) (h_sym : A_sym = (-A.1, -A.2)) : A_sym = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l129_129812


namespace probability_A_and_B_selected_l129_129335

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129335


namespace repeating_decimal_eq_fraction_l129_129263

noncomputable def repeating_decimal_to_fraction (x : ℝ) : ℝ :=
  let x : ℝ := 4.5656565656 -- * 0.5656... repeating
  (100*x - x) / (100 - 1)

-- Define the theorem we want to prove
theorem repeating_decimal_eq_fraction : 
  ∀ x : ℝ, x = 4.565656 -> x = (452 : ℝ) / (99 : ℝ) :=
by
  intro x h
  -- here we would provide the proof steps, but since it's omitted
  -- we'll use sorry to skip it.
  sorry

end repeating_decimal_eq_fraction_l129_129263


namespace range_of_m_l129_129071

noncomputable def p (x : ℝ) : Prop := abs (1 - (x - 1) / 3) ≤ 2
noncomputable def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0 ∧ m > 0

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ¬p x → ¬q x m) → (m ≥ 9) :=
by
  sorry

end range_of_m_l129_129071


namespace school_A_win_prob_expectation_X_is_13_l129_129702

-- Define the probabilities of school A winning individual events
def pA_event1 : ℝ := 0.5
def pA_event2 : ℝ := 0.4
def pA_event3 : ℝ := 0.8

-- Define the probability of school A winning the championship
def pA_win_championship : ℝ :=
  (pA_event1 * pA_event2 * pA_event3) +
  (pA_event1 * (1 - pA_event2) * pA_event3) +
  (pA_event1 * pA_event2 * (1 - pA_event3)) +
  ((1 - pA_event1) * pA_event2 * pA_event3)

-- Proof statement for the probability of school A winning the championship
theorem school_A_win_prob : pA_win_championship = 0.6 := sorry

-- Define the distribution and expectation for school B's total score
def X_prob : ℝ → ℝ
| 0  := (1 - pA_event1) * (1 - pA_event2) * (1 - pA_event3)
| 10 := pA_event1 * (1 - pA_event2) * (1 - pA_event3) +
        (1 - pA_event1) * pA_event2 * (1 - pA_event3) +
        (1 - pA_event1) * (1 - pA_event2) * pA_event3
| 20 := pA_event1 * pA_event2 * (1 - pA_event3) +
        pA_event1 * (1 - pA_event2) * pA_event3 +
        (1 - pA_event1) * pA_event2 * pA_event3
| 30 := pA_event1 * pA_event2 * pA_event3
| _  := 0

def expected_X : ℝ :=
  0 * X_prob 0 +
  10 * X_prob 10 +
  20 * X_prob 20 +
  30 * X_prob 30

-- Proof statement for the expectation of school B's total score
theorem expectation_X_is_13 : expected_X = 13 := sorry

end school_A_win_prob_expectation_X_is_13_l129_129702


namespace probability_A_and_B_selected_l129_129421

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129421


namespace factorize_quadratic_l129_129271

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l129_129271


namespace min_cubes_are_three_l129_129765

/-- 
  A toy construction set consists of cubes, each with one button on one side and socket holes on the other five sides.
  Prove that the minimum number of such cubes required to build a structure where all buttons are hidden, and only the sockets are visible is 3.
--/

def min_cubes_to_hide_buttons (num_cubes : ℕ) : Prop :=
  num_cubes = 3

theorem min_cubes_are_three : ∃ (n : ℕ), (∀ (num_buttons : ℕ), min_cubes_to_hide_buttons num_buttons) :=
by
  use 3
  sorry

end min_cubes_are_three_l129_129765


namespace distance_first_day_l129_129834

theorem distance_first_day (total_distance : ℕ) (q : ℚ) (n : ℕ) (a : ℚ) : total_distance = 378 ∧ q = 1 / 2 ∧ n = 6 → a = 192 :=
by
  -- Proof omitted, just provide the statement
  sorry

end distance_first_day_l129_129834


namespace total_courses_l129_129548

-- Define the conditions as variables
def max_courses : Nat := 40
def sid_courses : Nat := 4 * max_courses

-- State the theorem we want to prove
theorem total_courses : max_courses + sid_courses = 200 := 
  by
    -- This is where the actual proof would go
    sorry

end total_courses_l129_129548


namespace prob_select_A_and_B_l129_129326

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129326


namespace proof_op_nabla_l129_129665

def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem proof_op_nabla :
  op_nabla (op_nabla (1/2) (1/3)) (1/4) = 9 / 11 := by
  sorry

end proof_op_nabla_l129_129665


namespace probability_A_and_B_selected_l129_129495

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129495


namespace probability_of_selecting_A_and_B_l129_129465

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129465


namespace factorization_identity_l129_129282

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l129_129282


namespace triangle_angle_sum_l129_129571

theorem triangle_angle_sum (a : ℝ) (x : ℝ) :
  0 < 2 * a + 20 ∧ 0 < 3 * a - 15 ∧ 0 < 175 - 5 * a ∧
  2 * a + 20 + 3 * a - 15 + x = 180 → 
  x = 175 - 5 * a ∧ max (2 * a + 20) (max (3 * a - 15) (175 - 5 * a)) = 88 :=
sorry

end triangle_angle_sum_l129_129571


namespace gcd_polynomial_l129_129509

variable (a : ℕ)

def multiple_of_2345 (a : ℕ) := ∃ k : ℕ, a = 2345 * k

theorem gcd_polynomial (h : multiple_of_2345 a) : Nat.gcd (a^2 + 10*a + 25) (a + 5) = a + 5 := 
sorry

end gcd_polynomial_l129_129509


namespace scientific_notation_correct_l129_129594

def number_in_scientific_notation : ℝ := 1600000
def expected_scientific_notation : ℝ := 1.6 * 10^6

theorem scientific_notation_correct :
  number_in_scientific_notation = expected_scientific_notation := by
  sorry

end scientific_notation_correct_l129_129594


namespace Debby_daily_bottles_is_six_l129_129253

def daily_bottles (total_bottles : ℕ) (total_days : ℕ) : ℕ :=
  total_bottles / total_days

theorem Debby_daily_bottles_is_six : daily_bottles 12 2 = 6 := by
  sorry

end Debby_daily_bottles_is_six_l129_129253


namespace probability_of_selecting_A_and_B_l129_129390

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129390


namespace range_of_a_l129_129082

noncomputable def tangent_slopes (a x0 : ℝ) : ℝ × ℝ :=
  let k1 := (a * x0 + a - 1) * Real.exp x0
  let k2 := (x0 - 2) * Real.exp (-x0)
  (k1, k2)

theorem range_of_a (a x0 : ℝ) (h : x0 ∈ Set.Icc 0 (3 / 2))
  (h_perpendicular : (tangent_slopes a x0).1 * (tangent_slopes a x0).2 = -1)
  : 1 ≤ a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l129_129082


namespace angle_between_bisectors_of_trihedral_angle_l129_129592

noncomputable def angle_between_bisectors_trihedral (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) : ℝ :=
  60

theorem angle_between_bisectors_of_trihedral_angle (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) :
  angle_between_bisectors_trihedral α β γ hα hβ hγ = 60 := 
sorry

end angle_between_bisectors_of_trihedral_angle_l129_129592


namespace vector_subtraction_l129_129824

variables (a : ℝ × ℝ) (b : ℝ × ℝ)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def vector_scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem vector_subtraction (ha : a = (-1, 3)) (hb : b = (2, -1)) : 
  vector_sub a (vector_scalar_mult 2 b) = (-5, 5) :=
by sorry

end vector_subtraction_l129_129824


namespace probability_A_and_B_selected_l129_129334

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129334


namespace dirk_profit_l129_129785

theorem dirk_profit 
  (days : ℕ) 
  (amulets_per_day : ℕ) 
  (sale_price : ℕ) 
  (cost_price : ℕ) 
  (cut_percentage : ℕ) 
  (profit : ℕ) : 
  days = 2 → amulets_per_day = 25 → sale_price = 40 → cost_price = 30 → cut_percentage = 10 → profit = 300 :=
by
  intros h_days h_amulets_per_day h_sale_price h_cost_price h_cut_percentage
  -- Placeholder for the proof
  sorry

end dirk_profit_l129_129785


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129453

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129453


namespace matilda_fathers_chocolate_bars_l129_129693

/-- Matilda had 20 chocolate bars and shared them evenly amongst herself and her 4 sisters.
    When her father got home, he was upset that they did not put aside any chocolates for him.
    They felt bad, so they each gave up half of their chocolate bars for their father.
    Their father then gave 3 chocolate bars to their mother and ate some.
    Matilda's father had 5 chocolate bars left.
    Prove that Matilda's father ate 2 chocolate bars. -/
theorem matilda_fathers_chocolate_bars:
  ∀ (total_chocolates initial_people chocolates_per_person given_to_father chocolates_left chocolates_eaten: ℕ ),
    total_chocolates = 20 →
    initial_people = 5 →
    chocolates_per_person = total_chocolates / initial_people →
    given_to_father = (chocolates_per_person / 2) * initial_people →
    chocolates_left = given_to_father - 3 →
    chocolates_left - 5 = chocolates_eaten →
    chocolates_eaten = 2 :=
by
  intros
  sorry

end matilda_fathers_chocolate_bars_l129_129693


namespace find_g_l129_129094

open Real

def even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

theorem find_g 
  (f g : ℝ → ℝ) 
  (hf : even f) 
  (hg : odd g)
  (h : ∀ x, f x + g x = exp x) :
  ∀ x, g x = exp x - exp (-x) :=
by
  sorry

end find_g_l129_129094


namespace Miss_Darlington_total_blueberries_l129_129550

-- Conditions
def initial_basket := 20
def additional_baskets := 9

-- Definition and statement to be proved
theorem Miss_Darlington_total_blueberries :
  initial_basket + additional_baskets * initial_basket = 200 :=
by
  sorry

end Miss_Darlington_total_blueberries_l129_129550


namespace alpha_more_economical_l129_129238

theorem alpha_more_economical (n : ℕ) : n ≥ 12 → 80 + 12 * n < 10 + 18 * n := 
by
  sorry

end alpha_more_economical_l129_129238


namespace solveEquation_l129_129140

theorem solveEquation (x : ℝ) (hx : |x| ≥ 3) : (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (x₁ / 3 + x₁ / Real.sqrt (x₁ ^ 2 - 9) = 35 / 12) ∧ (x₂ / 3 + x₂ / Real.sqrt (x₂ ^ 2 - 9) = 35 / 12)) ∧ x₁ + x₂ = 8.75) :=
sorry

end solveEquation_l129_129140


namespace tan_half_alpha_eq_one_third_l129_129503

open Real

theorem tan_half_alpha_eq_one_third (α : ℝ) (h1 : 5 * sin (2 * α) = 6 * cos α) (h2 : 0 < α ∧ α < π / 2) :
  tan (α / 2) = 1 / 3 :=
by
  sorry

end tan_half_alpha_eq_one_third_l129_129503


namespace sqrt_of_mixed_number_l129_129788

theorem sqrt_of_mixed_number :
  (Real.sqrt (8 + 9 / 16)) = (Real.sqrt 137 / 4) :=
by
  sorry

end sqrt_of_mixed_number_l129_129788


namespace how_many_tuna_l129_129694

-- Definitions for conditions
variables (customers : ℕ) (weightPerTuna : ℕ) (weightPerCustomer : ℕ)
variables (unsatisfiedCustomers : ℕ)

-- Hypotheses based on the problem conditions
def conditions :=
  customers = 100 ∧
  weightPerTuna = 200 ∧
  weightPerCustomer = 25 ∧
  unsatisfiedCustomers = 20

-- Statement to prove how many tuna Mr. Ray needs
theorem how_many_tuna (h : conditions customers weightPerTuna weightPerCustomer unsatisfiedCustomers) : 
  ∃ n, n = 10 :=
by
  sorry

end how_many_tuna_l129_129694


namespace lemonade_in_pitcher_l129_129915

theorem lemonade_in_pitcher (iced_tea lemonade total_pitcher total_in_drink lemonade_ratio : ℚ)
  (h1 : iced_tea = 1/4)
  (h2 : lemonade = 5/4)
  (h3 : total_in_drink = iced_tea + lemonade)
  (h4 : lemonade_ratio = lemonade / total_in_drink)
  (h5 : total_pitcher = 18) :
  (total_pitcher * lemonade_ratio) = 15 :=
by
  sorry

end lemonade_in_pitcher_l129_129915


namespace cos_135_eq_neg_inv_sqrt_2_l129_129051

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129051


namespace find_c_l129_129504

theorem find_c (a b c : ℝ) (h1 : a + b = 5) (h2 : c^2 = a * b + b - 9) : c = 0 :=
by
  sorry

end find_c_l129_129504


namespace prob_select_A_and_B_l129_129302

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129302


namespace probability_of_A_and_B_selected_l129_129489

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129489


namespace solve_equation_l129_129713

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l129_129713


namespace eq_solution_set_l129_129929

theorem eq_solution_set (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^(a^a)) :
  (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) :=
by
  sorry

end eq_solution_set_l129_129929


namespace probability_A_and_B_selected_l129_129406

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129406


namespace monotonically_increasing_iff_l129_129146

noncomputable def f (x : ℝ) (a : ℝ) := x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_iff (a : ℝ) : 
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) := 
sorry

end monotonically_increasing_iff_l129_129146


namespace probability_A_and_B_selected_l129_129480

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129480


namespace probability_A_and_B_selected_l129_129498

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129498


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129452

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129452


namespace positive_difference_of_two_numbers_l129_129167

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129167


namespace sum_powers_eq_34_over_3_l129_129825

theorem sum_powers_eq_34_over_3 (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 3)
  (h3 : a^3 + b^3 + c^3 = 6):
  a^4 + b^4 + c^4 = 34 / 3 :=
by
  sorry

end sum_powers_eq_34_over_3_l129_129825


namespace cos_135_eq_neg_inv_sqrt_2_l129_129004

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129004


namespace remainder_equality_l129_129660

theorem remainder_equality 
  (Q Q' S S' E s s' : ℕ) 
  (Q_gt_Q' : Q > Q')
  (h1 : Q % E = S)
  (h2 : Q' % E = S')
  (h3 : (Q^2 * Q') % E = s)
  (h4 : (S^2 * S') % E = s') :
  s = s' :=
sorry

end remainder_equality_l129_129660


namespace false_statement_is_D_l129_129888

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

def is_scalene_triangle (a b c : ℝ) : Prop :=
  (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def is_right_isosceles_triangle (a b c : ℝ) : Prop :=
  is_right_triangle a b c ∧ is_isosceles_triangle a b c

-- Statements derived from conditions
def statement_A : Prop := ∀ (a b c : ℝ), is_isosceles_triangle a b c → a = b ∨ b = c ∨ c = a
def statement_B : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2
def statement_C : Prop := ∀ (a b c : ℝ), is_scalene_triangle a b c → a ≠ b ∧ b ≠ c ∧ c ≠ a
def statement_D : Prop := ∀ (a b c : ℝ), is_right_triangle a b c → is_isosceles_triangle a b c
def statement_E : Prop := ∀ (a b c : ℝ), is_right_isosceles_triangle a b c → ∃ (θ : ℝ), θ ≠ 90 ∧ θ = 45

-- Main theorem to be proved
theorem false_statement_is_D : statement_D = false :=
by
  sorry

end false_statement_is_D_l129_129888


namespace probability_A_B_l129_129356

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129356


namespace quarters_per_jar_l129_129681

/-- Jenn has 5 jars full of quarters. Each jar can hold a certain number of quarters.
    The bike costs 180 dollars, and she will have 20 dollars left over after buying it.
    Prove that each jar can hold 160 quarters. -/
theorem quarters_per_jar (num_jars : ℕ) (cost_bike : ℕ) (left_over : ℕ)
  (quarters_per_dollar : ℕ) (total_quarters : ℕ) (quarters_per_jar : ℕ) :
  num_jars = 5 → cost_bike = 180 → left_over = 20 → quarters_per_dollar = 4 →
  total_quarters = ((cost_bike + left_over) * quarters_per_dollar) →
  quarters_per_jar = (total_quarters / num_jars) →
  quarters_per_jar = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end quarters_per_jar_l129_129681


namespace probability_of_selecting_A_and_B_l129_129436

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129436


namespace compute_b_l129_129507

noncomputable def rational_coefficients (a b : ℚ) :=
∃ x : ℚ, (x^3 + a * x^2 + b * x + 15 = 0)

theorem compute_b (a b : ℚ) (h1 : (3 + Real.sqrt 5)∈{root : ℝ | root^3 + a * root^2 + b * root + 15 = 0}) 
(h2 : rational_coefficients a b) : b = -18.5 :=
by
  sorry

end compute_b_l129_129507


namespace factorization_of_polynomial_l129_129267

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l129_129267


namespace prob_select_A_and_B_l129_129330

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129330


namespace probability_of_A_and_B_selected_l129_129484

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129484


namespace factor_quadratic_l129_129278

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l129_129278


namespace positive_difference_l129_129209

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l129_129209


namespace all_tutors_work_together_in_90_days_l129_129692

theorem all_tutors_work_together_in_90_days :
  lcm 5 (lcm 6 (lcm 9 10)) = 90 := by
  sorry

end all_tutors_work_together_in_90_days_l129_129692


namespace select_3_from_5_prob_l129_129347

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129347


namespace harkamal_payment_l129_129952

noncomputable def calculate_total_cost : ℝ :=
  let price_grapes := 8 * 70
  let price_mangoes := 9 * 45
  let price_apples := 5 * 30
  let price_strawberries := 3 * 100
  let price_oranges := 10 * 40
  let price_kiwis := 6 * 60
  let total_grapes_and_apples := price_grapes + price_apples
  let discount_grapes_and_apples := 0.10 * total_grapes_and_apples
  let total_oranges_and_kiwis := price_oranges + price_kiwis
  let discount_oranges_and_kiwis := 0.05 * total_oranges_and_kiwis
  let total_mangoes_and_strawberries := price_mangoes + price_strawberries
  let tax_mangoes_and_strawberries := 0.12 * total_mangoes_and_strawberries
  let total_amount := price_grapes + price_mangoes + price_apples + price_strawberries + price_oranges + price_kiwis
  total_amount - discount_grapes_and_apples - discount_oranges_and_kiwis + tax_mangoes_and_strawberries

theorem harkamal_payment : calculate_total_cost = 2150.6 :=
by
  sorry

end harkamal_payment_l129_129952


namespace thor_fraction_correct_l129_129992

-- Define the initial conditions
def moes_money : ℕ := 12
def lokis_money : ℕ := 10
def nicks_money : ℕ := 8
def otts_money : ℕ := 6

def thor_received_from_each : ℕ := 2

-- Calculate total money each time
def total_initial_money : ℕ := moes_money + lokis_money + nicks_money + otts_money
def thor_total_received : ℕ := 4 * thor_received_from_each
def thor_fraction_of_total : ℚ := thor_total_received / total_initial_money

-- The theorem to prove
theorem thor_fraction_correct : thor_fraction_of_total = 2/9 :=
by
  sorry

end thor_fraction_correct_l129_129992


namespace probability_of_selecting_A_and_B_l129_129466

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129466


namespace number_for_B_expression_l129_129150

-- Define the number for A as a variable
variable (a : ℤ)

-- Define the number for B in terms of a
def number_for_B (a : ℤ) : ℤ := 2 * a - 1

-- Statement to prove
theorem number_for_B_expression (a : ℤ) : number_for_B a = 2 * a - 1 := by
  sorry

end number_for_B_expression_l129_129150


namespace value_of_E_l129_129675

variable {D E F : ℕ}

theorem value_of_E (h1 : D + E + F = 16) (h2 : F + D + 1 = 16) (h3 : E - 1 = D) : E = 1 :=
sorry

end value_of_E_l129_129675


namespace tan_diff_identity_l129_129815

theorem tan_diff_identity (α : ℝ) (hα : 0 < α ∧ α < π) (h : Real.sin α = 4 / 5) :
  Real.tan (π / 4 - α) = -1 / 7 ∨ Real.tan (π / 4 - α) = -7 :=
sorry

end tan_diff_identity_l129_129815


namespace probability_of_A_and_B_selected_l129_129485

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129485


namespace find_small_pack_size_l129_129863

-- Define the conditions of the problem
def soymilk_sold_in_packs (pack_size : ℕ) : Prop :=
  pack_size = 2 ∨ ∃ L : ℕ, pack_size = L

def cartons_bought (total_cartons : ℕ) (large_pack_size : ℕ) (num_large_packs : ℕ) (small_pack_size : ℕ) : Prop :=
  total_cartons = num_large_packs * large_pack_size + small_pack_size

-- The problem statement as a Lean theorem
theorem find_small_pack_size (total_cartons : ℕ) (num_large_packs : ℕ) (large_pack_size : ℕ) :
  soymilk_sold_in_packs 2 →
  soymilk_sold_in_packs large_pack_size →
  cartons_bought total_cartons large_pack_size num_large_packs 2 →
  total_cartons = 17 →
  num_large_packs = 3 →
  large_pack_size = 5 →
  ∃ S : ℕ, soymilk_sold_in_packs S ∧ S = 2 :=
by
  sorry

end find_small_pack_size_l129_129863


namespace necessary_but_insufficient_for_extreme_value_at_l129_129620

variable {α : Type*} [TopologicalSpace α] [NormedAddCommGroup α] [NormedSpace ℝ α]
variable {β : Type*} [NormedAddCommGroup β] [NormedSpace ℝ β]
variable {f : α → β} {x₀ : α}

theorem necessary_but_insufficient_for_extreme_value_at {f : α → ℝ} {x₀ : α}
  (hf : ContinuousAt f x₀)
  (h0 : f x₀ = 0) :
  IsExtremum f x₀ ↔ (∃ x₁, IsExtremum f x₁ ∧ f x₁ = 0) :=
sorry

end necessary_but_insufficient_for_extreme_value_at_l129_129620


namespace probability_A_and_B_selected_l129_129492

open Finset

noncomputable def total_ways : ℕ := (choose 5 3)

noncomputable def favorable_ways : ℕ := (choose 3 1)

noncomputable def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_A_and_B_selected :
  probability total_ways favorable_ways = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129492


namespace sum_a_c_eq_13_l129_129752

noncomputable def conditions (a b c d k : ℤ) :=
  d = a * b * c ∧
  1 < a ∧ a < b ∧ b < c ∧
  233 = d * k + 79

theorem sum_a_c_eq_13 (a b c d k : ℤ) (h : conditions a b c d k) : a + c = 13 := by
  sorry

end sum_a_c_eq_13_l129_129752


namespace solve_and_sum_solutions_l129_129138

theorem solve_and_sum_solutions :
  (∀ x : ℝ, x^2 > 9 → 
  (∃ S : ℝ, (∀ x : ℝ, (x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12) → S = 8.75))) :=
sorry

end solve_and_sum_solutions_l129_129138


namespace solve_equation_l129_129716

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 := by
  sorry

end solve_equation_l129_129716


namespace cos_135_eq_neg_inv_sqrt_2_l129_129011

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129011


namespace symmetric_point_origin_l129_129809

def symmetric_point (p: ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_point_origin : 
  (symmetric_point (3, -2)) = (-3, 2) :=
by
  sorry

end symmetric_point_origin_l129_129809


namespace find_percentage_reduction_l129_129758

-- Given the conditions of the problem.
def original_price : ℝ := 7500
def current_price: ℝ := 4800
def percentage_reduction (x : ℝ) : Prop := (original_price * (1 - x)^2 = current_price)

-- The statement we need to prove:
theorem find_percentage_reduction (x : ℝ) (h : percentage_reduction x) : x = 0.2 :=
by
  sorry

end find_percentage_reduction_l129_129758


namespace find_x_l129_129101

noncomputable def x_value : ℝ :=
  let x := 24
  x

theorem find_x (x : ℝ) (h : 7 * x + 3 * x + 4 * x + x = 360) : x = 24 := by
  sorry

end find_x_l129_129101


namespace positive_difference_of_numbers_l129_129201

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l129_129201


namespace calculate_smaller_sphere_radius_l129_129640

noncomputable def smaller_sphere_radius (r1 r2 r3 r4 : ℝ) : ℝ := 
  if h : r1 = 2 ∧ r2 = 2 ∧ r3 = 3 ∧ r4 = 3 then 
    6 / 11 
  else 
    0

theorem calculate_smaller_sphere_radius :
  smaller_sphere_radius 2 2 3 3 = 6 / 11 :=
by
  sorry

end calculate_smaller_sphere_radius_l129_129640


namespace probability_A_B_l129_129353

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129353


namespace remainder_when_divided_by_five_l129_129257

theorem remainder_when_divided_by_five :
  let E := 1250 * 1625 * 1830 * 2075 + 245
  E % 5 = 0 := by
  sorry

end remainder_when_divided_by_five_l129_129257


namespace identify_roles_l129_129997

-- Define the number of liars and truth-tellers
def num_liars : Nat := 1000
def num_truth_tellers : Nat := 1000

-- Define the properties of the individuals
def first_person_is_liar := true
def second_person_is_truth_teller := true

-- The main statement equivalent to the problem
theorem identify_roles : first_person_is_liar = true ∧ second_person_is_truth_teller = true := by
  sorry

end identify_roles_l129_129997


namespace age_difference_l129_129222

theorem age_difference (P M Mo : ℕ)
  (h1 : P = 3 * M / 5)
  (h2 : Mo = 5 * M / 3)
  (h3 : P + M + Mo = 196) :
  Mo - P = 64 := 
sorry

end age_difference_l129_129222


namespace christopher_age_l129_129247

theorem christopher_age (G C : ℕ) (h1 : C = 2 * G) (h2 : C - 9 = 5 * (G - 9)) : C = 24 := 
by
  sorry

end christopher_age_l129_129247


namespace positive_difference_l129_129180

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l129_129180


namespace positive_difference_of_two_numbers_l129_129168

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l129_129168


namespace probability_25_sixes_probability_at_least_one_one_expected_number_of_sixes_expected_sum_of_faces_l129_129934

-- Describe the conditions
def cube_formation : Prop :=
  ∃ (cubes : Fin 27 → Fin 6 → ℝ), -- each die face has an equal probability of 1/6
    (∀ i, (∑ j, cubes i j = 1) ∧ (∀ j, cubes i j = 1 / 6))  -- valid probabilities

-- Part a)
theorem probability_25_sixes (h : cube_formation) : 
  let p := (31 : ℝ) / (2^13 * 3^18) in p > 0 := sorry
  
-- Part b)
theorem probability_at_least_one_one (h : cube_formation) :
  let p := 1 - (5^6 : ℝ) / (2^2 * 3^18) in p > 0 := sorry

-- Part c)
theorem expected_number_of_sixes (h : cube_formation) :
  let e := 9 in e > 0 := sorry

-- Part d)
theorem expected_sum_of_faces (h : cube_formation) : 
  let e := 6 - (5^6 : ℝ) / (2 * 3^17) in e > 0 := sorry

end probability_25_sixes_probability_at_least_one_one_expected_number_of_sixes_expected_sum_of_faces_l129_129934


namespace negation_example_l129_129514

variable {I : Set ℝ}

theorem negation_example (h : ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) : ¬(∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 :=
by
  sorry

end negation_example_l129_129514


namespace buckets_with_original_size_l129_129581

variable (C : ℝ) (N : ℝ)

theorem buckets_with_original_size :
  (C : ℝ) > 0 → (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  intros hC hV
  have : C ≠ 0 := by linarith
  have h : (2 / 5) * 62.5 = 25 := by norm_num
  field_simp at hV
  rw h at hV
  assumption

end buckets_with_original_size_l129_129581


namespace Aaron_initial_erasers_l129_129767

/-- 
  Given:
  - Aaron gives 34 erasers to Doris.
  - Aaron ends with 47 erasers.
  Prove:
  - Aaron started with 81 erasers.
-/ 
theorem Aaron_initial_erasers (gives : ℕ) (ends : ℕ) (start : ℕ) :
  gives = 34 → ends = 47 → start = ends + gives → start = 81 :=
by
  intros h_gives h_ends h_start
  sorry

end Aaron_initial_erasers_l129_129767


namespace year_2023_not_lucky_l129_129784

def is_valid_date (month day year : ℕ) : Prop :=
  month * day = year % 100

def is_lucky_year (year : ℕ) : Prop :=
  ∃ month day, month ≤ 12 ∧ day ≤ 31 ∧ is_valid_date month day year

theorem year_2023_not_lucky : ¬ is_lucky_year 2023 :=
by sorry

end year_2023_not_lucky_l129_129784


namespace probability_A_and_B_selected_l129_129397

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129397


namespace positive_difference_of_two_numbers_l129_129165

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129165


namespace prob_select_A_and_B_l129_129328

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129328


namespace tensor_problem_l129_129572

namespace MathProof

def tensor (a b : ℚ) : ℚ := b^2 + 1

theorem tensor_problem (m : ℚ) : tensor m (tensor m 3) = 101 := by
  -- problem statement, proof not included
  sorry

end MathProof

end tensor_problem_l129_129572


namespace prob_select_A_and_B_l129_129327

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129327


namespace horizontal_distance_l129_129840

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end horizontal_distance_l129_129840


namespace probability_AB_selected_l129_129372

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129372


namespace general_term_of_sequence_l129_129943

theorem general_term_of_sequence 
  (a : ℕ → ℝ)
  (log_a : ℕ → ℝ)
  (h1 : ∀ n, log_a n = Real.log (a n)) 
  (h2 : ∃ d, ∀ n, log_a (n + 1) - log_a n = d)
  (h3 : d = Real.log 3)
  (h4 : log_a 0 + log_a 1 + log_a 2 = 6 * Real.log 3) : 
  ∀ n, a n = 3 ^ n :=
by
  sorry

end general_term_of_sequence_l129_129943


namespace eighth_grade_girls_l129_129152

theorem eighth_grade_girls
  (G : ℕ) 
  (boys : ℕ) 
  (h1 : boys = 2 * G - 16) 
  (h2 : G + boys = 68) : 
  G = 28 :=
by
  sorry

end eighth_grade_girls_l129_129152


namespace solution_set_inequality_l129_129159

theorem solution_set_inequality (x : ℝ) : 2 * x^2 - x - 3 > 0 ↔ x > 3 / 2 ∨ x < -1 := by
  sorry

end solution_set_inequality_l129_129159


namespace total_weight_proof_l129_129905

-- Definitions of the variables and conditions given in the problem
variable (M D C : ℕ)
variable (h1 : D + C = 60)  -- Daughter and grandchild together weigh 60 kg
variable (h2 : C = 1 / 5 * M)  -- Grandchild's weight is 1/5th of grandmother's weight
variable (h3 : D = 42)  -- Daughter's weight is 42 kg

-- The goal is to prove the total weight is 150 kg
theorem total_weight_proof (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 42) :
  M + D + C = 150 :=
by
  sorry

end total_weight_proof_l129_129905


namespace cos_135_eq_neg_inv_sqrt_2_l129_129052

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129052


namespace math_problem_l129_129974

theorem math_problem (x : ℝ) (h : x = 0.18 * 4750) : 1.5 * x = 1282.5 :=
by
  sorry

end math_problem_l129_129974


namespace prob_select_A_and_B_l129_129307

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129307


namespace probability_of_selecting_A_and_B_l129_129469

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129469


namespace division_problem_l129_129063

theorem division_problem : 160 / (10 + 11 * 2) = 5 := 
  by 
    sorry

end division_problem_l129_129063


namespace range_of_m_l129_129515

open Set

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 7 }
def B (m : ℝ) : Set ℝ := { x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1) }

theorem range_of_m (m : ℝ) : (A ∪ B m = A) → m ≤ 4 :=
by
  intro h
  sorry

end range_of_m_l129_129515


namespace yellow_paint_amount_l129_129117

theorem yellow_paint_amount (b y : ℕ) (h_ratio : y * 7 = 3 * b) (h_blue_amount : b = 21) : y = 9 :=
by
  sorry

end yellow_paint_amount_l129_129117


namespace monotonic_intervals_range_of_a_for_inequality_l129_129085

noncomputable def f (a x : ℝ) : ℝ := (x + a) / (a * Real.exp x)

theorem monotonic_intervals (a : ℝ) :
  (if a > 0 then
    ∀ x, (x < (1 - a) → 0 < deriv (f a) x) ∧ ((1 - a) < x → deriv (f a) x < 0)
  else
    ∀ x, (x < (1 - a) → deriv (f a) x < 0) ∧ ((1 - a) < x → 0 < deriv (f a) x)) := 
sorry

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x, 0 < x → (3 + 2 * Real.log x) / Real.exp x ≤ f a x + 2 * x) ↔
  a ∈ Set.Iio (-1/2) ∪ Set.Ioi 0 :=
sorry

end monotonic_intervals_range_of_a_for_inequality_l129_129085


namespace positive_difference_of_two_numbers_l129_129184

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129184


namespace babblian_word_count_l129_129558

theorem babblian_word_count (n : ℕ) (h1 : n = 6) : ∃ m, m = 258 := by
  sorry

end babblian_word_count_l129_129558


namespace marla_errand_total_time_l129_129120

theorem marla_errand_total_time :
  let d1 := 20 -- Driving to son's school
  let b := 30  -- Taking a bus to the grocery store
  let s := 15  -- Shopping at the grocery store
  let w := 10  -- Walking to the gas station
  let g := 5   -- Filling up gas
  let r := 25  -- Riding a bicycle to the school
  let p := 70  -- Attending parent-teacher night
  let c := 30  -- Catching up with a friend at a coffee shop
  let sub := 40-- Taking the subway home
  let d2 := 20 -- Driving home
  d1 + b + s + w + g + r + p + c + sub + d2 = 265 := by
  sorry

end marla_errand_total_time_l129_129120


namespace factorize_l129_129792

theorem factorize (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by 
  sorry

end factorize_l129_129792


namespace prob_select_A_and_B_l129_129311

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129311


namespace min_distance_proof_l129_129076

noncomputable def min_distance_squared : ℝ :=
  let a : ℝ := sorry -- some positive real number
  let b : ℝ := ln a / a
  let d : ℝ := sorry -- another real number
  let c : ℝ := d - 2
  in (a - c)^2 + (b - d)^2

theorem min_distance_proof : min_distance_squared = 9 / 2 :=
by
  sorry

end min_distance_proof_l129_129076


namespace positive_difference_of_two_numbers_l129_129193

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l129_129193


namespace sum_of_altitudes_l129_129057

theorem sum_of_altitudes (x y : ℝ) (hline : 10 * x + 8 * y = 80):
  let A := 1 / 2 * 8 * 10
  let hypotenuse := Real.sqrt (8 ^ 2 + 10 ^ 2)
  let third_altitude := 80 / hypotenuse
  let sum_altitudes := 8 + 10 + third_altitude
  sum_altitudes = 18 + 40 / Real.sqrt 41 := by
  sorry

end sum_of_altitudes_l129_129057


namespace probability_A_and_B_selected_l129_129398

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129398


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129455

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129455


namespace permutation_probability_l129_129098

theorem permutation_probability (total_digits: ℕ) (zeros: ℕ) (ones: ℕ) 
  (total_permutations: ℕ) (favorable_permutations: ℕ) (probability: ℚ)
  (h1: total_digits = 6) 
  (h2: zeros = 2) 
  (h3: ones = 4) 
  (h4: total_permutations = 2 ^ total_digits) 
  (h5: favorable_permutations = Nat.choose total_digits zeros) 
  (h6: probability = favorable_permutations / total_permutations) : 
  probability = 15 / 64 := 
sorry

end permutation_probability_l129_129098


namespace solution_set_of_inequality_l129_129158

theorem solution_set_of_inequality :
  {x : ℝ | 2 ≥ 1 / (x - 1)} = {x : ℝ | x < 1} ∪ {x : ℝ | x ≥ 3 / 2} :=
by
  sorry

end solution_set_of_inequality_l129_129158


namespace can_place_circles_l129_129590

theorem can_place_circles (r: ℝ) (h: r = 2008) :
  ∃ (n: ℕ), (n > 4016) ∧ ((n: ℝ) / 2 > r) :=
by 
  sorry

end can_place_circles_l129_129590


namespace select_3_from_5_prob_l129_129343

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129343


namespace solution_set_proof_l129_129829

theorem solution_set_proof {a b : ℝ} :
  (∀ x, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) →
  (∀ x, bx^2 - a * x - 1 > 0) →
  (∀ x, -1 / 2 < x ∧ x < -1 / 3) :=
by
  sorry

end solution_set_proof_l129_129829


namespace factorization_of_polynomial_l129_129268

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l129_129268


namespace problem_statement_l129_129115

def prop_p (x : ℝ) : Prop := x^2 >= x
def prop_q : Prop := ∃ x : ℝ, x^2 >= x

theorem problem_statement : (∀ x : ℝ, prop_p x) = false ∧ prop_q = true :=
by 
  sorry

end problem_statement_l129_129115


namespace seven_pow_k_minus_k_pow_seven_l129_129220

theorem seven_pow_k_minus_k_pow_seven (k : ℕ) (h : 21^k ∣ 435961) : 7^k - k^7 = 1 :=
sorry

end seven_pow_k_minus_k_pow_seven_l129_129220


namespace greatest_integer_radius_l129_129965

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
begin
  have h1 : r^2 < 75, from (lt_div_iff (real.pi_pos)).mp (by rwa [mul_div_cancel_left _ (ne_of_gt real.pi_pos)]),
  -- To show the highest integer r satisfies r^2 < 75 is 8:
  have : r ≤ nat.floor (real.sqrt 75), 
  { exact nat.le_floor_of_lt (by exact_mod_cast h1) },
  -- Since nat.floor (real.sqrt 75) == 8
  exact le_trans this (by norm_num)
end

end greatest_integer_radius_l129_129965


namespace positive_difference_l129_129181

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l129_129181


namespace solve_log_eq_l129_129861

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_log_eq (x : ℝ) (hx1 : x + 1 > 0) (hx2 : x - 1 > 0) :
  log_base (x + 1) (x^3 - 9 * x + 8) * log_base (x - 1) (x + 1) = 3 ↔ x = 3 := by
  sorry

end solve_log_eq_l129_129861


namespace geometric_series_sum_l129_129918

/-- 
The series is given as 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5 + 1/2^6 + 1/2^7 + 1/2^8.
First term a = 1/4 and common ratio r = 1/2 and number of terms n = 7. 
The sum should be 127/256.
-/
theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 2
  let n := 7
  let S := (a * (1 - r^n)) / (1 - r)
  S = 127 / 256 :=
by
  sorry

end geometric_series_sum_l129_129918


namespace cos_135_eq_neg_sqrt2_div_2_l129_129039

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129039


namespace triangle_has_120_degree_l129_129806

noncomputable def angles_of_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180

theorem triangle_has_120_degree (α β γ : Real)
    (h1 : angles_of_triangle α β γ)
    (h2 : Real.cos (3 * α) + Real.cos (3 * β) + Real.cos (3 * γ) = 1) :
  γ = 120 :=
  sorry

end triangle_has_120_degree_l129_129806


namespace quadratic_inequality_l129_129876

variables {a b c : ℝ}

theorem quadratic_inequality (h1 : ∀ x : ℝ, a * x^2 + b * x + c < 0) 
                            (h2 : a < 0) 
                            (h3 : b^2 - 4 * a * c < 0) : 
                            (b / a) < (c / a + 1) :=
begin
  sorry
end

end quadratic_inequality_l129_129876


namespace probability_A_B_l129_129360

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129360


namespace probability_A_B_l129_129357

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129357


namespace cos_135_eq_neg_sqrt2_div_2_l129_129032

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129032


namespace hyperbola_range_m_l129_129827

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (x^2 / (16 - m)) + (y^2 / (9 - m)) = 1) → 9 < m ∧ m < 16 :=
by 
  sorry

end hyperbola_range_m_l129_129827


namespace probability_both_A_B_selected_l129_129294

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129294


namespace Linda_has_24_classmates_l129_129850

theorem Linda_has_24_classmates 
  (cookies_per_student : ℕ := 10)
  (cookies_per_batch : ℕ := 48)
  (chocolate_chip_batches : ℕ := 2)
  (oatmeal_raisin_batches : ℕ := 1)
  (additional_batches : ℕ := 2) : 
  (chocolate_chip_batches * cookies_per_batch + oatmeal_raisin_batches * cookies_per_batch + additional_batches * cookies_per_batch) / cookies_per_student = 24 := 
by 
  sorry

end Linda_has_24_classmates_l129_129850


namespace monica_total_savings_l129_129123

noncomputable def weekly_savings : ℕ := 15
noncomputable def weeks_to_fill_moneybox : ℕ := 60
noncomputable def num_repeats : ℕ := 5
noncomputable def total_savings (weekly_savings weeks_to_fill_moneybox num_repeats : ℕ) : ℕ :=
  (weekly_savings * weeks_to_fill_moneybox) * num_repeats

theorem monica_total_savings :
  total_savings 15 60 5 = 4500 := by
  sorry

end monica_total_savings_l129_129123


namespace probability_both_A_and_B_selected_l129_129443

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129443


namespace probability_both_A_and_B_selected_l129_129446

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129446


namespace sasha_studies_more_avg_4_l129_129147

-- Define the differences recorded over the five days
def differences : List ℤ := [20, 0, 30, -20, -10]

-- Calculate the average difference
def average_difference (diffs : List ℤ) : ℚ :=
  (List.sum diffs : ℚ) / (List.length diffs : ℚ)

-- The statement to prove
theorem sasha_studies_more_avg_4 :
  average_difference differences = 4 := by
  sorry

end sasha_studies_more_avg_4_l129_129147


namespace volleyballs_remaining_l129_129735

def initial_volleyballs := 9
def lent_volleyballs := 5

theorem volleyballs_remaining : initial_volleyballs - lent_volleyballs = 4 := 
by
  sorry

end volleyballs_remaining_l129_129735


namespace total_wet_surface_area_correct_l129_129599

namespace Cistern

-- Define the dimensions of the cistern and the depth of the water
def length : ℝ := 10
def width : ℝ := 8
def depth : ℝ := 1.5

-- Calculate the individual surface areas
def bottom_surface_area : ℝ := length * width
def longer_side_surface_area : ℝ := length * depth * 2
def shorter_side_surface_area : ℝ := width * depth * 2

-- The total wet surface area is the sum of all individual wet surface areas
def total_wet_surface_area : ℝ := 
  bottom_surface_area + longer_side_surface_area + shorter_side_surface_area

-- Prove that the total wet surface area is 134 m^2
theorem total_wet_surface_area_correct : 
  total_wet_surface_area = 134 := 
by sorry

end Cistern

end total_wet_surface_area_correct_l129_129599


namespace infinite_product_value_l129_129886

noncomputable def infinite_product : ℝ :=
  ∏' n : ℕ, 9^(1/(3^n))

theorem infinite_product_value : infinite_product = 27 := 
  by sorry

end infinite_product_value_l129_129886


namespace problem_solved_probability_l129_129865

theorem problem_solved_probability :
  let PA := 1 / 2
  let PB := 1 / 3
  let PC := 1 / 4
  1 - ((1 - PA) * (1 - PB) * (1 - PC)) = 3 / 4 := 
sorry

end problem_solved_probability_l129_129865


namespace probability_of_selecting_A_and_B_l129_129439

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129439


namespace range_of_a_l129_129637

theorem range_of_a {
  a : ℝ
} :
  (∀ x ∈ Set.Ici (2 : ℝ), (x^2 + (2 - a) * x + 4 - 2 * a) > 0) ↔ a < 3 :=
by
  sorry

end range_of_a_l129_129637


namespace find_actual_average_height_l129_129564

noncomputable def actualAverageHeight (avg_height : ℕ) (num_boys : ℕ) (wrong_height : ℕ) (actual_height : ℕ) : Float :=
  let incorrect_total := avg_height * num_boys
  let difference := wrong_height - actual_height
  let correct_total := incorrect_total - difference
  (Float.ofInt correct_total) / (Float.ofNat num_boys)

theorem find_actual_average_height (avg_height num_boys wrong_height actual_height : ℕ) :
  avg_height = 185 ∧ num_boys = 35 ∧ wrong_height = 166 ∧ actual_height = 106 →
  actualAverageHeight avg_height num_boys wrong_height actual_height = 183.29 := by
  intros h
  have h_avg := h.1
  have h_num := h.2.1
  have h_wrong := h.2.2.1
  have h_actual := h.2.2.2
  rw [h_avg, h_num, h_wrong, h_actual]
  sorry

end find_actual_average_height_l129_129564


namespace card_moves_limit_l129_129853

theorem card_moves_limit:
  let total_moves := ∑ k in Finset.range 999, (k + 1) in   -- Summing from 2 to 1000, (k-1) corresponds to (k+1-1)
  total_moves ≤ 500000 := 
by
  sorry

end card_moves_limit_l129_129853


namespace q_investment_time_l129_129726

theorem q_investment_time (x t : ℝ)
  (h1 : (7 * 20 * x) / (5 * t * x) = 7 / 10) : t = 40 :=
by
  sorry

end q_investment_time_l129_129726


namespace find_whole_wheat_pastry_flour_l129_129698

variable (x : ℕ) -- where x is the pounds of whole-wheat pastry flour Sarah already had

-- Conditions
def rye_flour := 5
def whole_wheat_bread_flour := 10
def chickpea_flour := 3
def total_flour := 20

-- Total flour bought
def total_flour_bought := rye_flour + whole_wheat_bread_flour + chickpea_flour

-- Proof statement
theorem find_whole_wheat_pastry_flour (h : total_flour = total_flour_bought + x) : x = 2 :=
by
  -- The proof is omitted
  sorry

end find_whole_wheat_pastry_flour_l129_129698


namespace probability_A_wins_championship_distribution_and_expectation_B_l129_129700

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l129_129700


namespace relationship_between_y1_y2_l129_129526

theorem relationship_between_y1_y2 (y1 y2 : ℝ) :
    (y1 = -3 * 2 + 4 ∧ y2 = -3 * (-1) + 4) → y1 < y2 :=
by
  sorry

end relationship_between_y1_y2_l129_129526


namespace probability_A_B_l129_129355

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129355


namespace factorization_of_polynomial_l129_129269

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l129_129269


namespace prob_select_A_and_B_l129_129322

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129322


namespace value_of_f_neg6_l129_129808

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = -f x

theorem value_of_f_neg6 : f (-6) = 0 :=
by
  sorry

end value_of_f_neg6_l129_129808


namespace cos_135_eq_neg_inv_sqrt_2_l129_129016

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129016


namespace decorate_eggs_time_calculation_l129_129990

/-- Definition of Mia's and Billy's egg decorating rates, total number of eggs to be decorated, and the calculated time when working together --/
def MiaRate : ℕ := 24
def BillyRate : ℕ := 10
def totalEggs : ℕ := 170
def combinedRate : ℕ := MiaRate + BillyRate

theorem decorate_eggs_time_calculation :
  (totalEggs / combinedRate) = 5 := by
  sorry

end decorate_eggs_time_calculation_l129_129990


namespace probability_A_and_B_selected_l129_129475

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129475


namespace factor_quadratic_l129_129280

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l129_129280


namespace max_value_harmonic_series_l129_129954

theorem max_value_harmonic_series (k l m : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m)
  (h : 1/k + 1/l + 1/m < 1) : 
  (1/2 + 1/3 + 1/7) = 41/42 := 
sorry

end max_value_harmonic_series_l129_129954


namespace neither_sufficient_nor_necessary_condition_l129_129728

noncomputable def p (x : ℝ) : Prop := (x - 2) * (x - 1) > 0

noncomputable def q (x : ℝ) : Prop := x - 2 > 0 ∨ x - 1 > 0

theorem neither_sufficient_nor_necessary_condition (x : ℝ) : ¬(p x → q x) ∧ ¬(q x → p x) :=
by
  sorry

end neither_sufficient_nor_necessary_condition_l129_129728


namespace horizontal_distance_l129_129841

theorem horizontal_distance (elev_initial elev_final v_ratio h_ratio distance: ℝ)
  (h1 : elev_initial = 100)
  (h2 : elev_final = 1450)
  (h3 : v_ratio = 1)
  (h4 : h_ratio = 2)
  (h5 : distance = 1350) :
  distance * h_ratio = 2700 := by
  sorry

end horizontal_distance_l129_129841


namespace cube_fraction_inequality_l129_129649

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l129_129649


namespace select_3_from_5_prob_l129_129346

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129346


namespace range_of_m_l129_129656

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 2^x + 1 else 1 - Real.log (x) / Real.log 2

-- The problem is to find the range of m such that f(1 - m^2) > f(2m - 2). We assert the range of m as given in the correct answer.
theorem range_of_m : {m : ℝ | f (1 - m^2) > f (2 * m - 2)} = 
  {m : ℝ | -3 < m ∧ m < 1} ∪ {m : ℝ | m > 3 / 2} :=
sorry

end range_of_m_l129_129656


namespace max_number_of_squares_with_twelve_points_l129_129737

-- Define the condition: twelve marked points in a grid
def twelve_points_marked_on_grid : Prop := 
  -- Assuming twelve specific points represented in a grid-like structure
  -- (This will be defined concretely in the proof implementation context)
  sorry

-- Define the problem statement to be proved
theorem max_number_of_squares_with_twelve_points : 
  twelve_points_marked_on_grid → (∃ n, n = 11) :=
by 
  sorry

end max_number_of_squares_with_twelve_points_l129_129737


namespace evaluate_expression_l129_129624

theorem evaluate_expression (a b : ℕ) (h₁ : a = 3) (h₂ : b = 4) : ((a^b)^a - (b^a)^b) = -16246775 :=
by
  rw [h₁, h₂]
  sorry

end evaluate_expression_l129_129624


namespace best_model_is_model4_l129_129679

-- Define the R^2 values for each model
def R_squared_model1 : ℝ := 0.25
def R_squared_model2 : ℝ := 0.80
def R_squared_model3 : ℝ := 0.50
def R_squared_model4 : ℝ := 0.98

-- Define the highest R^2 value and which model it belongs to
theorem best_model_is_model4 (R1 R2 R3 R4 : ℝ) (h1 : R1 = R_squared_model1) (h2 : R2 = R_squared_model2) (h3 : R3 = R_squared_model3) (h4 : R4 = R_squared_model4) : 
  (R4 = 0.98) ∧ (R4 > R1) ∧ (R4 > R2) ∧ (R4 > R3) :=
by
  sorry

end best_model_is_model4_l129_129679


namespace positive_difference_of_numbers_l129_129202

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l129_129202


namespace probability_A_B_l129_129358

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129358


namespace greatest_radius_l129_129962

theorem greatest_radius (r : ℕ) : (π * (r : ℝ)^2 < 75 * π) ↔ r ≤ 8 := 
by
  sorry

end greatest_radius_l129_129962


namespace percentage_of_nine_hundred_l129_129756

theorem percentage_of_nine_hundred : (45 * 8 = 360) ∧ ((360 / 900) * 100 = 40) :=
by
  have h1 : 45 * 8 = 360 := by sorry
  have h2 : (360 / 900) * 100 = 40 := by sorry
  exact ⟨h1, h2⟩

end percentage_of_nine_hundred_l129_129756


namespace probability_of_selecting_A_and_B_l129_129432

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129432


namespace union_of_P_and_neg_RQ_l129_129950

noncomputable def R : Set ℝ := Set.univ

noncomputable def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

noncomputable def Q : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def neg_RQ : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem union_of_P_and_neg_RQ : 
  P ∪ neg_RQ = {x | x ≤ -2 ∨ 1 ≤ x ∧ x ≤ 3} :=
by
  sorry

end union_of_P_and_neg_RQ_l129_129950


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129048

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129048


namespace Rachel_average_speed_l129_129131

noncomputable def total_distance : ℝ := 2 + 4 + 6

noncomputable def time_to_Alicia : ℝ := 2 / 3
noncomputable def time_to_Lisa : ℝ := 4 / 5
noncomputable def time_to_Nicholas : ℝ := 1 / 2

noncomputable def total_time : ℝ := (20 / 30) + (24 / 30) + (15 / 30)

noncomputable def average_speed : ℝ := total_distance / total_time

theorem Rachel_average_speed : average_speed = 360 / 59 :=
by
  sorry

end Rachel_average_speed_l129_129131


namespace probability_both_A_B_selected_l129_129297

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129297


namespace sum_of_first_9_terms_l129_129807

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

axiom arithmetic_sequence_condition (h : is_arithmetic_sequence a) : a 5 = 2

theorem sum_of_first_9_terms (h : is_arithmetic_sequence a) (h5: a 5 = 2) : sum_of_first_n_terms a 9 = 18 := by
  sorry

end sum_of_first_9_terms_l129_129807


namespace probability_both_A_and_B_selected_l129_129447

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129447


namespace perpendicular_bisector_midpoint_l129_129687

theorem perpendicular_bisector_midpoint :
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  3 * R.1 - 2 * R.2 = -15 :=
by
  let P := (-8, 15)
  let Q := (6, -3)
  let R := ((-8 + 6) / 2, (15 - 3) / 2)
  sorry

end perpendicular_bisector_midpoint_l129_129687


namespace cube_fraction_inequality_l129_129648

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l129_129648


namespace cos_135_eq_neg_inv_sqrt_2_l129_129005

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129005


namespace positive_difference_of_two_numbers_l129_129164

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129164


namespace Julie_monthly_salary_l129_129537

theorem Julie_monthly_salary 
(hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (missed_days : ℕ) 
(h1 : hourly_rate = 5) (h2 : hours_per_day = 8) 
(h3 : days_per_week = 6) (h4 : weeks_per_month = 4) 
(h5 : missed_days = 1) : 
hourly_rate * hours_per_day * days_per_week * weeks_per_month - hourly_rate * hours_per_day * missed_days = 920 :=
by sorry

end Julie_monthly_salary_l129_129537


namespace hulk_jump_geometric_seq_l129_129563

theorem hulk_jump_geometric_seq :
  ∃ n : ℕ, (2 * 3^(n-1) > 2000) ∧ n = 8 :=
by
  sorry

end hulk_jump_geometric_seq_l129_129563


namespace foldable_shape_is_axisymmetric_l129_129093

def is_axisymmetric_shape (shape : Type) : Prop :=
  (∃ l : (shape → shape), (∀ x, l x = x))

theorem foldable_shape_is_axisymmetric (shape : Type) (l : shape → shape) 
  (h1 : ∀ x, l x = x) : is_axisymmetric_shape shape := by
  sorry

end foldable_shape_is_axisymmetric_l129_129093


namespace a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l129_129543

theorem a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b
  (a b : ℝ) :
  (∀ a b : ℝ, (a - b) * a ^ 2 < 0 → a < b) ∧ 
  (¬∀ a b : ℝ, a < b → (a - b) * a ^ 2 < 0) :=
sorry

end a_minus_b_a_squared_lt_zero_sufficient_not_necessary_for_a_lt_b_l129_129543


namespace complex_exp_sum_l129_129542

def w : ℂ := sorry  -- We define w as a complex number, satisfying the given condition.

theorem complex_exp_sum (h : w^2 - w + 1 = 0) : 
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2 * w :=
by
  sorry

end complex_exp_sum_l129_129542


namespace complex_number_quadrant_l129_129946

theorem complex_number_quadrant 
  (i : ℂ) (hi : i.im = 1 ∧ i.re = 0)
  (x y : ℝ) 
  (h : (x + i) * i = y - i) : 
  x < 0 ∧ y < 0 := 
sorry

end complex_number_quadrant_l129_129946


namespace school_A_win_prob_expectation_X_is_13_l129_129704

-- Define the probabilities of school A winning individual events
def pA_event1 : ℝ := 0.5
def pA_event2 : ℝ := 0.4
def pA_event3 : ℝ := 0.8

-- Define the probability of school A winning the championship
def pA_win_championship : ℝ :=
  (pA_event1 * pA_event2 * pA_event3) +
  (pA_event1 * (1 - pA_event2) * pA_event3) +
  (pA_event1 * pA_event2 * (1 - pA_event3)) +
  ((1 - pA_event1) * pA_event2 * pA_event3)

-- Proof statement for the probability of school A winning the championship
theorem school_A_win_prob : pA_win_championship = 0.6 := sorry

-- Define the distribution and expectation for school B's total score
def X_prob : ℝ → ℝ
| 0  := (1 - pA_event1) * (1 - pA_event2) * (1 - pA_event3)
| 10 := pA_event1 * (1 - pA_event2) * (1 - pA_event3) +
        (1 - pA_event1) * pA_event2 * (1 - pA_event3) +
        (1 - pA_event1) * (1 - pA_event2) * pA_event3
| 20 := pA_event1 * pA_event2 * (1 - pA_event3) +
        pA_event1 * (1 - pA_event2) * pA_event3 +
        (1 - pA_event1) * pA_event2 * pA_event3
| 30 := pA_event1 * pA_event2 * pA_event3
| _  := 0

def expected_X : ℝ :=
  0 * X_prob 0 +
  10 * X_prob 10 +
  20 * X_prob 20 +
  30 * X_prob 30

-- Proof statement for the expectation of school B's total score
theorem expectation_X_is_13 : expected_X = 13 := sorry

end school_A_win_prob_expectation_X_is_13_l129_129704


namespace smallest_positive_number_is_x2_l129_129933

noncomputable def x1 : ℝ := 14 - 4 * Real.sqrt 17
noncomputable def x2 : ℝ := 4 * Real.sqrt 17 - 14
noncomputable def x3 : ℝ := 23 - 7 * Real.sqrt 14
noncomputable def x4 : ℝ := 65 - 12 * Real.sqrt 34
noncomputable def x5 : ℝ := 12 * Real.sqrt 34 - 65

theorem smallest_positive_number_is_x2 :
  x2 = 4 * Real.sqrt 17 - 14 ∧
  (0 < x1 ∨ 0 < x2 ∨ 0 < x3 ∨ 0 < x4 ∨ 0 < x5) ∧
  (∀ x : ℝ, (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4 ∨ x = x5) → 0 < x → x2 ≤ x) := sorry

end smallest_positive_number_is_x2_l129_129933


namespace find_c_l129_129763

-- Definitions based on the conditions in the problem
def is_vertex (h k : ℝ) := (5, 1) = (h, k)
def passes_through (x y : ℝ) := (2, 3) = (x, y)

-- Lean theorem statement
theorem find_c (a b c : ℝ) (h k x y : ℝ) (hv : is_vertex h k) (hp : passes_through x y)
  (heq : ∀ y, x = a * y^2 + b * y + c) : c = 17 / 4 :=
by
  sorry

end find_c_l129_129763


namespace quadratic_zero_points_probability_l129_129945

theorem quadratic_zero_points_probability :
  let a_values : Set ℤ := {-1, 0, 1, 2}
  let b_values : Set ℤ := {-1, 0, 1, 2}
  let total_combinations := a_values.card * b_values.card
  let zero_points_count := a_values.card * b_values.card - 3 -- Calculated manually as shown in the steps
  let probability := (zero_points_count : ℚ) / total_combinations
  probability = 13/16 :=
by
  sorry

end quadratic_zero_points_probability_l129_129945


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129042

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129042


namespace felix_distance_l129_129565

theorem felix_distance : 
  ∀ (avg_speed: ℕ) (time: ℕ) (factor: ℕ), 
  avg_speed = 66 → factor = 2 → time = 4 → (factor * avg_speed * time = 528) := 
by
  intros avg_speed time factor h_avg_speed h_factor h_time
  rw [h_avg_speed, h_factor, h_time]
  norm_num
  sorry

end felix_distance_l129_129565


namespace cos_135_eq_neg_inv_sqrt_2_l129_129009

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129009


namespace booth_visibility_correct_l129_129896

noncomputable def booth_visibility (L : ℝ) : ℝ × ℝ :=
  let ρ_min := L
  let ρ_max := (1 + Real.sqrt 2) / 2 * L
  (ρ_min, ρ_max)

theorem booth_visibility_correct (L : ℝ) (hL : L > 0) :
  booth_visibility L = (L, (1 + Real.sqrt 2) / 2 * L) :=
by
  sorry

end booth_visibility_correct_l129_129896


namespace probability_of_selecting_A_and_B_l129_129385

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129385


namespace select_3_from_5_prob_l129_129342

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129342


namespace probability_A_and_B_selected_l129_129412

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129412


namespace lion_cubs_per_month_l129_129879

theorem lion_cubs_per_month
  (initial_lions : ℕ)
  (final_lions : ℕ)
  (months : ℕ)
  (lions_dying_per_month : ℕ)
  (net_increase : ℕ)
  (x : ℕ) : 
  initial_lions = 100 → 
  final_lions = 148 → 
  months = 12 → 
  lions_dying_per_month = 1 → 
  net_increase = 48 → 
  12 * (x - 1) = net_increase → 
  x = 5 := by
  intros initial_lions_eq final_lions_eq months_eq lions_dying_eq net_increase_eq equation
  sorry

end lion_cubs_per_month_l129_129879


namespace right_triangle_leg_length_l129_129231

theorem right_triangle_leg_length (a b c : ℕ) (h₁ : a = 8) (h₂ : c = 17) (h₃ : a^2 + b^2 = c^2) : b = 15 := 
by
  sorry

end right_triangle_leg_length_l129_129231


namespace cos_135_eq_neg_inv_sqrt_2_l129_129056

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129056


namespace greatest_integer_radius_l129_129971

theorem greatest_integer_radius (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 := 
by {
  have h1 : r^2 < 75, from (div_lt_div_right (by linarith [pi_pos])).mp h,
  have sqrt_75_lt_9 : sqrt 75 < 9 := by norm_num,
  have sqrt_75_lt_9' : r < 9 := lt_of_pow_lt_pow_of_lt_two h1 sqrt_75_lt_9,
  exact le_of_lt (int.le_iff_coe.mp sqrt_75_lt_9')
}

end greatest_integer_radius_l129_129971


namespace evaluate_expression_l129_129925

def a : ℕ := 3
def b : ℕ := 2

theorem evaluate_expression : (a^2 * a^5) / (b^2 / b^3) = 4374 := by
  sorry

end evaluate_expression_l129_129925


namespace total_yards_run_in_4_games_l129_129690

theorem total_yards_run_in_4_games (malik_ypg josiah_ypg darnell_avg : ℕ) (num_games : ℕ)
  (h1 : malik_ypg = 18) (h2 : josiah_ypg = 22) (h3 : darnell_avg = 11) (h4 : num_games = 4) :
  malik_ypg * num_games + josiah_ypg * num_games + darnell_avg * num_games = 204 := 
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end total_yards_run_in_4_games_l129_129690


namespace probability_A_and_B_selected_l129_129401

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129401


namespace cos_135_eq_neg_sqrt2_div_2_l129_129029

theorem cos_135_eq_neg_sqrt2_div_2 : cos (135 * real.pi / 180) = - (real.sqrt 2 / 2) :=
by
  have h1 : 135 = 180 - 45 := by norm_num
  have h2 : cos (180 * real.pi / 180 - 45 * real.pi / 180) = - cos (45 * real.pi / 180) :=
    by rw [cos_sub, cos_pi, cos_pi_div_four, sin_pi, sin_pi_div_four]; norm_num
  have h3 : cos (45 * real.pi / 180) = real.sqrt 2 / 2 := by norm_num
  rw [h1, ←h2, h3]
  norm_num
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129029


namespace intersection_distance_to_pole_l129_129677

theorem intersection_distance_to_pole (rho theta : ℝ) (h1 : rho > 0) (h2 : rho = 2 * theta + 1) (h3 : rho * theta = 1) : rho = 2 :=
by
  -- We replace "sorry" with actual proof steps, if necessary.
  sorry

end intersection_distance_to_pole_l129_129677


namespace taxi_ride_cost_l129_129604

def baseFare : ℝ := 1.50
def costPerMile : ℝ := 0.25
def milesTraveled : ℕ := 5
def totalCost := baseFare + (costPerMile * milesTraveled)

/-- The cost of a 5-mile taxi ride is $2.75. -/
theorem taxi_ride_cost : totalCost = 2.75 := by
  sorry

end taxi_ride_cost_l129_129604


namespace probability_A_and_B_selected_l129_129416

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129416


namespace speed_of_man_in_still_water_l129_129750

variable (V_m V_s : ℝ)

/-- The speed of a man in still water -/
theorem speed_of_man_in_still_water (h_downstream : 18 = (V_m + V_s) * 3)
                                     (h_upstream : 12 = (V_m - V_s) * 3) :
    V_m = 5 := 
sorry

end speed_of_man_in_still_water_l129_129750


namespace determine_x_l129_129258

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end determine_x_l129_129258


namespace Felix_distance_proof_l129_129566

def average_speed : ℕ := 66
def twice_speed : ℕ := 2 * average_speed
def driving_hours : ℕ := 4
def distance_covered : ℕ := twice_speed * driving_hours

theorem Felix_distance_proof : distance_covered = 528 := by
  sorry

end Felix_distance_proof_l129_129566


namespace probability_A_and_B_selected_l129_129316

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129316


namespace fourth_intersection_point_l129_129835

noncomputable def curve (p : ℝ × ℝ) : Prop := p.1 * p.2 = 2

def circle (a b r : ℝ) (p : ℝ × ℝ) : Prop := (p.1 - a)^2 + (p.2 - b)^2 = r^2

def point1 := (4 : ℝ, 1 / 2 : ℝ)
def point2 := (-2 : ℝ, -1 : ℝ)
def point3 := (1 / 4 : ℝ, 8 : ℝ)
def point4 := (-1 / 8 : ℝ, -16 : ℝ)

theorem fourth_intersection_point 
(a b r : ℝ) 
(h1 : curve point1)
(h2 : curve point2)
(h3 : curve point3)
(h4 : circle a b r point1)
(h5 : circle a b r point2)
(h6 : circle a b r point3) 
(h7 : curve point4) :
  circle a b r point4 :=
  sorry

end fourth_intersection_point_l129_129835


namespace cosine_135_eq_neg_sqrt_2_div_2_l129_129047

theorem cosine_135_eq_neg_sqrt_2_div_2 (cos180 : Real := -1) (cos90 : Real := 0) (cos45 : Real := Real.sqrt 2 / 2)
    (theta1 : Real := 180) (theta2 : Real := 45) (theta3 : Real := 135) :
    theta3 = theta1 - theta2 ∧ 
    Real.cos theta1 = cos180 ∧ 
    Real.cos theta2 = cos45 → 
    Real.cos theta3 = -cos45 := by
  sorry

end cosine_135_eq_neg_sqrt_2_div_2_l129_129047


namespace range_of_a_l129_129937

theorem range_of_a {a : ℝ} : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ - 3 / 5 < a ∧ a ≤ 1 := sorry

end range_of_a_l129_129937


namespace sin_double_angle_l129_129636

theorem sin_double_angle (α : ℝ) (h : Real.tan α = 3 / 4) : Real.sin (2 * α) = 24 / 25 := by
  sorry

end sin_double_angle_l129_129636


namespace percentage_girls_l129_129579

theorem percentage_girls (initial_boys : ℕ) (initial_girls : ℕ) (added_boys : ℕ) :
  initial_boys = 11 → initial_girls = 13 → added_boys = 1 → 
  100 * initial_girls / (initial_boys + added_boys + initial_girls) = 52 :=
by
  intros h_boys h_girls h_added
  sorry

end percentage_girls_l129_129579


namespace factor_quadratic_l129_129277

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l129_129277


namespace probability_of_selecting_A_and_B_l129_129369

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129369


namespace complex_imaginary_unit_sum_l129_129775

theorem complex_imaginary_unit_sum (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 = -1 := 
by sorry

end complex_imaginary_unit_sum_l129_129775


namespace g_four_times_of_three_l129_129985

noncomputable def g (x : ℕ) : ℕ :=
if x % 3 = 0 then x / 3 else 4 * x - 1

theorem g_four_times_of_three :
  g (g (g (g 3))) = 3 := by
  sorry

end g_four_times_of_three_l129_129985


namespace roots_of_polynomial_l129_129847

theorem roots_of_polynomial (c d : ℝ) (h1 : Polynomial.eval c (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0)
    (h2 : Polynomial.eval d (Polynomial.X ^ 4 - 6 * Polynomial.X + 3) = 0) :
    c * d + c + d = Real.sqrt 3 := 
sorry

end roots_of_polynomial_l129_129847


namespace six_hundred_billion_in_scientific_notation_l129_129237

theorem six_hundred_billion_in_scientific_notation (billion : ℕ) (h_billion : billion = 10^9) : 
  600 * billion = 6 * 10^11 :=
by
  rw [h_billion]
  sorry

end six_hundred_billion_in_scientific_notation_l129_129237


namespace find_supplementary_angle_l129_129575

noncomputable def degree (x : ℝ) : ℝ := x
noncomputable def complementary_angle (x : ℝ) : ℝ := 90 - x
noncomputable def supplementary_angle (x : ℝ) : ℝ := 180 - x

theorem find_supplementary_angle
  (x : ℝ)
  (h1 : degree x / complementary_angle x = 1 / 8) :
  supplementary_angle x = 170 :=
by
  sorry

end find_supplementary_angle_l129_129575


namespace zoo_ticket_problem_l129_129904

theorem zoo_ticket_problem :
  ∀ (total_amount adult_ticket_cost children_ticket_cost : ℕ)
    (num_adult_tickets : ℕ),
  total_amount = 119 →
  adult_ticket_cost = 21 →
  children_ticket_cost = 14 →
  num_adult_tickets = 4 →
  6 = (num_adult_tickets + (total_amount - num_adult_tickets * adult_ticket_cost) / children_ticket_cost) :=
by 
  intros total_amount adult_ticket_cost children_ticket_cost num_adult_tickets 
         total_amt_eq adult_ticket_cost_eq children_ticket_cost_eq num_adult_tickets_eq
  sorry

end zoo_ticket_problem_l129_129904


namespace cards_eaten_by_hippopotamus_l129_129980

-- Defining the initial and remaining number of cards
def initial_cards : ℕ := 72
def remaining_cards : ℕ := 11

-- Statement of the proof problem
theorem cards_eaten_by_hippopotamus (initial_cards remaining_cards : ℕ) : initial_cards - remaining_cards = 61 :=
by
  sorry

end cards_eaten_by_hippopotamus_l129_129980


namespace prob_select_A_and_B_l129_129308

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129308


namespace fran_avg_speed_l129_129107

theorem fran_avg_speed (Joann_speed : ℕ) (Joann_time : ℚ) (Fran_time : ℕ) (distance : ℕ) (s : ℚ) : 
  Joann_speed = 16 → 
  Joann_time = 3.5 → 
  Fran_time = 4 → 
  distance = Joann_speed * Joann_time → 
  distance = Fran_time * s → 
  s = 14 :=
by
  intros hJs hJt hFt hD hF
  sorry

end fran_avg_speed_l129_129107


namespace tangent_line_at_point_l129_129723

noncomputable def tangent_line_equation (x : ℝ) : Prop :=
  ∀ y : ℝ, y = x * (3 * Real.log x + 1) → (x = 1 ∧ y = 1) → y = 4 * x - 3

theorem tangent_line_at_point : tangent_line_equation 1 :=
sorry

end tangent_line_at_point_l129_129723


namespace solve_expression_l129_129862

def evaluation_inside_parentheses : ℕ := 3 - 3

def power_of_zero : ℝ := (5 : ℝ) ^ evaluation_inside_parentheses

theorem solve_expression :
  (3 : ℝ) - power_of_zero = 2 := by
  -- Utilize the conditions defined above
  sorry

end solve_expression_l129_129862


namespace cos_135_eq_neg_inv_sqrt_2_l129_129050

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129050


namespace probability_of_selecting_A_and_B_l129_129463

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129463


namespace inequality_proof_l129_129653

theorem inequality_proof (s r : ℝ) (h1 : s > 0) (h2 : r > 0) (h3 : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by
  sorry

end inequality_proof_l129_129653


namespace minimum_squares_and_perimeter_l129_129614

theorem minimum_squares_and_perimeter 
  (length width : ℕ) 
  (h_length : length = 90) 
  (h_width : width = 42) 
  (h_gcd : Nat.gcd length width = 6) 
  : 
  ((length / Nat.gcd length width) * (width / Nat.gcd length width) = 105) ∧ 
  (105 * (4 * Nat.gcd length width) = 2520) := 
by 
  sorry

end minimum_squares_and_perimeter_l129_129614


namespace rickshaw_distance_l129_129601

theorem rickshaw_distance (km1_charge : ℝ) (rate_per_km : ℝ) (total_km : ℝ) (total_charge : ℝ) :
  km1_charge = 13.50 → rate_per_km = 2.50 → total_km = 13 → total_charge = 103.5 → (total_charge - km1_charge) / rate_per_km = 36 :=
by
  intro h1 h2 h3 h4
  -- We would fill in proof steps here, but skipping as required.
  sorry

end rickshaw_distance_l129_129601


namespace problem_statement_l129_129813

-- Definitions of propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x = 1
def q : Prop := ∀ x : ℝ, x^2 > 0

-- The proof problem
theorem problem_statement : ¬ (¬ p ∧ ¬ q) :=
by 
  -- sorry here indicates that actual proof is omitted
  sorry

end problem_statement_l129_129813


namespace brendan_threw_back_l129_129245

-- Brendan's catches in the morning, throwing back x fish and catching more in the afternoon
def brendan_morning (x : ℕ) : ℕ := 8 - x
def brendan_afternoon : ℕ := 5

-- Brendan's and his dad's total catches
def brendan_total (x : ℕ) : ℕ := brendan_morning x + brendan_afternoon
def dad_total : ℕ := 13

-- Combined total fish caught by both
def total_fish (x : ℕ) : ℕ := brendan_total x + dad_total

-- The number of fish thrown back by Brendan
theorem brendan_threw_back : ∃ x : ℕ, total_fish x = 23 ∧ x = 3 :=
by
  sorry

end brendan_threw_back_l129_129245


namespace positive_difference_of_two_numbers_l129_129186

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129186


namespace find_total_games_l129_129621

-- Define the initial conditions
def avg_points_per_game : ℕ := 26
def games_played : ℕ := 15
def goal_avg_points : ℕ := 30
def required_avg_remaining : ℕ := 42

-- Statement of the proof problem
theorem find_total_games (G : ℕ) :
  avg_points_per_game * games_played + required_avg_remaining * (G - games_played) = goal_avg_points * G →
  G = 20 :=
by sorry

end find_total_games_l129_129621


namespace range_of_m_l129_129800

-- Definitions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := (x^2 - 4*x + 4 - m^2) ≤ 0

-- Theorem Statement
theorem range_of_m (m : ℝ) (h_m : m > 0) : 
  (¬(∃ x, ¬p x) → ¬(∃ x, ¬q x m)) → m ≥ 8 := 
sorry -- Proof not required

end range_of_m_l129_129800


namespace prod_three_consec_cubemultiple_of_504_l129_129580

theorem prod_three_consec_cubemultiple_of_504 (a : ℤ) : (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := by
  sorry

end prod_three_consec_cubemultiple_of_504_l129_129580


namespace cos_alpha_value_l129_129081

variable (α : ℝ)
variable (x y r : ℝ)

-- Conditions
def point_condition : Prop := (x = 1 ∧ y = -Real.sqrt 3 ∧ r = 2 ∧ r = Real.sqrt (x^2 + y^2))

-- Question/Proof Statement
theorem cos_alpha_value (h : point_condition x y r) : Real.cos α = 1 / 2 :=
sorry

end cos_alpha_value_l129_129081


namespace max_radius_of_circle_l129_129969

theorem max_radius_of_circle (r : ℕ) (h : π * r^2 < 75 * π) : r ≤ 8 :=
by
  sorry

end max_radius_of_circle_l129_129969


namespace setC_is_not_pythagorean_triple_l129_129769

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of numbers
def setA := (3, 4, 5)
def setB := (5, 12, 13)
def setC := (7, 25, 26)
def setD := (6, 8, 10)

-- The theorem stating that setC is not a Pythagorean triple
theorem setC_is_not_pythagorean_triple : ¬isPythagoreanTriple 7 25 26 := 
by sorry

end setC_is_not_pythagorean_triple_l129_129769


namespace prove_monomial_l129_129818

-- Definitions and conditions from step a)
def like_terms (x y : ℕ) := 
  x = 2 ∧ x + y = 5

-- Main statement to be proved
theorem prove_monomial (x y : ℕ) (h : like_terms x y) : 
  1 / 2 * x^3 - 1 / 6 * x * y^2 = 1 :=
by
  sorry

end prove_monomial_l129_129818


namespace triangle_side_lengths_l129_129598

theorem triangle_side_lengths (r : ℝ) (AC BC AB : ℝ) (y : ℝ) 
  (h1 : r = 3 * Real.sqrt 2)
  (h2 : AC = 5 * Real.sqrt y) 
  (h3 : BC = 13 * Real.sqrt y) 
  (h4 : AB = 10 * Real.sqrt y) : 
  r = 3 * Real.sqrt 2 → 
  (∃ (AC BC AB : ℝ), 
     AC = 5 * Real.sqrt (7) ∧ 
     BC = 13 * Real.sqrt (7) ∧ 
     AB = 10 * Real.sqrt (7)) :=
by
  sorry

end triangle_side_lengths_l129_129598


namespace evaluate_expression_l129_129623

theorem evaluate_expression : 
  let a := 3
  let b := 4
  (a^b)^a - (b^a)^b = -16245775 := 
by 
  sorry

end evaluate_expression_l129_129623


namespace pencil_color_change_l129_129135

-- Problem statement: Given several children each with a pencil of one of three colors,
-- prove that it is possible to assign pencils initially such that, after some exchanges,
-- no child retains the same pencil color.

theorem pencil_color_change (n : ℕ) (pencils : Fin n → Fin 3) (exchange : Perm (Fin n)) :
  (∀ i, pencils (exchange i) ≠ pencils i) → ∃ initial_assignment : Fin n → Fin 3,
  ∀ i, initial_assignment (exchange i) ≠ initial_assignment i := 
sorry

end pencil_color_change_l129_129135


namespace george_speed_to_school_l129_129502

theorem george_speed_to_school :
  ∀ (d1 d2 v1 v2 v_arrive : ℝ), 
  d1 = 1.0 → d2 = 0.5 → v1 = 3.0 → v2 * (d1 / v1 + d2 / v2) = (d1 + d2) / 4.0 → v_arrive = 12.0 :=
by sorry

end george_speed_to_school_l129_129502


namespace remainder_of_2365487_div_3_l129_129884

theorem remainder_of_2365487_div_3 : (2365487 % 3) = 2 := by
  sorry

end remainder_of_2365487_div_3_l129_129884


namespace probability_A_B_selected_l129_129426

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129426


namespace classroom_needs_more_money_l129_129908

theorem classroom_needs_more_money 
    (goal : ℕ) 
    (raised_from_two_families : ℕ) 
    (raised_from_eight_families : ℕ) 
    (raised_from_ten_families : ℕ) 
    (H : goal = 200) 
    (H1 : raised_from_two_families = 2 * 20) 
    (H2 : raised_from_eight_families = 8 * 10) 
    (H3 : raised_from_ten_families = 10 * 5) 
    (total_raised : ℕ := raised_from_two_families + raised_from_eight_families + raised_from_ten_families) : 
    (goal - total_raised) = 30 := 
by 
  sorry

end classroom_needs_more_money_l129_129908


namespace find_n_on_angle_bisector_l129_129511

theorem find_n_on_angle_bisector (M : ℝ × ℝ) (hM : M = (3 * n - 2, 2 * n + 7) ∧ M.1 + M.2 = 0) : 
    n = -1 :=
by
  sorry

end find_n_on_angle_bisector_l129_129511


namespace students_growth_rate_l129_129831

theorem students_growth_rate (x : ℝ) 
  (h_total : 728 = 200 + 200 * (1+x) + 200 * (1+x)^2) : 
  200 + 200 * (1+x) + 200*(1+x)^2 = 728 := 
  by
  sorry

end students_growth_rate_l129_129831


namespace total_courses_attended_l129_129547

-- Define the number of courses attended by Max
def maxCourses : ℕ := 40

-- Define the number of courses attended by Sid (four times as many as Max)
def sidCourses : ℕ := 4 * maxCourses

-- Define the total number of courses attended by both Max and Sid
def totalCourses : ℕ := maxCourses + sidCourses

-- The proof statement
theorem total_courses_attended : totalCourses = 200 := by
  sorry

end total_courses_attended_l129_129547


namespace probability_of_selecting_A_and_B_l129_129367

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129367


namespace math_problem_l129_129748

def a : ℕ := 2013
def b : ℕ := 2014

theorem math_problem :
  (a^3 - 2 * a^2 * b + 3 * a * b^2 - b^3 + 1) / (a * b) = a := by
  sorry

end math_problem_l129_129748


namespace probability_A_and_B_selected_l129_129315

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129315


namespace cos_135_eq_correct_l129_129017

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129017


namespace positive_difference_of_two_numbers_l129_129191

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l129_129191


namespace positive_difference_of_two_numbers_l129_129174

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l129_129174


namespace factor_quadratic_l129_129279

theorem factor_quadratic : ∀ (x : ℝ), 16*x^2 - 40*x + 25 = (4*x - 5)^2 := by
  intro x
  sorry

end factor_quadratic_l129_129279


namespace Julie_monthly_salary_l129_129536

theorem Julie_monthly_salary 
(hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (missed_days : ℕ) 
(h1 : hourly_rate = 5) (h2 : hours_per_day = 8) 
(h3 : days_per_week = 6) (h4 : weeks_per_month = 4) 
(h5 : missed_days = 1) : 
hourly_rate * hours_per_day * days_per_week * weeks_per_month - hourly_rate * hours_per_day * missed_days = 920 :=
by sorry

end Julie_monthly_salary_l129_129536


namespace last_digit_to_appear_is_6_l129_129249

def modified_fib (n : ℕ) : ℕ :=
match n with
| 1 => 2
| 2 => 3
| n + 3 => modified_fib (n + 2) + modified_fib (n + 1)
| _ => 0 -- To silence the "missing cases" warning; won't be hit.

theorem last_digit_to_appear_is_6 :
  ∃ N : ℕ, ∀ n : ℕ, (n < N → ∃ d, d < 10 ∧ 
    (∀ m < n, (modified_fib m) % 10 ≠ d) ∧ d = 6) := sorry

end last_digit_to_appear_is_6_l129_129249


namespace positive_difference_of_two_numbers_l129_129161

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129161


namespace pie_eating_contest_l129_129882

theorem pie_eating_contest :
  (7 / 8 : ℚ) - (5 / 6 : ℚ) = (1 / 24 : ℚ) :=
sorry

end pie_eating_contest_l129_129882


namespace max_number_of_squares_l129_129738

theorem max_number_of_squares (points : finset (fin 12)) : 
  ∃ (sq_count : ℕ), sq_count = 11 := 
sorry

end max_number_of_squares_l129_129738


namespace probability_of_selecting_A_and_B_l129_129371

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129371


namespace at_least_240_students_l129_129911

-- Define the total number of students
def total_students : ℕ := 1200

-- Define the 80th percentile score
def percentile_80_score : ℕ := 103

-- Define the number of students below the 80th percentile
def students_below_80th_percentile : ℕ := total_students * 80 / 100

-- Define the number of students with at least the 80th percentile score
def students_at_least_80th_percentile : ℕ := total_students - students_below_80th_percentile

-- The theorem to prove
theorem at_least_240_students : students_at_least_80th_percentile ≥ 240 :=
by
  -- Placeholder proof, to be filled in as the actual proof
  sorry

end at_least_240_students_l129_129911


namespace chord_intercept_length_l129_129567

noncomputable def focus_of_parabola (x : ℝ) (y : ℝ) : Prop :=
x = 0 ∧ y = 1

noncomputable def directrix_of_parabola (y : ℝ) : Prop :=
y = -1

noncomputable def circle (x : ℝ) (y : ℝ) : Prop :=
x^2 + (y - 1)^2 = 4

theorem chord_intercept_length :
  let focus_x := 0
  let focus_y := 1
  let directrix_y := -1
  let radius := 2
  let center_x := focus_x
  let center_y := focus_y
  let circle_eq := circle x y
  -- Find y-intercepts
  let y_top := 3
  let y_bottom := -1
  -- Compute chord length
  y_top - y_bottom = 4 :=
by sorry

end chord_intercept_length_l129_129567


namespace joint_probability_bound_l129_129754

open ProbabilityTheory

theorem joint_probability_bound {Ω : Type*} [measure_space Ω] (P : measure Ω) [probability_measure P]
  (A B : set Ω) :
  |P[A ∩ B] - P[A] * P[B]| ≤ 1/4 := 
sorry

end joint_probability_bound_l129_129754


namespace positive_difference_of_two_numbers_l129_129188

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129188


namespace probability_A_and_B_selected_l129_129414

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129414


namespace factorization_of_polynomial_l129_129266

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l129_129266


namespace probability_A_and_B_selected_l129_129405

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129405


namespace select_3_from_5_prob_l129_129350

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129350


namespace positive_difference_of_two_numbers_l129_129182

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129182


namespace find_sum_l129_129804

variable (a b c d : ℝ)

theorem find_sum (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end find_sum_l129_129804


namespace probability_A_and_B_selected_l129_129318

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129318


namespace band_song_arrangements_l129_129868

theorem band_song_arrangements (n : ℕ) (t : ℕ) (r : ℕ) 
  (h1 : n = 8) (h2 : t = 3) (h3 : r = 5) : 
  ∃ (ways : ℕ), ways = 14400 := by
  sorry

end band_song_arrangements_l129_129868


namespace maximum_distance_product_l129_129833

theorem maximum_distance_product (α : ℝ) (hα : 0 < α ∧ α < π / 2) :
  let ρ1 := 4 * Real.cos α
  let ρ2 := 2 * Real.sin α
  |ρ1 * ρ2| ≤ 4 :=
by
  -- The proof would go here
  sorry

end maximum_distance_product_l129_129833


namespace red_balls_count_l129_129977

theorem red_balls_count (R W : ℕ) (h1 : R / W = 4 / 5) (h2 : W = 20) : R = 16 := sorry

end red_balls_count_l129_129977


namespace log3_cubicroot_of_3_l129_129262

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem log3_cubicroot_of_3 :
  log_base_3 (3 ^ (1/3 : ℝ)) = 1 / 3 :=
by
  sorry

end log3_cubicroot_of_3_l129_129262


namespace ways_to_divide_day_l129_129760

theorem ways_to_divide_day (n m : ℕ+) : n * m = 86400 → 96 = 96 :=
by
  sorry

end ways_to_divide_day_l129_129760


namespace correct_phrase_l129_129871

-- Define statements representing each option
def option_A : String := "as twice much"
def option_B : String := "much as twice"
def option_C : String := "twice as much"
def option_D : String := "as much twice"

-- The correct option
def correct_option : String := "twice as much"

-- The main theorem statement
theorem correct_phrase : option_C = correct_option :=
by
  sorry

end correct_phrase_l129_129871


namespace maximum_area_rectangular_backyard_l129_129858

theorem maximum_area_rectangular_backyard (x : ℕ) (y : ℕ) (h_perimeter : 2 * (x + y) = 100) : 
  x * y ≤ 625 :=
by
  sorry

end maximum_area_rectangular_backyard_l129_129858


namespace probability_both_A_B_selected_l129_129293

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129293


namespace find_breadth_l129_129973

theorem find_breadth (p l : ℕ) (h_p : p = 600) (h_l : l = 100) (h_perimeter : p = 2 * (l + b)) : b = 200 :=
by
  sorry

end find_breadth_l129_129973


namespace probability_A_and_B_selected_l129_129419

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129419


namespace solve_congruence_y37_x3_11_l129_129860

theorem solve_congruence_y37_x3_11 (p : ℕ) (hp_pr : Nat.Prime p) (hp_le100 : p ≤ 100) : 
  ∃ (x y : ℕ), y^37 ≡ x^3 + 11 [MOD p] := 
sorry

end solve_congruence_y37_x3_11_l129_129860


namespace total_students_mrs_mcgillicuddy_l129_129126

-- Define the conditions as variables
def students_registered_morning : ℕ := 25
def students_absent_morning : ℕ := 3
def students_registered_afternoon : ℕ := 24
def students_absent_afternoon : ℕ := 4

-- Prove the total number of students present over the two sessions
theorem total_students_mrs_mcgillicuddy : 
  students_registered_morning - students_absent_morning + students_registered_afternoon - students_absent_afternoon = 42 :=
by
  sorry

end total_students_mrs_mcgillicuddy_l129_129126


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129458

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129458


namespace probability_A_B_l129_129354

open Nat

-- Define a function for combination
def combination (n r : Nat) : Nat :=
  factorial n / (factorial r * factorial (n - r))

-- Define the set of students and the selection conditions
def num_students : Nat := 5
def num_selected : Nat := 3

-- Define A and B as selected students
def selected_A_B : Nat := combination 3 1

-- Define the probability calculation
def probability_A_B_selected : Rat :=
  (selected_A_B : Rat) / (combination num_students num_selected : Rat)

-- The statement of the problem
theorem probability_A_B : probability_A_B_selected = 3 / 10 := by
  sorry

end probability_A_B_l129_129354


namespace eval_expr_l129_129622

theorem eval_expr (a b : ℕ) (ha : a = 3) (hb : b = 4) : (a^b)^a - (b^a)^b = -16245775 := by
  sorry

end eval_expr_l129_129622


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129454

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129454


namespace four_cells_same_color_rectangle_l129_129718

theorem four_cells_same_color_rectangle (color : Fin 3 → Fin 7 → Bool) :
  ∃ (r₁ r₂ r₃ r₄ : Fin 3) (c₁ c₂ c₃ c₄ : Fin 7), 
    r₁ ≠ r₂ ∧ r₃ ≠ r₄ ∧ c₁ ≠ c₂ ∧ c₃ ≠ c₄ ∧ 
    r₁ = r₃ ∧ r₂ = r₄ ∧ c₁ = c₃ ∧ c₂ = c₄ ∧
    color r₁ c₁ = color r₁ c₂ ∧ color r₂ c₁ = color r₂ c₂ := sorry

end four_cells_same_color_rectangle_l129_129718


namespace probability_A_and_B_selected_l129_129474

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129474


namespace probability_of_selecting_A_and_B_l129_129438

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129438


namespace probability_of_A_and_B_selected_l129_129486

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129486


namespace bill_sunday_miles_l129_129695

variable (B : ℕ)

-- Conditions
def miles_Bill_Saturday : ℕ := B
def miles_Bill_Sunday : ℕ := B + 4
def miles_Julia_Sunday : ℕ := 2 * (B + 4)
def total_miles : ℕ := miles_Bill_Saturday B + miles_Bill_Sunday B + miles_Julia_Sunday B

theorem bill_sunday_miles (h : total_miles B = 32) : miles_Bill_Sunday B = 9 := by
  sorry

end bill_sunday_miles_l129_129695


namespace smallest_positive_e_for_polynomial_l129_129060

theorem smallest_positive_e_for_polynomial :
  ∃ a b c d e : ℤ, e = 168 ∧
  (a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e = 0) ∧
  (a * (x + 3) * (x - 7) * (x - 8) * (4 * x + 1) = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e) := sorry

end smallest_positive_e_for_polynomial_l129_129060


namespace cos_135_eq_correct_l129_129024

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129024


namespace cube_fraction_inequality_l129_129647

theorem cube_fraction_inequality (s r : ℝ) (h1 : s > r) (h2 : r > 0) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by 
  sorry

end cube_fraction_inequality_l129_129647


namespace factorize_quadratic_l129_129270

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l129_129270


namespace probability_A_B_selected_l129_129425

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129425


namespace cars_difference_proof_l129_129554

theorem cars_difference_proof (U M : ℕ) :
  let initial_cars := 150
  let total_cars := 196
  let cars_from_uncle := U
  let cars_from_grandpa := 2 * U
  let cars_from_dad := 10
  let cars_from_auntie := U + 1
  let cars_from_mum := M
  let total_given_cars := cars_from_dad + cars_from_auntie + cars_from_uncle + cars_from_grandpa + cars_from_mum
  initial_cars + total_given_cars = total_cars ->
  (cars_from_mum - cars_from_dad = 5) := 
by
  sorry

end cars_difference_proof_l129_129554


namespace julie_monthly_salary_l129_129535

def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 6
def missed_days : ℕ := 1
def weeks_per_month : ℕ := 4

theorem julie_monthly_salary :
  (hourly_rate * hours_per_day * (days_per_week - missed_days) * weeks_per_month) = 920 :=
by
  sorry

end julie_monthly_salary_l129_129535


namespace min_expression_value_l129_129541

theorem min_expression_value (a b c : ℝ) (ha : 1 ≤ a) (hbc : b ≥ a) (hcb : c ≥ b) (hc5 : c ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * Real.sqrt (5^(1/4)) + 4 :=
sorry

end min_expression_value_l129_129541


namespace parallel_lines_slope_l129_129885

theorem parallel_lines_slope (d : ℝ) (h : 3 = 4 * d) : d = 3 / 4 :=
by
  sorry

end parallel_lines_slope_l129_129885


namespace probability_of_selecting_A_and_B_l129_129437

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129437


namespace total_courses_attended_l129_129546

-- Define the number of courses attended by Max
def maxCourses : ℕ := 40

-- Define the number of courses attended by Sid (four times as many as Max)
def sidCourses : ℕ := 4 * maxCourses

-- Define the total number of courses attended by both Max and Sid
def totalCourses : ℕ := maxCourses + sidCourses

-- The proof statement
theorem total_courses_attended : totalCourses = 200 := by
  sorry

end total_courses_attended_l129_129546


namespace T_perimeter_l129_129148

theorem T_perimeter (l w : ℝ) (h1 : l = 4) (h2 : w = 2) :
  let rect_perimeter := 2 * l + 2 * w
  let overlap := 2 * w
  2 * rect_perimeter - overlap = 20 :=
by
  -- Proof will be added here
  sorry

end T_perimeter_l129_129148


namespace cookies_per_child_l129_129671

theorem cookies_per_child (total_cookies : ℕ) (adults : ℕ) (children : ℕ) (fraction_eaten_by_adults : ℚ) 
  (h1 : total_cookies = 120) (h2 : adults = 2) (h3 : children = 4) (h4 : fraction_eaten_by_adults = 1/3) :
  total_cookies * (1 - fraction_eaten_by_adults) / children = 20 := 
by
  sorry

end cookies_per_child_l129_129671


namespace tangent_line_at_point_l129_129721

noncomputable theory

open Real

def f (x : ℝ) : ℝ := x * (3 * log x + 1)

def f_deriv (x : ℝ) : ℝ := deriv f x

theorem tangent_line_at_point :
  f 1 = 1 ∧ f_deriv 1 = 4 → ∀ x : ℝ, (1 : ℝ) - 1 = 4 * (x - 1) → (4 : ℝ) * x - 3 = x * (3 * log x + 1) :=
by
  sorry

end tangent_line_at_point_l129_129721


namespace probability_A_and_B_selected_l129_129417

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129417


namespace positive_difference_of_two_numbers_l129_129190

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l129_129190


namespace arithmetic_sum_eight_terms_l129_129776

theorem arithmetic_sum_eight_terms :
  ∀ (a d : ℤ) (n : ℕ), a = -3 → d = 6 → n = 8 → 
  (last_term = a + (n - 1) * d) →
  (last_term = 39) →
  (sum = (n * (a + last_term)) / 2) →
  sum = 144 :=
by
  intros a d n ha hd hn hlast_term hlast_term_value hsum
  sorry

end arithmetic_sum_eight_terms_l129_129776


namespace division_value_of_712_5_by_12_5_is_57_l129_129751

theorem division_value_of_712_5_by_12_5_is_57 : 712.5 / 12.5 = 57 :=
  by
    sorry

end division_value_of_712_5_by_12_5_is_57_l129_129751


namespace ratio_of_60_to_12_l129_129743

theorem ratio_of_60_to_12 : 60 / 12 = 5 := 
by 
  sorry

end ratio_of_60_to_12_l129_129743


namespace impossible_even_sum_l129_129143

theorem impossible_even_sum (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 :=
sorry

end impossible_even_sum_l129_129143


namespace probability_A_and_B_selected_l129_129337

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129337


namespace brenda_peaches_remaining_l129_129243

theorem brenda_peaches_remaining (total_peaches : ℕ) (percent_fresh : ℚ) (thrown_away : ℕ) (fresh_peaches : ℕ) (remaining_peaches : ℕ) :
    total_peaches = 250 → 
    percent_fresh = 0.60 → 
    thrown_away = 15 → 
    fresh_peaches = total_peaches * percent_fresh → 
    remaining_peaches = fresh_peaches - thrown_away → 
    remaining_peaches = 135 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end brenda_peaches_remaining_l129_129243


namespace ellipse_value_l129_129102

noncomputable def a_c_ratio (a c : ℝ) : ℝ :=
  (a + c) / (a - c)

theorem ellipse_value (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2) 
  (h2 : a^2 + b^2 - 3 * c^2 = 0) :
  a_c_ratio a c = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end ellipse_value_l129_129102


namespace simplify_sqrt_of_mixed_number_l129_129790

noncomputable def sqrt_fraction := λ (a b : ℕ), (Real.sqrt a) / (Real.sqrt b)

theorem simplify_sqrt_of_mixed_number : sqrt_fraction 137 16 = (Real.sqrt 137) / 4 := by
  sorry

end simplify_sqrt_of_mixed_number_l129_129790


namespace probability_A_and_B_selected_l129_129402

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129402


namespace probability_A_and_B_selected_l129_129411

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129411


namespace rational_function_value_l129_129866

theorem rational_function_value (g : ℚ → ℚ) (h : ∀ x : ℚ, x ≠ 0 → 4 * g (x⁻¹) + 3 * g x / x = 2 * x^3) : g (-1) = -2 :=
sorry

end rational_function_value_l129_129866


namespace power_mod_residue_l129_129744

theorem power_mod_residue (n : ℕ) (h : n = 1234) : (7^n) % 19 = 9 := by
  sorry

end power_mod_residue_l129_129744


namespace probability_A_and_B_selected_l129_129340

/-- There are 5 students including A and B. The probability of selecting both A and B 
    when 3 students are selected randomly from 5 is 3/10. -/
theorem probability_A_and_B_selected :
  (∃ (students : List String) (h : students.length = 5),
    ∃ (selected : List String) (h' : selected.length = 3),
    "A" ∈ selected ∧ "B" ∈ selected) →
  ∃ (P : ℚ), P = 3 / 10 :=
by
  sorry

end probability_A_and_B_selected_l129_129340


namespace positive_difference_of_two_numbers_l129_129172

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l129_129172


namespace equal_five_digit_number_sets_l129_129593

def five_digit_numbers_not_div_5 : ℕ :=
  9 * 10^3 * 8

def five_digit_numbers_first_two_not_5 : ℕ :=
  8 * 9 * 10^3

theorem equal_five_digit_number_sets :
  five_digit_numbers_not_div_5 = five_digit_numbers_first_two_not_5 :=
by
  repeat { sorry }

end equal_five_digit_number_sets_l129_129593


namespace find_d_l129_129639

theorem find_d (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h1 : a^2 = c * (d + 20)) (h2 : b^2 = c * (d - 18)) : d = 2 :=
by
  sorry

end find_d_l129_129639


namespace max_squares_from_twelve_points_l129_129739

/-- Twelve points are marked on a grid paper. Prove that the maximum number of squares 
that can be formed by connecting four of these points is 11. -/
theorem max_squares_from_twelve_points : ∀ (points : list (ℝ × ℝ)), points.length = 12 → ∃ (squares : set (set (ℝ × ℝ))), squares.card = 11 ∧ ∀ square ∈ squares, ∃ (p₁ p₂ p₃ p₄ : (ℝ × ℝ)), p₁ ∈ points ∧ p₂ ∈ points ∧ p₃ ∈ points ∧ p₄ ∈ points ∧ set.to_finset {p₁, p₂, p₃, p₄}.card = 4 ∧ is_square {p₁, p₂, p₃, p₄} :=
by
  sorry

end max_squares_from_twelve_points_l129_129739


namespace probability_of_A_and_B_selected_l129_129483

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129483


namespace probability_AB_selected_l129_129374

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129374


namespace probability_A_and_B_selected_l129_129413

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129413


namespace multiplication_problem_l129_129532

-- Definitions for different digits A, B, C, D
def is_digit (n : ℕ) := n < 10

theorem multiplication_problem 
  (A B C D : ℕ) 
  (hA : is_digit A) 
  (hB : is_digit B) 
  (hC : is_digit C) 
  (hD : is_digit D) 
  (h_diff : ∀ x y : ℕ, x ≠ y → is_digit x → is_digit y → x ≠ A → y ≠ B → x ≠ C → y ≠ D)
  (hD1 : D = 1)
  (h_mult : A * D = A) 
  (hC_eq : C = A + B) :
  A + C = 5 := sorry

end multiplication_problem_l129_129532


namespace dot_product_a_b_equals_neg5_l129_129086

-- Defining vectors and conditions
structure vector2 := (x : ℝ) (y : ℝ)

def a : vector2 := ⟨2, 1⟩
def b (x : ℝ) : vector2 := ⟨x, -1⟩

-- Collinearity condition
def parallel (v w : vector2) : Prop :=
  v.x * w.y = v.y * w.x

-- Dot product definition
def dot_product (v w : vector2) : ℝ :=
  v.x * w.x + v.y * w.y

-- Given condition
theorem dot_product_a_b_equals_neg5 (x : ℝ) (h : parallel a ⟨a.x - x, a.y - (-1)⟩) : dot_product a (b x) = -5 :=
sorry

end dot_product_a_b_equals_neg5_l129_129086


namespace intersection_points_count_l129_129520

noncomputable def f1 (x : ℝ) : ℝ := abs (3 * x - 2)
noncomputable def f2 (x : ℝ) : ℝ := -abs (2 * x + 5)

theorem intersection_points_count : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f1 x1 = f2 x1 ∧ f1 x2 = f2 x2 ∧ 
    (∀ x : ℝ, f1 x = f2 x → x = x1 ∨ x = x2)) :=
sorry

end intersection_points_count_l129_129520


namespace probability_A_and_B_selected_l129_129472

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129472


namespace positive_difference_of_two_numbers_l129_129163

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129163


namespace relationship_among_abc_l129_129984

noncomputable def a : ℝ := 2 ^ 0.3
def b : ℝ := 0.3 ^ 2
noncomputable def c : ℝ := Real.log 0.3 / Real.log 2

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l129_129984


namespace factorization_identity_l129_129286

theorem factorization_identity (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
  sorry

end factorization_identity_l129_129286


namespace cos_135_eq_neg_inv_sqrt_2_l129_129010

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129010


namespace cos_135_eq_neg_inv_sqrt_2_l129_129014

theorem cos_135_eq_neg_inv_sqrt_2 :
  cos (135 * Real.pi / 180) = - (1 / Real.sqrt 2) :=
by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129014


namespace probability_A_and_B_selected_l129_129481

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129481


namespace ball_color_arrangement_l129_129130

-- Definitions for the conditions
variable (balls_in_red_box balls_in_white_box balls_in_yellow_box : Nat)
variable (red_balls white_balls yellow_balls : Nat)

-- Conditions as assumptions
axiom more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls
axiom different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls
axiom fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box

-- The main theorem to prove
theorem ball_color_arrangement
  (more_balls_in_yellow_box_than_yellow_balls : balls_in_yellow_box > yellow_balls)
  (different_balls_in_red_box_than_white_balls : balls_in_red_box ≠ white_balls)
  (fewer_white_balls_than_balls_in_white_box : white_balls < balls_in_white_box) :
  (balls_in_red_box, balls_in_white_box, balls_in_yellow_box) = (yellow_balls, red_balls, white_balls) :=
sorry

end ball_color_arrangement_l129_129130


namespace prob_select_A_and_B_l129_129310

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129310


namespace city_partition_exists_l129_129878

-- Define a market and street as given
structure City where
  markets : Type
  street : markets → markets → Prop
  leaves_exactly_two : ∀ (m : markets), ∃ (m1 m2 : markets), street m m1 ∧ street m m2

-- Our formal proof statement
theorem city_partition_exists (C : City) : 
  ∃ (partition : C.markets → Fin 1014), 
    (∀ (m1 m2 : C.markets), C.street m1 m2 → partition m1 ≠ partition m2) ∧
    (∀ (d1 d2 : Fin 1014) (m1 m2 : C.markets), (partition m1 = d1) ∧ (partition m2 = d2) → 
     (C.street m1 m2 ∨ C.street m2 m1) →  (∀ (k l : Fin 1014), (k = d1) → (l = d2) → (∀ (a b : C.markets), (partition a = k) → (partition b = l) → (C.street a b ∨ C.street b a)))) :=
sorry

end city_partition_exists_l129_129878


namespace polynomial_remainder_l129_129111

theorem polynomial_remainder (P : Polynomial ℝ) (H1 : P.eval 1 = 2) (H2 : P.eval 2 = 1) :
  ∃ Q : Polynomial ℝ, P = Q * (Polynomial.X - 1) * (Polynomial.X - 2) + (3 - Polynomial.X) :=
by
  sorry

end polynomial_remainder_l129_129111


namespace sqrt_mixed_fraction_l129_129789

theorem sqrt_mixed_fraction (a b : ℤ) (h_a : a = 8) (h_b : b = 9) : 
  (√(a + b / 16)) = (√137) / 4 := 
by 
  sorry

end sqrt_mixed_fraction_l129_129789


namespace prob_select_A_and_B_l129_129303

theorem prob_select_A_and_B (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_card : students.card = 5) :
  (students.choose 3).count {comb | A ∈ comb ∧ B ∈ comb} / (students.choose 3).card = 3 / 10 := sorry

end prob_select_A_and_B_l129_129303


namespace find_sum_on_si_l129_129894

noncomputable def sum_invested_on_si (r1 r2 r3 : ℝ) (years_si: ℕ) (ci_rate: ℝ) (principal_ci: ℝ) (years_ci: ℕ) (times_compounded: ℕ) :=
  let ci_rate_period := ci_rate / times_compounded
  let amount_ci := principal_ci * (1 + ci_rate_period / 1)^(years_ci * times_compounded)
  let ci := amount_ci - principal_ci
  let si := ci / 2
  let total_si_rate := r1 / 100 + r2 / 100 + r3 / 100
  let principle_si := si / total_si_rate
  principle_si

theorem find_sum_on_si :
  sum_invested_on_si 0.05 0.06 0.07 3 0.10 4000 2 2 = 2394.51 :=
by
  sorry

end find_sum_on_si_l129_129894


namespace solve_system_of_equations_l129_129717

theorem solve_system_of_equations :
    ∀ (x y : ℝ), 
    (x^3 * y + x * y^3 = 10) ∧ (x^4 + y^4 = 17) ↔
    (x = 2 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = -1 ∧ y = -2) ∨ (x = -2 ∧ y = -1) :=
by
    sorry

end solve_system_of_equations_l129_129717


namespace cos_135_eq_correct_l129_129018

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129018


namespace matrix_identity_l129_129686

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 4, 3]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  B^4 = -3 • B + 2 • I :=
by
  sorry

end matrix_identity_l129_129686


namespace positive_difference_of_numbers_l129_129199

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end positive_difference_of_numbers_l129_129199


namespace melanie_total_dimes_l129_129121

theorem melanie_total_dimes (d_1 d_2 d_3 : ℕ) (h₁ : d_1 = 19) (h₂ : d_2 = 39) (h₃ : d_3 = 25) : d_1 + d_2 + d_3 = 83 := by
  sorry

end melanie_total_dimes_l129_129121


namespace average_rate_dan_trip_l129_129895

/-- 
Given:
- Dan runs along a 4-mile stretch of river and then swims back along the same route.
- Dan runs at a rate of 10 miles per hour.
- Dan swims at a rate of 6 miles per hour.

Prove:
Dan's average rate for the entire trip is 0.125 miles per minute.
-/
theorem average_rate_dan_trip :
  let distance := 4 -- miles
  let run_rate := 10 -- miles per hour
  let swim_rate := 6 -- miles per hour
  let time_run_hours := distance / run_rate -- hours
  let time_swim_hours := distance / swim_rate -- hours
  let time_run_minutes := time_run_hours * 60 -- minutes
  let time_swim_minutes := time_swim_hours * 60 -- minutes
  let total_distance := distance + distance -- miles
  let total_time := time_run_minutes + time_swim_minutes -- minutes
  let average_rate := total_distance / total_time -- miles per minute
  average_rate = 0.125 :=
by sorry

end average_rate_dan_trip_l129_129895


namespace factorize_quadratic_l129_129275

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l129_129275


namespace cost_of_apples_is_2_l129_129627

variable (A : ℝ)

def cost_of_apples (A : ℝ) : ℝ := 5 * A
def cost_of_sugar (A : ℝ) : ℝ := 3 * (A - 1)
def cost_of_walnuts : ℝ := 0.5 * 6
def total_cost (A : ℝ) : ℝ := cost_of_apples A + cost_of_sugar A + cost_of_walnuts

theorem cost_of_apples_is_2 (A : ℝ) (h : total_cost A = 16) : A = 2 := 
by 
  sorry

end cost_of_apples_is_2_l129_129627


namespace quadrilateral_area_proof_l129_129241

-- Assume we have a rectangle with area 24 cm^2 and two triangles with total area 7.5 cm^2.
-- We want to prove the area of the quadrilateral ABCD is 16.5 cm^2 inside this rectangle.

def rectangle_area : ℝ := 24
def triangles_area : ℝ := 7.5
def quadrilateral_area : ℝ := rectangle_area - triangles_area

theorem quadrilateral_area_proof : quadrilateral_area = 16.5 := 
by
  exact sorry

end quadrilateral_area_proof_l129_129241


namespace interior_diagonals_sum_l129_129602

theorem interior_diagonals_sum (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + c * a) = 112)
  (h2 : 4 * (a + b + c) = 60) : 
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := 
by 
  sorry

end interior_diagonals_sum_l129_129602


namespace geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l129_129794

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ} (hq : 0 < q) (hq2 : q ≠ 1)

-- ① If $a_{1}=1$ and the common ratio is $\frac{1}{2}$, then $S_{n} < 2$;
theorem geom_seq_sum_lt_two (h₁ : a 1 = 1) (hq_half : q = 1 / 2) (n : ℕ) : S n < 2 := sorry

-- ② The sequence $\{a_{n}^{2}\}$ must be a geometric sequence
theorem geom_seq_squared (h_geom : ∀ n, a (n + 1) = q * a n) : ∃ r : ℝ, ∀ n, a n ^ 2 = r ^ n := sorry

-- ④ For any positive integer $n$, $a{}_{n}^{2}+a{}_{n+2}^{2}\geqslant 2a{}_{n+1}^{2}$
theorem geom_seq_square_inequality (h_geom : ∀ n, a (n + 1) = q * a n) (n : ℕ) (hn : 0 < n) : 
  a n ^ 2 + a (n + 2) ^ 2 ≥ 2 * a (n + 1) ^ 2 := sorry

end geom_seq_sum_lt_two_geom_seq_squared_geom_seq_square_inequality_l129_129794


namespace positive_difference_l129_129176

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l129_129176


namespace diff_of_squares_example_l129_129780

theorem diff_of_squares_example : (262^2 - 258^2 = 2080) :=
by
  let a := 262
  let b := 258
  have h1 : a^2 - b^2 = (a + b) * (a - b) := by rw [pow_two, pow_two, sub_mul, add_comm a b, mul_sub]
  have h2 : (a + b) = 520 := by norm_num
  have h3 : (a - b) = 4 := by norm_num
  have h4 : (262 + 258) * (262 - 258) = 520 * 4 := congr (congr_arg (*) h2) h3
  rw [h1, h4]
  norm_num

end diff_of_squares_example_l129_129780


namespace probability_of_selecting_A_and_B_l129_129386

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129386


namespace Kiley_ate_two_slices_l129_129260

-- Definitions of conditions
def calories_per_slice := 350
def total_calories := 2800
def percentage_eaten := 0.25

-- Statement of the problem
theorem Kiley_ate_two_slices (h1 : total_calories = 2800)
                            (h2 : calories_per_slice = 350)
                            (h3 : percentage_eaten = 0.25) :
  (total_calories / calories_per_slice * percentage_eaten) = 2 := 
sorry

end Kiley_ate_two_slices_l129_129260


namespace probability_A_and_B_selected_l129_129395

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129395


namespace probability_both_A_B_selected_l129_129299

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129299


namespace prob_select_A_and_B_l129_129325

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129325


namespace solve_equation_l129_129715

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 := by
  sorry

end solve_equation_l129_129715


namespace tangency_of_abs_and_circle_l129_129828

theorem tangency_of_abs_and_circle (a : ℝ) (ha_pos : a > 0) (ha_ne_two : a ≠ 2) :
    (y = abs x ∧ ∀ x, y = abs x → x^2 + (y - a)^2 = 2 * (a - 2)^2)
    → (a = 4/3 ∨ a = 4) := sorry

end tangency_of_abs_and_circle_l129_129828


namespace cos_135_eq_neg_inv_sqrt_2_l129_129006

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129006


namespace probability_A_and_B_selected_l129_129320

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129320


namespace problem_1_problem_2_l129_129610

theorem problem_1 : ((1 / 3 - 3 / 4 + 5 / 6) / (1 / 12)) = 5 := 
  sorry

theorem problem_2 : ((-1 : ℤ) ^ 2023 + |(1 : ℝ) - 0.5| * (-4 : ℝ) ^ 2) = 7 := 
  sorry

end problem_1_problem_2_l129_129610


namespace minimum_value_of_w_l129_129618

noncomputable def w (x y : ℝ) : ℝ := 3 * x ^ 2 + 3 * y ^ 2 + 9 * x - 6 * y + 27

theorem minimum_value_of_w : (∃ x y : ℝ, w x y = 20.25) := sorry

end minimum_value_of_w_l129_129618


namespace tangent_line_at_A_increasing_intervals_decreasing_interval_l129_129084

noncomputable def f (x : ℝ) := 2 * x^3 + 3 * x^2 + 1

-- Define the derivatives at x
noncomputable def f' (x : ℝ) := 6 * x^2 + 6 * x

-- Define the tangent line equation at a point
noncomputable def tangent_line (x : ℝ) := 12 * x - 6

theorem tangent_line_at_A :
  tangent_line 1 = 6 :=
  by
    -- proof omitted
    sorry

theorem increasing_intervals :
  (∀ x ∈ Set.Ioi 0, f' x > 0) ∧
  (∀ x ∈ Set.Iio (-1), f' x > 0) :=
  by
    -- proof omitted
    sorry

theorem decreasing_interval :
  ∀ x ∈ Set.Ioo (-1) 0, f' x < 0 :=
  by
    -- proof omitted
    sorry

end tangent_line_at_A_increasing_intervals_decreasing_interval_l129_129084


namespace calculate_expression_l129_129778

theorem calculate_expression :
  8^8 + 8^8 + 8^8 + 8^8 + 8^5 = 4 * 8^8 + 8^5 := 
by sorry

end calculate_expression_l129_129778


namespace problem_1_problem_2_problem_3_l129_129897

section basketball_team

-- Definition of conditions
def num_games : ℕ := 6
def prob_win : ℚ := 1 / 3
def prob_loss : ℚ := 2 / 3

-- Problem 1
theorem problem_1 : 
  (prob_loss ^ 2 * prob_win) = 4 / 27 := sorry

-- Problem 2
theorem problem_2 : 
  (nat.choose num_games 3 * (prob_win ^ 3) * (prob_loss ^ 3)) = 160 / 729 := sorry

-- Problem 3
theorem problem_3 : 
  (num_games * prob_win) = 2 := sorry

end basketball_team

end problem_1_problem_2_problem_3_l129_129897


namespace phone_sales_total_amount_l129_129768

theorem phone_sales_total_amount
  (vivienne_phones : ℕ)
  (aliyah_more_phones : ℕ)
  (price_per_phone : ℕ)
  (aliyah_phones : ℕ := vivienne_phones + aliyah_more_phones)
  (total_phones : ℕ := vivienne_phones + aliyah_phones)
  (total_amount : ℕ := total_phones * price_per_phone) :
  vivienne_phones = 40 → aliyah_more_phones = 10 → price_per_phone = 400 → total_amount = 36000 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end phone_sales_total_amount_l129_129768


namespace probability_of_selecting_A_and_B_l129_129464

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129464


namespace proof_solution_l129_129720

noncomputable def proof_problem (x : ℝ) : Prop :=
  (⌈2 * x⌉₊ : ℝ) - (⌊2 * x⌋₊ : ℝ) = 0 → (⌈2 * x⌉₊ : ℝ) - 2 * x = 0

theorem proof_solution (x : ℝ) : proof_problem x :=
by
  sorry

end proof_solution_l129_129720


namespace probability_A_and_B_selected_l129_129394

theorem probability_A_and_B_selected :
  let n := 5
  let r := 3
  let total_ways := Nat.choose n r
  let favorable_ways := Nat.choose 3 1
  let probability := favorable_ways / total_ways
  probability = (3 / 10) := 
by
  sorry

end probability_A_and_B_selected_l129_129394


namespace percent_increase_in_sales_l129_129603

theorem percent_increase_in_sales :
  let new := 416
  let old := 320
  (new - old) / old * 100 = 30 := by
  sorry

end percent_increase_in_sales_l129_129603


namespace triangle_expression_negative_l129_129666

theorem triangle_expression_negative {a b c : ℝ} (habc : a > 0 ∧ b > 0 ∧ c > 0) (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a^2 + b^2 - c^2 - 2 * a * b < 0 :=
sorry

end triangle_expression_negative_l129_129666


namespace tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l129_129955

theorem tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m (m : ℝ) (h : Real.cos (80 * Real.pi / 180) = m) :
    Real.tan (-440 * Real.pi / 180) = - (Real.sqrt (1 - m^2) / m) :=
by
  -- proof goes here
  sorry

end tan_neg440_eq_neg_sqrt_one_minus_m_sq_div_m_l129_129955


namespace sum_of_squares_is_289_l129_129574

theorem sum_of_squares_is_289 (x y : ℤ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 :=
by
  sorry

end sum_of_squares_is_289_l129_129574


namespace projection_matrix_3_4_l129_129289

def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let ⟨vx, vy⟩ := v in
  let denom := vx * vx + vy * vy in
  Matrix.of (λ i j, ([9, 12, 12, 16][(2 * i.val + j.val)] / denom))

theorem projection_matrix_3_4 :
  projection_matrix (3, 4) = Matrix.of (λ i j, [9/25, 12/25, 12/25, 16/25][2 * i.val + j.val]) :=
sorry

end projection_matrix_3_4_l129_129289


namespace cos_135_eq_neg_inv_sqrt_2_l129_129054

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129054


namespace real_roots_iff_integer_roots_iff_l129_129930

noncomputable def discriminant (k : ℝ) : ℝ := (k + 1)^2 - 4 * k * (k - 1)

theorem real_roots_iff (k : ℝ) : 
  (discriminant k ≥ 0) ↔ (∃ (a b : ℝ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) := sorry

theorem integer_roots_iff (k : ℝ) : 
  (∃ (a b : ℤ), kx ^ 2 + (k + 1) * x + (k - 1) = 0) ↔ 
  (k = 0 ∨ k = 1 ∨ k = -1/7) := sorry

-- These theorems need to be proven within Lean 4 itself

end real_roots_iff_integer_roots_iff_l129_129930


namespace tank_capacity_l129_129235

theorem tank_capacity (C : ℝ) :
  (3/4 * C - 0.4 * (3/4 * C) + 0.3 * (3/4 * C - 0.4 * (3/4 * C))) = 4680 → C = 8000 :=
by
  sorry

end tank_capacity_l129_129235


namespace minimize_potato_cost_l129_129626

def potatoes_distribution (x1 x2 x3 : ℚ) : Prop :=
  x1 ≥ 0 ∧ x2 ≥ 0 ∧ x3 ≥ 0 ∧
  x1 + x2 + x3 = 12 ∧
  x1 + 4 * x2 + 3 * x3 ≤ 40 ∧
  x1 ≤ 10 ∧ x2 ≤ 8 ∧ x3 ≤ 6 ∧
  4 * x1 + 3 * x2 + 1 * x3 = (74 / 3)

theorem minimize_potato_cost :
  ∃ x1 x2 x3 : ℚ, potatoes_distribution x1 x2 x3 ∧ x1 = (2/3) ∧ x2 = (16/3) ∧ x3 = 6 :=
by
  sorry

end minimize_potato_cost_l129_129626


namespace find_n_l129_129585

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 25 ∧ (-250 ≡ n [MOD 23]) ∧ n = 3 := by
  use 3
  -- Proof omitted
  sorry

end find_n_l129_129585


namespace positive_difference_l129_129179

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l129_129179


namespace rhombus_diagonal_length_l129_129870

theorem rhombus_diagonal_length (d1 d2 : ℝ) (Area : ℝ) 
  (h1 : d1 = 12) (h2 : Area = 60) 
  (h3 : Area = (d1 * d2) / 2) : d2 = 10 := 
by
  sorry

end rhombus_diagonal_length_l129_129870


namespace prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l129_129706

/-- Let us define the probabilities for school A winning the events -/
def prob_A_wins_event_1 : ℝ := 0.5
def prob_A_wins_event_2 : ℝ := 0.4
def prob_A_wins_event_3 : ℝ := 0.8

/-- The total probability of school A winning the championship -/
noncomputable def prob_A_championship_wins : ℝ :=
  prob_A_wins_event_1 * prob_A_wins_event_2 * prob_A_wins_event_3 +   -- All three events
  (prob_A_wins_event_1 * prob_A_wins_event_2 * (1 - prob_A_wins_event_3) + -- First two events
   prob_A_wins_event_1 * (1 - prob_A_wins_event_2) * prob_A_wins_event_3 + -- First and third event
   (1 - prob_A_wins_event_1) * prob_A_wins_event_2 * prob_A_wins_event_3)  -- Second and third events

/-- The distribution for school B's scores -/
def score_dist_B : List (ℕ × ℝ) :=
  [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)]

/-- The expectation of X (total score of school B) -/
noncomputable def expectation_X : ℝ :=
  0 * 0.16 + 10 * 0.44 + 20 * 0.34 + 30 * 0.06

/-- The proofs for the derived results -/
theorem prob_A_championship_win_is_correct : prob_A_championship_wins = 0.6 := sorry

theorem expectation_X_is_correct : expectation_X = 13 := sorry

theorem distribution_X_is_correct :
  score_dist_B = [(0, 0.16), (10, 0.44), (20, 0.34), (30, 0.06)] := sorry

end prob_A_championship_win_is_correct_expectation_X_is_correct_distribution_X_is_correct_l129_129706


namespace min_w_value_l129_129617

open Real

noncomputable def w (x y : ℝ) : ℝ := 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27

theorem min_w_value : ∃ x y : ℝ, w x y = 81 / 4 :=
by
  use [-3/2, 1]
  dsimp [w]
  norm_num
  done

end min_w_value_l129_129617


namespace probability_both_A_B_selected_l129_129295

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129295


namespace probability_of_selecting_A_and_B_l129_129383

open_locale classical

noncomputable def select_probability (n k : ℕ) : ℚ :=
  (nat.choose n k)⁻¹

theorem probability_of_selecting_A_and_B :
  let total_students := 5,
      students_to_select := 3,
      remaining_students := 3 in
  select_probability total_students students_to_select * nat.choose remaining_students 1 = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_l129_129383


namespace initial_customers_l129_129914

theorem initial_customers (x : ℕ) (h : x - 3 + 39 = 50) : x = 14 :=
by
  sorry

end initial_customers_l129_129914


namespace find_blue_shirts_l129_129798

-- Statements of the problem conditions
def total_shirts : ℕ := 23
def green_shirts : ℕ := 17

-- Definition that we want to prove
def blue_shirts : ℕ := total_shirts - green_shirts

-- Proof statement (no need to include the proof itself)
theorem find_blue_shirts : blue_shirts = 6 := by
  sorry

end find_blue_shirts_l129_129798


namespace average_additional_minutes_per_day_l129_129976

def daily_differences : List ℤ := [20, 5, -5, 0, 15, -10, 10]

theorem average_additional_minutes_per_day :
  (List.sum daily_differences / daily_differences.length) = 5 := by
  sorry

end average_additional_minutes_per_day_l129_129976


namespace probability_of_A_and_B_selected_l129_129487

variable {α : Type}

/-- From 5 students, we randomly select 3 students to participate in community service work. 
  Let A and B be two specific students among these five. The probability that 
  both A and B are selected is 3 / 10. -/
theorem probability_of_A_and_B_selected (A B : α) (s : Finset α) (h : s.card = 5) :
  (∃ t : Finset α, t ⊆ s ∧ A ∈ t ∧ B ∈ t ∧ (t.card = 3) ∧ (t.card = 3) →
   (@Finset.exists_subset_card_eq α t s 3) →
   (@Finset.exists_subset_card_eq α t (Finset.erase s A) 2) ∧
   (@Finset.exists_subset_card_eq α t (Finset.erase (Finset.erase s A) B) 1)) →
  (3 : ℚ) / 10 :=
by
  sorry -- Proof omitted as per instructions.

end probability_of_A_and_B_selected_l129_129487


namespace max_tied_teams_for_most_wins_l129_129100

-- Definitions based on conditions
def num_teams : ℕ := 7
def total_games_played : ℕ := num_teams * (num_teams - 1) / 2

-- Proposition stating the problem and the expected answer
theorem max_tied_teams_for_most_wins : 
  (∀ (t : ℕ), t ≤ num_teams → ∃ w : ℕ, t * w = total_games_played / num_teams) → 
  t = 7 :=
by
  sorry

end max_tied_teams_for_most_wins_l129_129100


namespace units_digit_of_7_power_exp_is_1_l129_129632

-- Define the periodicity of units digits of powers of 7
def units_digit_seq : List ℕ := [7, 9, 3, 1]

-- Define the function to calculate the units digit of 7^n
def units_digit_power_7 (n : ℕ) : ℕ :=
  units_digit_seq.get! (n % 4)

-- Define the exponent
def exp : ℕ := 8^5

-- Define the modular operation result
def exp_modulo : ℕ := exp % 4

-- Define the main statement
theorem units_digit_of_7_power_exp_is_1 :
  units_digit_power_7 exp = 1 :=
by
  simp [units_digit_power_7, units_digit_seq, exp, exp_modulo]
  sorry

end units_digit_of_7_power_exp_is_1_l129_129632


namespace number_of_boys_in_class_l129_129145

theorem number_of_boys_in_class (n : ℕ) (h : 182 * n - 166 + 106 = 180 * n) : n = 30 :=
by {
  sorry
}

end number_of_boys_in_class_l129_129145


namespace rectangle_area_l129_129678

/-- 
In the rectangle \(ABCD\), \(AD - AB = 9\) cm. The area of trapezoid \(ABCE\) is 5 times 
the area of triangle \(ADE\). The perimeter of triangle \(ADE\) is 68 cm less than the 
perimeter of trapezoid \(ABCE\). Prove that the area of the rectangle \(ABCD\) 
is 3060 square centimeters.
-/
theorem rectangle_area (AB AD : ℝ) (S_ABC : ℝ) (S_ADE : ℝ) (P_ADE : ℝ) (P_ABC : ℝ) :
  AD - AB = 9 →
  S_ABC = 5 * S_ADE →
  P_ADE = P_ABC - 68 →
  (AB * AD = 3060) :=
by
  sorry

end rectangle_area_l129_129678


namespace smallest_n_for_convex_100gon_l129_129932

def isConvexPolygon (P : List (Real × Real)) : Prop := sorry -- Assumption for polygon convexity
def canBeIntersectedByTriangles (P : List (Real × Real)) (n : ℕ) : Prop := sorry -- Assumption for intersection by n triangles

theorem smallest_n_for_convex_100gon :
  ∀ (P : List (Real × Real)),
  isConvexPolygon P →
  List.length P = 100 →
  (∀ n, canBeIntersectedByTriangles P n → n ≥ 50) ∧ canBeIntersectedByTriangles P 50 :=
sorry

end smallest_n_for_convex_100gon_l129_129932


namespace area_of_10th_square_l129_129259

noncomputable def area_of_square (n: ℕ) : ℚ :=
  if n = 1 then 4
  else 2 * (1 / 2)^(n - 1)

theorem area_of_10th_square : area_of_square 10 = 1 / 256 := 
  sorry

end area_of_10th_square_l129_129259


namespace find_sum_l129_129805

variable (a b c d : ℝ)

theorem find_sum (h1 : a * b + b * c + c * d + d * a = 48) (h2 : b + d = 6) : a + c = 8 :=
sorry

end find_sum_l129_129805


namespace max_dist_2_minus_2i_l129_129638

open Complex

noncomputable def max_dist (z1 : ℂ) : ℝ :=
  Complex.abs 1 + Complex.abs z1

theorem max_dist_2_minus_2i :
  max_dist (2 - 2*I) = 1 + 2 * Real.sqrt 2 := by
  sorry

end max_dist_2_minus_2i_l129_129638


namespace geom_sequence_common_ratio_l129_129528

variable {α : Type*} [LinearOrderedField α]

theorem geom_sequence_common_ratio (a1 q : α) (h : a1 > 0) (h_eq : a1 + a1 * q + a1 * q^2 + a1 * q = 9 * a1 * q^2) : q = 1 / 2 :=
by sorry

end geom_sequence_common_ratio_l129_129528


namespace numberOfCorrectRelations_l129_129749

open ProbabilityTheory Set

def eventA : Set (Set String) := {{"HH"}}
def eventB : Set (Set String) := {{"HH", "TT"}}

theorem numberOfCorrectRelations : ∃ n : Nat, n = 1 ∧
  (eventA ⊆ eventB ∧
   ¬ (P(eventA) * P(eventB) = P(eventA ∩ eventB))) :=
by
  exists 1
  sorry

end numberOfCorrectRelations_l129_129749


namespace scale_length_l129_129232

theorem scale_length (length_of_part : ℕ) (number_of_parts : ℕ) (h1 : number_of_parts = 2) (h2 : length_of_part = 40) :
  number_of_parts * length_of_part = 80 := 
by
  sorry

end scale_length_l129_129232


namespace solution_to_inequality_system_l129_129160

theorem solution_to_inequality_system (x : ℝ) :
  (x + 3 ≥ 2 ∧ (3 * x - 1) / 2 < 4) ↔ -1 ≤ x ∧ x < 3 :=
by
  sorry

end solution_to_inequality_system_l129_129160


namespace percentage_increase_each_year_is_50_l129_129734

-- Definitions based on conditions
def students_passed_three_years_ago : ℕ := 200
def students_passed_this_year : ℕ := 675

-- The prove statement
theorem percentage_increase_each_year_is_50
    (N3 N0 : ℕ)
    (P : ℚ)
    (h1 : N3 = students_passed_three_years_ago)
    (h2 : N0 = students_passed_this_year)
    (h3 : N0 = N3 * (1 + P)^3) :
  P = 0.5 :=
by
  sorry

end percentage_increase_each_year_is_50_l129_129734


namespace blue_tshirt_count_per_pack_l129_129252

theorem blue_tshirt_count_per_pack :
  ∀ (total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack : ℕ), 
    white_packs = 3 →
    blue_packs = 2 → 
    tshirts_per_white_pack = 6 → 
    total_tshirts = 26 →
    total_tshirts = white_packs * tshirts_per_white_pack + blue_packs * tshirts_per_blue_pack →
  tshirts_per_blue_pack = 4 :=
by
  intros total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack
  intros h1 h2 h3 h4 h5
  sorry

end blue_tshirt_count_per_pack_l129_129252


namespace disproved_option_a_disproved_option_b_disproved_option_c_proved_option_d_l129_129587

theorem disproved_option_a : ¬ ∀ m n : ℤ, abs m = abs n → m = n := 
by 
  intro h
  have h1 : abs (-3) = abs 3 := by norm_num
  have h2 : h (-3) 3 h1
  contradiction

theorem disproved_option_b : ¬ ∀ m n : ℤ, m > n → abs m > abs n := 
by 
  intro h
  have h1 : 1 > -3 := by norm_num
  have h2 : abs 1 = 1 := by norm_num
  have h3 : abs (-3) = 3 := by norm_num
  have h4 : h 1 (-3) h1
  contradiction

theorem disproved_option_c : ¬ ∀ m n : ℤ, abs m > abs n → m > n :=
by 
  intro h
  have h1 : abs (-3) > abs 1 := by norm_num
  have h2 : (-3) < 1 := by norm_num
  exact not_lt_of_gt h2 (h (-3) 1 h1)

theorem proved_option_d : ∀ m n : ℤ, m < n → n < 0 → abs m > abs n := 
by 
  intro m n hmn hn0
  simp only [abs_lt, int.lt_neg, int.abs]
  have h1 : -n < -m := by exact neg_lt_neg hmn
  exact_mod_cast h1

end disproved_option_a_disproved_option_b_disproved_option_c_proved_option_d_l129_129587


namespace cos_135_eq_correct_l129_129019

noncomputable def cos_135_eq: Prop :=
  real.cos ((135 * real.pi) / 180) = -real.sqrt 2 / 2

theorem cos_135_eq_correct : cos_135_eq :=
  sorry

end cos_135_eq_correct_l129_129019


namespace probability_both_A_B_selected_l129_129300

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end probability_both_A_B_selected_l129_129300


namespace greatest_number_zero_l129_129793

-- Define the condition (inequality)
def inequality (x : ℤ) : Prop :=
  3 * x + 2 < 5 - 2 * x

-- Define the property of being the greatest whole number satisfying the inequality
def greatest_whole_number (x : ℤ) : Prop :=
  inequality x ∧ (∀ y : ℤ, inequality y → y ≤ x)

-- The main theorem stating the greatest whole number satisfying the inequality is 0
theorem greatest_number_zero : greatest_whole_number 0 :=
by
  sorry

end greatest_number_zero_l129_129793


namespace find_price_max_profit_l129_129900

/-
Part 1: Prove the price per unit of type A and B
-/

def price_per_unit (x y : ℕ) : Prop :=
  (2 * x + 3 * y = 690) ∧ (x + 4 * y = 720)

theorem find_price :
  ∃ x y : ℕ, price_per_unit x y ∧ x = 120 ∧ y = 150 :=
by
  sorry

/-
Part 2: Prove the maximum profit with constraints
-/

def conditions (m : ℕ) : Prop :=
  m ≤ 3 * (40 - m) ∧ 120 * m + 150 * (40 - m) ≤ 5400

def profit (m : ℕ) : ℕ :=
  (160 - 120) * m + (200 - 150) * (40 - m)

theorem max_profit :
  ∃ m : ℕ, 20 ≤ m ∧ m ≤ 30 ∧ conditions m ∧ profit m = profit 20 :=
by
  sorry

end find_price_max_profit_l129_129900


namespace cos_135_eq_neg_sqrt2_div_2_l129_129035

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l129_129035


namespace Maryann_total_minutes_worked_l129_129851

theorem Maryann_total_minutes_worked (c a t : ℕ) (h1 : c = 70) (h2 : a = 7 * c) (h3 : t = c + a) : t = 560 := by
  sorry

end Maryann_total_minutes_worked_l129_129851


namespace factorize_quadratic_l129_129274

theorem factorize_quadratic (x : ℝ) : 
  16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 :=
sorry

end factorize_quadratic_l129_129274


namespace probability_A_and_B_selected_l129_129473

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129473


namespace eq_iff_squared_eq_l129_129676

theorem eq_iff_squared_eq (a b : ℝ) : a = b ↔ a^2 + b^2 = 2 * a * b :=
by
  sorry

end eq_iff_squared_eq_l129_129676


namespace sum_of_solutions_of_quadratic_l129_129777

theorem sum_of_solutions_of_quadratic :
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  roots_sum = 3 / 2 :=
by
  let a := 18
  let b := -27
  let c := -45
  let roots_sum := (-b / a : ℚ)
  have h1 : roots_sum = 3 / 2 := by sorry
  exact h1

end sum_of_solutions_of_quadratic_l129_129777


namespace probability_A_and_B_selected_l129_129420

theorem probability_A_and_B_selected (students : Finset ℕ) (A B : ℕ) (h : A ∈ students ∧ B ∈ students ∧ students.card = 5) : 
  (students.choose 3).count (λ s, {A, B} ⊆ s) / (students.choose 3).card = 3 / 10 := by
  sorry

end probability_A_and_B_selected_l129_129420


namespace triangle_inequality_circumradius_l129_129505

theorem triangle_inequality_circumradius (a b c R : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) 
  (circumradius_def : R = (a * b * c) / (4 * (Real.sqrt ((a + b + c) * (a + b - c) * (a - b + c) * (-a + b + c))))) :
  (1 / (a * b)) + (1 / (b * c)) + (1 / (c * a)) ≥ (1 / (R ^ 2)) :=
sorry

end triangle_inequality_circumradius_l129_129505


namespace select_3_from_5_prob_l129_129345

theorem select_3_from_5_prob {students : Finset ℕ} 
  (h : students.card = 5) 
  {A B : ℕ} 
  (hA : A ∈ students) 
  (hB : B ∈ students) 
  (H : 3 ≤ students.card) : 
  (finset.card (students.filter (λ x, x ≠ A ∧ x ≠ B)) = 3) → 
  (finset.card ((students.filter (λ x, x ≠ A ∧ x ≠ B)).subtype (λ x, True)) = 3) → 
  ((finset.card (finset.filter (λ (students : Finset ℕ) A ∈ students ∧ B ∈ students ↔ 3 = students.card) 5)) = ↑10) → 
  ∃ students', (students'.card = 3 ∧ A ∈ students' ∧ B ∈ students') → 
  ∃ (p : ℚ), p = 3 / 10 ∧ students.filter (λ x, x = A ∨ x = B) = 2 :=
  sorry

end select_3_from_5_prob_l129_129345


namespace probability_of_selecting_A_and_B_l129_129462

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129462


namespace probability_of_selecting_A_and_B_l129_129468

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129468


namespace complement_union_example_l129_129688

open Set

variable (I : Set ℕ) (A : Set ℕ) (B : Set ℕ)

noncomputable def complement (U : Set ℕ) (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem complement_union_example
    (hI : I = {0, 1, 2, 3, 4})
    (hA : A = {0, 1, 2, 3})
    (hB : B = {2, 3, 4}) :
    (complement I A) ∪ (complement I B) = {0, 1, 4} := by
  sorry

end complement_union_example_l129_129688


namespace probability_A_and_B_selected_l129_129478

theorem probability_A_and_B_selected 
  (students : Finset ℕ) (A B : ℕ) 
  (h_students_size : students.card = 5) 
  (h_A_in_students : A ∈ students) 
  (h_B_in_students : B ∈ students)
  : (↑(Nat.choose 3) : ℚ) / (↑(Nat.choose 5) : ℚ) = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129478


namespace total_value_of_goods_l129_129236

theorem total_value_of_goods (V : ℝ) (tax_paid : ℝ) (tax_exemption : ℝ) (tax_rate : ℝ) :
  tax_exemption = 600 → tax_rate = 0.11 → tax_paid = 123.2 → 0.11 * (V - 600) = tax_paid → V = 1720 :=
by
  sorry

end total_value_of_goods_l129_129236


namespace probability_of_selecting_A_and_B_l129_129433

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129433


namespace kevin_expected_away_time_l129_129983

theorem kevin_expected_away_time
  (leak_rate : ℝ)
  (bucket_capacity : ℝ)
  (bucket_factor : ℝ)
  (leak_rate_eq : leak_rate = 1.5)
  (bucket_capacity_eq : bucket_capacity = 36)
  (bucket_factor_eq : bucket_factor = 2)
  : ((bucket_capacity / bucket_factor) / leak_rate) = 12 :=
by
  rw [bucket_capacity_eq, leak_rate_eq, bucket_factor_eq]
  sorry

end kevin_expected_away_time_l129_129983


namespace probability_of_selecting_A_and_B_l129_129435

theorem probability_of_selecting_A_and_B (students : Finset ℕ) (A B : ℕ)
  (h_size : students.card = 5) (h_in : A ∈ students ∧ B ∈ students) : 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3 ∧ A ∈ s ∧ B ∈ s}.to_set) / 
  (Finset.card {s : Finset ℕ // s ⊆ students ∧ s.card = 3}.to_set) = 3 / 10 :=
sorry

end probability_of_selecting_A_and_B_l129_129435


namespace cos_135_eq_neg_inv_sqrt_2_l129_129049

theorem cos_135_eq_neg_inv_sqrt_2 : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129049


namespace binomial_sum_alternating_l129_129787

theorem binomial_sum_alternating :
  (finset.range 51).sum (λ k, (-1 : ℤ) ^ k * (k + 1) * (nat.choose 50 k)) = 0 :=
begin
  sorry
end

end binomial_sum_alternating_l129_129787


namespace positive_difference_l129_129178

theorem positive_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end positive_difference_l129_129178


namespace valentines_left_l129_129994

theorem valentines_left (initial valentines_to_children valentines_to_neighbors valentines_to_coworkers : ℕ) (h_initial : initial = 30)
  (h1 : valentines_to_children = 8) (h2 : valentines_to_neighbors = 5) (h3 : valentines_to_coworkers = 3) : initial - (valentines_to_children + valentines_to_neighbors + valentines_to_coworkers) = 14 := by
  sorry

end valentines_left_l129_129994


namespace identity_function_l129_129928

theorem identity_function {f : ℕ → ℕ} (h : ∀ a b : ℕ, 0 < a → 0 < b → a - f b ∣ a * f a - b * f b) :
  ∀ a : ℕ, 0 < a → f a = a :=
by
  sorry

end identity_function_l129_129928


namespace probability_A_and_B_selected_l129_129314

/-- Let there be a group of 5 students including A and B. 
    If 3 students are selected randomly from these 5, the probability of selecting both A and B is 3/10. -/
theorem probability_A_and_B_selected : 
  let students := 5 in
  let selected := 3 in
  let combinations := students.choose selected in
  let favorable_combinations := (students - 2).choose (selected - 2) in
  let probability := (favorable_combinations : ℚ) / (combinations : ℚ) in
  probability = 3 / 10 := 
by sorry

end probability_A_and_B_selected_l129_129314


namespace basketball_free_throws_l129_129573

/-
Given the following conditions:
1. The players scored twice as many points with three-point shots as with two-point shots: \( 3b = 2a \).
2. The number of successful free throws was one more than the number of successful two-point shots: \( x = a + 1 \).
3. The team’s total score was 84 points: \( 2a + 3b + x = 84 \).

Prove that the number of free throws \( x \) equals 16.
-/
theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a) 
  (h2 : x = a + 1) 
  (h3 : 2 * a + 3 * b + x = 84) : 
  x = 16 := 
  sorry

end basketball_free_throws_l129_129573


namespace calc_fraction_l129_129919
-- Import necessary libraries

-- Define the necessary fractions and the given expression
def expr := (5 / 6) * (1 / (7 / 8 - 3 / 4))

-- State the theorem
theorem calc_fraction : expr = 20 / 3 := 
by
  sorry

end calc_fraction_l129_129919


namespace distinct_prime_factors_count_l129_129256

theorem distinct_prime_factors_count :
  ∀ (a b c d : ℕ),
  (a = 79) → (b = 3^4) → (c = 5 * 17) → (d = 3 * 29) →
  (∃ s : Finset ℕ, ∀ n ∈ s, Nat.Prime n ∧ 79 * 81 * 85 * 87 = s.prod id) :=
sorry

end distinct_prime_factors_count_l129_129256


namespace probability_both_A_and_B_selected_l129_129442

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129442


namespace parabola_directrix_value_l129_129822

noncomputable def parabola_p_value (p : ℝ) : Prop :=
(∀ y : ℝ, y^2 = 2 * p * (-2 - (-2)))

theorem parabola_directrix_value : parabola_p_value 4 :=
by
  -- proof steps here
  sorry

end parabola_directrix_value_l129_129822


namespace positive_difference_l129_129206

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l129_129206


namespace andre_flowers_given_l129_129561

variable (initialFlowers totalFlowers flowersGiven : ℕ)

theorem andre_flowers_given (h1 : initialFlowers = 67) (h2 : totalFlowers = 90) :
  flowersGiven = totalFlowers - initialFlowers → flowersGiven = 23 :=
by
  intro h3
  rw [h1, h2] at h3
  simp at h3
  exact h3

end andre_flowers_given_l129_129561


namespace evaluate_expression_l129_129782

def g (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 7

theorem evaluate_expression : 3 * g 2 + 4 * g (-2) = 113 := by
  sorry

end evaluate_expression_l129_129782


namespace probability_A_B_selected_l129_129424

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129424


namespace probability_of_A_not_losing_l129_129855

/-- The probability of player A winning is 0.3,
    and the probability of a draw between player A and player B is 0.4.
    Hence, the probability of player A not losing is 0.7. -/
theorem probability_of_A_not_losing (pA_win p_draw : ℝ) (hA_win : pA_win = 0.3) (h_draw : p_draw = 0.4) : 
  (pA_win + p_draw = 0.7) :=
by
  sorry

end probability_of_A_not_losing_l129_129855


namespace positive_difference_of_two_numbers_l129_129187

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end positive_difference_of_two_numbers_l129_129187


namespace trapezium_other_parallel_side_l129_129630

theorem trapezium_other_parallel_side (a : ℝ) (b d : ℝ) (area : ℝ) 
  (h1 : a = 18) (h2 : d = 15) (h3 : area = 285) : b = 20 :=
by
  sorry

end trapezium_other_parallel_side_l129_129630


namespace positive_difference_l129_129207

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l129_129207


namespace min_value_abc_l129_129696

theorem min_value_abc : 
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a^b % 10 = 4) ∧ (b^c % 10 = 2) ∧ (c^a % 10 = 9) ∧ 
    (a + b + c = 17) :=
  by {
    sorry
  }

end min_value_abc_l129_129696


namespace class_percentage_of_girls_l129_129576

/-
Given:
- Initial number of boys in the class: 11
- Number of girls in the class: 13
- 1 boy is added to the class, resulting in the new total number of boys being 12

Prove:
- The percentage of the class that are girls is 52%.
-/
theorem class_percentage_of_girls (initial_boys : ℕ) (girls : ℕ) (added_boy : ℕ)
  (new_boy_total : ℕ) (total_students : ℕ) (percent_girls : ℕ) (h1 : initial_boys = 11) 
  (h2 : girls = 13) (h3 : added_boy = 1) (h4 : new_boy_total = initial_boys + added_boy) 
  (h5 : total_students = new_boy_total + girls) 
  (h6 : percent_girls = (girls * 100) / total_students) : percent_girls = 52 :=
sorry

end class_percentage_of_girls_l129_129576


namespace pencils_multiple_of_30_l129_129873

-- Defines the conditions of the problem
def num_pens : ℕ := 2010
def max_students : ℕ := 30
def equal_pens_per_student := num_pens % max_students = 0

-- Proves that the number of pencils must be a multiple of 30
theorem pencils_multiple_of_30 (P : ℕ) (h1 : equal_pens_per_student) (h2 : ∀ n, n ≤ max_students → ∃ m, n * m = num_pens) : ∃ k : ℕ, P = max_students * k :=
sorry

end pencils_multiple_of_30_l129_129873


namespace book_page_count_l129_129926

def total_pages_in_book (pages_three_nights_ago pages_two_nights_ago pages_last_night pages_tonight total_pages : ℕ) : Prop :=
  pages_three_nights_ago = 15 ∧
  pages_two_nights_ago = 2 * pages_three_nights_ago ∧
  pages_last_night = pages_two_nights_ago + 5 ∧
  pages_tonight = 20 ∧
  total_pages = pages_three_nights_ago + pages_two_nights_ago + pages_last_night + pages_tonight

theorem book_page_count : total_pages_in_book 15 30 35 20 100 :=
by {
  sorry
}

end book_page_count_l129_129926


namespace probability_A_and_B_selected_l129_129404

theorem probability_A_and_B_selected (A B : Type) (S : Finset (Type)) (h : S.card = 5) (hA : A ∈ S) (hB : B ∈ S) (T : Finset (Type)) (hT : T.card = 3) :
  (∃ T ⊆ S, A ∈ T ∧ B ∈ T ∧ T.card = 3) → 
  ∃ T ⊆ S, T.card = 3 ∧ (finset.card (T ∩ {A, B}) = 2) → 
  (finset.card (T ∩ {A, B}) = 2) = (3 / 10 : Real) := sorry

end probability_A_and_B_selected_l129_129404


namespace binom_26_6_l129_129079

theorem binom_26_6 (h₁ : Nat.choose 25 5 = 53130) (h₂ : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 :=
by
  sorry

end binom_26_6_l129_129079


namespace exchange_candies_l129_129553

-- Define the problem conditions and calculate the required values
def chocolates := 7
def caramels := 9
def exchange := 5

-- Combinatorial function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem exchange_candies (h1 : chocolates = 7) (h2 : caramels = 9) (h3 : exchange = 5) :
  binomial chocolates exchange * binomial caramels exchange = 2646 := by
  sorry

end exchange_candies_l129_129553


namespace range_of_s_l129_129742

noncomputable def s (x : ℝ) := 1 / (2 + x)^3

theorem range_of_s :
  Set.range s = {y : ℝ | y < 0} ∪ {y : ℝ | y > 0} :=
by
  sorry

end range_of_s_l129_129742


namespace Rohan_earning_after_6_months_l129_129133

def farm_area : ℕ := 20
def trees_per_sqm : ℕ := 2
def coconuts_per_tree : ℕ := 6
def harvest_interval : ℕ := 3
def sale_price : ℝ := 0.50
def total_months : ℕ := 6

theorem Rohan_earning_after_6_months :
  farm_area * trees_per_sqm * coconuts_per_tree * (total_months / harvest_interval) * sale_price 
    = 240 := by
  sorry

end Rohan_earning_after_6_months_l129_129133


namespace smallest_period_sin_cos_l129_129157

theorem smallest_period_sin_cos (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x + Real.cos x) :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = 2 * Real.pi :=
sorry

end smallest_period_sin_cos_l129_129157


namespace solveEquation_l129_129139

theorem solveEquation (x : ℝ) (hx : |x| ≥ 3) : (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (x₁ / 3 + x₁ / Real.sqrt (x₁ ^ 2 - 9) = 35 / 12) ∧ (x₂ / 3 + x₂ / Real.sqrt (x₂ ^ 2 - 9) = 35 / 12)) ∧ x₁ + x₂ = 8.75) :=
sorry

end solveEquation_l129_129139


namespace positive_difference_of_two_numbers_l129_129171

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l129_129171


namespace ribbon_problem_l129_129239

variable (Ribbon1 Ribbon2 : ℕ)
variable (L : ℕ)

theorem ribbon_problem
    (h1 : Ribbon1 = 8)
    (h2 : ∀ L, L > 0 → Ribbon1 % L = 0 → Ribbon2 % L = 0)
    (h3 : ∀ k, (k > 0 ∧ Ribbon1 % k = 0 ∧ Ribbon2 % k = 0) → k ≤ 8) :
    Ribbon2 = 8 := by
  sorry

end ribbon_problem_l129_129239


namespace smallest_value_of_N_l129_129099

theorem smallest_value_of_N :
  ∃ N : ℕ, ∀ (P1 P2 P3 P4 P5 : ℕ) (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : ℕ),
    (P1 = 1 ∧ P2 = 2 ∧ P3 = 3 ∧ P4 = 4 ∧ P5 = 5) →
    (x1 = a_1 ∧ x2 = N + a_2 ∧ x3 = 2 * N + a_3 ∧ x4 = 3 * N + a_4 ∧ x5 = 4 * N + a_5) →
    (y1 = 5 * (a_1 - 1) + 1 ∧ y2 = 5 * (a_2 - 1) + 2 ∧ y3 = 5 * (a_3 - 1) + 3 ∧ y4 = 5 * (a_4 - 1) + 4 ∧ y5 = 5 * (a_5 - 1) + 5) →
    (x1 = y2 ∧ x2 = y1 ∧ x3 = y4 ∧ x4 = y5 ∧ x5 = y3) →
    N = 149 :=
sorry

end smallest_value_of_N_l129_129099


namespace farmer_loss_representative_value_l129_129227

def check_within_loss_range (S L : ℝ) : Prop :=
  (S = 100000) → (20000 ≤ L ∧ L ≤ 25000)

theorem farmer_loss_representative_value : check_within_loss_range 100000 21987.53 :=
by
  intros hs
  sorry

end farmer_loss_representative_value_l129_129227


namespace eggs_in_each_basket_is_15_l129_129118
open Nat

theorem eggs_in_each_basket_is_15 :
  ∃ n : ℕ, (n ∣ 30) ∧ (n ∣ 45) ∧ (n ≥ 5) ∧ (n = 15) :=
sorry

end eggs_in_each_basket_is_15_l129_129118


namespace maria_younger_than_ann_l129_129119

variable (M A : ℕ)

def maria_current_age : Prop := M = 7

def age_relation_four_years_ago : Prop := M - 4 = (1 / 2) * (A - 4)

theorem maria_younger_than_ann :
  maria_current_age M → age_relation_four_years_ago M A → A - M = 3 :=
by
  sorry

end maria_younger_than_ann_l129_129119


namespace find_angle_C_l129_129941

-- Given conditions
variable {A B C : ℝ}
variable (h_triangle : A + B + C = π)
variable (h_tanA : Real.tan A = 1/2)
variable (h_cosB : Real.cos B = 3 * Real.sqrt 10 / 10)

-- The proof statement
theorem find_angle_C :
  C = 3 * π / 4 := by
  sorry

end find_angle_C_l129_129941


namespace joan_original_seashells_l129_129683

-- Definitions based on the conditions
def seashells_left : ℕ := 27
def seashells_given_away : ℕ := 43

-- Theorem statement
theorem joan_original_seashells : 
  seashells_left + seashells_given_away = 70 := 
by
  sorry

end joan_original_seashells_l129_129683


namespace students_in_classroom_l129_129733

/-- There are some students in a classroom. Half of them have 5 notebooks each and the other half have 3 notebooks each. There are 112 notebooks in total in the classroom. Prove the number of students is 28. -/
theorem students_in_classroom (S : ℕ) (h1 : (S / 2) * 5 + (S / 2) * 3 = 112) : S = 28 := 
sorry

end students_in_classroom_l129_129733


namespace reaction_completion_l129_129931

-- Definitions from conditions
def NaOH_moles : ℕ := 2
def H2O_moles : ℕ := 2

-- Given the balanced equation
-- 2 NaOH + H2SO4 → Na2SO4 + 2 H2O

theorem reaction_completion (H2SO4_moles : ℕ) :
  (2 * (NaOH_moles / 2)) = H2O_moles → H2SO4_moles = 1 :=
by 
  -- Skip proof
  sorry

end reaction_completion_l129_129931


namespace setC_not_pythagorean_l129_129772

/-- Defining sets of numbers as options -/
def SetA := (3, 4, 5)
def SetB := (5, 12, 13)
def SetC := (7, 25, 26)
def SetD := (6, 8, 10)

/-- Function to check if a set is a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem stating set C is not a Pythagorean triple -/
theorem setC_not_pythagorean :
  ¬isPythagoreanTriple 7 25 26 :=
by {
  -- This slot will be filled with the concrete proof steps in Lean.
  sorry
}

end setC_not_pythagorean_l129_129772


namespace part1_part2_l129_129986

noncomputable def f (a x : ℝ) := Real.exp (2 * x) - 4 * a * Real.exp x - 2 * a * x
noncomputable def g (a x : ℝ) := x^2 + 5 * a^2
noncomputable def F (a x : ℝ) := f a x + g a x

theorem part1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ a ≤ 0 :=
by sorry

theorem part2 (a : ℝ) : ∀ x : ℝ, F a x ≥ 4 * (1 - Real.log 2)^2 / 5 :=
by sorry

end part1_part2_l129_129986


namespace sin_double_angle_neg_l129_129936

variable (α : Real)
variable (h1 : Real.tan α < 0)
variable (h2 : Real.sin α = -Real.sqrt 3 / 3)

theorem sin_double_angle_neg (h1 : Real.tan α < 0) (h2 : Real.sin α = -Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 * Real.sqrt 2 / 3 := 
by 
  sorry

end sin_double_angle_neg_l129_129936


namespace cubic_vs_square_ratio_l129_129643

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end cubic_vs_square_ratio_l129_129643


namespace no_real_roots_f_of_f_x_eq_x_l129_129659

theorem no_real_roots_f_of_f_x_eq_x (a b c : ℝ) (h: (b - 1)^2 - 4 * a * c < 0) : 
  ¬(∃ x : ℝ, (a * (a * x^2 + b * x + c)^2 + b * (a * x^2 + b * x + c) + c = x)) := 
by
  sorry

end no_real_roots_f_of_f_x_eq_x_l129_129659


namespace g_of_f_of_3_is_217_l129_129110

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2 - 4
def g (x : ℝ) : ℝ := x^3 + 3 * x^2 + 3 * x + 2

-- The theorem we need to prove
theorem g_of_f_of_3_is_217 : g (f 3) = 217 := by
  sorry

end g_of_f_of_3_is_217_l129_129110


namespace prob_select_A_and_B_l129_129329

theorem prob_select_A_and_B : 
  let total_students := 5
  let to_select := 3
  let favorable_outcomes := Nat.div_fact (Nat.factorial 3) (Nat.factorial 1)
  let total_outcomes := Nat.div_fact (Nat.factorial 5) (Nat.factorial 3 * Nat.factorial (5 - 3))
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  sorry

end prob_select_A_and_B_l129_129329


namespace probability_of_selecting_A_and_B_is_three_tenths_l129_129459

-- Define the set of students
def students : Finset ℕ := {0, 1, 2, 3, 4}

-- Define subsets A and B
def student_A : ℕ := 0
def student_B : ℕ := 1

-- Total number of ways to select 3 students out of 5
def total_ways : ℕ := students.card.choose 3

-- Number of ways to select 3 students including A and B
def favorable_ways : ℕ := {2, 3, 4}.card.choose 1

-- The probability
def probability_of_selecting_A_and_B : ℚ := (favorable_ways : ℚ) / (total_ways : ℚ)

theorem probability_of_selecting_A_and_B_is_three_tenths :
  probability_of_selecting_A_and_B = 3 / 10 :=
by
  sorry

end probability_of_selecting_A_and_B_is_three_tenths_l129_129459


namespace positive_difference_of_two_numbers_l129_129189

theorem positive_difference_of_two_numbers 
  (a b : ℝ) 
  (h₁ : a + b = 10) 
  (h₂ : a^2 - b^2 = 40) : 
  |a - b| = 4 := 
sorry

end positive_difference_of_two_numbers_l129_129189


namespace cos_135_eq_neg_inv_sqrt_2_l129_129001

theorem cos_135_eq_neg_inv_sqrt_2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 :=
sorry

end cos_135_eq_neg_inv_sqrt_2_l129_129001


namespace ball_distribution_l129_129128

theorem ball_distribution: 
  let n := 18 in
  let k := 5 in
  -- placing at least 3 balls in each of 5 boxes
  ∃ ways, ways = Nat.choose (n - 3 * k + k - 1) (k - 1) ∧ ways = 35 :=
begin
  let n := 18,
  let k := 5,
  -- calculating the remaining balls
  let remaining_balls := n - 3 * k,
  -- using the stars and bars method
  let ways := Nat.choose (remaining_balls + k - 1) (k - 1),
  use ways,
  split,
  {
    refl,
  },
  {
    -- calculation of the binomial coefficient
    have h : remaining_balls = 3 := rfl, -- 18 - 15 = 3
    rw h,
    have h' : Nat.choose (7) (4) = 35 := by rnorm,
    rw h',
  },
end

end ball_distribution_l129_129128


namespace probability_A_B_selected_l129_129429

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129429


namespace gavin_shirts_l129_129797

theorem gavin_shirts (t g b : ℕ) (h_total : t = 23) (h_green : g = 17) (h_blue : b = t - g) : b = 6 :=
by sorry

end gavin_shirts_l129_129797


namespace max_value_m_l129_129142

noncomputable def exists_triangle_with_sides (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem max_value_m (a b c : ℝ) (m : ℝ) (h1 : 0 < m) (h2 : abc ≤ 1/4) (h3 : 1/(a^2) + 1/(b^2) + 1/(c^2) < m) :
  m ≤ 9 ↔ exists_triangle_with_sides a b c :=
sorry

end max_value_m_l129_129142


namespace rico_more_dogs_than_justin_l129_129920

theorem rico_more_dogs_than_justin 
  (justin_dogs : ℕ := 14) 
  (camden_legs : ℕ := 72) 
  (camden_ratio : ℚ := 3/4) :
  let camden_dogs := camden_legs / 4 in
  let rico_dogs := camden_dogs * (4/3) in
  rico_dogs - justin_dogs = 10 := 
by
  sorry

end rico_more_dogs_than_justin_l129_129920


namespace probability_A_B_selected_l129_129430

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def combination (n r : ℕ) : ℕ :=
  factorial n / (factorial r * factorial (n - r))

theorem probability_A_B_selected :
  let total_ways := combination 5 3 in
  let ways_A_B := combination 3 1 in
  (ways_A_B : ℚ) / total_ways = 3 / 10 :=
by
  sorry

end probability_A_B_selected_l129_129430


namespace unit_stratified_sampling_l129_129597

theorem unit_stratified_sampling 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (selected_elderly : ℕ)
  (total : ℕ) (n : ℕ)
  (h1 : elderly = 27)
  (h2 : middle_aged = 54)
  (h3 : young = 81)
  (h4 : selected_elderly = 3)
  (h5 : total = elderly + middle_aged + young)
  (h6 : 3 / 27 = selected_elderly / elderly)
  (h7 : n / total = selected_elderly / elderly) : 
  n = 18 := 
by
  sorry

end unit_stratified_sampling_l129_129597


namespace probability_AB_selected_l129_129381

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129381


namespace solve_equation_l129_129714

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l129_129714


namespace equilateral_triangle_of_arith_geo_seq_l129_129680

def triangle (A B C : ℝ) (a b c : ℝ) (α β γ : ℝ) :=
  (α + β + γ = Real.pi) ∧
  (2 * β = α + γ) ∧
  (b^2 = a * c)

theorem equilateral_triangle_of_arith_geo_seq
  (A B C : ℝ) (a b c α β γ : ℝ)
  (h1 : triangle A B C a b c α β γ)
  : (a = c) ∧ (A = B) ∧ (B = C) ∧ (a = b) :=
  sorry

end equilateral_triangle_of_arith_geo_seq_l129_129680


namespace sum_of_roots_of_quadratic_l129_129089

theorem sum_of_roots_of_quadratic (m n : ℝ) (h1 : m = 2 * n) (h2 : ∀ x : ℝ, x ^ 2 + m * x + n = 0) :
    m + n = 3 / 2 :=
sorry

end sum_of_roots_of_quadratic_l129_129089


namespace probability_A_wins_championship_distribution_and_expectation_B_l129_129699

noncomputable def prob_event_1 : ℝ := 0.5
noncomputable def prob_event_2 : ℝ := 0.4
noncomputable def prob_event_3 : ℝ := 0.8

noncomputable def prob_A_wins_all : ℝ := prob_event_1 * prob_event_2 * prob_event_3
noncomputable def prob_A_wins_exactly_2 : ℝ :=
  prob_event_1 * prob_event_2 * (1 - prob_event_3) +
  prob_event_1 * (1 - prob_event_2) * prob_event_3 +
  (1 - prob_event_1) * prob_event_2 * prob_event_3

noncomputable def prob_A_wins_champ : ℝ := prob_A_wins_all + prob_A_wins_exactly_2

theorem probability_A_wins_championship : prob_A_wins_champ = 0.6 := by
  sorry

noncomputable def prob_B_wins_0 : ℝ := prob_A_wins_all
noncomputable def prob_B_wins_1 : ℝ := prob_event_1 * (1 - prob_event_2) * (1 - prob_event_3) +
                                        (1 - prob_event_1) * prob_event_2 * (1 - prob_event_3) +
                                        (1 - prob_event_1) * (1 - prob_event_2) * prob_event_3
noncomputable def prob_B_wins_2 : ℝ := (1 - prob_event_1) * prob_event_2 * prob_event_3 +
                                        prob_event_1 * (1 - prob_event_2) * prob_event_3 + 
                                        prob_event_1 * prob_event_2 * (1 - prob_event_3)
noncomputable def prob_B_wins_3 : ℝ := (1 - prob_event_1) * (1 - prob_event_2) * (1 - prob_event_3)

noncomputable def expected_score_B : ℝ :=
  0 * prob_B_wins_0 + 10 * prob_B_wins_1 +
  20 * prob_B_wins_2 + 30 * prob_B_wins_3

theorem distribution_and_expectation_B : 
  prob_B_wins_0 = 0.16 ∧
  prob_B_wins_1 = 0.44 ∧
  prob_B_wins_2 = 0.34 ∧
  prob_B_wins_3 = 0.06 ∧
  expected_score_B = 13 := by
  sorry

end probability_A_wins_championship_distribution_and_expectation_B_l129_129699


namespace ephraim_keiko_same_heads_probability_l129_129846

def coin_toss_probability_same_heads : ℚ :=
  let keiko_prob_0 := 1 / 4
  let keiko_prob_1 := 1 / 2
  let keiko_prob_2 := 1 / 4
  let ephraim_prob_0 := 1 / 8
  let ephraim_prob_1 := 3 / 8
  let ephraim_prob_2 := 3 / 8
  let ephraim_prob_3 := 1 / 8
  (keiko_prob_0 * ephraim_prob_0) 
  + (keiko_prob_1 * ephraim_prob_1) 
  + (keiko_prob_2 * ephraim_prob_2)

theorem ephraim_keiko_same_heads_probability : 
  coin_toss_probability_same_heads = 11 / 32 :=
by 
  unfold coin_toss_probability_same_heads
  norm_num
  sorry

end ephraim_keiko_same_heads_probability_l129_129846


namespace time_to_decorate_l129_129989

variable (mia_rate billy_rate total_eggs : ℕ)

theorem time_to_decorate (h_mia : mia_rate = 24) (h_billy : billy_rate = 10) (h_total : total_eggs = 170) :
  total_eggs / (mia_rate + billy_rate) = 5 :=
by
  sorry

end time_to_decorate_l129_129989


namespace factorization_of_polynomial_l129_129264

theorem factorization_of_polynomial (x : ℝ) : 16 * x ^ 2 - 40 * x + 25 = (4 * x - 5) ^ 2 := by 
  sorry

end factorization_of_polynomial_l129_129264


namespace probability_both_A_and_B_selected_l129_129449

-- The universe of students we consider
inductive Student
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

open Student

-- Function to count combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- The set of all students
def students : List Student := [A, B, C, D, E]

-- The total number of ways to select 3 students from 5
def total_selections := combinations 5 3

-- The number of ways A and B are both selected by selecting 1 more from the remaining 3 students
def selections_AB := combinations 3 1

-- The probability that both A and B are selected
def probability_AB := (selections_AB : ℚ) / (total_selections : ℚ)

-- The proof statement
theorem probability_both_A_and_B_selected :
  probability_AB = 3 / 10 :=
sorry

end probability_both_A_and_B_selected_l129_129449


namespace probability_AB_selected_l129_129380

def factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)

def choose (n k : Nat) : Nat := factorial n / (factorial k * factorial (n - k))

theorem probability_AB_selected (h1 : 5 = 5) (h2 : 3 = 3) : (choose 3 1) / (choose 5 3) = 3 / 10 := by
  sorry

end probability_AB_selected_l129_129380


namespace top_four_cards_probability_l129_129234

def num_cards : ℕ := 52

def num_hearts : ℕ := 13

def num_diamonds : ℕ := 13

def num_clubs : ℕ := 13

def prob_first_heart := (num_hearts : ℚ) / num_cards
def prob_second_heart := (num_hearts - 1 : ℚ) / (num_cards - 1)
def prob_third_diamond := (num_diamonds : ℚ) / (num_cards - 2)
def prob_fourth_club := (num_clubs : ℚ) / (num_cards - 3)

def combined_prob :=
  prob_first_heart * prob_second_heart * prob_third_diamond * prob_fourth_club

theorem top_four_cards_probability :
  combined_prob = 39 / 63875 := by
  sorry

end top_four_cards_probability_l129_129234


namespace school_adding_seats_l129_129764

theorem school_adding_seats (row_seats : ℕ) (seat_cost : ℕ) (discount_rate : ℝ) (total_cost : ℕ) (n : ℕ) 
                         (total_seats : ℕ) (discounted_seat_cost : ℕ)
                         (total_groups : ℕ) (rows : ℕ) :
  row_seats = 8 →
  seat_cost = 30 →
  discount_rate = 0.10 →
  total_cost = 1080 →
  discounted_seat_cost = seat_cost * (1 - discount_rate) →
  total_seats = total_cost / discounted_seat_cost →
  total_groups = total_seats / 10 →
  rows = total_seats / row_seats →
  rows = 5 :=
by
  intros hrowseats hseatcost hdiscountrate htotalcost hdiscountedseatcost htotalseats htotalgroups hrows
  sorry

end school_adding_seats_l129_129764


namespace probability_of_selecting_A_and_B_l129_129467

open_locale classical

-- Define the number of students
def num_students := 5

-- Define the number of students to be selected
def students_to_select := 3

def combination (n k : ℕ) : ℕ := nat.choose n k

-- Define the correct answer
def correct_probability := 3 / 10

-- Define the probability calculation
def calc_probability := (combination 3 1) / (combination 5 3)

-- Statement of the theorem to be proved
theorem probability_of_selecting_A_and_B :
  calc_probability = correct_probability :=
by sorry

end probability_of_selecting_A_and_B_l129_129467
