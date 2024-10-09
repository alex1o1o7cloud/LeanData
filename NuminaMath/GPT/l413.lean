import Mathlib

namespace calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l413_41372

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

end calculation_a_squared_plus_b_squared_calculation_a_minus_b_squared_l413_41372


namespace inches_of_rain_received_so_far_l413_41389

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

end inches_of_rain_received_so_far_l413_41389


namespace factorize_expression_l413_41332

theorem factorize_expression (m : ℝ) : 
  4 * m^2 - 64 = 4 * (m + 4) * (m - 4) :=
sorry

end factorize_expression_l413_41332


namespace triangle_is_isosceles_right_l413_41317

theorem triangle_is_isosceles_right (a b c : ℝ) (A B C : ℝ) (h1 : b = a * Real.sin C) (h2 : c = a * Real.cos B) : 
  A = π / 2 ∧ b = c := 
sorry

end triangle_is_isosceles_right_l413_41317


namespace find_min_value_l413_41338

noncomputable def min_value (x y z : ℝ) : ℝ :=
  x^4 * y^3 * z^2

theorem find_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (cond : 2/x + 3/y + 5/z = 10) : min_value x y z = 390625 / 1296 :=
sorry

end find_min_value_l413_41338


namespace polygon_sides_l413_41360

theorem polygon_sides (s : ℕ) (h : 180 * (s - 2) = 720) : s = 6 :=
by
  sorry

end polygon_sides_l413_41360


namespace recipe_butter_per_cup_l413_41327

theorem recipe_butter_per_cup (coconut_oil_to_butter_substitution : ℝ)
  (remaining_butter : ℝ)
  (planned_baking_mix : ℝ)
  (used_coconut_oil : ℝ)
  (butter_per_cup : ℝ)
  (h1 : coconut_oil_to_butter_substitution = 1)
  (h2 : remaining_butter = 4)
  (h3 : planned_baking_mix = 6)
  (h4 : used_coconut_oil = 8) :
  butter_per_cup = 4 / 3 := 
by 
  sorry

end recipe_butter_per_cup_l413_41327


namespace stamps_total_l413_41370

def Lizette_stamps : ℕ := 813
def Minerva_stamps : ℕ := Lizette_stamps - 125
def Jermaine_stamps : ℕ := Lizette_stamps + 217

def total_stamps : ℕ := Minerva_stamps + Lizette_stamps + Jermaine_stamps

theorem stamps_total :
  total_stamps = 2531 := by
  sorry

end stamps_total_l413_41370


namespace agnes_weekly_hours_l413_41374

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

end agnes_weekly_hours_l413_41374


namespace find_n_l413_41308

-- Define the polynomial function
def polynomial (n : ℤ) : ℤ :=
  n^4 + 2 * n^3 + 6 * n^2 + 12 * n + 25

-- Define the condition that n is a positive integer
def is_positive_integer (n : ℤ) : Prop :=
  n > 0

-- Define the condition that polynomial is a perfect square
def is_perfect_square (k : ℤ) : Prop :=
  ∃ m : ℤ, m^2 = k

-- The theorem we need to prove
theorem find_n (n : ℤ) (h1 : is_positive_integer n) (h2 : is_perfect_square (polynomial n)) : n = 8 :=
sorry

end find_n_l413_41308


namespace Jackie_apples_count_l413_41385

variable (Adam_apples Jackie_apples : ℕ)

-- Conditions
axiom Adam_has_14_apples : Adam_apples = 14
axiom Adam_has_5_more_than_Jackie : Adam_apples = Jackie_apples + 5

-- Theorem to prove
theorem Jackie_apples_count : Jackie_apples = 9 := by
  -- Use the conditions to derive the answer
  sorry

end Jackie_apples_count_l413_41385


namespace probability_five_dice_same_l413_41311

-- Define a function that represents the probability problem
noncomputable def probability_all_dice_same : ℚ :=
  (1 / 6) * (1 / 6) * (1 / 6) * (1 / 6)

-- The main theorem to state the proof problem
theorem probability_five_dice_same : probability_all_dice_same = 1 / 1296 :=
by
  sorry

end probability_five_dice_same_l413_41311


namespace third_stick_length_l413_41382

theorem third_stick_length (x : ℝ) (h1 : 2 > 0) (h2 : 5 > 0) (h3 : 3 < x) (h4 : x < 7) : x = 4 :=
by
  sorry

end third_stick_length_l413_41382


namespace smallest_m_exists_l413_41309

theorem smallest_m_exists :
  ∃ (m : ℕ), 0 < m ∧ (∃ k : ℕ, 5 * m = k^2) ∧ (∃ l : ℕ, 3 * m = l^3) ∧ m = 243 :=
by
  sorry

end smallest_m_exists_l413_41309


namespace pow2_gt_square_for_all_n_ge_5_l413_41335

theorem pow2_gt_square_for_all_n_ge_5 (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
by
  sorry

end pow2_gt_square_for_all_n_ge_5_l413_41335


namespace find_divisor_l413_41376

theorem find_divisor
  (n : ℕ) (h1 : n > 0)
  (h2 : (n + 1) % 6 = 4)
  (h3 : ∃ d : ℕ, n % d = 1) :
  ∃ d : ℕ, (n % d = 1) ∧ d = 2 :=
by
  sorry

end find_divisor_l413_41376


namespace perpendicular_vectors_l413_41391

noncomputable def a (k : ℝ) : ℝ × ℝ := (2 * k - 4, 3)
noncomputable def b (k : ℝ) : ℝ × ℝ := (-3, k)

theorem perpendicular_vectors (k : ℝ) (h : (2 * k - 4) * (-3) + 3 * k = 0) : k = 4 :=
sorry

end perpendicular_vectors_l413_41391


namespace var_X_is_86_over_225_l413_41383

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

end var_X_is_86_over_225_l413_41383


namespace valid_rearrangements_count_l413_41386

noncomputable def count_valid_rearrangements : ℕ := sorry

theorem valid_rearrangements_count :
  count_valid_rearrangements = 7 :=
sorry

end valid_rearrangements_count_l413_41386


namespace relationship_m_n_l413_41315

variable (a b : ℝ)
variable (m n : ℝ)

theorem relationship_m_n (h1 : a > b) (h2 : b > 0) (hm : m = Real.sqrt a - Real.sqrt b) (hn : n = Real.sqrt (a - b)) : m < n := sorry

end relationship_m_n_l413_41315


namespace function_value_proof_l413_41363

theorem function_value_proof (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : ∀ x, f (x + 1) = -f (-x + 1))
    (h2 : ∀ x, f (x + 2) = f (-x + 2))
    (h3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b)
    (h4 : ∀ x y : ℝ, x - y - 3 = 0)
    : f (9/2) = 5/4 := by
  sorry

end function_value_proof_l413_41363


namespace triangle_is_right_triangle_l413_41371

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

end triangle_is_right_triangle_l413_41371


namespace charles_remaining_skittles_l413_41302

def c : ℕ := 25
def d : ℕ := 7
def remaining_skittles : ℕ := c - d

theorem charles_remaining_skittles : remaining_skittles = 18 := by
  sorry

end charles_remaining_skittles_l413_41302


namespace intersection_point_of_lines_l413_41330

theorem intersection_point_of_lines :
  ∃ x y : ℝ, 
    2 * x + y - 7 = 0 ∧ 
    x + 2 * y - 5 = 0 ∧ 
    x = 3 ∧ 
    y = 1 := 
by {
  sorry
}

end intersection_point_of_lines_l413_41330


namespace prime_arithmetic_sequence_l413_41344

theorem prime_arithmetic_sequence {p1 p2 p3 d : ℕ} 
  (hp1 : Nat.Prime p1) 
  (hp2 : Nat.Prime p2) 
  (hp3 : Nat.Prime p3)
  (h3_p1 : 3 < p1)
  (h3_p2 : 3 < p2)
  (h3_p3 : 3 < p3)
  (h_seq1 : p2 = p1 + d)
  (h_seq2 : p3 = p1 + 2 * d) : 
  d % 6 = 0 :=
by sorry

end prime_arithmetic_sequence_l413_41344


namespace constant_term_of_second_eq_l413_41307

theorem constant_term_of_second_eq (x y : ℝ) 
  (h1 : 7*x + y = 19) 
  (h2 : 2*x + y = 5) : 
  ∃ k : ℝ, x + 3*y = k ∧ k = 15 := 
by
  sorry

end constant_term_of_second_eq_l413_41307


namespace binomial_square_formula_l413_41388

theorem binomial_square_formula (a b : ℝ) :
  let e1 := (4 * a + b) * (4 * a - 2 * b)
  let e2 := (a - 2 * b) * (2 * b - a)
  let e3 := (2 * a - b) * (-2 * a + b)
  let e4 := (a - b) * (a + b)
  (e4 = a^2 - b^2) :=
by
  sorry

end binomial_square_formula_l413_41388


namespace sum_infinite_geometric_series_l413_41384

theorem sum_infinite_geometric_series :
  ∑' (n : ℕ), (3 : ℝ) * ((1 / 3) ^ n) = (9 / 2 : ℝ) :=
sorry

end sum_infinite_geometric_series_l413_41384


namespace xz_less_than_half_l413_41377

theorem xz_less_than_half (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : xy + yz + zx = 1) : x * z < 1 / 2 :=
  sorry

end xz_less_than_half_l413_41377


namespace value_of_frac_sum_l413_41396

theorem value_of_frac_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : (x + y) / 3 = 11 / 9 :=
by
  sorry

end value_of_frac_sum_l413_41396


namespace problem_statement_l413_41397

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem problem_statement : f (g (-3)) = 961 := by
  sorry

end problem_statement_l413_41397


namespace two_digit_solution_l413_41379

def two_digit_number (x y : ℕ) : ℕ := 10 * x + y

theorem two_digit_solution :
  ∃ (x y : ℕ), 
    two_digit_number x y = 24 ∧ 
    two_digit_number x y = x^3 + y^2 ∧ 
    0 ≤ x ∧ x ≤ 9 ∧ 
    0 ≤ y ∧ y ≤ 9 :=
by
  sorry

end two_digit_solution_l413_41379


namespace sector_perimeter_ratio_l413_41380

theorem sector_perimeter_ratio (α : ℝ) (r R : ℝ) 
  (h1 : α > 0) 
  (h2 : r > 0) 
  (h3 : R > 0) 
  (h4 : (1/2) * α * r^2 / ((1/2) * α * R^2) = 1/4) :
  (2 * r + α * r) / (2 * R + α * R) = 1 / 2 := 
sorry

end sector_perimeter_ratio_l413_41380


namespace Foster_Farms_donated_45_chickens_l413_41353

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

end Foster_Farms_donated_45_chickens_l413_41353


namespace prob_allergic_prescribed_l413_41368

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

end prob_allergic_prescribed_l413_41368


namespace side_c_possibilities_l413_41364

theorem side_c_possibilities (A : ℝ) (a b c : ℝ) (hA : A = 30) (ha : a = 4) (hb : b = 4 * Real.sqrt 3) :
  c = 4 ∨ c = 8 :=
sorry

end side_c_possibilities_l413_41364


namespace problem_part1_problem_part2_l413_41387

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

end problem_part1_problem_part2_l413_41387


namespace rotated_curve_eq_l413_41301

theorem rotated_curve_eq :
  let θ := Real.pi / 4  -- Rotation angle 45 degrees in radians
  let cos_theta := Real.sqrt 2 / 2
  let sin_theta := Real.sqrt 2 / 2
  let x' := cos_theta * x - sin_theta * y
  let y' := sin_theta * x + cos_theta * y
  x + y^2 = 1 → x' ^ 2 + y' ^ 2 - 2 * x' * y' + Real.sqrt 2 * x' + Real.sqrt 2 * y' - 2 = 0 := 
sorry  -- Proof to be provided.

end rotated_curve_eq_l413_41301


namespace strictly_positive_integer_le_36_l413_41336

theorem strictly_positive_integer_le_36 (n : ℕ) (h_pos : n > 0) :
  (∀ a : ℤ, (a % 2 = 1) → (a * a ≤ n) → (a ∣ n)) → n ≤ 36 := by
  sorry

end strictly_positive_integer_le_36_l413_41336


namespace satisfying_lines_l413_41390

theorem satisfying_lines (x y : ℝ) : (y^2 - 2*y = x^2 + 2*x) ↔ (y = x + 2 ∨ y = -x) :=
by
  sorry

end satisfying_lines_l413_41390


namespace shadow_boundary_eqn_l413_41346

noncomputable def boundary_of_shadow (x : ℝ) : ℝ := x^2 / 10 - 1

theorem shadow_boundary_eqn (radius : ℝ) (center : ℝ × ℝ × ℝ) (light_source : ℝ × ℝ × ℝ) (x y: ℝ) :
  radius = 2 →
  center = (0, 0, 2) →
  light_source = (0, -2, 3) →
  y = boundary_of_shadow x :=
by
  intros hradius hcenter hlight
  sorry

end shadow_boundary_eqn_l413_41346


namespace abc_sum_eq_sixteen_l413_41373

theorem abc_sum_eq_sixteen (a b c : ℤ) (h1 : a ≠ b ∨ a ≠ c ∨ b ≠ c) (h2 : a ≥ 4 ∧ b ≥ 4 ∧ c ≥ 4) (h3 : 4 * a * b * c = (a + 3) * (b + 3) * (c + 3)) : a + b + c = 16 :=
by 
  sorry

end abc_sum_eq_sixteen_l413_41373


namespace option_d_correct_l413_41347

theorem option_d_correct (x : ℝ) : (-3 * x + 2) * (-3 * x - 2) = 9 * x^2 - 4 := 
  sorry

end option_d_correct_l413_41347


namespace geometric_sequence_const_k_l413_41333

noncomputable def sum_of_terms (n : ℕ) (k : ℤ) : ℤ := 3 * 2^n + k
noncomputable def a1 (k : ℤ) : ℤ := sum_of_terms 1 k
noncomputable def a2 (k : ℤ) : ℤ := sum_of_terms 2 k - sum_of_terms 1 k
noncomputable def a3 (k : ℤ) : ℤ := sum_of_terms 3 k - sum_of_terms 2 k

theorem geometric_sequence_const_k :
  (∀ (k : ℤ), (a1 k * a3 k = a2 k * a2 k) → k = -3) :=
by
  sorry

end geometric_sequence_const_k_l413_41333


namespace seats_selection_l413_41304

theorem seats_selection (n k d : ℕ) (hn : n ≥ 4) (hk : k ≥ 2) (hd : d ≥ 2) (hkd : k * d ≤ n) :
  ∃ ways : ℕ, ways = (n / k) * Nat.choose (n - k * d + k - 1) (k - 1) :=
sorry

end seats_selection_l413_41304


namespace inequality_solution_l413_41365

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (3 * x - 1) > 1 ↔ (1 / 3 < x ∧ x < 1) :=
sorry

end inequality_solution_l413_41365


namespace average_mark_of_excluded_students_l413_41305

theorem average_mark_of_excluded_students 
  (N A E A_remaining : ℕ) 
  (hN : N = 25) 
  (hA : A = 80) 
  (hE : E = 5) 
  (hA_remaining : A_remaining = 95) : 
  ∃ A_excluded : ℕ, A_excluded = 20 :=
by
  -- Use the conditions in the proof.
  sorry

end average_mark_of_excluded_students_l413_41305


namespace central_angle_radian_measure_l413_41354

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

end central_angle_radian_measure_l413_41354


namespace total_pints_l413_41343

variables (Annie Kathryn Ben Sam : ℕ)

-- Conditions
def condition1 := Annie = 16
def condition2 (Annie : ℕ) := Kathryn = 2 * Annie + 2
def condition3 (Kathryn : ℕ) := Ben = Kathryn / 2 - 3
def condition4 (Ben Kathryn : ℕ) := Sam = 2 * (Ben + Kathryn) / 3

-- Statement to prove
theorem total_pints (Annie Kathryn Ben Sam : ℕ) 
  (h1 : condition1 Annie) 
  (h2 : condition2 Annie Kathryn) 
  (h3 : condition3 Kathryn Ben) 
  (h4 : condition4 Ben Kathryn Sam) : 
  Annie + Kathryn + Ben + Sam = 96 :=
sorry

end total_pints_l413_41343


namespace solution_set_ineq1_solution_set_ineq2_l413_41319

theorem solution_set_ineq1 (x : ℝ) : 
  (-3 * x ^ 2 + x + 1 > 0) ↔ (x ∈ Set.Ioo ((1 - Real.sqrt 13) / 6) ((1 + Real.sqrt 13) / 6)) := 
sorry

theorem solution_set_ineq2 (x : ℝ) : 
  (x ^ 2 - 2 * x + 1 ≤ 0) ↔ (x = 1) := 
sorry

end solution_set_ineq1_solution_set_ineq2_l413_41319


namespace find_digits_l413_41351

theorem find_digits :
  ∃ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1098) :=
by {
  sorry
}

end find_digits_l413_41351


namespace min_value_frac_add_x_l413_41316

theorem min_value_frac_add_x (x : ℝ) (h : x > 3) : (∃ m, (∀ (y : ℝ), y > 3 → (4 / y - 3 + y) ≥ m) ∧ m = 7) :=
sorry

end min_value_frac_add_x_l413_41316


namespace correct_assignment_statement_l413_41366

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

end correct_assignment_statement_l413_41366


namespace ellipse_eccentricity_l413_41318

theorem ellipse_eccentricity :
  (∃ (e : ℝ), (∀ (x y : ℝ), ((x^2 / 9) + y^2 = 1) → (e = 2 * Real.sqrt 2 / 3))) :=
by
  sorry

end ellipse_eccentricity_l413_41318


namespace equal_real_roots_possible_values_l413_41310

theorem equal_real_roots_possible_values (a : ℝ): 
  (∀ x : ℝ, x^2 + a * x + 1 = 0) → (a = 2 ∨ a = -2) :=
by
  sorry

end equal_real_roots_possible_values_l413_41310


namespace feed_days_l413_41392

theorem feed_days (morning_food evening_food total_food : ℕ) (h1 : morning_food = 1) (h2 : evening_food = 1) (h3 : total_food = 32)
: (total_food / (morning_food + evening_food)) = 16 := by
  sorry

end feed_days_l413_41392


namespace solve_quadratic_eq_l413_41352

theorem solve_quadratic_eq (x : ℝ) :
  x^2 + 4 * x + 2 = 0 ↔ (x = -2 + Real.sqrt 2 ∨ x = -2 - Real.sqrt 2) :=
by
  -- This is a statement only. No proof is required.
  sorry

end solve_quadratic_eq_l413_41352


namespace coordinates_of_B_l413_41356

-- Definitions of the points and vectors are given as conditions.
def A : ℝ × ℝ := (-1, -1)
def a : ℝ × ℝ := (2, 3)

-- Statement of the problem translated to Lean
theorem coordinates_of_B (B : ℝ × ℝ) (h : B = (5, 8)) :
  (B.1 + 1, B.2 + 1) = (3 * a.1, 3 * a.2) :=
sorry

end coordinates_of_B_l413_41356


namespace part_one_solution_part_two_solution_l413_41357

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part (1): "When a = 1, find the solution set of the inequality f(x) ≥ 3"
theorem part_one_solution (x : ℝ) : f x 1 ≥ 3 ↔ x ≤ 0 ∨ x ≥ 3 :=
by sorry

-- Part (2): "If f(x) ≥ 2a - 1, find the range of values for a"
theorem part_two_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ 2 * a - 1) ↔ a ≤ 1 :=
by sorry

end part_one_solution_part_two_solution_l413_41357


namespace claudia_ratio_of_kids_l413_41320

def claudia_art_class :=
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  sunday_kids / saturday_kids = 1 / 2

theorem claudia_ratio_of_kids :
  let saturday_kids := 20
  let sunday_kids := (300 - saturday_kids * 10) / 10
  (sunday_kids / saturday_kids = 1 / 2) :=
by
  sorry

end claudia_ratio_of_kids_l413_41320


namespace parallel_implies_not_contained_l413_41340

variables {Line Plane : Type} (l : Line) (α : Plane)

-- Define the predicate for a line being parallel to a plane
def parallel (l : Line) (α : Plane) : Prop := sorry

-- Define the predicate for a line not being contained in a plane
def not_contained (l : Line) (α : Plane) : Prop := sorry

theorem parallel_implies_not_contained (l : Line) (α : Plane) (h : parallel l α) : not_contained l α :=
sorry

end parallel_implies_not_contained_l413_41340


namespace number_of_people_in_room_l413_41399

theorem number_of_people_in_room (P : ℕ) 
  (h1 : 1/4 * P = P / 4) 
  (h2 : 3/4 * P = 3 * P / 4) 
  (h3 : P / 4 = 20) : 
  P = 80 :=
sorry

end number_of_people_in_room_l413_41399


namespace cos_alpha_value_l413_41378

theorem cos_alpha_value
  (a : ℝ) (h1 : π < a ∧ a < 3 * π / 2)
  (h2 : Real.tan a = 2) :
  Real.cos a = - (Real.sqrt 5) / 5 :=
sorry

end cos_alpha_value_l413_41378


namespace avery_egg_cartons_l413_41393

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end avery_egg_cartons_l413_41393


namespace area_triangle_ABC_l413_41348

-- Definitions of the lengths and height
def BD : ℝ := 3
def DC : ℝ := 2 * BD
def BC : ℝ := BD + DC
def h_A_BC : ℝ := 4

-- The triangle area formula
def areaOfTriangle (base height : ℝ) : ℝ := 0.5 * base * height

-- The goal to prove that the area of triangle ABC is 18 square units
theorem area_triangle_ABC : areaOfTriangle BC h_A_BC = 18 := by
  sorry

end area_triangle_ABC_l413_41348


namespace tan_alpha_plus_pi_div_four_l413_41358

theorem tan_alpha_plus_pi_div_four (α : ℝ) (h : (3 * Real.sin α + 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 3) : 
  Real.tan (α + Real.pi / 4) = -3 := 
by 
  sorry

end tan_alpha_plus_pi_div_four_l413_41358


namespace incorrect_axis_symmetry_l413_41362

noncomputable def quadratic_function (x : ℝ) : ℝ := - (x + 2)^2 - 3

theorem incorrect_axis_symmetry :
  (∀ x : ℝ, quadratic_function x < 0) ∧
  (∀ x : ℝ, x > -1 → (quadratic_function x < quadratic_function (-2))) ∧
  (¬∃ x : ℝ, quadratic_function x = 0) ∧
  (¬ ∀ x : ℝ, x = 2) →
  false :=
by
  sorry

end incorrect_axis_symmetry_l413_41362


namespace fraction_arithmetic_l413_41334

theorem fraction_arithmetic :
  (3 / 4) / (5 / 8) + (1 / 8) = 53 / 40 :=
by
  sorry

end fraction_arithmetic_l413_41334


namespace sum_of_radical_conjugates_l413_41329

theorem sum_of_radical_conjugates : 
  (8 - Real.sqrt 1369) + (8 + Real.sqrt 1369) = 16 :=
by
  sorry

end sum_of_radical_conjugates_l413_41329


namespace simplify_poly_l413_41323

-- Define the polynomial expressions
def poly1 (r : ℝ) := 2 * r^3 + 4 * r^2 + 5 * r - 3
def poly2 (r : ℝ) := r^3 + 6 * r^2 + 8 * r - 7

-- Simplification goal
theorem simplify_poly (r : ℝ) : (poly1 r) - (poly2 r) = r^3 - 2 * r^2 - 3 * r + 4 :=
by 
  -- We declare the proof is omitted using sorry
  sorry

end simplify_poly_l413_41323


namespace range_of_2a_minus_b_l413_41342

theorem range_of_2a_minus_b (a b : ℝ) (h1 : 1 < a) (h2 : a < 3) (h3 : 2 < b) (h4 : b < 4) : 
  -2 < 2 * a - b ∧ 2 * a - b < 4 := 
by 
  sorry

end range_of_2a_minus_b_l413_41342


namespace statement_A_statement_B_statement_C_l413_41331

variable {a b : ℝ}
variable (ha : a > 0) (hb : b > 0)

theorem statement_A : (ab ≤ 1) → (1/a + 1/b ≥ 2) :=
by
  sorry

theorem statement_B : (a + b = 4) → (∀ x, (x = 1/a + 9/b) → (x ≥ 4)) :=
by
  sorry

theorem statement_C : (a^2 + b^2 = 4) → (ab ≤ 2) :=
by
  sorry

end statement_A_statement_B_statement_C_l413_41331


namespace find_the_number_l413_41369

theorem find_the_number 
  (x y n : ℤ)
  (h : 19 * (x + y) + 17 = 19 * (-x + y) - n)
  (hx : x = 1) :
  n = -55 :=
by
  sorry

end find_the_number_l413_41369


namespace min_u_value_l413_41322

theorem min_u_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 2 * y = 1) : 
  (x + 1 / x) * (y + 1 / (4 * y)) ≥ 25 / 8 :=
by
  sorry

end min_u_value_l413_41322


namespace arithmetic_sequence_sum_l413_41321

theorem arithmetic_sequence_sum :
  ∃ (a : ℕ → ℝ) (d : ℝ), 
  (∀ n, a n = a 0 + n * d) ∧ 
  (∃ b c, b^2 - 6*b + 5 = 0 ∧ c^2 - 6*c + 5 = 0 ∧ a 3 = b ∧ a 15 = c) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 :=
by
  sorry

end arithmetic_sequence_sum_l413_41321


namespace christine_needs_32_tablespoons_l413_41345

-- Define the conditions
def tablespoons_per_egg_white : ℕ := 2
def egg_whites_per_cake : ℕ := 8
def number_of_cakes : ℕ := 2

-- Define the calculation for total tablespoons of aquafaba needed
def total_tbs_aquafaba : ℕ :=
  tablespoons_per_egg_white * (egg_whites_per_cake * number_of_cakes)

-- The theorem to prove
theorem christine_needs_32_tablespoons :
  total_tbs_aquafaba = 32 :=
by 
  -- Placeholder for proof, as proof steps are not required
  sorry

end christine_needs_32_tablespoons_l413_41345


namespace can_invent_1001_sad_stories_l413_41350

-- Definitions
def is_natural (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 17

def is_sad_story (a b c : ℕ) : Prop :=
  ∀ x y : ℤ, a * x + b * y ≠ c

-- The Statement
theorem can_invent_1001_sad_stories :
  ∃ stories : Finset (ℕ × ℕ × ℕ),
    (∀ (a b c : ℕ), (a, b, c) ∈ stories → is_natural a ∧ is_natural b ∧ is_natural c ∧ is_sad_story a b c) ∧
    stories.card ≥ 1001 :=
by
  sorry

end can_invent_1001_sad_stories_l413_41350


namespace inequality_proof_l413_41300

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  1 / (b * c + a + 1 / a) + 1 / (a * c + b + 1 / b) + 1 / (a * b + c + 1 / c) ≤ 27 / 31 :=
sorry

end inequality_proof_l413_41300


namespace max_value_function_l413_41339

theorem max_value_function (x : ℝ) (h : x > 4) : -x + (1 / (4 - x)) ≤ -6 :=
sorry

end max_value_function_l413_41339


namespace moles_of_C2H6_formed_l413_41341

-- Definitions of the quantities involved
def moles_H2 : ℕ := 3
def moles_C2H4 : ℕ := 3
def moles_C2H6 : ℕ := 3

-- Stoichiometry condition stated in a way that Lean can understand.
axiom stoichiometry : moles_H2 = moles_C2H4

theorem moles_of_C2H6_formed : moles_C2H6 = 3 :=
by
  -- Assume the constraints and state the final result
  have h : moles_H2 = moles_C2H4 := stoichiometry
  show moles_C2H6 = 3
  sorry

end moles_of_C2H6_formed_l413_41341


namespace rectangle_ratio_l413_41359

theorem rectangle_ratio (a b c d : ℝ) (h₀ : a = 4)
  (h₁ : b = (4 / 3)) (h₂ : c = (8 / 3)) (h₃ : d = 4) :
  (∃ XY YZ, XY * YZ = a * a ∧ XY / YZ = 0.9) :=
by
  -- Proof to be filled
  sorry

end rectangle_ratio_l413_41359


namespace buying_beams_l413_41337

theorem buying_beams (x : ℕ) (h : 3 * (x - 1) * x = 6210) :
  3 * (x - 1) * x = 6210 :=
by {
  sorry
}

end buying_beams_l413_41337


namespace sues_answer_l413_41375

theorem sues_answer (x : ℕ) (hx : x = 6) : 
  let b := 2 * (x + 1)
  let s := 2 * (b - 1)
  s = 26 :=
by
  sorry

end sues_answer_l413_41375


namespace intersection_complement_l413_41303

def real_set_M : Set ℝ := {x | 1 < x}
def real_set_N : Set ℝ := {x | x > 4}

theorem intersection_complement (x : ℝ) : x ∈ (real_set_M ∩ (real_set_Nᶜ)) ↔ 1 < x ∧ x ≤ 4 :=
by
  sorry

end intersection_complement_l413_41303


namespace odd_function_behavior_l413_41312

variable {f : ℝ → ℝ}

theorem odd_function_behavior (h1 : ∀ x : ℝ, f (-x) = -f x) 
                             (h2 : ∀ x : ℝ, 0 < x → f x = x * (1 + x)) 
                             (x : ℝ)
                             (hx : x < 0) : 
  f x = x * (1 - x) :=
by
  -- Insert proof here
  sorry

end odd_function_behavior_l413_41312


namespace congruence_a_b_mod_1008_l413_41325

theorem congruence_a_b_mod_1008
  (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : a ^ b - b ^ a = 1008) : a ≡ b [MOD 1008] :=
sorry

end congruence_a_b_mod_1008_l413_41325


namespace wristwatch_cost_proof_l413_41395

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

end wristwatch_cost_proof_l413_41395


namespace ab_cd_zero_l413_41394

theorem ab_cd_zero {a b c d : ℝ} (h1 : a^2 + b^2 = 1) (h2 : c^2 + d^2 = 1) (h3 : ac + bd = 0) : ab + cd = 0 :=
sorry

end ab_cd_zero_l413_41394


namespace chalk_breaking_probability_l413_41367

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

end chalk_breaking_probability_l413_41367


namespace impossible_permuted_sum_l413_41306

def isPermutation (X Y : ℕ) : Prop :=
  -- Define what it means for two numbers to be permutations of each other.
  sorry

theorem impossible_permuted_sum (X Y : ℕ) (h1 : isPermutation X Y) (h2 : X + Y = (10^1111 - 1)) : false :=
  sorry

end impossible_permuted_sum_l413_41306


namespace system_of_linear_eq_l413_41361

theorem system_of_linear_eq :
  ∃ (x y : ℝ), x + y = 5 ∧ y = 2 :=
sorry

end system_of_linear_eq_l413_41361


namespace transform_equation_l413_41314

theorem transform_equation (x y : ℝ) (h : y = x + x⁻¹) :
  x^4 + x^3 - 5 * x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 := 
sorry

end transform_equation_l413_41314


namespace inequality_and_equality_l413_41398

variables {x y z : ℝ}

theorem inequality_and_equality (x y z : ℝ) :
  (x^2 + y^4 + z^6 >= x * y^2 + y^2 * z^3 + x * z^3) ∧ (x^2 + y^4 + z^6 = x * y^2 + y^2 * z^3 + x * z^3 ↔ x = y^2 ∧ y^2 = z^3) :=
by sorry

end inequality_and_equality_l413_41398


namespace hexagon_largest_angle_l413_41326

variable (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ)
theorem hexagon_largest_angle (h : a₁ = 3)
                             (h₀ : a₂ = 3)
                             (h₁ : a₃ = 3)
                             (h₂ : a₄ = 4)
                             (h₃ : a₅ = 5)
                             (h₄ : a₆ = 6)
                             (sum_angles : 3*a₁ + 3*a₀ + 3*a₁ + 4*a₂ + 5*a₃ + 6*a₄ = 720) :
                             6 * 30 = 180 := by
    sorry

end hexagon_largest_angle_l413_41326


namespace polygon_problem_l413_41313

theorem polygon_problem
  (sum_interior_angles : ℕ → ℝ)
  (sum_exterior_angles : ℝ)
  (condition : ∀ n, sum_interior_angles n = (3 * sum_exterior_angles) - 180) :
  (∃ n : ℕ, sum_interior_angles n = 180 * (n - 2) ∧ n = 7) ∧
  (∃ n : ℕ, n = 7 → (n * (n - 3) / 2) = 14) :=
by
  sorry

end polygon_problem_l413_41313


namespace detergent_for_9_pounds_l413_41381

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

end detergent_for_9_pounds_l413_41381


namespace soccer_ball_seams_l413_41355

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

end soccer_ball_seams_l413_41355


namespace sqrt_of_16_is_4_l413_41328

theorem sqrt_of_16_is_4 : Real.sqrt 16 = 4 :=
sorry

end sqrt_of_16_is_4_l413_41328


namespace range_of_a_l413_41349

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → a ≥ Real.exp x) ∧ (∃ x : ℝ, x^2 - 4 * x + a ≤ 0) →
  a ∈ Set.Icc (Real.exp 1) 4 :=
by
  sorry

end range_of_a_l413_41349


namespace find_fraction_l413_41324

theorem find_fraction (f : ℝ) (n : ℝ) (h : n = 180) (eqn : f * ((1 / 3) * (1 / 5) * n) + 6 = (1 / 15) * n) : f = 1 / 2 :=
by
  -- Definitions and assumptions provided above will be used here.
  sorry

end find_fraction_l413_41324
